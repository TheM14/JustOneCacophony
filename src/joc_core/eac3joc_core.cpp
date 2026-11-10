#define EJOC_BUILD_DLL
#include "eac3joc_core.h"
#include "qmf_tables.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include <vector>

namespace ejoc {

struct Complex {
    double re;
    double im;
};

inline Complex mul(const Complex a, const Complex b) noexcept {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

inline double clamp_unit(double value) noexcept {
    return value < -1.0 ? -1.0 : (value > 1.0 ? 1.0 : value);
}

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kLfeDelay = 1217;
constexpr int kMaxBands = EJOC_MAX_PARAMETER_BANDS;

const uint8_t* parameter_band_map(const int bands) noexcept {
    using namespace tables;
    switch (bands) {
    case 1: return kPbMap1;
    case 3: return kPbMap3;
    case 5: return kPbMap5;
    case 7: return kPbMap7;
    case 9: return kPbMap9;
    case 12: return kPbMap12;
    case 15: return kPbMap15;
    case 23: return kPbMap23;
    default: return nullptr;
    }
}

class Renderer final {
public:
    Renderer() noexcept {
        initialize_tables();
        reset();
    }

    ~Renderer() noexcept {
        stop_workers();
    }

    int reset() noexcept {
        std::memset(analysis_fifo_, 0, sizeof(analysis_fifo_));
        std::memset(analysis_delay_, 0, sizeof(analysis_delay_));
        std::memset(surround_delay_, 0, sizeof(surround_delay_));
        std::memset(surround_history_, 0, sizeof(surround_history_));
        std::memset(lfe_delay_, 0, sizeof(lfe_delay_));
        std::memset(matrix_previous_, 0, sizeof(matrix_previous_));
        std::memset(synthesis_state_, 0, sizeof(synthesis_state_));
        std::memset(x_, 0, sizeof(x_));
        std::memset(z_, 0, sizeof(z_));
        // Frame-level state: per-object previous matrices and presence markers,
        // the seq state machine, the 8-plane passthrough history, the LFE ramp
        // start value and the previous actual core-channel count.
        std::memset(prev_matrix_, 0, sizeof(prev_matrix_));
        std::memset(prev_present_, 0, sizeof(prev_present_));
        std::memset(core_history_, 0, sizeof(core_history_));
        state_ = 0;
        expected_seq_ = 0;
        invalid_count_ = 0;
        last_frame_valid_ = false;
        n_channels_prev_ = -1;
        has_lfe_clipgain_ = false;
        lfe_clipgain_ = 0.0;
        // Sentinel: the first frame after reset takes the new scalar directly,
        // without an intra-frame transition.
        analysis_phase_ = -0.5f;
        error_[0] = '\0';
        return 0;
    }

    int set_threads(uint32_t total_threads) noexcept {
        if (total_threads < 1) {
            total_threads = 1;
        }
        if (total_threads > EJOC_MAX_OBJECTS) {
            total_threads = EJOC_MAX_OBJECTS;
        }
        stop_workers();
        job_count_ = 0;
        next_job_.store(0, std::memory_order_relaxed);
        if (total_threads == 1) {
            return 0;
        }
        try {
            stop_.store(false, std::memory_order_relaxed);
            pool_ready_.store(false, std::memory_order_relaxed);
            work_barrier_ = std::make_unique<std::barrier<>>(static_cast<std::ptrdiff_t>(total_threads));
            workers_.reserve(total_threads - 1);
            for (uint32_t index = 1; index < total_threads; ++index) {
                workers_.emplace_back([this]() noexcept { worker_loop(); });
            }
            pool_ready_.store(true, std::memory_order_release);
            // Startup rendezvous: ensure every worker has entered the two-phase
            // barrier loop before set_threads returns, so an immediate destroy
            // cannot race a worker that exits before reaching the barrier.
            work_barrier_->arrive_and_wait();
            work_barrier_->arrive_and_wait();
        } catch (...) {
            stop_.store(true, std::memory_order_release);
            pool_ready_.store(true, std::memory_order_release);
            for (std::thread& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            workers_.clear();
            work_barrier_.reset();
            stop_.store(false, std::memory_order_relaxed);
            return fail("failed to create native worker threads");
        }
        return 0;
    }

    uint32_t thread_count() const noexcept {
        return static_cast<uint32_t>(workers_.size() + 1);
    }

    const char* error() const noexcept {
        return error_[0] ? error_ : "";
    }

    int process(
        const float* bed_planar,
        const float* lfe,
        const uint32_t core_channel_count,
        const uint32_t phase_shift,
        const uint32_t object_mask,
        const uint8_t* n_bands,
        const uint8_t* n_dpoints,
        const uint8_t* slope_idx,
        const uint8_t* offset_ts,
        const double* dq,
        const double clipgain,
        const float phase_new,
        const float output_scale,
        float* output16) noexcept {

        error_[0] = '\0';
        if (!bed_planar || !n_bands || !n_dpoints || !slope_idx || !offset_ts || !dq || !output16) {
            return fail("null pointer passed to ejoc_renderer_process");
        }
        if (core_channel_count < EJOC_CORE_CHANNELS || core_channel_count > EJOC_MAX_CORE_CHANNELS) {
            return fail("core_channel_count must be 5 or 7");
        }
        core_channels_ = core_channel_count;
        phase_shift_enabled_ = phase_shift != 0u;
        if (object_mask & ~((1u << EJOC_MAX_OBJECTS) - 1u)) {
            return fail("object_mask contains an object index above 14");
        }
        if (!std::isfinite(clipgain) || !std::isfinite(phase_new) || !std::isfinite(output_scale)) {
            return fail("clipgain, phase_new, and output_scale must be finite");
        }
        for (int object = 0; object < EJOC_MAX_OBJECTS; ++object) {
            if ((object_mask & (1u << object)) == 0) {
                continue;
            }
            if (!parameter_band_map(n_bands[object])) {
                return fail("unsupported parameter-band count");
            }
            if (n_dpoints[object] < 1 || n_dpoints[object] > 2) {
                return fail("n_dpoints must be 1 or 2");
            }
            if (slope_idx[object] > 1) {
                return fail("slope_idx must be 0 or 1");
            }
        }

        std::memset(output16, 0, sizeof(float) * EJOC_OUTPUT_CHANNELS * EJOC_FRAME_SAMPLES);
        // Analysis-side scalar = the structural 1/16 times this frame's clipgain.
        // clipgain enters the analysis at this stage and the x16 at the output
        // cancels the 1/16, so the net effect is a multiply by joc_clipgain.
        analysis(bed_planar, static_cast<float>(phase_new * clipgain));
        decode_lfe(lfe, clipgain, output16, output_scale);
        process_objects(object_mask, n_bands, n_dpoints, slope_idx, offset_ts,
                        dq, output_scale, output16);
        return 0;
    }

    // Frame-level entry point; the state machine and per-object history live
    // inside the renderer (see render_frame_impl()).
    int render_frame(const ejoc_frame_input& input) noexcept {
        return render_frame_impl(input);
    }

private:
    void initialize_tables() noexcept {
        for (int i = 0; i < 64; ++i) {
            int value = i;
            int reversed = 0;
            for (int bit = 0; bit < 6; ++bit) {
                reversed = (reversed << 1) | (value & 1);
                value >>= 1;
            }
            bit_reverse_[i] = static_cast<uint8_t>(reversed);
            const double theta = kPi * static_cast<double>(i) / 128.0;
            rotation_sin_[i] = 0.5 * std::sin(theta);
            rotation_cos_[i] = 0.5 * std::cos(theta);
        }
        for (int i = 0; i < 32; ++i) {
            const double angle = -2.0 * kPi * static_cast<double>(i) / 64.0;
            fft_twiddle_[i] = {std::cos(angle), std::sin(angle)};
        }
    }

    int fail(const char* message) noexcept {
        std::snprintf(error_, sizeof(error_), "%s", message);
        return -1;
    }

    void fft64(Complex* values) const noexcept {
        for (int i = 0; i < 64; ++i) {
            const int j = bit_reverse_[i];
            if (j > i) {
                const Complex temp = values[i];
                values[i] = values[j];
                values[j] = temp;
            }
        }
        for (int length = 2; length <= 64; length <<= 1) {
            const int half = length >> 1;
            const int twiddle_step = 64 / length;
            for (int base = 0; base < 64; base += length) {
                for (int j = 0; j < half; ++j) {
                    const Complex even = values[base + j];
                    const Complex odd = mul(values[base + j + half], fft_twiddle_[j * twiddle_step]);
                    values[base + j] = {even.re + odd.re, even.im + odd.im};
                    values[base + j + half] = {even.re - odd.re, even.im - odd.im};
                }
            }
        }
    }

    void analysis_slot(const int channel, const float* ring, const int timeslot) noexcept {
        double v36[64];
        double v40[64];
        Complex frequency[64];

        for (int sample = 0; sample < 64; ++sample) {
            v36[sample] =
                analysis_fifo_[channel][0][sample] * tables::kAnalysisWindow[8 * 64 + sample] +
                analysis_fifo_[channel][2][sample] * tables::kAnalysisWindow[6 * 64 + sample] +
                analysis_fifo_[channel][4][sample] * tables::kAnalysisWindow[4 * 64 + sample] +
                analysis_fifo_[channel][6][sample] * tables::kAnalysisWindow[2 * 64 + sample] +
                analysis_fifo_[channel][8][sample] * tables::kAnalysisWindow[0 * 64 + sample];
            v40[sample] =
                analysis_fifo_[channel][1][sample] * tables::kAnalysisWindow[7 * 64 + sample] +
                analysis_fifo_[channel][3][sample] * tables::kAnalysisWindow[5 * 64 + sample] +
                analysis_fifo_[channel][5][sample] * tables::kAnalysisWindow[3 * 64 + sample] +
                analysis_fifo_[channel][7][sample] * tables::kAnalysisWindow[1 * 64 + sample] +
                static_cast<double>(ring[sample]) * tables::kAnalysisWindow[9 * 64 + sample];
        }

        for (int k = 0; k < 64; ++k) {
            const int source = 63 - k;
            const double re = v40[source];
            const double im = v36[source];
            const double a = rotation_sin_[k];
            const double b = rotation_cos_[k];
            frequency[k] = {im * a - re * b, im * b + re * a};
        }
        fft64(frequency);
        constexpr double scale = 1.0 / 64.0;
        for (int k = 0; k < 32; ++k) {
            x_[channel][2 * k][timeslot] = {frequency[k].re * scale, -frequency[k].im * scale};
            x_[channel][2 * k + 1][timeslot] = {
                frequency[63 - k].re * scale,
                frequency[63 - k].im * scale};
        }

        for (int history = 8; history > 0; --history) {
            std::memcpy(analysis_fifo_[channel][history], analysis_fifo_[channel][history - 1],
                        sizeof(analysis_fifo_[channel][history]));
        }
        for (int sample = 0; sample < 64; ++sample) {
            analysis_fifo_[channel][0][sample] = static_cast<double>(ring[sample]);
        }
    }

    void surround_post() noexcept {
        static constexpr double kDcA[21] = {
            -.0006242550443857908, -.0019234686624258757, -.0042654648423194885,
            -.008168308064341545, -.014327201060950756, -.023759860545396805,
            -.03757232800126076, -.05577569454908371, -.07568276673555374,
            -.09172472357749939, -.5979374051094055, -.09172472357749939,
            -.07568276673555374, -.05577569454908371, -.03757232800126076,
            -.023759860545396805, -.014327201060950756, -.008168308064341545,
            -.0042654648423194885, -.0019234686624258757, -.0006242550443857908,
        };
        static constexpr double kDcB[21] = {
            .0013996040215715766, .003839150769636035, .007512642536312342,
            .012419373728334904, .018367428332567215, .0249701626598835,
            .03167900815606117, .03785000368952751, .04283412545919418,
            .04607561603188515, .047200120985507965, .04607561603188515,
            .04283412545919418, .03785000368952751, .03167900815606117,
            .0249701626598835, .018367428332567215, .012419373728334904,
            .007512642536312342, .003839150769636035, .0013996040215715766,
        };

        for (int surround = 0; surround < 2; ++surround) {
            const int channel = surround + 3;
            for (int group = 0; group < 24; group += 4) {
                Complex current[4][64];
                Complex dc_buffer[24];
                for (int slot = 0; slot < 4; ++slot) {
                    for (int band = 0; band < 64; ++band) {
                        current[slot][band] = x_[channel][band][group + slot];
                        const Complex delayed = surround_delay_[surround][slot][band];
                        x_[channel][band][group + slot] =
                            phase_shift_enabled_ ? Complex{delayed.im, -delayed.re} : delayed;
                    }
                }

                for (int i = 0; i < 20; ++i) {
                    dc_buffer[i] = surround_history_[surround][i];
                }
                for (int i = 0; i < 4; ++i) {
                    dc_buffer[20 + i] = current[i][0];
                }
                if (phase_shift_enabled_) {
                    for (int slot = 0; slot < 4; ++slot) {
                        Complex sum{0.0, 0.0};
                        for (int tap = 0; tap < 21; ++tap) {
                            const Complex sample = dc_buffer[slot + tap];
                            const double cr = kDcB[tap];
                            const double ci = kDcA[tap];
                            sum.re += sample.re * cr - sample.im * ci;
                            sum.im += sample.re * ci + sample.im * cr;
                        }
                        x_[channel][0][group + slot] = {2.0 * sum.re, 2.0 * sum.im};
                    }
                }
                for (int i = 0; i < 20; ++i) {
                    surround_history_[surround][i] = dc_buffer[i + 4];
                }

                std::memmove(&surround_delay_[surround][0][0],
                             &surround_delay_[surround][4][0],
                             sizeof(Complex) * 6 * 64);
                for (int slot = 0; slot < 4; ++slot) {
                    std::memcpy(surround_delay_[surround][6 + slot], current[slot],
                                sizeof(Complex) * 64);
                }
            }
        }
    }

    void analysis_channel(const int channel) noexcept {
        const float* bed_planar = job_bed_;
        const float phase_old = job_phase_old_;
        const float phase_new = job_phase_new_;
        const bool ramp_phase = phase_old != phase_new;
        const float phase_step = static_cast<float>((phase_new - phase_old) / 256.0f);
        float scaled[64];
        float delayed[64];

        for (int timeslot = 0; timeslot < 24; ++timeslot) {
            for (int sample = 0; sample < 64; ++sample) {
                const int frame_sample = timeslot * 64 + sample;
                float gain = phase_new;
                if (ramp_phase && frame_sample < 256) {
                    const float product = static_cast<float>(static_cast<float>(frame_sample) * phase_step);
                    gain = static_cast<float>(phase_old + product);
                }
                scaled[sample] = static_cast<float>(bed_planar[channel * 1536 + frame_sample] * gain);
            }
            // L/R/C and, for 7 core channels, the extra pair (5 and 6) take the
            // ten-slot delay here; Ls/Rs take the same delay inside surround_post().
            const float* analysis_input = scaled;
            const int delay_lane = channel < 3 ? channel : channel - 2;
            if (channel < 3 || channel >= 5) {
                std::memcpy(delayed, analysis_delay_[delay_lane][0], sizeof(delayed));
                std::memmove(&analysis_delay_[delay_lane][0][0],
                             &analysis_delay_[delay_lane][1][0],
                             sizeof(float) * 9 * 64);
                std::memcpy(analysis_delay_[delay_lane][9], scaled, sizeof(scaled));
                analysis_input = delayed;
            }
            analysis_slot(channel, analysis_input, timeslot);
        }
    }

    // Frames the analysis QMF of the core PCM with the given analysis-side
    // scalar (the structural 1/16 times joc_clipgain).  The scalar ramps over the
    // frame's first 256 samples from the previous frame's value; the first frame
    // after a reset takes it directly.
    void analysis(const float* bed_planar, const float scalar) noexcept {
        job_bed_ = bed_planar;
        job_phase_old_ = analysis_phase_;
        if (job_phase_old_ <= 0.0f && job_phase_old_ != 0.0f) {
            job_phase_old_ = scalar;
        }
        job_phase_new_ = scalar;
        job_kind_ = JobKind::Analysis;
        job_count_ = static_cast<int>(core_channels_);
        for (uint32_t channel = 0; channel < core_channels_; ++channel) {
            job_objects_[channel] = static_cast<int>(channel);
        }
        dispatch_jobs();
        analysis_phase_ = scalar;
        surround_post();
    }

    static std::size_t dq_index(const int object, const int point, const int channel, const int band) noexcept {
        return static_cast<std::size_t>(
            (((object * 2 + point) * EJOC_MAX_CORE_CHANNELS + channel) * kMaxBands) + band);
    }

    void matrix_object(
        const int object,
        const uint8_t* n_bands,
        const uint8_t* n_dpoints,
        const uint8_t* slope_idx,
        const uint8_t* offset_ts,
        const double* dq) noexcept {

        std::memset(z_[object], 0, sizeof(z_[object]));
        const int bands = n_bands[object];
        const int points = n_dpoints[object];
        const int slope = slope_idx[object];
        const uint8_t* pb_map = parameter_band_map(bands);

        for (uint32_t channel = 0; channel < core_channels_; ++channel) {
            for (int subband = 0; subband < 64; ++subband) {
                const int parameter_band = pb_map[subband];
                const double previous = matrix_previous_[object][channel][subband];
                const double target0 = dq[dq_index(object, 0, channel, parameter_band)];
                const double target1 = points == 2
                    ? dq[dq_index(object, 1, channel, parameter_band)]
                    : target0;
                double last = previous;

                for (int timeslot = 0; timeslot < 24; ++timeslot) {
                    double coefficient;
                    if (slope == 0) {
                        if (points == 1) {
                            const double alpha = static_cast<double>(timeslot + 1) / 24.0;
                            coefficient = previous * (1.0 - alpha) + target0 * alpha;
                        } else if (timeslot < 12) {
                            const double alpha = static_cast<double>(timeslot + 1) / 12.0;
                            coefficient = previous * (1.0 - alpha) + target0 * alpha;
                        } else {
                            const double alpha = static_cast<double>(timeslot - 11) / 12.0;
                            coefficient = target0 * (1.0 - alpha) + target1 * alpha;
                        }
                    } else if (points == 1) {
                        coefficient = timeslot < offset_ts[object * 2] ? previous : target0;
                    } else {
                        coefficient = timeslot < offset_ts[object * 2] ? previous : target0;
                        if (timeslot >= offset_ts[object * 2 + 1]) {
                            coefficient = target1;
                        }
                    }
                    z_[object][subband][timeslot].re += x_[channel][subband][timeslot].re * coefficient;
                    z_[object][subband][timeslot].im += x_[channel][subband][timeslot].im * coefficient;
                    last = coefficient;
                }
                matrix_previous_[object][channel][subband] = last;
            }
        }
    }

    void qmf5_step(double* state, const double* rotated, double* output) noexcept {
        for (int block = 0; block < 16; ++block) {
            for (int lane = 0; lane < 4; ++lane) {
                const int sample = block * 4 + lane;
                const int state_base = block * 36 + lane;
                const double even = rotated[block * 8 + lane * 2];
                const double odd = rotated[block * 8 + lane * 2 + 1];
                output[sample] = 2.0 * (
                    tables::kQmf5Window[sample] * even + state[state_base]);

                state[state_base] = tables::kQmf5Window[1 * 64 + sample] * odd + state[state_base + 4];
                for (int slot = 0; slot < 7; ++slot) {
                    const double alternating = (slot & 1) == 0 ? even : odd;
                    state[state_base + (slot + 1) * 4] =
                        tables::kQmf5Window[(slot + 2) * 64 + sample] * alternating +
                        state[state_base + (slot + 2) * 4];
                }
                state[state_base + 8 * 4] = tables::kQmf5Window[9 * 64 + sample] * odd;
            }
        }
    }

    void synthesis_object(
        const int object,
        const float output_scale,
        float* output16) noexcept {

        Complex frequency[64];
        double rotated[128];
        double pcm64[64];
        double* state = synthesis_state_[object];
        float* destination = output16 + (object + 1) * 1536;
        for (int timeslot = 0; timeslot < 24; ++timeslot) {
            for (int k = 0; k < 32; ++k) {
                const Complex even = z_[object][2 * k][timeslot];
                const Complex odd = z_[object][2 * k + 1][timeslot];
                frequency[k] = {even.re, -even.im};
                frequency[63 - k] = {odd.re, odd.im};
            }
            fft64(frequency);
            for (int k = 0; k < 64; ++k) {
                const double re = frequency[k].re;
                const double im = frequency[k].im;
                const double sin_component = rotation_sin_[k];
                const double cos_component = rotation_cos_[k];
                rotated[2 * k] = 2.0 * (re * cos_component + im * sin_component);
                rotated[2 * k + 1] = 2.0 * (im * cos_component - re * sin_component);
            }
            qmf5_step(state, rotated, pcm64);
            for (int sample = 0; sample < 64; ++sample) {
                // The clamp is on the ×16 object PCM; joc_clipgain already
                // entered the analysis, so only the user output gain remains, and
                // it is a float32 multiply like the reference.
                const double clipped = clamp_unit(16.0 * pcm64[sample]);
                destination[timeslot * 64 + sample] = static_cast<float>(clipped) * output_scale;
            }
        }
    }

    void process_one_object(const int object) noexcept {
        matrix_object(object, job_n_bands_, job_n_dpoints_, job_slope_idx_,
                      job_offset_ts_, job_dq_);
        synthesis_object(object, job_output_scale_, job_output16_);
    }

    void execute_job_loop() noexcept {
        while (true) {
            const int index = next_job_.fetch_add(1, std::memory_order_relaxed);
            if (index >= job_count_) {
                break;
            }
            const int item = job_objects_[index];
            if (job_kind_ == JobKind::Analysis) {
                analysis_channel(item);
            } else if (job_kind_ == JobKind::Objects) {
                process_one_object(item);
            } else {
                process_one_object_frame(item);
            }
        }
    }

    void dispatch_jobs() noexcept {
        if (workers_.empty() || job_count_ == 1) {
            next_job_.store(0, std::memory_order_relaxed);
            execute_job_loop();
            return;
        }
        next_job_.store(0, std::memory_order_relaxed);
        work_barrier_->arrive_and_wait();
        execute_job_loop();
        work_barrier_->arrive_and_wait();
    }

    void worker_loop() noexcept {
        while (!pool_ready_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        while (true) {
            work_barrier_->arrive_and_wait();
            if (stop_.load(std::memory_order_acquire)) {
                return;
            }
            execute_job_loop();
            work_barrier_->arrive_and_wait();
        }
    }

    void stop_workers() noexcept {
        if (workers_.empty()) {
            work_barrier_.reset();
            stop_.store(false, std::memory_order_relaxed);
            pool_ready_.store(false, std::memory_order_relaxed);
            return;
        }
        stop_.store(true, std::memory_order_release);
        work_barrier_->arrive_and_wait();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        work_barrier_.reset();
        stop_.store(false, std::memory_order_relaxed);
        pool_ready_.store(false, std::memory_order_relaxed);
    }

    void process_objects(
        const uint32_t object_mask,
        const uint8_t* n_bands,
        const uint8_t* n_dpoints,
        const uint8_t* slope_idx,
        const uint8_t* offset_ts,
        const double* dq,
        const float output_scale,
        float* output16) noexcept {

        int count = 0;
        for (int object = 0; object < EJOC_MAX_OBJECTS; ++object) {
            if (object_mask & (1u << object)) {
                job_objects_[count++] = object;
            }
        }
        if (count == 0) {
            return;
        }
        job_kind_ = JobKind::Objects;
        job_count_ = count;
        job_n_bands_ = n_bands;
        job_n_dpoints_ = n_dpoints;
        job_slope_idx_ = slope_idx;
        job_offset_ts_ = offset_ts;
        job_dq_ = dq;
        job_output_scale_ = output_scale;
        job_output16_ = output16;

        dispatch_jobs();
    }

    // Core LFE: the full joc_clipgain is applied as a sample-wise ramp before the
    // 1217-sample delay, and the read-out is only clamped.  The ramp is
    // G_old + n*(G_new-G_old)/256 for n < 256 and G_new afterwards; the first
    // frame after a reset uses G_old = G_new.
    void decode_lfe(const float* lfe, const double clipgain, float* output16,
                    const float output_scale) noexcept {
        const double new_gain = clipgain;
        const double old_gain = has_lfe_clipgain_ ? lfe_clipgain_ : new_gain;
        const bool ramped = old_gain != new_gain;
        const double step = (new_gain - old_gain) / 256.0;
        double ramp[256];
        for (int n = 0; n < 256; ++n) {
            ramp[n] = ramped ? old_gain + static_cast<double>(n) * step : new_gain;
        }
        has_lfe_clipgain_ = true;
        lfe_clipgain_ = new_gain;

        float* channel0 = output16;
        for (int sample = 0; sample < kLfeDelay; ++sample) {
            channel0[sample] =
                static_cast<float>(clamp_unit(lfe_delay_[sample])) * output_scale;
        }
        for (int sample = kLfeDelay; sample < EJOC_FRAME_SAMPLES; ++sample) {
            const int input = sample - kLfeDelay;
            const double gain = input < 256 ? ramp[input] : new_gain;
            const double value = (lfe != nullptr) ? static_cast<double>(lfe[input]) : 0.0;
            channel0[sample] =
                static_cast<float>(clamp_unit(value * gain)) * output_scale;
        }
        for (int sample = 0; sample < kLfeDelay; ++sample) {
            const int input = sample + (EJOC_FRAME_SAMPLES - kLfeDelay);
            const double gain = input < 256 ? ramp[input] : new_gain;
            const double value = (lfe != nullptr) ? static_cast<double>(lfe[input]) : 0.0;
            lfe_delay_[sample] = value * gain;
        }
    }

    enum class FrameMode : uint8_t { Normal, Hold, HardCut, Passthrough };

    // ---- frame-level state machine ------------------------------------- //
    // The Renderer owns the seq state machine (normal / hold / hard cut /
    // passthrough), the per-object previous matrix and its presence marker, the
    // analysis/synthesis state, the LFE delay ring with its clipgain ramp, and
    // the 8-plane core history of the passthrough path, so one audio frame in
    // produces the complete 16-channel object frame.

    float held_analysis_scalar() const noexcept {
        // Sentinel: a renderer that has not seen a JOC frame yet reports the
        // structural 1/16, i.e. clipgain 1.0.
        return (analysis_phase_ <= 0.0f && analysis_phase_ != 0.0f) ? 0.0625f
                                                                   : analysis_phase_;
    }

    double held_clipgain() const noexcept {
        return static_cast<double>(held_analysis_scalar()) * 16.0;
    }

    // The 10-bit JOC sequence counter wraps from 1023 back to 1.
    static uint32_t next_seq(const uint32_t value) noexcept {
        return value == 1023u ? 1u : value + 1u;
    }

    // Drops the state of the core channels a smaller layout does not use.
    void clear_unused_channels(const uint32_t n_channels) noexcept {
        if (n_channels >= EJOC_MAX_CORE_CHANNELS) {
            return;
        }
        for (uint32_t channel = n_channels; channel < EJOC_MAX_CORE_CHANNELS; ++channel) {
            std::memset(analysis_fifo_[channel], 0, sizeof(analysis_fifo_[channel]));
        }
        for (int object = 0; object < EJOC_MAX_OBJECTS; ++object) {
            for (uint32_t channel = n_channels; channel < EJOC_MAX_CORE_CHANNELS; ++channel) {
                std::memset(prev_matrix_[object][channel], 0,
                            sizeof(prev_matrix_[object][channel]));
            }
        }
        if (n_channels == 5u) {
            // A 5-channel frame does not drive the extra-pair delay lanes.
            std::memset(analysis_delay_[3], 0, sizeof(analysis_delay_[3]));
            std::memset(analysis_delay_[4], 0, sizeof(analysis_delay_[4]));
        }
    }

    // A declared core-channel-count change (renderer status 9) clears the LFE
    // ring; the clipgain ramp state is kept, so the gain does not retune.
    void maybe_clear_lfe_ring(const uint32_t status) noexcept {
        if (status == 9u) {
            std::memset(lfe_delay_, 0, sizeof(lfe_delay_));
        }
    }

    // Advances the seq state machine for this frame and reports the state the
    // renderer was in before the frame.  status 8 (no metadata) neither advances
    // the expected counter nor compares the sequence.
    uint32_t advance_state(const uint32_t status, const uint32_t seq) noexcept {
        const uint32_t pre_state = state_;
        const bool valid = status == 0u || status == 9u;
        if (status == EJOC_FRAME_STATUS_NO_METADATA && pre_state != 0u) {
            state_ = 3;
            return pre_state;
        }
        expected_seq_ = next_seq(expected_seq_);
        if (valid && seq == 0u && pre_state != 0u) {
            // The programme restarted: hard cut and drop the object history.
            state_ = 4;
            return pre_state;
        }
        uint32_t state = state_;
        if (state == 0u) {
            if (valid) {
                state = 1u;
                expected_seq_ = seq;
            }
        } else if (state == 1u) {
            if (valid) {
                if (seq != expected_seq_) {
                    state = 4u;
                }
            } else {
                state = 2u;
                invalid_count_ = 1u;
            }
        } else if (state == 2u) {
            if (valid) {
                state = 1u;
                expected_seq_ = seq;
            } else if (invalid_count_ == 5u) {
                state = 3u;
            } else {
                ++invalid_count_;
            }
        } else if (state == 3u) {
            if (valid) {
                state = 1u;
                expected_seq_ = seq;
            } else {
                ++invalid_count_;
            }
        } else {
            // state 4: the hard cut lasted one frame; a valid frame resyncs and
            // an invalid one starts the invalid-frame count.
            if (valid) {
                state = 1u;
                expected_seq_ = seq;
            } else {
                state = 2u;
                invalid_count_ = 1u;
            }
        }
        state_ = state;
        if (status == 9u) {
            state_ = 4u;
        }
        last_frame_valid_ = valid;
        return pre_state;
    }

    // Compatibility note for frames without JOC metadata (status 8).
    //
    // The state machine used to map status 8 to Passthrough unconditionally,
    // including in the middle of a stream.  That is only meaningful before the
    // first valid JOC frame: a plain E-AC-3 stream (or a stream whose first JOC
    // frame comes late) has no object state yet, so the core planes are all
    // there is and the delayed core PCM is the right output.  In the middle of
    // a JOC stream a frame without metadata is a damaged frame, not "no
    // objects": the Passthrough output replaces the object render with the core
    // planes, and the object synthesis restarts from silence on the next frame.
    // Every such frame therefore produced a step at the frame boundary and a
    // second one when the metadata came back; encoders that are not careful
    // with the JOC EMDF container emit exactly those frames.
    //
    // After the change status 8 follows the state machine like the other
    // invalid statuses: state 0 (no valid frame rendered yet) stays Passthrough,
    // and state 3 (mid-stream) selects Hold, which reuses the previous object
    // matrices and keeps the bed and object rendering continuous.
    //
    // This is a compatibility choice, not a claim about the normative reading
    // of the specification: a frame without metadata can be argued to be the
    // Passthrough case.  It exists so that non-standard streams stay click-free;
    // if the reference behaviour is preferred, re-add the unconditional
    // Passthrough branch here.
    FrameMode select_mode(const uint32_t status) const noexcept {
        if (status == EJOC_FRAME_STATUS_NO_METADATA) {
            // See the note above: only a stream that has not rendered a valid
            // frame yet has nothing to hold.
            return state_ == 0u ? FrameMode::Passthrough : FrameMode::Hold;
        }
        if (state_ == 4u) {
            return FrameMode::HardCut;
        }
        if (state_ == 2u || state_ == 3u) {
            return FrameMode::Hold;
        }
        if (state_ == 1u && status == 0u) {
            return FrameMode::Normal;
        }
        return FrameMode::Passthrough;
    }

    // JOC input order 5/7 channels (without LFE) -> 8-plane core order
    // L,R,C,LFE,Ls,Rs,(Lrs,Rrs).  Used by the passthrough path and its history.
    void build_core8(const float* bed_planar, const float* lfe,
                     const uint32_t n_channels) noexcept {
        std::memset(core8_, 0, sizeof(core8_));
        for (uint32_t channel = 0; channel < 3u && channel < n_channels; ++channel) {
            for (int sample = 0; sample < EJOC_FRAME_SAMPLES; ++sample) {
                core8_[channel][sample] =
                    static_cast<double>(bed_planar[channel * EJOC_FRAME_SAMPLES + sample]);
            }
        }
        if (lfe != nullptr) {
            for (int sample = 0; sample < EJOC_FRAME_SAMPLES; ++sample) {
                core8_[EJOC_LFE_SLOT][sample] = static_cast<double>(lfe[sample]);
            }
        }
        const uint32_t rest = n_channels > 3u ? n_channels - 3u : 0u;
        for (uint32_t index = 0; index < rest; ++index) {
            const uint32_t channel = 3u + index;
            for (int sample = 0; sample < EJOC_FRAME_SAMPLES; ++sample) {
                core8_[4u + index][sample] =
                    static_cast<double>(bed_planar[channel * EJOC_FRAME_SAMPLES + sample]);
            }
        }
    }

    // Passthrough output: slots 0..7 take the previous frame's 8 core planes
    // delayed by 1217 samples plus this frame's leading 319 samples; slots 8..15
    // stay zero.
    void write_passthrough(float* output16, const float output_scale) const noexcept {
        for (int channel = 0; channel < EJOC_CORE_OUT_CHANNELS; ++channel) {
            float* destination = output16 + channel * EJOC_FRAME_SAMPLES;
            for (int sample = 0; sample < kLfeDelay; ++sample) {
                destination[sample] = static_cast<float>(
                    core_history_[channel][EJOC_FRAME_SAMPLES - kLfeDelay + sample]) *
                    output_scale;
            }
            for (int sample = kLfeDelay; sample < EJOC_FRAME_SAMPLES; ++sample) {
                destination[sample] =
                    static_cast<float>(core8_[channel][sample - kLfeDelay]) * output_scale;
            }
        }
    }

    // Normal rendering: interpolate the per-object matrix from its previous
    // frame-end value (or hard cut when there is none) and publish objects that
    // disappeared as an all-zero matrix until they come back.
    void matrix_object_frame(const int object) noexcept {
        std::memset(z_[object], 0, sizeof(z_[object]));
        const int n_channels = static_cast<int>(core_channels_);
        const bool present = (job_object_mask_ & (1u << object)) != 0u;
        if (!present) {
            if (prev_present_[object]) {
                for (int channel = 0; channel < n_channels; ++channel) {
                    std::memset(prev_matrix_[object][channel], 0,
                                sizeof(prev_matrix_[object][channel]));
                }
                prev_present_[object] = true;
            }
            return;
        }
        const int bands = job_n_bands_[object];
        const int points = job_n_dpoints_[object];
        const int slope = job_slope_idx_[object];
        const uint8_t* pb_map = parameter_band_map(bands);
        const bool has_previous = prev_present_[object];
        for (int channel = 0; channel < n_channels; ++channel) {
            for (int subband = 0; subband < EJOC_SUBBANDS; ++subband) {
                const int parameter_band = pb_map[subband];
                const double target0 = job_dq_[dq_index(object, 0, channel, parameter_band)];
                const double target1 = points == 2
                                           ? job_dq_[dq_index(object, 1, channel, parameter_band)]
                                           : target0;
                const double previous =
                    has_previous ? prev_matrix_[object][channel][subband] : 0.0;
                double last = target0;
                for (int timeslot = 0; timeslot < EJOC_TIMESLOTS; ++timeslot) {
                    double coefficient = 0.0;
                    if (slope == 0) {
                        if (!has_previous) {
                            if (points == 1 || timeslot < EJOC_TIMESLOTS / 2) {
                                // Hard cut: no fade-in from zero.  With two data
                                // points the second half still interpolates.
                                coefficient = target0;
                            } else {
                                const double alpha =
                                    static_cast<double>(timeslot - (EJOC_TIMESLOTS / 2 - 1)) /
                                    static_cast<double>(EJOC_TIMESLOTS / 2);
                                coefficient = target0 * (1.0 - alpha) + target1 * alpha;
                            }
                        } else if (points == 1) {
                            const double alpha =
                                static_cast<double>(timeslot + 1) /
                                static_cast<double>(EJOC_TIMESLOTS);
                            coefficient = previous * (1.0 - alpha) + target0 * alpha;
                        } else if (timeslot < EJOC_TIMESLOTS / 2) {
                            const double alpha = static_cast<double>(timeslot + 1) /
                                                 (EJOC_TIMESLOTS / 2);
                            coefficient = previous * (1.0 - alpha) + target0 * alpha;
                        } else {
                            const double alpha =
                                static_cast<double>(timeslot - (EJOC_TIMESLOTS / 2 - 1)) /
                                static_cast<double>(EJOC_TIMESLOTS / 2);
                            coefficient = target0 * (1.0 - alpha) + target1 * alpha;
                        }
                    } else {
                        // Step mode switches targets at the raw offset slots.
                        const double base = has_previous ? previous : target0;
                        coefficient = timeslot < job_offset_ts_[object * EJOC_MAX_DPOINTS]
                                          ? base
                                          : target0;
                        if (points == 2 &&
                            timeslot >= job_offset_ts_[object * EJOC_MAX_DPOINTS + 1]) {
                            coefficient = target1;
                        }
                    }
                    z_[object][subband][timeslot].re +=
                        x_[channel][subband][timeslot].re * coefficient;
                    z_[object][subband][timeslot].im +=
                        x_[channel][subband][timeslot].im * coefficient;
                    last = coefficient;
                }
                prev_matrix_[object][channel][subband] = last;
            }
        }
        prev_present_[object] = true;
    }

    // Hold rendering: all 24 slots reuse the previous frame-end matrix and the
    // previous state is not updated.
    void matrix_object_hold(const int object) noexcept {
        std::memset(z_[object], 0, sizeof(z_[object]));
        const int n_channels = static_cast<int>(core_channels_);
        for (int channel = 0; channel < n_channels; ++channel) {
            for (int subband = 0; subband < EJOC_SUBBANDS; ++subband) {
                const double previous = prev_matrix_[object][channel][subband];
                for (int timeslot = 0; timeslot < EJOC_TIMESLOTS; ++timeslot) {
                    z_[object][subband][timeslot].re +=
                        x_[channel][subband][timeslot].re * previous;
                    z_[object][subband][timeslot].im +=
                        x_[channel][subband][timeslot].im * previous;
                }
            }
        }
    }

    void process_one_object_frame(const int object) noexcept {
        if (frame_mode_ == FrameMode::Normal) {
            matrix_object_frame(object);
        } else if (frame_mode_ == FrameMode::Hold) {
            matrix_object_hold(object);
        } else {
            // Hard cut and passthrough publish silence, but every object slot is
            // still synthesised so the synthesis state stays aligned.
            std::memset(z_[object], 0, sizeof(z_[object]));
        }
        synthesis_object(object, job_output_scale_, job_output16_);
    }

    // One E-AC-3 audio frame: validates the description, advances the state
    // machine and dispatches one of the four rendering modes.
    int render_frame_impl(const ejoc_frame_input& input) noexcept {
        error_[0] = '\0';
        if (input.struct_size != sizeof(ejoc_frame_input) ||
            input.struct_version != EJOC_FRAME_INPUT_VERSION) {
            return fail("unsupported ejoc_frame_input struct version");
        }
        if (input.bed_planar == nullptr || input.output16_planar == nullptr) {
            return fail("null bed or output pointer passed to ejoc_renderer_render_frame");
        }
        const uint32_t n_channels = input.core_channel_count;
        if (n_channels != EJOC_CORE_CHANNELS && n_channels != EJOC_MAX_CORE_CHANNELS) {
            return fail("core_channel_count must be 5 or 7");
        }
        if (!std::isfinite(input.clipgain) || !std::isfinite(input.output_scale)) {
            return fail("clipgain and output_scale must be finite");
        }
        const bool has_metadata = input.has_metadata != 0u;
        uint32_t status = input.frame_status;
        if (!has_metadata) {
            if (status != EJOC_FRAME_STATUS_NO_METADATA) {
                return fail("a frame without JOC metadata must use frame_status 8");
            }
        } else if (status != 0u && (status < 2u || status > 7u)) {
            return fail("frame_status is not a JOC parse status");
        }
        if ((input.object_mask & ~((1u << EJOC_MAX_OBJECTS) - 1u)) != 0u) {
            return fail("object_mask contains an object index above 14");
        }
        if (has_metadata && status == 0u) {
            for (int object = 0; object < EJOC_MAX_OBJECTS; ++object) {
                if ((input.object_mask & (1u << object)) == 0u) {
                    continue;
                }
                if (parameter_band_map(input.n_bands[object]) == nullptr) {
                    return fail("unsupported parameter-band count");
                }
                if (input.n_dpoints[object] < 1u || input.n_dpoints[object] > 2u) {
                    return fail("n_dpoints must be 1 or 2");
                }
                if (input.slope_idx[object] > 1u) {
                    return fail("slope_idx must be 0 or 1");
                }
                if (input.slope_idx[object] == 1u) {
                    const uint8_t* offsets =
                        input.offset_ts + object * EJOC_MAX_DPOINTS;
                    for (uint32_t point = 0; point < input.n_dpoints[object]; ++point) {
                        if (offsets[point] >= EJOC_TIMESLOTS) {
                            return fail("offset_ts leaves the 0..23 range");
                        }
                        if (point != 0u && offsets[point] <= offsets[point - 1u]) {
                            return fail("offset_ts is not strictly increasing");
                        }
                    }
                }
            }
        }

        if (n_channels_prev_ >= 0 && n_channels != static_cast<uint32_t>(n_channels_prev_)) {
            clear_unused_channels(n_channels);
        }
        // The parse status decides whether the frame's clipgain is usable; a
        // declared-channel-count change (status 9) is only the renderer's own
        // dispatch tag, so the new gain still applies to that frame.
        const uint32_t parse_status = status;
        // A declared core-channel-count change is the renderer's status 9: hard
        // cut with the LFE ring cleared, while the clipgain ramp state is kept.
        if (has_metadata && status == 0u && input.declared_core_channels != 0u &&
            n_channels_prev_ >= 0 &&
            input.declared_core_channels != static_cast<uint32_t>(n_channels_prev_)) {
            status = 9u;
        }
        advance_state(status, input.seq);
        maybe_clear_lfe_ring(status);
        const FrameMode mode = select_mode(status);
        if (input.phase_shift_valid != 0u) {
            phase_shift_enabled_ = input.phase_shift != 0u;
        }
        const bool valid_metadata =
            has_metadata && parse_status == 0u && input.declared_core_channels != 0u;
        double clipgain;
        float phase_new;
        if ((mode == FrameMode::Normal || mode == FrameMode::HardCut) && valid_metadata) {
            clipgain = input.clipgain;
            phase_new = static_cast<float>(0.0625 * clipgain);
        } else {
            // Held frames reuse the current clipgain; they do not retune it.
            clipgain = held_clipgain();
            phase_new = held_analysis_scalar();
        }

        core_channels_ = n_channels;
        std::memset(input.output16_planar, 0,
                    sizeof(float) * EJOC_OUTPUT_CHANNELS * EJOC_FRAME_SAMPLES);
        build_core8(input.bed_planar, input.lfe, n_channels);
        analysis(input.bed_planar, phase_new);

        job_kind_ = JobKind::FrameObjects;
        frame_mode_ = mode;
        job_object_mask_ = mode == FrameMode::Normal ? input.object_mask : 0u;
        job_n_bands_ = input.n_bands;
        job_n_dpoints_ = input.n_dpoints;
        job_slope_idx_ = input.slope_idx;
        job_offset_ts_ = input.offset_ts;
        job_dq_ = input.dq;
        job_output_scale_ = input.output_scale;
        job_output16_ = input.output16_planar;
        job_count_ = EJOC_MAX_OBJECTS;
        for (int object = 0; object < EJOC_MAX_OBJECTS; ++object) {
            job_objects_[object] = object;
        }
        dispatch_jobs();

        decode_lfe(input.lfe, clipgain, input.output16_planar, input.output_scale);
        if (mode == FrameMode::HardCut) {
            // All matrices are zero this frame; the presence markers stay set so
            // the next valid frame fades in from zero.
            std::memset(input.output16_planar, 0,
                        sizeof(float) * EJOC_OUTPUT_CHANNELS * EJOC_FRAME_SAMPLES);
            for (int object = 0; object < EJOC_MAX_OBJECTS; ++object) {
                prev_present_[object] = true;
                for (int channel = 0; channel < EJOC_MAX_CORE_CHANNELS; ++channel) {
                    std::memset(prev_matrix_[object][channel], 0,
                                sizeof(prev_matrix_[object][channel]));
                }
            }
        } else if (mode == FrameMode::Passthrough) {
            write_passthrough(input.output16_planar, input.output_scale);
        }

        // The 8-plane history is updated on every frame so a later passthrough
        // frame is continuous with the last core PCM.
        std::memcpy(core_history_, core8_, sizeof(core_history_));
        n_channels_prev_ = static_cast<int>(n_channels);
        return 0;
    }

    enum class JobKind : uint8_t { Analysis, Objects, FrameObjects };

    std::vector<std::thread> workers_;
    std::unique_ptr<std::barrier<>> work_barrier_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> pool_ready_{false};
    std::atomic<int> next_job_{0};
    int job_objects_[EJOC_MAX_OBJECTS]{};
    int job_count_ = 0;
    JobKind job_kind_ = JobKind::Objects;
    const float* job_bed_ = nullptr;
    float job_phase_old_ = 0.0625f;
    float job_phase_new_ = 0.0625f;
    const uint8_t* job_n_bands_ = nullptr;
    const uint8_t* job_n_dpoints_ = nullptr;
    const uint8_t* job_slope_idx_ = nullptr;
    const uint8_t* job_offset_ts_ = nullptr;
    const double* job_dq_ = nullptr;
    float job_output_scale_ = 1.0f;
    float* job_output16_ = nullptr;
    uint32_t job_object_mask_ = 0;
    FrameMode frame_mode_ = FrameMode::Normal;

    uint32_t core_channels_ = EJOC_CORE_CHANNELS;
    alignas(64) double analysis_fifo_[EJOC_MAX_CORE_CHANNELS][9][64];
    // Delay lanes: 0..2 hold L/R/C, 3..4 hold the extra pair of a 7-channel frame.
    alignas(64) float analysis_delay_[5][10][64];
    float analysis_phase_ = -0.5f;
    alignas(64) Complex surround_delay_[2][10][64];
    alignas(64) Complex surround_history_[2][20];
    // sigma(k) switch; the caller sets it for downmix configurations 3 and 4.
    bool phase_shift_enabled_ = true;
    alignas(64) double lfe_delay_[kLfeDelay];
    // LFE gain ramp start; cleared by reset() and cleared/unset only there, so a
    // held frame keeps the last applied gain.
    double lfe_clipgain_ = 0.0;
    bool has_lfe_clipgain_ = false;
    alignas(64) double matrix_previous_[15][EJOC_MAX_CORE_CHANNELS][64];
    alignas(64) double synthesis_state_[15][640];
    alignas(64) Complex x_[EJOC_MAX_CORE_CHANNELS][64][24];
    alignas(64) Complex z_[15][64][24];

    // Frame-level state: the seq state machine and the per-object interpolation
    // previous matrix (frame-end timeslot of the last published matrix).
    uint32_t state_ = 0;
    uint32_t expected_seq_ = 0;
    uint32_t invalid_count_ = 0;
    bool last_frame_valid_ = false;
    int n_channels_prev_ = -1;
    alignas(64) double prev_matrix_[EJOC_MAX_OBJECTS][EJOC_MAX_CORE_CHANNELS][64];
    bool prev_present_[EJOC_MAX_OBJECTS];
    // Passthrough history: the 8 core planes of the previous frame and of this
    // frame, in the order L,R,C,LFE,Ls,Rs,(Lrs,Rrs).
    alignas(64) double core_history_[EJOC_CORE_OUT_CHANNELS][EJOC_FRAME_SAMPLES];
    alignas(64) double core8_[EJOC_CORE_OUT_CHANNELS][EJOC_FRAME_SAMPLES];

    uint8_t bit_reverse_[64];
    Complex fft_twiddle_[32];
    double rotation_sin_[64];
    double rotation_cos_[64];
    char error_[256];
};

}  // namespace ejoc

extern "C" {

uint32_t EJOC_CALL ejoc_abi_version(void) {
    return EJOC_ABI_VERSION;
}

const char* EJOC_CALL ejoc_build_info(void) {
#if defined(_MSC_VER)
    return "eac3joc-core abi=4 compiler=MSVC fft=fixed64 speaker=double binaural=double crt=static-by-build";
#elif defined(__clang__)
    return "eac3joc-core abi=4 compiler=Clang fft=fixed64 speaker=double binaural=double";
#elif defined(__GNUC__)
    return "eac3joc-core abi=4 compiler=GCC fft=fixed64 speaker=double binaural=double";
#else
    return "eac3joc-core abi=4 compiler=unknown fft=fixed64 speaker=double binaural=double";
#endif
}

ejoc_renderer_handle EJOC_CALL ejoc_renderer_create(void) {
    return new (std::nothrow) ejoc::Renderer();
}

void EJOC_CALL ejoc_renderer_destroy(ejoc_renderer_handle handle) {
    delete static_cast<ejoc::Renderer*>(handle);
}

int EJOC_CALL ejoc_renderer_reset(ejoc_renderer_handle handle) {
    if (!handle) {
        return -1;
    }
    return static_cast<ejoc::Renderer*>(handle)->reset();
}

int EJOC_CALL ejoc_renderer_set_threads(ejoc_renderer_handle handle, uint32_t total_threads) {
    if (!handle) {
        return -1;
    }
    return static_cast<ejoc::Renderer*>(handle)->set_threads(total_threads);
}

uint32_t EJOC_CALL ejoc_renderer_thread_count(ejoc_renderer_handle handle) {
    if (!handle) {
        return 0;
    }
    return static_cast<ejoc::Renderer*>(handle)->thread_count();
}

const char* EJOC_CALL ejoc_renderer_last_error(ejoc_renderer_handle handle) {
    if (!handle) {
        return "null renderer handle";
    }
    return static_cast<ejoc::Renderer*>(handle)->error();
}

int EJOC_CALL ejoc_renderer_process(
    ejoc_renderer_handle handle,
    const float* bed_planar,
    const float* lfe,
    const uint32_t core_channel_count,
    const uint32_t phase_shift,
    const uint32_t object_mask,
    const uint8_t* n_bands,
    const uint8_t* n_dpoints,
    const uint8_t* slope_idx,
    const uint8_t* offset_ts,
    const double* dq,
    const double clipgain,
    const float phase_new,
    const float output_scale,
    float* output16_planar) {
    if (!handle) {
        return -1;
    }
    return static_cast<ejoc::Renderer*>(handle)->process(
        bed_planar, lfe, core_channel_count, phase_shift, object_mask, n_bands, n_dpoints,
        slope_idx, offset_ts, dq, clipgain, phase_new, output_scale, output16_planar);
}

int EJOC_CALL ejoc_renderer_render_frame(ejoc_renderer_handle handle,
                                         const ejoc_frame_input* input) {
    if (!handle) {
        return -1;
    }
    if (!input) {
        return -1;
    }
    return static_cast<ejoc::Renderer*>(handle)->render_frame(*input);
}

}  // extern "C"
