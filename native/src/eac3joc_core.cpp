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
        analysis_phase_ = 0.0625f;
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
        const float* bed5,
        const float* lfe,
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
        if (!bed5 || !n_bands || !n_dpoints || !slope_idx || !offset_ts || !dq || !output16) {
            return fail("null pointer passed to ejoc_renderer_process");
        }
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
        analysis(bed5, phase_new);
        render_lfe(lfe, output_scale, output16);
        process_objects(object_mask, n_bands, n_dpoints, slope_idx, offset_ts,
                        dq, clipgain, output_scale, output16);
        return 0;
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
                        x_[channel][band][group + slot] = {delayed.im, -delayed.re};
                    }
                }

                for (int i = 0; i < 20; ++i) {
                    dc_buffer[i] = surround_history_[surround][i];
                }
                for (int i = 0; i < 4; ++i) {
                    dc_buffer[20 + i] = current[i][0];
                }
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
        const float* bed5 = job_bed5_;
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
                scaled[sample] = static_cast<float>(bed5[channel * 1536 + frame_sample] * gain);
            }
            const float* analysis_input = scaled;
            if (channel < 3) {
                std::memcpy(delayed, analysis_delay_[channel][0], sizeof(delayed));
                std::memmove(&analysis_delay_[channel][0][0],
                             &analysis_delay_[channel][1][0],
                             sizeof(float) * 9 * 64);
                std::memcpy(analysis_delay_[channel][9], scaled, sizeof(scaled));
                analysis_input = delayed;
            }
            analysis_slot(channel, analysis_input, timeslot);
        }
    }

    void analysis(const float* bed5, const float phase_new) noexcept {
        job_bed5_ = bed5;
        job_phase_old_ = analysis_phase_;
        job_phase_new_ = phase_new;
        job_kind_ = JobKind::Analysis;
        job_count_ = 5;
        for (int channel = 0; channel < 5; ++channel) {
            job_objects_[channel] = channel;
        }
        dispatch_jobs();
        analysis_phase_ = phase_new;
        surround_post();
    }

    static std::size_t dq_index(const int object, const int point, const int channel, const int band) noexcept {
        return static_cast<std::size_t>((((object * 2 + point) * 5 + channel) * kMaxBands) + band);
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

        for (int channel = 0; channel < 5; ++channel) {
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
        const double clipgain,
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
                const double clipped = clamp_unit(16.0 * pcm64[sample]);
                const float value = static_cast<float>(clipped * clipgain);
                destination[timeslot * 64 + sample] = static_cast<float>(value * output_scale);
            }
        }
    }

    void process_one_object(const int object) noexcept {
        matrix_object(object, job_n_bands_, job_n_dpoints_, job_slope_idx_,
                      job_offset_ts_, job_dq_);
        synthesis_object(object, job_clipgain_, job_output_scale_, job_output16_);
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
            } else {
                process_one_object(item);
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
        const double clipgain,
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
        job_clipgain_ = clipgain;
        job_output_scale_ = output_scale;
        job_output16_ = output16;

        dispatch_jobs();
    }

    void render_lfe(const float* lfe, const float output_scale, float* output16) noexcept {
        if (!lfe) {
            return;
        }
        for (int sample = 0; sample < kLfeDelay; ++sample) {
            const float value = static_cast<float>(clamp_unit(lfe_delay_[sample]));
            output16[sample] = static_cast<float>(value * output_scale);
        }
        for (int sample = kLfeDelay; sample < 1536; ++sample) {
            const float value = static_cast<float>(clamp_unit(static_cast<double>(lfe[sample - kLfeDelay])));
            output16[sample] = static_cast<float>(value * output_scale);
        }
        for (int sample = 0; sample < kLfeDelay; ++sample) {
            lfe_delay_[sample] = static_cast<double>(lfe[sample + (1536 - kLfeDelay)]);
        }
    }

    enum class JobKind : uint8_t { Analysis, Objects };

    std::vector<std::thread> workers_;
    std::unique_ptr<std::barrier<>> work_barrier_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> pool_ready_{false};
    std::atomic<int> next_job_{0};
    int job_objects_[EJOC_MAX_OBJECTS]{};
    int job_count_ = 0;
    JobKind job_kind_ = JobKind::Objects;
    const float* job_bed5_ = nullptr;
    float job_phase_old_ = 0.0625f;
    float job_phase_new_ = 0.0625f;
    const uint8_t* job_n_bands_ = nullptr;
    const uint8_t* job_n_dpoints_ = nullptr;
    const uint8_t* job_slope_idx_ = nullptr;
    const uint8_t* job_offset_ts_ = nullptr;
    const double* job_dq_ = nullptr;
    double job_clipgain_ = 1.0;
    float job_output_scale_ = 1.0f;
    float* job_output16_ = nullptr;

    alignas(64) double analysis_fifo_[5][9][64];
    alignas(64) float analysis_delay_[3][10][64];
    float analysis_phase_;
    alignas(64) Complex surround_delay_[2][10][64];
    alignas(64) Complex surround_history_[2][20];
    alignas(64) double lfe_delay_[kLfeDelay];
    alignas(64) double matrix_previous_[15][5][64];
    alignas(64) double synthesis_state_[15][640];
    alignas(64) Complex x_[5][64][24];
    alignas(64) Complex z_[15][64][24];

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
    return "eac3joc-core abi=1 compiler=MSVC fft=fixed64 speaker=double crt=static-by-build";
#elif defined(__clang__)
    return "eac3joc-core abi=1 compiler=Clang fft=fixed64 speaker=double";
#elif defined(__GNUC__)
    return "eac3joc-core abi=1 compiler=GCC fft=fixed64 speaker=double";
#else
    return "eac3joc-core abi=1 compiler=unknown fft=fixed64 speaker=double";
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
    const float* bed5_planar,
    const float* lfe,
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
        bed5_planar, lfe, object_mask, n_bands, n_dpoints, slope_idx,
        offset_ts, dq, clipgain, phase_new, output_scale, output16_planar);
}

}  // extern "C"
