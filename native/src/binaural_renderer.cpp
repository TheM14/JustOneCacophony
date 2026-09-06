#define EJOC_BUILD_DLL
#include "eac3joc_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

namespace ejoc::binaural {

struct Complex {
    double re;
    double im;
};

inline Complex add(Complex a, Complex b) noexcept {
    return {a.re + b.re, a.im + b.im};
}

inline Complex mul(Complex a, Complex b) noexcept {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

inline Complex scale(Complex value, double gain) noexcept {
    return {value.re * gain, value.im * gain};
}

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kChannels = EJOC_BINAURAL_INPUT_CHANNELS;
constexpr int kEars = EJOC_BINAURAL_OUTPUT_CHANNELS;
constexpr int kBlock = EJOC_BINAURAL_BLOCK_SAMPLES;
constexpr int kSlots = kBlock / 64;
constexpr int kQmf = EJOC_BINAURAL_QMF_BANDS;
constexpr int kHybrid = EJOC_BINAURAL_HYBRID_BANDS;
constexpr int kRank = 4;

class Renderer final {
public:
    Renderer() noexcept {
        initialize_fft();
        reset();
    }

    int configure_kernels(
        const double* qmf_analysis,
        const double* hybrid_low,
        const int16_t* hybrid_indices,
        const double* hybrid_values,
        uint32_t hybrid_count,
        const double* qmf_basis,
        const double* qmf_taps) noexcept {
        if (!qmf_analysis || !hybrid_low || !hybrid_indices || !hybrid_values ||
            !qmf_basis || !qmf_taps || hybrid_count == 0) {
            return fail("invalid binaural kernel configuration");
        }
        std::memcpy(qmf_analysis_.data(), qmf_analysis,
                    qmf_analysis_.size() * sizeof(double));
        hybrid_low_.assign(hybrid_low, hybrid_low + 3 * 2 * 13 * 16 * 2);
        hybrid_indices_.assign(hybrid_indices, hybrid_indices + hybrid_count * 4);
        hybrid_values_.assign(hybrid_values, hybrid_values + hybrid_count);
        std::memcpy(qmf_basis_.data(), qmf_basis,
                    qmf_basis_.size() * sizeof(double));
        std::memcpy(qmf_taps_.data(), qmf_taps,
                    qmf_taps_.size() * sizeof(double));
        kernels_ready_ = true;
        reset();
        return 0;
    }

    int configure_room(
        uint32_t bands,
        uint32_t allpass_count,
        const uint32_t* allpass_delays,
        const double* allpass_gains,
        const uint32_t* fdn_delays,
        const double* fdn_matrix,
        uint32_t output_tap_delay,
        const double* feedback_complex,
        const double* output_taps,
        const double* output_complex,
        uint32_t extra_count,
        const uint32_t* extra_delays,
        const double* extra_fields_complex,
        const double* extra_matrices) noexcept {
        if (bands != 64 || !fdn_delays || !fdn_matrix || !feedback_complex ||
            !output_taps || !output_complex ||
            (allpass_count && (!allpass_delays || !allpass_gains)) ||
            (extra_count && (!extra_delays || !extra_fields_complex || !extra_matrices))) {
            return fail("invalid binaural room configuration");
        }
        room_bands_ = bands;
        if (allpass_count) {
            allpass_delays_.assign(allpass_delays, allpass_delays + allpass_count);
            allpass_gains_.assign(allpass_gains, allpass_gains + allpass_count);
        } else {
            allpass_delays_.clear();
            allpass_gains_.clear();
        }
        allpass_offsets_.resize(allpass_count);
        allpass_positions_.assign(allpass_count, 0);
        size_t allpass_size = 0;
        for (uint32_t index = 0; index < allpass_count; ++index) {
            if (allpass_delays_[index] == 0) {
                return fail("binaural allpass delay must be positive");
            }
            allpass_offsets_[index] = allpass_size;
            allpass_size += static_cast<size_t>(allpass_delays_[index]) * bands;
        }
        allpass_memory_.assign(allpass_size, {});

        room_capacity_ = 0;
        for (int branch = 0; branch < 4; ++branch) {
            fdn_delays_[branch] = fdn_delays[branch];
            room_capacity_ = std::max(room_capacity_, fdn_delays_[branch]);
        }
        if (room_capacity_ == 0) {
            return fail("binaural room delay must be positive");
        }
        std::copy(fdn_matrix, fdn_matrix + 16, fdn_matrix_.begin());
        output_tap_delay_ = output_tap_delay;
        for (int band = 0; band < 64; ++band) {
            for (int branch = 0; branch < 4; ++branch) {
                const size_t complex_index = (static_cast<size_t>(band) * 4 + branch) * 2;
                feedback_[band][branch] = {
                    feedback_complex[complex_index], feedback_complex[complex_index + 1]};
                output_taps_[band][branch] = output_taps[band * 4 + branch];
                for (int ear = 0; ear < 2; ++ear) {
                    const size_t output_index =
                        ((static_cast<size_t>(ear) * 64 + band) * 4 + branch) * 2;
                    output_matrix_[ear][band][branch] = {
                        output_complex[output_index], output_complex[output_index + 1]};
                }
            }
        }
        room_memory_.assign(static_cast<size_t>(room_capacity_) * 64 * 4, {});
        if (extra_count) {
            extra_delays_.assign(extra_delays, extra_delays + extra_count);
        } else {
            extra_delays_.clear();
        }
        extra_fields_.resize(static_cast<size_t>(extra_count) * 64);
        extra_matrices_.resize(static_cast<size_t>(extra_count) * 16);
        for (uint32_t extra = 0; extra < extra_count; ++extra) {
            for (int band = 0; band < 64; ++band) {
                const size_t source = (static_cast<size_t>(extra) * 64 + band) * 2;
                extra_fields_[static_cast<size_t>(extra) * 64 + band] = {
                    extra_fields_complex[source], extra_fields_complex[source + 1]};
            }
            std::copy(extra_matrices + static_cast<size_t>(extra) * 16,
                      extra_matrices + static_cast<size_t>(extra + 1) * 16,
                      extra_matrices_.begin() + static_cast<size_t>(extra) * 16);
        }
        room_ready_ = true;
        reset();
        return 0;
    }

    int reset() noexcept {
        qmf_history_.fill(0.0);
        hybrid_low_history_.fill({});
        hybrid_high_history_.fill({});
        synthesis_history_.fill(0.0);
        std::fill(allpass_memory_.begin(), allpass_memory_.end(), Complex{});
        std::fill(allpass_positions_.begin(), allpass_positions_.end(), 0u);
        std::fill(room_memory_.begin(), room_memory_.end(), Complex{});
        room_position_ = 0;
        error_[0] = '\0';
        return 0;
    }

    const char* error() const noexcept {
        return error_[0] ? error_ : "";
    }

    int process(
        const double* input,
        const double* gains,
        const double* room_sends,
        double output_gain,
        double* output) noexcept {
        if (!kernels_ready_ || !room_ready_) {
            return fail("binaural renderer is not configured");
        }
        if (!input || !gains || !room_sends || !output || !std::isfinite(output_gain)) {
            return fail("invalid binaural process arguments");
        }
        for (int slot = 0; slot < kSlots; ++slot) {
            std::array<Complex, kChannels * kQmf> qmf{};
            std::array<Complex, kChannels * kHybrid> hybrid{};
            analyze_qmf(input + static_cast<size_t>(slot) * 64 * kChannels, qmf);
            analyze_hybrid(qmf, hybrid);

            std::array<Complex, kEars * kHybrid> rendered{};
            std::array<Complex, kHybrid> room_input{};
            for (int source = kChannels - 1; source >= 0; --source) {
                for (int band = 0; band < kHybrid; ++band) {
                    const Complex value = hybrid[source * kHybrid + band];
                    room_input[band] = add(room_input[band], scale(value, room_sends[source]));
                    for (int ear = 0; ear < kEars; ++ear) {
                        const size_t gain_index =
                            (((static_cast<size_t>(source) * kEars + ear) * kHybrid + band) * 2);
                        const Complex gain{gains[gain_index], gains[gain_index + 1]};
                        rendered[ear * kHybrid + band] = add(
                            rendered[ear * kHybrid + band], mul(value, gain));
                    }
                }
            }
            const auto room = process_room(room_input);
            for (size_t index = 0; index < rendered.size(); ++index) {
                rendered[index] = add(rendered[index], room[index]);
            }

            std::array<Complex, kEars * kQmf> qmf_output{};
            synthesize_hybrid(rendered, qmf_output);
            for (int ear = 0; ear < kEars; ++ear) {
                std::array<double, 64> samples{};
                synthesize_qmf(qmf_output.data() + ear * kQmf, ear, samples);
                for (int sample = 0; sample < 64; ++sample) {
                    output[(static_cast<size_t>(slot) * 64 + sample) * 2 + ear] =
                        samples[sample] * output_gain;
                }
            }
        }
        return 0;
    }

private:
    int fail(const char* message) noexcept {
        std::snprintf(error_, sizeof(error_), "%s", message);
        return -1;
    }

    void initialize_fft() noexcept {
        for (int index = 0; index < 128; ++index) {
            int value = index;
            int reversed = 0;
            for (int bit = 0; bit < 7; ++bit) {
                reversed = (reversed << 1) | (value & 1);
                value >>= 1;
            }
            bit_reverse_[index] = static_cast<uint8_t>(reversed);
        }
        for (int phase = 0; phase < 64; ++phase) {
            const double angle = -kPi * static_cast<double>(phase) / 128.0;
            premod_[phase] = {std::cos(angle), std::sin(angle)};
            const double post_angle =
                -3.0 * (static_cast<double>(phase) + 0.5) * kPi / 128.0;
            post_[phase] = {std::cos(post_angle), std::sin(post_angle)};
            even_post_[phase] = {0.0, (phase & 1) ? -1.0 : 1.0};
        }
    }

    void fft128(std::array<Complex, 128>& values) const noexcept {
        for (int index = 0; index < 128; ++index) {
            const int reversed = bit_reverse_[index];
            if (reversed > index) {
                std::swap(values[index], values[reversed]);
            }
        }
        for (int length = 2; length <= 128; length <<= 1) {
            const double angle = -2.0 * kPi / static_cast<double>(length);
            const Complex step{std::cos(angle), std::sin(angle)};
            for (int start = 0; start < 128; start += length) {
                Complex rotation{1.0, 0.0};
                for (int offset = 0; offset < length / 2; ++offset) {
                    const Complex even = values[start + offset];
                    const Complex odd = mul(values[start + offset + length / 2], rotation);
                    values[start + offset] = {even.re + odd.re, even.im + odd.im};
                    values[start + offset + length / 2] = {
                        even.re - odd.re, even.im - odd.im};
                    rotation = mul(rotation, step);
                }
            }
        }
    }

    void qmf_transform(const std::array<double, 64>& source,
                       std::array<Complex, 64>& target) const noexcept {
        std::array<Complex, 128> work{};
        for (int phase = 0; phase < 64; ++phase) {
            work[phase] = scale(premod_[phase], source[phase]);
        }
        fft128(work);
        for (int band = 0; band < 64; ++band) {
            target[band] = mul(work[band], post_[band]);
        }
    }

    void analyze_qmf(const double* input,
                     std::array<Complex, kChannels * kQmf>& output) noexcept {
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int lag = 9; lag > 0; --lag) {
                for (int phase = 0; phase < 64; ++phase) {
                    qmf_history_[qmf_history_index(lag, channel, phase)] =
                        qmf_history_[qmf_history_index(lag - 1, channel, phase)];
                }
            }
            for (int phase = 0; phase < 64; ++phase) {
                qmf_history_[qmf_history_index(0, channel, phase)] =
                    input[phase * kChannels + channel];
            }
            std::array<double, 64> even{};
            std::array<double, 64> odd{};
            for (int phase = 0; phase < 64; ++phase) {
                for (int lag = 0; lag < 10; ++lag) {
                    const double value =
                        qmf_history_[qmf_history_index(lag, channel, phase)] *
                        qmf_analysis_[phase * 10 + lag];
                    (lag & 1 ? odd[phase] : even[phase]) += value;
                }
            }
            std::array<Complex, 64> even_fft{};
            std::array<Complex, 64> odd_fft{};
            qmf_transform(even, even_fft);
            qmf_transform(odd, odd_fft);
            for (int band = 0; band < 64; ++band) {
                output[channel * 64 + band] = add(
                    odd_fft[band], mul(even_fft[band], even_post_[band]));
            }
        }
    }

    void analyze_hybrid(
        const std::array<Complex, kChannels * kQmf>& qmf,
        std::array<Complex, kChannels * kHybrid>& output) noexcept {
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int lag = 12; lag > 0; --lag) {
                for (int band = 0; band < 3; ++band) {
                    hybrid_low_history_[hybrid_low_history_index(lag, channel, band)] =
                        hybrid_low_history_[hybrid_low_history_index(lag - 1, channel, band)];
                }
            }
            for (int band = 0; band < 3; ++band) {
                hybrid_low_history_[hybrid_low_history_index(0, channel, band)] =
                    qmf[channel * 64 + band];
            }
            for (int output_band = 0; output_band < 16; ++output_band) {
                Complex value{};
                for (int lag = 0; lag < 13; ++lag) {
                    for (int input_band = 0; input_band < 3; ++input_band) {
                        const Complex source = hybrid_low_history_[
                            hybrid_low_history_index(lag, channel, input_band)];
                        const double components[2]{source.re, source.im};
                        for (int input_component = 0; input_component < 2; ++input_component) {
                            value.re += components[input_component] * hybrid_low_[
                                hybrid_low_kernel_index(input_band, input_component, lag,
                                                        output_band, 0)];
                            value.im += components[input_component] * hybrid_low_[
                                hybrid_low_kernel_index(input_band, input_component, lag,
                                                        output_band, 1)];
                        }
                    }
                }
                output[channel * kHybrid + output_band] = value;
            }
            for (int band = 0; band < 61; ++band) {
                output[channel * kHybrid + 16 + band] =
                    hybrid_high_history_[hybrid_high_history_index(0, channel, band)];
                for (int delay = 0; delay < 5; ++delay) {
                    hybrid_high_history_[hybrid_high_history_index(delay, channel, band)] =
                        hybrid_high_history_[hybrid_high_history_index(delay + 1, channel, band)];
                }
                hybrid_high_history_[hybrid_high_history_index(5, channel, band)] =
                    qmf[channel * 64 + 3 + band];
            }
        }
    }

    std::array<Complex, kEars * kHybrid> process_room(
        const std::array<Complex, kHybrid>& input) noexcept {
        std::array<Complex, 64> filtered{};
        for (int band = 0; band < 64; ++band) {
            filtered[band] = scale(input[band], 0.70710677);
        }
        for (size_t stage = 0; stage < allpass_delays_.size(); ++stage) {
            const uint32_t position = allpass_positions_[stage];
            const double gain = allpass_gains_[stage];
            for (int band = 0; band < 64; ++band) {
                Complex& memory = allpass_memory_[
                    allpass_offsets_[stage] + static_cast<size_t>(position) * 64 + band];
                const Complex residual = add(filtered[band], scale(memory, -gain));
                filtered[band] = add(scale(residual, gain), memory);
                memory = residual;
            }
            allpass_positions_[stage] = (position + 1) % allpass_delays_[stage];
        }

        std::array<Complex, 64 * 4> branches{};
        std::array<Complex, 64 * 4> taps{};
        for (int band = 0; band < 64; ++band) {
            for (int branch = 0; branch < 4; ++branch) {
                Complex value = filtered[band];
                for (int source = 0; source < 4; ++source) {
                    const uint32_t position =
                        (room_position_ + room_capacity_ - fdn_delays_[source]) % room_capacity_;
                    value = add(value, scale(room_memory_[
                        room_memory_index(position, band, source)],
                        fdn_matrix_[branch * 4 + source]));
                }
                branches[band * 4 + branch] = value;
                const uint32_t tap_position =
                    (room_position_ + room_capacity_ -
                     (output_tap_delay_ % room_capacity_)) % room_capacity_;
                taps[band * 4 + branch] =
                    room_memory_[room_memory_index(tap_position, band, branch)];
            }
        }
        for (int band = 0; band < 64; ++band) {
            for (int branch = 0; branch < 4; ++branch) {
                room_memory_[room_memory_index(room_position_, band, branch)] =
                    mul(branches[band * 4 + branch], feedback_[band][branch]);
            }
        }
        room_position_ = (room_position_ + 1) % room_capacity_;

        std::array<Complex, 64 * 4> extra{};
        for (size_t index = 0; index < extra_delays_.size(); ++index) {
            const uint32_t position =
                (room_position_ + room_capacity_ -
                 ((extra_delays_[index] + 1) % room_capacity_)) % room_capacity_;
            for (int band = 0; band < 64; ++band) {
                for (int target = 0; target < 4; ++target) {
                    Complex mixed{};
                    for (int source = 0; source < 4; ++source) {
                        mixed = add(mixed, scale(room_memory_[
                            room_memory_index(position, band, source)],
                            extra_matrices_[index * 16 + target * 4 + source]));
                    }
                    extra[band * 4 + target] = add(
                        extra[band * 4 + target],
                        mul(mixed, extra_fields_[index * 64 + band]));
                }
            }
        }

        std::array<Complex, kEars * kHybrid> output{};
        for (int ear = 0; ear < 2; ++ear) {
            for (int band = 0; band < 64; ++band) {
                Complex value{};
                for (int branch = 0; branch < 4; ++branch) {
                    const Complex signal = add(
                        scale(taps[band * 4 + branch], output_taps_[band][branch]),
                        extra[band * 4 + branch]);
                    value = add(value, mul(
                        signal, output_matrix_[ear][band][branch]));
                }
                output[ear * kHybrid + band] = value;
            }
        }
        return output;
    }

    void synthesize_hybrid(
        const std::array<Complex, kEars * kHybrid>& input,
        std::array<Complex, kEars * kQmf>& output) const noexcept {
        for (size_t mapping = 0; mapping < hybrid_values_.size(); ++mapping) {
            const int16_t* index = hybrid_indices_.data() + mapping * 4;
            const int input_band = index[0];
            const int input_component = index[1];
            const int output_band = index[2];
            const int output_component = index[3];
            const double gain = hybrid_values_[mapping];
            for (int ear = 0; ear < 2; ++ear) {
                const Complex source = input[ear * kHybrid + input_band];
                Complex& target = output[ear * kQmf + output_band];
                const double component = input_component == 0 ? source.re : source.im;
                (output_component == 0 ? target.re : target.im) += component * gain;
            }
        }
    }

    void synthesize_qmf(const Complex* input, int ear,
                        std::array<double, 64>& output) noexcept {
        std::array<double, 64 * kRank> features{};
        std::array<double, 128> flat{};
        for (int band = 0; band < 64; ++band) {
            flat[band * 2] = input[band].re;
            flat[band * 2 + 1] = input[band].im;
        }
        for (int phase = 0; phase < 64; ++phase) {
            for (int rank = 0; rank < kRank; ++rank) {
                double value = 0.0;
                const size_t base = (static_cast<size_t>(phase) * kRank + rank) * 128;
                for (int component = 0; component < 128; ++component) {
                    value += flat[component] * qmf_basis_[base + component];
                }
                features[phase * kRank + rank] = value;
            }
        }
        for (int phase = 0; phase < 64; ++phase) {
            double value = 0.0;
            for (int lag = 0; lag < 10; ++lag) {
                for (int rank = 0; rank < kRank; ++rank) {
                    const double feature = lag == 0
                        ? features[phase * kRank + rank]
                        : synthesis_history_[synthesis_history_index(
                            ear, lag - 1, phase, rank)];
                    value += feature * qmf_taps_[
                        ((static_cast<size_t>(phase) * 10 + lag) * kRank + rank)];
                }
            }
            output[phase] = value;
        }
        for (int lag = 8; lag > 0; --lag) {
            for (int phase = 0; phase < 64; ++phase) {
                for (int rank = 0; rank < kRank; ++rank) {
                    synthesis_history_[synthesis_history_index(ear, lag, phase, rank)] =
                        synthesis_history_[synthesis_history_index(
                            ear, lag - 1, phase, rank)];
                }
            }
        }
        for (int phase = 0; phase < 64; ++phase) {
            for (int rank = 0; rank < kRank; ++rank) {
                synthesis_history_[synthesis_history_index(ear, 0, phase, rank)] =
                    features[phase * kRank + rank];
            }
        }
    }

    static size_t qmf_history_index(int lag, int channel, int phase) noexcept {
        return (static_cast<size_t>(lag) * kChannels + channel) * 64 + phase;
    }

    static size_t hybrid_low_history_index(int lag, int channel, int band) noexcept {
        return (static_cast<size_t>(lag) * kChannels + channel) * 3 + band;
    }

    static size_t hybrid_high_history_index(int delay, int channel, int band) noexcept {
        return (static_cast<size_t>(delay) * kChannels + channel) * 61 + band;
    }

    static size_t hybrid_low_kernel_index(
        int input_band, int input_component, int lag,
        int output_band, int output_component) noexcept {
        return (((static_cast<size_t>(input_band) * 2 + input_component) * 13 + lag) *
                16 + output_band) * 2 + output_component;
    }

    size_t room_memory_index(uint32_t position, int band, int branch) const noexcept {
        return (static_cast<size_t>(position) * 64 + band) * 4 + branch;
    }

    static size_t synthesis_history_index(
        int ear, int lag, int phase, int rank) noexcept {
        return (((static_cast<size_t>(ear) * 9 + lag) * 64 + phase) * kRank + rank);
    }

    bool kernels_ready_ = false;
    bool room_ready_ = false;
    std::array<double, 64 * 10> qmf_analysis_{};
    std::vector<double> hybrid_low_;
    std::vector<int16_t> hybrid_indices_;
    std::vector<double> hybrid_values_;
    std::array<double, 64 * kRank * 128> qmf_basis_{};
    std::array<double, 64 * 10 * kRank> qmf_taps_{};

    std::array<double, 10 * kChannels * 64> qmf_history_{};
    std::array<Complex, 13 * kChannels * 3> hybrid_low_history_{};
    std::array<Complex, 6 * kChannels * 61> hybrid_high_history_{};
    std::array<double, kEars * 9 * 64 * kRank> synthesis_history_{};

    uint32_t room_bands_ = 0;
    std::vector<uint32_t> allpass_delays_;
    std::vector<double> allpass_gains_;
    std::vector<size_t> allpass_offsets_;
    std::vector<uint32_t> allpass_positions_;
    std::vector<Complex> allpass_memory_;
    std::array<uint32_t, 4> fdn_delays_{};
    std::array<double, 16> fdn_matrix_{};
    uint32_t room_capacity_ = 0;
    uint32_t output_tap_delay_ = 0;
    std::array<std::array<Complex, 4>, 64> feedback_{};
    std::array<std::array<double, 4>, 64> output_taps_{};
    std::array<std::array<std::array<Complex, 4>, 64>, 2> output_matrix_{};
    std::vector<Complex> room_memory_;
    uint32_t room_position_ = 0;
    std::vector<uint32_t> extra_delays_;
    std::vector<Complex> extra_fields_;
    std::vector<double> extra_matrices_;

    std::array<uint8_t, 128> bit_reverse_{};
    std::array<Complex, 64> premod_{};
    std::array<Complex, 64> post_{};
    std::array<Complex, 64> even_post_{};
    char error_[256]{};
};

}  // namespace ejoc::binaural

extern "C" {

ejoc_binaural_renderer_handle EJOC_CALL ejoc_binaural_renderer_create(void) {
    return new (std::nothrow) ejoc::binaural::Renderer();
}

void EJOC_CALL ejoc_binaural_renderer_destroy(ejoc_binaural_renderer_handle handle) {
    delete static_cast<ejoc::binaural::Renderer*>(handle);
}

int EJOC_CALL ejoc_binaural_renderer_reset(ejoc_binaural_renderer_handle handle) {
    return handle ? static_cast<ejoc::binaural::Renderer*>(handle)->reset() : -1;
}

const char* EJOC_CALL ejoc_binaural_renderer_last_error(
    ejoc_binaural_renderer_handle handle) {
    return handle ? static_cast<ejoc::binaural::Renderer*>(handle)->error()
                  : "null binaural renderer handle";
}

int EJOC_CALL ejoc_binaural_renderer_configure_kernels(
    ejoc_binaural_renderer_handle handle,
    const double* qmf_analysis,
    const double* hybrid_low,
    const int16_t* hybrid_indices,
    const double* hybrid_values,
    uint32_t hybrid_count,
    const double* qmf_basis,
    const double* qmf_taps) {
    return handle ? static_cast<ejoc::binaural::Renderer*>(handle)->configure_kernels(
        qmf_analysis, hybrid_low, hybrid_indices, hybrid_values,
        hybrid_count, qmf_basis, qmf_taps) : -1;
}

int EJOC_CALL ejoc_binaural_renderer_configure_room(
    ejoc_binaural_renderer_handle handle,
    uint32_t bands,
    uint32_t allpass_count,
    const uint32_t* allpass_delays,
    const double* allpass_gains,
    const uint32_t* fdn_delays,
    const double* fdn_matrix,
    uint32_t output_tap_delay,
    const double* feedback_complex,
    const double* output_taps,
    const double* output_complex,
    uint32_t extra_count,
    const uint32_t* extra_delays,
    const double* extra_fields_complex,
    const double* extra_matrices) {
    return handle ? static_cast<ejoc::binaural::Renderer*>(handle)->configure_room(
        bands, allpass_count, allpass_delays, allpass_gains,
        fdn_delays, fdn_matrix, output_tap_delay,
        feedback_complex, output_taps, output_complex,
        extra_count, extra_delays, extra_fields_complex, extra_matrices) : -1;
}

int EJOC_CALL ejoc_binaural_renderer_process(
    ejoc_binaural_renderer_handle handle,
    const double* input16_interleaved,
    const double* gains_complex,
    const double* room_sends,
    double output_gain,
    double* output_stereo_interleaved) {
    return handle ? static_cast<ejoc::binaural::Renderer*>(handle)->process(
        input16_interleaved, gains_complex, room_sends,
        output_gain, output_stereo_interleaved) : -1;
}

}  // extern "C"
