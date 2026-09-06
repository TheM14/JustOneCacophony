#define EJOC_BUILD_DLL
#include "eac3joc_core.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace ejoc::sofa_binaural {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kChannels = EJOC_BINAURAL_INPUT_CHANNELS;
constexpr int kHop = 64;
constexpr int kQmf = EJOC_BINAURAL_QMF_BANDS;
constexpr int kHybrid = EJOC_BINAURAL_HYBRID_BANDS;
constexpr int kRank = 4;
constexpr int kLatency = 961;
constexpr int kOrder = 5;
constexpr int kTerms = (kOrder + 1) * (kOrder + 1);
constexpr int kEarlyHistory = 256;
constexpr int kTransitionSlots = 8;
constexpr double kSampleRate = 48000.0;
constexpr int kFftSize = 128;
constexpr int kMaxBlockSamples = 512;

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

inline Complex mulr(Complex a, double b) noexcept {
    return {a.re * b, a.im * b};
}

inline Complex addmul(Complex acc, Complex b, Complex c) noexcept {
    return {acc.re + b.re * c.re - b.im * c.im, acc.im + b.re * c.im + b.im * c.re};
}

// ---------------------------------------------------------------------------
// 128-point FFT, forward: X[k] = sum_p x[p] exp(-2j pi k p / N) (radix-2 DIT)
// ---------------------------------------------------------------------------
class Fft128 final {
public:
    Fft128() noexcept {
        for (int k = 0; k < kFftSize; ++k) {
            const double angle = -2.0 * kPi * k / kFftSize;
            twiddle_[k] = {std::cos(angle), std::sin(angle)};
        }
    }

    void forward(const Complex* input, Complex* output) const noexcept {
        // bit-reversal permutation for the 7-bit index (DIT)
        for (int index = 0; index < kFftSize; ++index) {
            unsigned int reversed = 0;
            for (int bit = 0; bit < 7; ++bit) {
                reversed = (reversed << 1)
                    | ((static_cast<unsigned int>(index) >> bit) & 1u);
            }
            output[static_cast<int>(reversed)] = input[index];
        }
        for (int size = 2; size <= kFftSize; size <<= 1) {
            const int half = size >> 1;
            const int step = kFftSize / size;
            for (int base = 0; base < kFftSize; base += size) {
                for (int offset = 0; offset < half; ++offset) {
                    const Complex w = twiddle_[offset * step];
                    const Complex even = output[base + offset];
                    const Complex odd = mul(output[base + offset + half], w);
                    output[base + offset] = add(even, odd);
                    output[base + offset + half] = {
                        even.re - odd.re, even.im - odd.im};
                }
            }
        }
    }

private:
    Complex twiddle_[kFftSize];
};

// ---------------------------------------------------------------------------
// QMF analysis (mirrors QmfAnalysis: joined history, lag 0..9, modulations)
// ---------------------------------------------------------------------------
class QmfAnalysis final {
public:
    void configure(const double* coefficients) noexcept {
        std::memcpy(coeff_, coefficients, sizeof(coeff_));
    }

    void reset() noexcept {
        std::memset(history_, 0, sizeof(history_));
    }

    // input: interleaved [samples][16]; output: [slots][16][64] complex
    void process(const double* input, int slots, Complex* output) noexcept {
        // joined[9 + slots][16][64]
        for (int lag = 0; lag < 9; ++lag) {
            std::memcpy(joined_[lag], history_[lag], sizeof(joined_[0]));
        }
        for (int slot = 0; slot < slots; ++slot) {
            for (int channel = 0; channel < kChannels; ++channel) {
                for (int p = 0; p < 64; ++p) {
                    joined_[9 + slot][channel][p] =
                        input[(static_cast<size_t>(slot) * 64 + p) * kChannels
                              + channel];
                }
            }
        }
        for (int slot = 0; slot < slots; ++slot) {
            for (int channel = 0; channel < kChannels; ++channel) {
                Complex even[kFftSize];
                Complex odd[kFftSize];
                for (int p = 0; p < kFftSize; ++p) {
                    even[p] = {0.0, 0.0};
                    odd[p] = {0.0, 0.0};
                }
                // 64 polyphase positions; the 128-point FFT zero-pads the rest
                for (int p = 0; p < kQmf; ++p) {
                    double even_re = 0.0;
                    double odd_re = 0.0;
                    for (int lag = 0; lag < 5; ++lag) {
                        even_re += joined_[9 + slot - 2 * lag][channel][p]
                            * coeff_[p][2 * lag];
                    }
                    for (int lag = 0; lag < 5; ++lag) {
                        odd_re += joined_[9 + slot - (2 * lag + 1)][channel][p]
                            * coeff_[p][2 * lag + 1];
                    }
                    const double pre_angle = -kPi * p / kFftSize;
                    even[p] = {even_re * std::cos(pre_angle),
                               even_re * std::sin(pre_angle)};
                    odd[p] = {odd_re * std::cos(pre_angle), odd_re * std::sin(pre_angle)};
                }
                Complex even_fft[kFftSize];
                Complex odd_fft[kFftSize];
                fft_.forward(even, even_fft);
                fft_.forward(odd, odd_fft);
                Complex* band = output
                    + (static_cast<size_t>(slot) * kChannels + channel) * kQmf;
                for (int b = 0; b < kQmf; ++b) {
                    const double post_angle = -3.0 * (b + 0.5) * kPi / kFftSize;
                    const Complex post = {std::cos(post_angle), std::sin(post_angle)};
                    const Complex even_post = {0.0, (b % 2 == 0) ? 1.0 : -1.0};
                    // python: (odd_fft + even_fft * even_post) * post
                    band[b] = mul(post, add(odd_fft[b],
                                            mul(even_fft[b], even_post)));
                }
            }
        }
        for (int lag = 0; lag < 9; ++lag) {
            std::memcpy(history_[lag], joined_[slots + lag], sizeof(history_[0]));
        }
    }

private:
    double coeff_[kQmf][10];
    double history_[9][kChannels][64];
    double joined_[9 + kMaxBlockSamples / kHop][kChannels][64];
    Fft128 fft_;
};

// ---------------------------------------------------------------------------
// Hybrid analysis (mirrors HybridAnalysis: 13-tap low join, 6-slot high join)
// ---------------------------------------------------------------------------
class HybridAnalysis final {
public:
    void configure(const double* low_kernel) noexcept {
        std::memcpy(low_, low_kernel, sizeof(low_));
    }

    void reset() noexcept {
        std::memset(history_, 0, sizeof(history_));
        std::memset(high_history_, 0, sizeof(high_history_));
    }

    // qmf: [slots][16][64] complex; output: [slots][16][77] complex
    void process(const Complex* qmf, int slots, Complex* output) noexcept {
        for (int lag = 0; lag < 12; ++lag) {
            std::memcpy(low_joined_[lag], history_[lag], sizeof(low_joined_[0]));
        }
        for (int lag = 0; lag < 6; ++lag) {
            std::memcpy(high_joined_[lag], high_history_[lag],
                        sizeof(high_joined_[0]));
        }
        for (int slot = 0; slot < slots; ++slot) {
            for (int channel = 0; channel < kChannels; ++channel) {
                const Complex* band = qmf
                    + (static_cast<size_t>(slot) * kChannels + channel) * kQmf;
                for (int parent = 0; parent < 3; ++parent) {
                    low_joined_[12 + slot][channel][parent][0] = band[parent].re;
                    low_joined_[12 + slot][channel][parent][1] = band[parent].im;
                }
                for (int b = 0; b < 61; ++b) {
                    high_joined_[6 + slot][channel][b] = band[3 + b];
                }
            }
        }
        for (int slot = 0; slot < slots; ++slot) {
            for (int channel = 0; channel < kChannels; ++channel) {
                Complex* out = output
                    + (static_cast<size_t>(slot) * kChannels + channel) * kHybrid;
                for (int hb = 0; hb < 16; ++hb) {
                    Complex value = {0.0, 0.0};
                    for (int parent = 0; parent < 3; ++parent) {
                        for (int comp = 0; comp < 2; ++comp) {
                            for (int lag = 0; lag < 13; ++lag) {
                                const double source =
                                    low_joined_[12 + slot - lag][channel][parent][comp];
                                value.re += source * low_[parent][comp][lag][hb][0];
                                value.im += source * low_[parent][comp][lag][hb][1];
                            }
                        }
                    }
                    out[hb] = value;
                }
                for (int b = 0; b < 61; ++b) {
                    out[16 + b] = high_joined_[slot][channel][b];
                }
            }
        }
        for (int lag = 0; lag < 12; ++lag) {
            std::memcpy(history_[lag], low_joined_[slots + lag], sizeof(history_[0]));
        }
        for (int lag = 0; lag < 6; ++lag) {
            std::memcpy(high_history_[lag], high_joined_[slots + lag],
                        sizeof(high_history_[0]));
        }
    }

private:
    double low_[3][2][13][16][2];
    double history_[12][kChannels][3][2];
    double low_joined_[12 + kMaxBlockSamples / kHop][kChannels][3][2];
    Complex high_history_[6][kChannels][61];
    Complex high_joined_[6 + kMaxBlockSamples / kHop][kChannels][61];
};

// ---------------------------------------------------------------------------
// Hybrid synthesis sparse map (mirrors HybridSynthesis)
// ---------------------------------------------------------------------------
struct SynthesisEntry {
    int in_band;
    int in_comp;
    int out_band;
    int out_comp;
    double gain;
};

class HybridSynthesis final {
public:
    void configure(const int16_t* indices, const double* values,
                   uint32_t count) noexcept {
        entries_.clear();
        for (uint32_t i = 0; i < count; ++i) {
            entries_.push_back({static_cast<int>(indices[i * 4]),
                                static_cast<int>(indices[i * 4 + 1]),
                                static_cast<int>(indices[i * 4 + 2]),
                                static_cast<int>(indices[i * 4 + 3]),
                                values[i]});
        }
    }

    // hybrid: [slots][2][77]; output: [slots][2][64]
    void process(const Complex* hybrid, int slots, Complex* output) noexcept {
        for (int slot = 0; slot < slots; ++slot) {
            for (int channel = 0; channel < 2; ++channel) {
                const Complex* source = hybrid
                    + (static_cast<size_t>(slot) * 2 + channel) * kHybrid;
                Complex* target = output
                    + (static_cast<size_t>(slot) * 2 + channel) * kQmf;
                for (int b = 0; b < kQmf; ++b) {
                    target[b] = {0.0, 0.0};
                }
                for (const auto& entry : entries_) {
                    const Complex value = source[entry.in_band];
                    const double component =
                        (entry.in_comp == 0) ? value.re : value.im;
                    if (entry.out_comp == 0) {
                        target[entry.out_band].re += component * entry.gain;
                    } else {
                        target[entry.out_band].im += component * entry.gain;
                    }
                }
            }
        }
    }

private:
    std::vector<SynthesisEntry> entries_;
};

// ---------------------------------------------------------------------------
// QMF synthesis, rank-4 (mirrors QmfSynthesis: joined history, lag 0..9)
// ---------------------------------------------------------------------------
class QmfSynthesis final {
public:
    void configure(const double* basis, const double* taps) noexcept {
        std::memcpy(basis_, basis, sizeof(basis_));
        std::memcpy(taps_, taps, sizeof(taps_));
    }

    void reset() noexcept {
        std::memset(history_, 0, sizeof(history_));
    }

    // qmf: [slots][2][64]; output: [slots][2][64] real
    void process(const Complex* qmf, int slots, double* output) noexcept {
        for (int lag = 0; lag < 9; ++lag) {
            std::memcpy(joined_[lag], history_[lag], sizeof(joined_[0]));
        }
        for (int slot = 0; slot < slots; ++slot) {
            for (int channel = 0; channel < 2; ++channel) {
                const Complex* values = qmf
                    + (static_cast<size_t>(slot) * 2 + channel) * kQmf;
                double flat[kFftSize];
                for (int b = 0; b < kQmf; ++b) {
                    flat[2 * b] = values[b].re;
                    flat[2 * b + 1] = values[b].im;
                }
                for (int b = 0; b < kQmf; ++b) {
                    for (int r = 0; r < kRank; ++r) {
                        double value = 0.0;
                        for (int j = 0; j < kFftSize; ++j) {
                            value += flat[j] * basis_[b][r][j];
                        }
                        joined_[9 + slot][channel][b][r] = value;
                    }
                }
            }
        }
        for (int slot = 0; slot < slots; ++slot) {
            for (int channel = 0; channel < 2; ++channel) {
                for (int b = 0; b < kQmf; ++b) {
                    double value = 0.0;
                    for (int lag = 0; lag < 10; ++lag) {
                        for (int r = 0; r < kRank; ++r) {
                            value += joined_[9 + slot - lag][channel][b][r]
                                * taps_[b][lag][r];
                        }
                    }
                    output[(static_cast<size_t>(slot) * 2 + channel) * 64 + b]
                        = value;
                }
            }
        }
        for (int lag = 0; lag < 9; ++lag) {
            std::memcpy(history_[lag], joined_[slots + lag], sizeof(history_[0]));
        }
    }

private:
    double basis_[kQmf][kRank][kFftSize];
    double taps_[kQmf][10][kRank];
    double history_[9][2][kQmf][kRank];
    double joined_[9 + kMaxBlockSamples / kHop][2][kQmf][kRank];
};

// ---------------------------------------------------------------------------
// Fifth-order ACN/N3D real spherical harmonics
// ---------------------------------------------------------------------------
class RealSh final {
public:
    static void evaluate(const double* direction, double* basis) noexcept {
        const double azimuth = std::atan2(direction[1], direction[0]);
        const double x = std::max(-1.0, std::min(1.0, direction[2]));
        int index = 0;
        for (int degree = 0; degree <= kOrder; ++degree) {
            for (int order = -degree; order <= degree; ++order) {
                const int absolute = std::abs(order);
                const double normalization = std::sqrt(
                    (2.0 * degree + 1.0) / (4.0 * kPi)
                    * factorial_ratio(degree, absolute));
                const double legendre = associated_legendre(
                    degree, absolute, x, pmm_value(absolute, x));
                if (order > 0) {
                    basis[index] = std::sqrt(2.0) * normalization * legendre
                        * std::cos(order * azimuth);
                } else if (order < 0) {
                    basis[index] = std::sqrt(2.0) * normalization * legendre
                        * std::sin(absolute * azimuth);
                } else {
                    basis[index] = normalization * legendre;
                }
                ++index;
            }
        }
    }

private:
    static double factorial_ratio(int degree, int order) noexcept {
        double ratio = 1.0;
        for (int value = degree - order + 1; value <= degree + order; ++value) {
            ratio /= static_cast<double>(value);
        }
        return ratio;
    }

    static double pmm_value(int m, double x) noexcept {
        if (m == 0) {
            return 1.0;
        }
        double double_factorial = 1.0;
        for (int value = 1; value <= 2 * m - 1; value += 2) {
            double_factorial *= static_cast<double>(value);
        }
        double value = double_factorial
            * std::pow(std::max(0.0, 1.0 - x * x), 0.5 * m);
        if (m % 2 == 1) {
            value = -value;
        }
        return value;
    }

    static double associated_legendre(int degree, int order, double x,
                                      double pmm) noexcept {
        if (degree == order) {
            return pmm;
        }
        double pm1 = pmm;
        double pmm1 = (2.0 * order + 1.0) * x * pmm;
        if (degree == order + 1) {
            return pmm1;
        }
        double result = 0.0;
        for (int l = order + 2; l <= degree; ++l) {
            result = ((2.0 * l - 1.0) * x * pmm1
                      - (l + order - 1.0) * pm1) / (l - order);
            pm1 = pmm1;
            pmm1 = result;
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------
struct Path {
    int delay_slots[2];
    Complex transfer[2][kHybrid];
};

struct SourceState {
    double position[3];
    int profile;
    double gain;
    int enabled;
    int special_lfe;
    std::vector<Path> current;
    std::vector<Path> target;
    int fade_position;
    int fade_total;
    double late_current;
    double late_start;
    double late_target;
    int64_t late_fade_position;
    int64_t late_fade_total;
};

constexpr double kDistanceM[3] = {1.00000465, 2.19327927, 6.40177584};
constexpr double kMinimumDistance = 0.10;
constexpr double kCoupling = 0.01318359375;
constexpr double kRoomCalibration = 1.4;
constexpr double kLateSend[3] = {0.06, 0.16, 0.28};
constexpr double kAir = 0.002;

class Renderer final {
public:
    Renderer() noexcept {
        reset_state();
    }

    int configure_kernels(const double* qmf_analysis, const double* hybrid_low,
                          const int16_t* hybrid_indices, const double* hybrid_values,
                          uint32_t hybrid_count, const double* qmf_basis,
                          const double* qmf_taps) noexcept {
        if (!qmf_analysis || !hybrid_low || !hybrid_indices || !hybrid_values
            || !qmf_basis || !qmf_taps || hybrid_count == 0) {
            return fail("invalid sofa binaural kernel configuration");
        }
        analysis_.configure(qmf_analysis);
        hybrid_analysis_.configure(hybrid_low);
        hybrid_synthesis_.configure(hybrid_indices, hybrid_values, hybrid_count);
        synthesis_.configure(qmf_basis, qmf_taps);
        kernels_ready_ = true;
        reset();
        return 0;
    }

    int configure_field(const double* coefficients, const double* delay_coefficients,
                        const double* delay_bounds, const double* band_centers,
                        double measurement_radius_m) noexcept {
        if (!coefficients || !delay_coefficients || !delay_bounds || !band_centers
            || !std::isfinite(measurement_radius_m) || measurement_radius_m <= 0.0) {
            return fail("invalid sofa binaural field configuration");
        }
        for (int term = 0; term < kTerms; ++term) {
            for (int ear = 0; ear < 2; ++ear) {
                for (int band = 0; band < kHybrid; ++band) {
                    const size_t source =
                        ((static_cast<size_t>(term) * 2 + ear) * kHybrid + band) * 2;
                    field_coeff_[term][ear][band] = {
                        coefficients[source], coefficients[source + 1]};
                }
                delay_coeff_[term][ear] =
                    delay_coefficients[static_cast<size_t>(term) * 2 + ear];
            }
        }
        std::memcpy(delay_bounds_, delay_bounds, sizeof(delay_bounds_));
        std::memcpy(centers_, band_centers, sizeof(centers_));
        measurement_radius_ = measurement_radius_m;
        const double maximum = std::max(delay_bounds_[0][1], delay_bounds_[1][1]);
        hrtf_slots_ = static_cast<uint32_t>(
            std::ceil(maximum / static_cast<double>(kHop)));
        field_ready_ = true;
        reset();
        return 0;
    }

    int configure_room(const double* dims, const double* listener,
                       const double* wall_gains, double speed,
                       const uint32_t* fdn_delays, const double* fdn_feedback,
                       double damping, double fdn_gain,
                       const uint32_t* allpass_delays, const double* allpass_gains,
                       uint32_t enable_early, uint32_t enable_late) noexcept {
        if (!dims || !listener || !wall_gains || !fdn_delays || !fdn_feedback
            || !allpass_delays || !allpass_gains || !(speed > 0.0)
            || !(damping >= 0.0 && damping < 1.0)) {
            return fail("invalid sofa binaural room configuration");
        }
        std::memcpy(dims_, dims, sizeof(dims_));
        std::memcpy(listener_, listener, sizeof(listener_));
        std::memcpy(wall_gain_, wall_gains, sizeof(wall_gain_));
        speed_ = speed;
        std::memcpy(fdn_delays_, fdn_delays, sizeof(fdn_delays_));
        std::memcpy(fdn_feedback_, fdn_feedback, sizeof(fdn_feedback_));
        damping_ = damping;
        fdn_gain_ = fdn_gain;
        std::memcpy(allpass_delays_, allpass_delays, sizeof(allpass_delays_));
        std::memcpy(allpass_gains_, allpass_gains, sizeof(allpass_gains_));
        enable_early_reflections_ = enable_early != 0;
        enable_late_room_ = enable_late != 0;
        for (int line = 0; line < 4; ++line) {
            fdn_buffers_[line].assign(fdn_delays_[line], 0.0);
        }
        for (int line = 0; line < 2; ++line) {
            allpass_buffers_[line].assign(allpass_delays_[line], 0.0);
        }
        room_ready_ = true;
        reset();
        return 0;
    }

    int reset() noexcept {
        analysis_.reset();
        hybrid_analysis_.reset();
        synthesis_.reset();
        reset_state();
        return 0;
    }

    const char* error() const noexcept {
        return error_[0] ? error_ : "";
    }

    int set_source(uint32_t source, const double* position, uint32_t profile,
                   double gain, uint32_t enabled, uint32_t special_lfe,
                   uint32_t fade) noexcept {
        if (!kernels_ready_ || !field_ready_ || !room_ready_) {
            return fail("sofa binaural renderer is not configured");
        }
        if (source >= kChannels || !position || profile > 2) {
            return fail("invalid sofa binaural source update");
        }
        SourceState& state = sources_[source];
        state.position[0] = position[0];
        state.position[1] = position[1];
        state.position[2] = position[2];
        state.profile = static_cast<int>(profile);
        state.gain = gain;
        state.enabled = enabled != 0;
        state.special_lfe = special_lfe != 0;

        const double effective = enabled ? gain : 0.0;
        std::vector<Path> paths;
        double late_send = 0.0;
        double direction[3];
        double radius;
        normalize_adm(position, direction, radius);
        const double distance =
            std::max(kMinimumDistance, radius * kDistanceM[state.profile]);
        if (state.special_lfe) {
            paths.push_back(lfe_path(effective));
        } else {
            paths = ordinary_paths(direction, distance, state.profile, effective);
            if (enabled && enable_late_room_) {
                const double base = kLateSend[state.profile];
                const double radial = std::min(
                    std::max(std::sqrt(std::max(radius, 0.0)), 0.25), 1.5);
                late_send = effective * kRoomCalibration * base * radial;
            }
        }
        set_late_target(source, late_send, fade != 0);

        const int fade_slots = (fade && processed_slots_ > 0) ? kTransitionSlots : 0;
        if (!state.target.empty()) {
            state.current = state.target;
            state.target.clear();
        }
        if (processed_slots_ == 0 || fade_slots == 0) {
            state.current = paths;
            state.target.clear();
            state.fade_position = 0;
            state.fade_total = 0;
        } else {
            state.target = paths;
            state.fade_position = 0;
            state.fade_total = fade_slots;
        }
        return 0;
    }

    int process(const double* input, uint32_t sample_count, double output_gain,
                double* output) noexcept {
        if (!kernels_ready_ || !field_ready_ || !room_ready_) {
            return fail("sofa binaural renderer is not configured");
        }
        if (sample_count == 0 || sample_count > kMaxBlockSamples
            || sample_count % kHop != 0 || !output) {
            return fail("sofa binaural process requires 64-sample alignment");
        }
        const int slots = static_cast<int>(sample_count / kHop);
        process_internal(input, slots, output_gain);
        uint32_t skip = std::min(latency_to_discard_, sample_count);
        latency_to_discard_ -= skip;
        processed_input_samples_ += sample_count;
        std::memcpy(output, output_.data() + static_cast<size_t>(skip) * 2,
                    static_cast<size_t>(sample_count - skip) * 2 * sizeof(double));
        return static_cast<int>(sample_count - skip);
    }

    int finish(uint32_t flush_samples, double* output, uint32_t capacity) noexcept {
        if (!kernels_ready_ || !field_ready_ || !room_ready_) {
            return fail("sofa binaural renderer is not configured");
        }
        if (flush_samples == 0 || flush_samples > kMaxBlockSamples
            || flush_samples % kHop != 0 || !output || capacity < flush_samples) {
            return fail("invalid sofa binaural finish request");
        }
        const int slots = static_cast<int>(flush_samples / kHop);
        process_internal(nullptr, slots, 1.0);
        uint32_t skip = std::min(latency_to_discard_, flush_samples);
        latency_to_discard_ -= skip;
        processed_input_samples_ += flush_samples;
        std::memcpy(output, output_.data() + static_cast<size_t>(skip) * 2,
                    static_cast<size_t>(flush_samples - skip) * 2 * sizeof(double));
        return static_cast<int>(flush_samples - skip);
    }

private:
    static void normalize_adm(const double* position, double* direction,
                              double& radius) noexcept {
        const double r = std::sqrt(position[0] * position[0]
                                   + position[1] * position[1]
                                   + position[2] * position[2]);
        radius = r;
        if (r > 1.0e-15) {
            direction[0] = position[0] / r;
            direction[1] = position[1] / r;
            direction[2] = position[2] / r;
        } else {
            direction[0] = 0.0;
            direction[1] = 1.0;
            direction[2] = 0.0;
        }
    }

    static double direct_level_gain(int profile, double distance) noexcept {
        if (profile == 0) {
            return 1.0;
        }
        return 1.0 / std::sqrt(1.0 + kCoupling * distance * distance);
    }

    double inverse_distance_gain(double path_distance) const noexcept {
        return measurement_radius_ / path_distance;
    }

    Path make_path(const double* direction, double extra_delay,
                   double amplitude) noexcept {
        double basis[kTerms];
        // ADM (+X right, +Y front, +Z up) -> SOFA listener (+X front, +Y left):
        // (x, y, z)_sofa = (y_adm, -x_adm, z_adm)
        const double listener_direction[3] = {
            direction[1], -direction[0], direction[2]};
        RealSh::evaluate(listener_direction, basis);
        Complex aligned[2][kHybrid];
        double delay[2];
        for (int ear = 0; ear < 2; ++ear) {
            double delay_value = 0.0;
            for (int band = 0; band < kHybrid; ++band) {
                aligned[ear][band] = {0.0, 0.0};
            }
            for (int term = 0; term < kTerms; ++term) {
                delay_value += basis[term] * delay_coeff_[term][ear];
                for (int band = 0; band < kHybrid; ++band) {
                    aligned[ear][band].re +=
                        field_coeff_[term][ear][band].re * basis[term];
                    aligned[ear][band].im +=
                        field_coeff_[term][ear][band].im * basis[term];
                }
            }
            delay[ear] = std::min(std::max(delay_value, delay_bounds_[ear][0]),
                                  delay_bounds_[ear][1]);
        }
        Path path;
        for (int ear = 0; ear < 2; ++ear) {
            const double total = delay[ear] + extra_delay;
            const double slots = std::floor(total / kHop);
            path.delay_slots[ear] = static_cast<int>(slots);
            const double residual = total - slots * kHop;
            for (int band = 0; band < kHybrid; ++band) {
                // Python transfer = raw aligned gains x residual-delay phase x
                // amplitude (the integer-slot part is the delay_slots history).
                const double phase = -2.0 * kPi * centers_[band]
                    * residual / kSampleRate;
                const Complex rotated = {
                    aligned[ear][band].re * std::cos(phase)
                        - aligned[ear][band].im * std::sin(phase),
                    aligned[ear][band].re * std::sin(phase)
                        + aligned[ear][band].im * std::cos(phase)};
                path.transfer[ear][band] = mulr(rotated, amplitude);
            }
        }
        return path;
    }

    Path lfe_path(double gain) noexcept {
        Path path;
        path.delay_slots[0] = 0;
        path.delay_slots[1] = 0;
        for (int band = 0; band < kHybrid; ++band) {
            const double frequency = centers_[band];
            double lowpass = 1.0;
            if (frequency >= 180.0) {
                lowpass = 0.0;
            } else if (frequency > 120.0) {
                const double amount = (frequency - 120.0) / 60.0;
                const double value = std::cos(0.5 * kPi * amount);
                lowpass = value * value;
            }
            path.transfer[0][band] = {gain * lowpass / std::sqrt(2.0), 0.0};
            path.transfer[1][band] = {gain * lowpass / std::sqrt(2.0), 0.0};
        }
        return path;
    }

    std::vector<Path> ordinary_paths(const double* direction, double distance,
                                     int profile, double gain) noexcept {
        std::vector<Path> paths;
        paths.push_back(
            make_path(direction, 0.0, gain * direct_level_gain(profile, distance)));
        if (enable_early_reflections_) {
            double source[3] = {
                listener_[0] + direction[0] * distance,
                listener_[1] + direction[1] * distance,
                listener_[2] + direction[2] * distance,
            };
            for (int axis = 0; axis < 3; ++axis) {
                for (int side = 0; side < 2; ++side) {
                    double image[3] = {source[0], source[1], source[2]};
                    image[axis] = (side == 0) ? -image[axis]
                                              : 2.0 * dims_[axis] - image[axis];
                    double vector[3] = {
                        image[0] - listener_[0],
                        image[1] - listener_[1],
                        image[2] - listener_[2],
                    };
                    const double path_distance = std::sqrt(
                        vector[0] * vector[0] + vector[1] * vector[1]
                        + vector[2] * vector[2]);
                    double path_direction[3] = {
                        vector[0] / path_distance,
                        vector[1] / path_distance,
                        vector[2] / path_distance,
                    };
                    const double extra = std::max(
                        0.0, (path_distance - distance) * kSampleRate / speed_);
                    const double air =
                        std::exp(-kAir * std::max(path_distance - distance, 0.0));
                    const double amplitude = gain * kRoomCalibration
                        * wall_gain_[axis * 2 + side] * air
                        * inverse_distance_gain(path_distance);
                    paths.push_back(make_path(path_direction, extra, amplitude));
                    maximum_early_delay_ = std::max(maximum_early_delay_, extra);
                }
            }
        }
        return paths;
    }

    void set_late_target(int source, double value, int fade) noexcept {
        const int64_t fade_samples =
            (fade && processed_input_samples_ > 0) ? kTransitionSlots * kHop : 0;
        SourceState& state = sources_[source];
        if (fade_samples == 0) {
            state.late_current = value;
            state.late_start = value;
            state.late_target = value;
            state.late_fade_position = 0;
            state.late_fade_total = 0;
        } else {
            state.late_start = state.late_current;
            state.late_target = value;
            state.late_fade_position = 0;
            state.late_fade_total = fade_samples;
        }
    }

    void process_internal(const double* input, int slots, double output_gain) noexcept {
        const uint32_t sample_count = static_cast<uint32_t>(slots) * kHop;
        // 1) late send envelopes -> mono
        std::vector<double> mono(sample_count, 0.0);
        for (int source = 0; source < kChannels; ++source) {
            SourceState& state = sources_[source];
            const int64_t total = state.late_fade_total;
            if (total == 0) {
                if (input && state.late_current != 0.0) {
                    for (uint32_t s = 0; s < sample_count; ++s) {
                        mono[s] += input[static_cast<size_t>(s) * kChannels + source]
                            * state.late_current;
                    }
                }
                continue;
            }
            int64_t position = state.late_fade_position;
            for (uint32_t s = 0; s < sample_count; ++s) {
                position += 1;
                double amount = static_cast<double>(position)
                    / static_cast<double>(total);
                amount = std::min(std::max(amount, 0.0), 1.0);
                const double value = state.late_start * (1.0 - amount)
                    + state.late_target * amount;
                if (input) {
                    mono[s] += input[static_cast<size_t>(s) * kChannels + source]
                        * value;
                }
            }
            if (position >= total) {
                state.late_current = state.late_target;
                state.late_start = state.late_target;
                state.late_fade_position = 0;
                state.late_fade_total = 0;
            } else {
                double amount = static_cast<double>(position)
                    / static_cast<double>(total);
                amount = std::min(std::max(amount, 0.0), 1.0);
                state.late_current =
                    state.late_start * (1.0 - amount) + state.late_target * amount;
                state.late_fade_position = position;
            }
        }

        // 2) FDN + 961-sample stereo delay
        std::vector<double> late_pcm(static_cast<size_t>(sample_count) * 2, 0.0);
        if (enable_late_room_) {
            std::vector<double> diffused = mono;
            for (int line = 0; line < 2; ++line) {
                diffused = allpass(line, diffused);
            }
            for (uint32_t s = 0; s < sample_count; ++s) {
                const double value = diffused[s];
                double delayed[4];
                for (int line = 0; line < 4; ++line) {
                    delayed[line] = fdn_buffers_[line][fdn_positions_[line]];
                }
                double damping_state[4];
                for (int line = 0; line < 4; ++line) {
                    damping_state[line] = damping_ * damping_state_[line]
                        + (1.0 - damping_) * delayed[line];
                    damping_state_[line] = damping_state[line];
                }
                const double fdn_left = damping_state[0] + damping_state[1]
                    - damping_state[2] - damping_state[3];
                const double fdn_right = damping_state[0] - damping_state[1]
                    + damping_state[2] - damping_state[3];
                const double out_left =
                    late_ring_[static_cast<size_t>(late_pos_) * 2];
                const double out_right =
                    late_ring_[static_cast<size_t>(late_pos_) * 2 + 1];
                late_pcm[s * 2] = out_left;
                late_pcm[s * 2 + 1] = out_right;
                late_ring_[static_cast<size_t>(late_pos_) * 2] =
                    fdn_gain_ * 0.5 * fdn_left;
                late_ring_[static_cast<size_t>(late_pos_) * 2 + 1] =
                    fdn_gain_ * 0.5 * fdn_right;
                late_pos_ = (late_pos_ + 1) % kLatency;
                // feedback = hadamard4/2 @ (damping_state * feedback_gain)
                // H4[i][j] = -1 when popcount(i & j) is odd
                double feedback[4];
                for (int target = 0; target < 4; ++target) {
                    double acc = 0.0;
                    for (int source_line = 0; source_line < 4; ++source_line) {
                        const bool odd = (static_cast<unsigned>(target & source_line)
                                          ? (std::popcount(static_cast<unsigned>(
                                                 target & source_line)) & 1u) != 0u
                                          : false);
                        const double sign = odd ? -1.0 : 1.0;
                        acc += sign * damping_state[source_line]
                            * fdn_feedback_[source_line];
                    }
                    feedback[target] = 0.5 * acc;
                }
                const double input_vector[4] = {0.5, -0.5, 0.5, 0.5};
                for (int line = 0; line < 4; ++line) {
                    const double write =
                        input_vector[line] * value + feedback[line];
                    fdn_buffers_[line][fdn_positions_[line]] = write;
                    fdn_positions_[line] =
                        (fdn_positions_[line] + 1) % fdn_delays_[line];
                }
            }
        }

        // 3) analysis chain
        analysis_.process(input ? input : zeros_.data(), slots, qmf_work_.data());
        hybrid_analysis_.process(qmf_work_.data(), slots, hybrid_work_.data());
        // 4) per-object direct/early with crossfade
        std::vector<Complex> direct_and_early(
            static_cast<size_t>(slots) * 2 * kHybrid, Complex{0.0, 0.0});
        for (int slot = 0; slot < slots; ++slot) {
            const Complex* hybrid_slot =
                hybrid_work_.data() + static_cast<size_t>(slot) * kChannels * kHybrid;
            for (int source = 0; source < kChannels; ++source) {
                std::memcpy(history_[source].data()
                                + static_cast<size_t>(position_) * kHybrid,
                            hybrid_slot + static_cast<size_t>(source) * kHybrid,
                            kHybrid * sizeof(Complex));
            }
            Complex* out_slot =
                direct_and_early.data() + static_cast<size_t>(slot) * 2 * kHybrid;
            for (int source = kChannels - 1; source >= 0; --source) {
                SourceState& state = sources_[source];
                if (state.target.empty()) {
                    render_paths(source, state.current, 1.0, out_slot);
                    continue;
                }
                state.fade_position += 1;
                double amount = static_cast<double>(state.fade_position)
                    / static_cast<double>(state.fade_total);
                amount = std::min(std::max(amount, 0.0), 1.0);
                render_paths(source, state.current, 1.0 - amount, out_slot);
                render_paths(source, state.target, amount, out_slot);
                if (state.fade_position >= state.fade_total) {
                    state.current = state.target;
                    state.target.clear();
                    state.fade_position = 0;
                    state.fade_total = 0;
                }
            }
            position_ = (position_ + 1) % history_slots_;
            processed_slots_ += 1;
        }
        // 5) synthesis chain
        hybrid_synthesis_.process(direct_and_early.data(), slots,
                                  qmf_synth_work_.data());
        synthesis_.process(qmf_synth_work_.data(), slots, pcm_work_.data());
        // 6) mix, gain, interleave [samples][2]
        for (uint32_t s = 0; s < sample_count; ++s) {
            const int slot = static_cast<int>(s / kHop);
            const int b = static_cast<int>(s % kHop);
            const double direct_left =
                pcm_work_[(static_cast<size_t>(slot) * 2) * 64 + b];
            const double direct_right =
                pcm_work_[(static_cast<size_t>(slot) * 2 + 1) * 64 + b];
            output_[static_cast<size_t>(s) * 2] =
                (direct_left + late_pcm[static_cast<size_t>(s) * 2]) * output_gain;
            output_[static_cast<size_t>(s) * 2 + 1] =
                (direct_right + late_pcm[static_cast<size_t>(s) * 2 + 1]) * output_gain;
        }
    }

    void render_paths(int source, const std::vector<Path>& paths, double scale,
                      Complex* out) const noexcept {
        for (const Path& path : paths) {
            const int slots = static_cast<int>(history_slots_);
            const int index0 = (static_cast<int>(position_) + slots
                                - path.delay_slots[0]) % slots;
            const int index1 = (static_cast<int>(position_) + slots
                                - path.delay_slots[1]) % slots;
            const Complex* history = history_[source].data();
            for (int band = 0; band < kHybrid; ++band) {
                const Complex source0 = history[index0 * kHybrid + band];
                const Complex source1 = history[index1 * kHybrid + band];
                out[band].re += source0.re * path.transfer[0][band].re * scale
                    - source0.im * path.transfer[0][band].im * scale;
                out[band].im += source0.re * path.transfer[0][band].im * scale
                    + source0.im * path.transfer[0][band].re * scale;
                out[kHybrid + band].re +=
                    source1.re * path.transfer[1][band].re * scale
                    - source1.im * path.transfer[1][band].im * scale;
                out[kHybrid + band].im +=
                    source1.re * path.transfer[1][band].im * scale
                    + source1.im * path.transfer[1][band].re * scale;
            }
        }
    }

    std::vector<double> allpass(int line, const std::vector<double>& source) noexcept {
        std::vector<double> output(source.size(), 0.0);
        const double gain = allpass_gains_[line];
        const uint32_t delay = allpass_delays_[line];
        for (size_t i = 0; i < source.size(); ++i) {
            const double delayed = allpass_buffers_[line][allpass_positions_[line]];
            const double result = delayed - gain * source[i];
            allpass_buffers_[line][allpass_positions_[line]] =
                source[i] + gain * result;
            allpass_positions_[line] = (allpass_positions_[line] + 1) % delay;
            output[i] = result;
        }
        return output;
    }

    int fail(const char* message) noexcept {
        std::snprintf(error_, sizeof(error_), "%s", message);
        return -1;
    }

    void reset_state() noexcept {
        latency_to_discard_ = kLatency;
        processed_input_samples_ = 0;
        processed_slots_ = 0;
        position_ = 0;
        maximum_early_delay_ = 0.0;
        history_slots_ = kEarlyHistory + hrtf_slots_;
        history_.assign(kChannels,
                        std::vector<Complex>(
                            static_cast<size_t>(history_slots_) * kHybrid,
                            Complex{0.0, 0.0}));
        sources_.assign(kChannels, SourceState{});
        for (int source = 0; source < kChannels; ++source) {
            sources_[source].position[0] = 0.0;
            sources_[source].position[1] = 1.0;
            sources_[source].position[2] = 0.0;
            sources_[source].profile = 1;
            sources_[source].gain = 1.0;
            sources_[source].enabled = 1;
            sources_[source].special_lfe = 0;
            sources_[source].fade_position = 0;
            sources_[source].fade_total = 0;
            sources_[source].late_current = 0.0;
            sources_[source].late_start = 0.0;
            sources_[source].late_target = 0.0;
            sources_[source].late_fade_position = 0;
            sources_[source].late_fade_total = 0;
        }
        for (int line = 0; line < 4; ++line) {
            fdn_positions_[line] = 0;
            damping_state_[line] = 0.0;
            if (!fdn_buffers_[line].empty()) {
                std::fill(fdn_buffers_[line].begin(), fdn_buffers_[line].end(), 0.0);
            }
        }
        for (int line = 0; line < 2; ++line) {
            allpass_positions_[line] = 0;
            if (!allpass_buffers_[line].empty()) {
                std::fill(allpass_buffers_[line].begin(),
                          allpass_buffers_[line].end(), 0.0);
            }
        }
        late_pos_ = 0;
        late_ring_.fill(0.0);
        error_[0] = '\0';
    }

    QmfAnalysis analysis_;
    HybridAnalysis hybrid_analysis_;
    HybridSynthesis hybrid_synthesis_;
    QmfSynthesis synthesis_;
    bool kernels_ready_ = false;
    bool field_ready_ = false;
    bool room_ready_ = false;

    Complex field_coeff_[kTerms][2][kHybrid];
    double delay_coeff_[kTerms][2];
    double delay_bounds_[2][2];
    double centers_[kHybrid];
    double measurement_radius_ = 1.0;
    uint32_t hrtf_slots_ = 0;

    double dims_[3];
    double listener_[3];
    double wall_gain_[6];
    double speed_ = 343.3;
    uint32_t fdn_delays_[4];
    double fdn_feedback_[4];
    double damping_ = 0.32;
    double fdn_gain_ = 0.22;
    uint32_t allpass_delays_[2];
    double allpass_gains_[2];

    uint32_t latency_to_discard_ = kLatency;
    uint64_t processed_input_samples_ = 0;
    int64_t processed_slots_ = 0;
    uint32_t history_slots_ = kEarlyHistory;
    uint32_t position_ = 0;
    double maximum_early_delay_ = 0.0;
    std::vector<std::vector<Complex>> history_;
    std::vector<SourceState> sources_;

    std::vector<double> fdn_buffers_[4];
    uint32_t fdn_positions_[4];
    double damping_state_[4];
    std::vector<double> allpass_buffers_[2];
    uint32_t allpass_positions_[2];
    std::array<double, kLatency * 2> late_ring_{};
    uint32_t late_pos_ = 0;

    std::array<Complex, kMaxBlockSamples / kHop * kChannels * kQmf> qmf_work_{};
    std::array<Complex, kMaxBlockSamples / kHop * kChannels * kHybrid> hybrid_work_{};
    std::array<Complex, kMaxBlockSamples / kHop * 2 * kHybrid> qmf_synth_work_{};
    std::array<double, kMaxBlockSamples / kHop * 2 * 64> pcm_work_{};
    std::array<double, kMaxBlockSamples * 2> output_{};
    std::array<double, kMaxBlockSamples * kChannels> zeros_{};

    char error_[256];
    bool enable_early_reflections_ = true;
    bool enable_late_room_ = true;
};

}  // namespace ejoc::sofa_binaural

using ejoc::sofa_binaural::Renderer;

extern "C" {

ejoc_sofa_binaural_handle EJOC_CALL ejoc_sofa_binaural_create(void) {
    try {
        return new (std::nothrow) Renderer();
    } catch (...) {
        return nullptr;
    }
}

void EJOC_CALL ejoc_sofa_binaural_destroy(ejoc_sofa_binaural_handle handle) {
    delete static_cast<Renderer*>(handle);
}

int EJOC_CALL ejoc_sofa_binaural_reset(ejoc_sofa_binaural_handle handle) {
    return handle ? static_cast<Renderer*>(handle)->reset() : -1;
}

const char* EJOC_CALL ejoc_sofa_binaural_last_error(
    ejoc_sofa_binaural_handle handle) {
    return handle ? static_cast<Renderer*>(handle)->error() : "null handle";
}

int EJOC_CALL ejoc_sofa_binaural_configure_kernels(
    ejoc_sofa_binaural_handle handle, const double* qmf_analysis,
    const double* hybrid_low, const int16_t* hybrid_indices,
    const double* hybrid_values, uint32_t hybrid_count, const double* qmf_basis,
    const double* qmf_taps) {
    return handle ? static_cast<Renderer*>(handle)->configure_kernels(
        qmf_analysis, hybrid_low, hybrid_indices, hybrid_values, hybrid_count,
        qmf_basis, qmf_taps) : -1;
}

int EJOC_CALL ejoc_sofa_binaural_configure_field(
    ejoc_sofa_binaural_handle handle, const double* coefficients,
    const double* delay_coefficients, const double* delay_bounds,
    const double* band_centers, double measurement_radius_m) {
    return handle ? static_cast<Renderer*>(handle)->configure_field(
        coefficients, delay_coefficients, delay_bounds, band_centers,
        measurement_radius_m) : -1;
}

int EJOC_CALL ejoc_sofa_binaural_configure_room(
    ejoc_sofa_binaural_handle handle, const double* room_dims,
    const double* listener_pos, const double* wall_gains, double speed_of_sound,
    const uint32_t* fdn_delays, const double* fdn_feedback, double damping,
    double fdn_output_gain, const uint32_t* allpass_delays,
    const double* allpass_gains, uint32_t enable_early, uint32_t enable_late) {
    return handle ? static_cast<Renderer*>(handle)->configure_room(
        room_dims, listener_pos, wall_gains, speed_of_sound, fdn_delays,
        fdn_feedback, damping, fdn_output_gain, allpass_delays, allpass_gains,
        enable_early, enable_late) : -1;
}

int EJOC_CALL ejoc_sofa_binaural_set_source(
    ejoc_sofa_binaural_handle handle, uint32_t source, const double* position_adm,
    uint32_t profile, double gain, uint32_t enabled, uint32_t special_lfe,
    uint32_t fade) {
    return handle ? static_cast<Renderer*>(handle)->set_source(
        source, position_adm, profile, gain, enabled, special_lfe, fade) : -1;
}

int EJOC_CALL ejoc_sofa_binaural_process(
    ejoc_sofa_binaural_handle handle, const double* input16_interleaved,
    uint32_t sample_count, double output_gain, double* output_stereo_interleaved) {
    return handle ? static_cast<Renderer*>(handle)->process(
        input16_interleaved, sample_count, output_gain, output_stereo_interleaved)
        : -1;
}

int EJOC_CALL ejoc_sofa_binaural_finish(
    ejoc_sofa_binaural_handle handle, uint32_t flush_samples,
    double* output_stereo_interleaved, uint32_t capacity) {
    return handle ? static_cast<Renderer*>(handle)->finish(
        flush_samples, output_stereo_interleaved, capacity) : -1;
}








}  // extern "C"
