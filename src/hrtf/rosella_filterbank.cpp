#include "hrtf/rosella_filterbank.h"

#include <algorithm>
#include <array>

#include "foundation/fft.h"
#include "hrtf/rosella_filterbank_tables.h"

namespace joc::hrtf {
namespace {

constexpr std::size_t kBands = 64;
constexpr std::size_t kQmfTaps = 10;
constexpr std::size_t kHybridTaps = 11;
constexpr std::size_t kHybridDelay = 3;
constexpr std::array<std::size_t, 16> kHybridOrder = {0,  7, 1,  6,  2, 9,  3,  4,
                                                      10, 5, 11, 15, 8, 12, 13, 14};

const dsp::FftPlan& qmf_plan() {
    static const dsp::FftPlan plan(kBands, false);
    return plan;
}

const dsp::FftPlan& hybrid_plan(std::size_t bands) {
    static const dsp::FftPlan eight(8, false);
    static const dsp::FftPlan four(4, false);
    return bands == 8 ? eight : four;
}

double tap_sign(std::size_t tap) { return (tap / 2) % 2 == 0 ? 1.0 : -1.0; }

}

RosellaFilterbank::RosellaFilterbank(std::size_t channels)
    : channels_(channels), analysis_history_(kQmfTaps * channels * kBands),
      hybrid_history_(kHybridTaps * channels * kBands),
      synthesis_history_(kQmfTaps * channels * kBands), spectrum_(kBands), hybrid_eight_(8),
      hybrid_four_(4) {}

void RosellaFilterbank::reset() {
    analysis_position_ = 0;
    hybrid_position_ = 0;
    synthesis_position_ = 0;
    std::fill(analysis_history_.begin(), analysis_history_.end(), 0.0);
    std::fill(hybrid_history_.begin(), hybrid_history_.end(), Complex{});
    std::fill(synthesis_history_.begin(), synthesis_history_.end(), Complex{});
}

void RosellaFilterbank::analyze_qmf(const std::vector<double>& hops, std::size_t slots,
                                    std::vector<Complex>* qmf) {
    const std::size_t stride = channels_ * kBands;
    qmf->resize(slots * stride);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        std::copy_n(hops.data() + slot * stride, stride,
                    analysis_history_.data() + analysis_position_ * stride);
        for (std::size_t channel = 0; channel < channels_; ++channel) {
            for (std::size_t band = 0; band < kBands; ++band) {
                double even = 0.0;
                double odd = 0.0;
                for (std::size_t tap = 0; tap < kQmfTaps; ++tap) {
                    const std::size_t history_slot = (analysis_position_ + tap + 1) % kQmfTaps;
                    const double value =
                        analysis_history_[history_slot * stride + channel * kBands + band];
                    const double contribution =
                        value * 0.5 * rosella_filterbank_tables::kPrototype[band * kQmfTaps + tap] *
                        tap_sign(tap);
                    (tap % 2 == 0 ? even : odd) += contribution;
                }
                spectrum_[band] = Complex(odd, even) * rosella_filterbank_tables::kRotation[band];
            }
            dsp::fft_radix2(&spectrum_, qmf_plan());
            Complex* destination = qmf->data() + slot * stride + channel * kBands;
            for (std::size_t band = 0; band < kBands / 2; ++band) {
                destination[2 * band] = spectrum_[band] / 64.0;
                destination[2 * band + 1] = std::conj(spectrum_[63 - band]) / 64.0;
            }
        }
        analysis_position_ = (analysis_position_ + 1) % kQmfTaps;
    }
}

void RosellaFilterbank::analyze_hybrid(const std::vector<Complex>& qmf, std::size_t slots,
                                       std::vector<Complex>* hybrid) {
    const std::size_t stride = channels_ * kBands;
    hybrid->resize(slots * channels_ * kHybridBandCount);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        std::copy_n(qmf.data() + slot * stride, stride,
                    hybrid_history_.data() + hybrid_position_ * stride);
        for (std::size_t channel = 0; channel < channels_; ++channel) {
            std::array<Complex, 16> low{};
            for (std::size_t band = 0; band < 3; ++band) {
                std::vector<Complex>& spectrum = band == 0 ? hybrid_eight_ : hybrid_four_;
                std::fill(spectrum.begin(), spectrum.end(), Complex{});
                for (std::size_t tap = 0; tap < kHybridTaps; ++tap) {
                    const std::size_t history_slot = (hybrid_position_ + tap + 1) % kHybridTaps;
                    const std::size_t bin =
                        (tap + kHybridDelay + 1 + 2 * spectrum.size() - kHybridTaps) %
                        spectrum.size();
                    spectrum[bin] +=
                        hybrid_history_[history_slot * stride + channel * kBands + band] *
                        rosella_filterbank_tables::kHybrid[(band == 0 ? 0 : kHybridTaps) + tap];
                }
                dsp::fft_radix2(&spectrum, hybrid_plan(spectrum.size()));
                std::copy(spectrum.begin(), spectrum.end(),
                          low.begin() + (band == 0 ? 0 : 4 + band * 4));
            }
            Complex* destination = hybrid->data() + (slot * channels_ + channel) * kHybridBandCount;
            for (std::size_t band = 0; band < low.size(); ++band) {
                destination[band] = low[kHybridOrder[band]];
            }
            destination[1] = std::conj(destination[1]);
            destination[3] = std::conj(destination[3]);
            const std::size_t delayed =
                (hybrid_position_ + kHybridTaps - kHybridDelay) % kHybridTaps;
            std::copy_n(hybrid_history_.data() + delayed * stride + channel * kBands + 3,
                        kBands - 3, destination + 16);
        }
        hybrid_position_ = (hybrid_position_ + 1) % kHybridTaps;
    }
}

void RosellaFilterbank::synthesize_hybrid(const std::vector<Complex>& hybrid, std::size_t slots,
                                          std::vector<Complex>* qmf) {
    qmf->resize(slots * channels_ * kBands);
    for (std::size_t row = 0; row < slots * channels_; ++row) {
        const Complex* source = hybrid.data() + row * kHybridBandCount;
        Complex* destination = qmf->data() + row * kBands;
        destination[0] = source[0] + source[2] + source[4] + source[6] + source[7] + source[9] +
                         std::conj(source[3]) + std::conj(source[1]);
        destination[1] = source[12] + source[5] + source[8] + source[10];
        destination[2] = source[13] + source[14] + source[15] + source[11];
        std::copy_n(source + 16, kBands - 3, destination + 3);
    }
}

void RosellaFilterbank::synthesize_qmf(const std::vector<Complex>& qmf, std::size_t slots,
                                       std::vector<double>* hops) {
    const std::size_t stride = channels_ * kBands;
    hops->resize(slots * stride);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        for (std::size_t channel = 0; channel < channels_; ++channel) {
            const Complex* source = qmf.data() + slot * stride + channel * kBands;
            for (std::size_t band = 0; band < kBands / 2; ++band) {
                spectrum_[band] = std::conj(source[2 * band]);
                spectrum_[32 + band] = source[63 - 2 * band];
            }
            dsp::fft_radix2(&spectrum_, qmf_plan());
            Complex* current =
                synthesis_history_.data() + synthesis_position_ * stride + channel * kBands;
            for (std::size_t band = 0; band < kBands; ++band) {
                current[band] = spectrum_[band] * rosella_filterbank_tables::kRotation[band];
            }
            for (std::size_t band = 0; band < kBands; ++band) {
                double value = 0.0;
                for (std::size_t lag = 0; lag < kQmfTaps; ++lag) {
                    const std::size_t history_slot =
                        (synthesis_position_ + kQmfTaps - lag) % kQmfTaps;
                    const Complex sample =
                        synthesis_history_[history_slot * stride + channel * kBands + band];
                    const double component = lag % 2 == 0 ? sample.real() : sample.imag();
                    value +=
                        component *
                        rosella_filterbank_tables::kPrototype[(63 - band) * kQmfTaps + 9 - lag] *
                        tap_sign(lag) * 2.0;
                }
                (*hops)[slot * stride + channel * kBands + band] = value;
            }
        }
        synthesis_position_ = (synthesis_position_ + 1) % kQmfTaps;
    }
}

}
