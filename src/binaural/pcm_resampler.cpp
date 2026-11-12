#include "binaural/pcm_resampler.h"

#include <algorithm>
#include <cmath>

namespace joc::multichannel {

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr int kHalfTaps = 32;

double bessel_i0(double value) {
    double sum = 1.0;
    double term = 1.0;
    const double half = value * 0.5;
    for (int k = 1; k < 32; ++k) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < sum * 1.0e-12) {
            break;
        }
    }
    return sum;
}

double kaiser_window(double x) {
    constexpr double kBeta = 8.6;
    const double ratio = std::max(0.0, 1.0 - (x / kHalfTaps) * (x / kHalfTaps));
    return bessel_i0(kBeta * std::sqrt(ratio)) / bessel_i0(kBeta);
}

double sinc(double x) {
    if (std::fabs(x) < 1.0e-12) {
        return 1.0;
    }
    return std::sin(kPi * x) / (kPi * x);
}

}  // namespace

Status resample_interleaved(const float* input, std::size_t frames, int channels,
                            std::uint32_t source_rate, std::uint32_t target_rate,
                            std::vector<float>* output) {
    if (output == nullptr || channels <= 0 || frames == 0) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kFoundation,
                            "resampler needs samples, channels and an output vector");
    }
    if (source_rate == 0 || target_rate == 0) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kFoundation,
                            "resampler sample rates must be positive");
    }
    if (source_rate == target_rate) {
        output->assign(input, input + frames * static_cast<std::size_t>(channels));
        return Status::success();
    }

    const double step = static_cast<double>(source_rate) / static_cast<double>(target_rate);
    const double cutoff = (target_rate < source_rate)
                              ? static_cast<double>(target_rate) / source_rate
                              : 1.0;
    const std::size_t output_frames = static_cast<std::size_t>(std::llround(
        static_cast<double>(frames) * target_rate / source_rate));
    output->assign(output_frames * static_cast<std::size_t>(channels), 0.0f);

    for (std::size_t out_frame = 0; out_frame < output_frames; ++out_frame) {
        const double position = static_cast<double>(out_frame) * step;
        const std::int64_t base = static_cast<std::int64_t>(std::floor(position));
        const double fraction = position - static_cast<double>(base);
        double weights[2 * kHalfTaps + 1];
        double weight_sum = 0.0;
        for (int tap = -kHalfTaps + 1; tap <= kHalfTaps; ++tap) {
            const double distance = tap - fraction;
            const double weight = cutoff * sinc(cutoff * distance) * kaiser_window(distance);
            weights[tap + kHalfTaps] = weight;
            weight_sum += weight;
        }
        if (std::fabs(weight_sum) < 1.0e-12) {
            continue;
        }
        for (int channel = 0; channel < channels; ++channel) {
            double sum = 0.0;
            for (int tap = -kHalfTaps + 1; tap <= kHalfTaps; ++tap) {
                const std::int64_t index = base + tap;
                if (index < 0 || index >= static_cast<std::int64_t>(frames)) {
                    continue;
                }
                sum += static_cast<double>(
                           input[static_cast<std::size_t>(index) *
                                     static_cast<std::size_t>(channels) +
                                 channel]) *
                       weights[tap + kHalfTaps];
            }
            output->at(out_frame * static_cast<std::size_t>(channels) + channel) =
                static_cast<float>(sum / weight_sum);
        }
    }
    return Status::success();
}

}  // namespace joc::multichannel
