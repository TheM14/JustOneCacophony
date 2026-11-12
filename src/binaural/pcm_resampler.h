// Rational sample-rate conversion for the multichannel binaural frontend.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "foundation/status.h"

namespace joc::multichannel {

// Resamples interleaved PCM with a windowed-sinc low-pass.  The binaural
// runtime is native 48 kHz, so any other input rate passes through this first.
Status resample_interleaved(const float* input, std::size_t frames, int channels,
                            std::uint32_t source_rate, std::uint32_t target_rate,
                            std::vector<float>* output);

}  // namespace joc::multichannel
