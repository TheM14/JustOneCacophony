// Minimal RIFF/WAVE reader and float32 writer for the multichannel binaural
// frontend.  It accepts the formats ffmpeg writes for PCM input: 16/24/32-bit
// integer, 32/64-bit float, including WAVE_FORMAT_EXTENSIBLE.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"

namespace joc::io {

struct WavAudio {
    std::uint32_t sample_rate = 0;
    std::uint16_t channels = 0;
    std::vector<float> samples;  // interleaved
};

Status read_wav_file(const std::string& path, WavAudio* out);

Status write_wav_float32(const std::string& path, std::uint32_t sample_rate,
                         std::uint16_t channels, const float* interleaved, std::size_t frames);

}  // namespace joc::io
