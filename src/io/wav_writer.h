// Port of src/speaker_wav.py.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "foundation/status.h"
#include "joc_core.h"

namespace joc::io {

enum class SampleFormat { Float32, Int24 };

struct WavInfo {
    SampleFormat format = SampleFormat::Float32;
    std::uint32_t bits_per_sample = 32;
    std::uint32_t bytes_per_sample = 4;
    std::uint32_t block_align = 0;
    std::uint64_t data_bytes = 0;
    bool rf64 = false;
};

// int24 packing shared by the WAV and ADM writers:
//   trunc(clip(v, -1, 1) * 8388607.0f) with the low three bytes written LE.
// NaN follows NumPy's float->int cast (INT32_MIN) so that the C++ conversion is
// never undefined; the reference passes it through unguarded (plan TD-3.11).
void pack_int24(const float* interleaved, std::size_t frames, std::size_t channels,
                std::string* out);

class WavWriter {
public:
    WavWriter() = default;
    ~WavWriter();

    WavWriter(const WavWriter&) = delete;
    WavWriter& operator=(const WavWriter&) = delete;

    // `total_frames` must be known up front: the header depends on it.
    Status open(const std::string& path, std::uint32_t channels, std::uint32_t rate,
                SampleFormat format, std::uint64_t total_frames);

    Status write(const double* interleaved, std::size_t frames);

    Status finalize();

    const WavInfo& info() const { return info_; }
    std::uint64_t frames_written() const { return frames_written_; }

private:
    std::FILE* file_ = nullptr;
    std::string path_;
    WavInfo info_;
    std::uint32_t channels_ = 0;
    std::uint64_t total_frames_ = 0;
    std::uint64_t frames_written_ = 0;
    bool finalized_ = false;
};

}  // namespace joc::io
