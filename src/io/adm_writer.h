
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "foundation/status.h"
#include "joc_core.h"

namespace joc::io {

class AdmBwfWriter {
public:
    static constexpr std::uint32_t kChannels = 25;
    static constexpr std::uint32_t kRate = 48000;
    static constexpr std::size_t kDefaultBlockSamples = 131072;

    AdmBwfWriter() = default;
    ~AdmBwfWriter();

    AdmBwfWriter(const AdmBwfWriter&) = delete;
    AdmBwfWriter& operator=(const AdmBwfWriter&) = delete;

    Status open(const std::string& path, std::size_t block_samples = kDefaultBlockSamples);

    Status write_objects16(const float* planar16);

    Status finalize(const std::string& axml, const std::string& chna, const std::string& dbmd);

    // Closes and removes a file that was never finalized (plan 28.3: abort must
    void abort();

    std::uint64_t frames() const { return frames_; }
    bool open_ok() const { return file_ != nullptr; }

private:
    Status write_chunk(const char id[4], const std::string& body);
    Status flush();

    std::FILE* file_ = nullptr;
    std::string path_;
    std::size_t block_samples_ = kDefaultBlockSamples;
    std::size_t used_ = 0;
    std::uint64_t frames_ = 0;
    std::vector<float> buffer_;
    std::string packed_;
    bool finalized_ = false;
};

}  // namespace joc::io
