#include "io/wav_writer.h"
#include "foundation/fs_utf8.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

namespace joc::io {

namespace {

constexpr std::uint16_t kWaveFormatPcm = 0x0001;
constexpr std::uint16_t kWaveFormatIeeeFloat = 0x0003;
constexpr std::uint16_t kWaveFormatExtensible = 0xFFFE;
constexpr std::uint8_t kPcmGuid[16] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
                                  0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
constexpr std::uint8_t kFloatGuid[16] = {0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
                                    0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};

void put_u16(std::string* out, std::uint16_t value) {
    char buffer[2];
    std::memcpy(buffer, &value, 2);
    out->append(buffer, 2);
}

void put_u32(std::string* out, std::uint32_t value) {
    char buffer[4];
    std::memcpy(buffer, &value, 4);
    out->append(buffer, 4);
}

void put_u64(std::string* out, std::uint64_t value) {
    char buffer[8];
    std::memcpy(buffer, &value, 8);
    out->append(buffer, 8);
}

// Port of speaker_wav._fmt_chunk.
std::string fmt_chunk(std::uint32_t channels, std::uint32_t rate, SampleFormat format, WavInfo* info) {
    std::uint16_t simple_tag = 0;
    const std::uint8_t* guid = nullptr;
    if (format == SampleFormat::Float32) {
        info->bits_per_sample = 32;
        info->bytes_per_sample = 4;
        simple_tag = kWaveFormatIeeeFloat;
        guid = kFloatGuid;
    } else {
        info->bits_per_sample = 24;
        info->bytes_per_sample = 3;
        simple_tag = kWaveFormatPcm;
        guid = kPcmGuid;
    }
    const std::uint32_t block_align = channels * info->bytes_per_sample;
    const std::uint32_t byte_rate = rate * block_align;
    info->block_align = block_align;

    std::string body;
    if (channels <= 2) {
        put_u16(&body, simple_tag);
        put_u16(&body, static_cast<std::uint16_t>(channels));
        put_u32(&body, rate);
        put_u32(&body, byte_rate);
        put_u16(&body, static_cast<std::uint16_t>(block_align));
        put_u16(&body, static_cast<std::uint16_t>(info->bits_per_sample));
    } else {
        put_u16(&body, kWaveFormatExtensible);
        put_u16(&body, static_cast<std::uint16_t>(channels));
        put_u32(&body, rate);
        put_u32(&body, byte_rate);
        put_u16(&body, static_cast<std::uint16_t>(block_align));
        put_u16(&body, static_cast<std::uint16_t>(info->bits_per_sample));
        put_u16(&body, 22);
        put_u16(&body, static_cast<std::uint16_t>(info->bits_per_sample));
        put_u32(&body, 0);
        body.append(reinterpret_cast<const char*>(guid), 16);
    }
    return body;
}

// int32 conversion identical to NumPy's float32 -> int32 cast after clipping.
std::int32_t to_int32(const float value) {
    if (!std::isfinite(value)) {
        return std::numeric_limits<std::int32_t>::min();
    }
    return static_cast<std::int32_t>(value);
}

}  // namespace

void pack_int24(const float* interleaved, std::size_t frames, std::size_t channels,
                std::string* out) {
    const std::size_t count = frames * channels;
    out->resize(count * 3);
    char* target = out->data();
    for (std::size_t i = 0; i < count; ++i) {
        float value = interleaved[i];
        if (value > 1.0f) {
            value = 1.0f;
        } else if (value < -1.0f) {
            value = -1.0f;
        }
        const std::int32_t scaled = to_int32(value * 8388607.0f);
        const std::uint32_t bits = static_cast<std::uint32_t>(scaled);
        target[i * 3 + 0] = static_cast<char>(bits & 0xFFu);
        target[i * 3 + 1] = static_cast<char>((bits >> 8) & 0xFFu);
        target[i * 3 + 2] = static_cast<char>((bits >> 16) & 0xFFu);
    }
}

WavWriter::~WavWriter() {
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

Status WavWriter::open(const std::string& path, std::uint32_t channels, std::uint32_t rate,
                       SampleFormat format, std::uint64_t total_frames) {
    if (channels == 0 || rate == 0) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput,
                            "WAV writer needs a positive channel count and rate");
    }
    path_ = path;
    channels_ = channels;
    total_frames_ = total_frames;
    frames_written_ = 0;
    finalized_ = false;
    info_ = WavInfo{};
    info_.format = format;

    const std::string fmt = fmt_chunk(channels, rate, format, &info_);
    const std::uint64_t data_size = total_frames * info_.block_align;
    info_.data_bytes = data_size;
    const std::uint64_t riff_file_size = 12u + 8u + fmt.size() + 8u + data_size;
    const bool rf64 = (riff_file_size - 8u) > 0xFFFFFFFFull;
    info_.rf64 = rf64;

    std::string header;
    if (rf64) {
        const std::uint64_t file_size = 12u + 36u + 8u + fmt.size() + 8u + data_size;
        header.append("RF64", 4);
        put_u32(&header, 0xFFFFFFFFu);
        header.append("WAVE", 4);
        header.append("ds64", 4);
        put_u32(&header, 28);
        put_u64(&header, file_size - 8u);
        put_u64(&header, data_size);
        put_u64(&header, total_frames);
        put_u32(&header, 0);
    } else {
        header.append("RIFF", 4);
        put_u32(&header, static_cast<std::uint32_t>(riff_file_size - 8u));
        header.append("WAVE", 4);
    }
    header.append("fmt ", 4);
    put_u32(&header, static_cast<std::uint32_t>(fmt.size()));
    header.append(fmt);
    header.append("data", 4);
    put_u32(&header, rf64 ? 0xFFFFFFFFu : static_cast<std::uint32_t>(data_size));

    file_ = fs_utf8::fopen(path, "wb");
    if (file_ == nullptr) {
        // The reference creates the parent directory itself.
        std::error_code ignored;
        const std::filesystem::path parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ignored);
        }
        file_ = fs_utf8::fopen(path, "wb");
    }
    if (file_ == nullptr) {
        return Status::fail(JOC_ERR_OUTPUT_OPEN, stage::kOutput, "cannot open " + path);
    }
    if (std::fwrite(header.data(), 1, header.size(), file_) != header.size()) {
        std::fclose(file_);
        file_ = nullptr;
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "cannot write header to " + path);
    }
    return Status::success();
}

Status WavWriter::write(const double* interleaved, std::size_t frames) {
    if (file_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kOutput, "WAV writer is not open");
    }
    if (frames == 0) {
        return Status::success();
    }
    if (frames_written_ + frames > total_frames_) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput,
                            "WAV writer received more frames than the header declared (declared " +
                                std::to_string(total_frames_) + ", written " +
                                std::to_string(frames_written_) + ", requested " +
                                std::to_string(frames) + ")");
    }
    const std::size_t count = frames * channels_;
    if (info_.format == SampleFormat::Float32) {
        std::vector<float> converted(count);
        for (std::size_t i = 0; i < count; ++i) {
            converted[i] = static_cast<float>(interleaved[i]);
        }
        if (std::fwrite(converted.data(), sizeof(float), count, file_) != count) {
            return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "write failed for " + path_);
        }
    } else {
        std::vector<float> converted(count);
        for (std::size_t i = 0; i < count; ++i) {
            converted[i] = static_cast<float>(interleaved[i]);
        }
        std::string packed;
        pack_int24(converted.data(), frames, channels_, &packed);
        if (std::fwrite(packed.data(), 1, packed.size(), file_) != packed.size()) {
            return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "write failed for " + path_);
        }
    }
    frames_written_ += frames;
    return Status::success();
}

Status WavWriter::finalize() {
    if (file_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kOutput, "WAV writer is not open");
    }
    if (frames_written_ != total_frames_) {
        std::fclose(file_);
        file_ = nullptr;
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput,
                            "WAV writer wrote " + std::to_string(frames_written_) + " of " +
                                std::to_string(total_frames_) + " frames");
    }
    const int result = std::fclose(file_);
    file_ = nullptr;
    finalized_ = true;
    if (result != 0) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "close failed for " + path_);
    }
    return Status::success();
}

}  // namespace joc::io
