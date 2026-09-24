#include "io/adm_writer.h"
#include "foundation/fs_utf8.h"

#include <cstring>
#include <filesystem>

#include "io/wav_writer.h"  // pack_int24 (shared int24 quantisation)

namespace joc::io {

namespace {

constexpr long kDs64BodyOffset = 20;
constexpr long kDataSizeOffset = 76;

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

}  // namespace

AdmBwfWriter::~AdmBwfWriter() { abort(); }

Status AdmBwfWriter::open(const std::string& path, std::size_t block_samples) {
    if (block_samples < JOC_FRAME_SAMPLES) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput,
                            "ADM block size must hold at least one E-AC-3 frame");
    }
    path_ = path;
    block_samples_ = block_samples;
    used_ = 0;
    frames_ = 0;
    finalized_ = false;
    buffer_.assign(block_samples * kChannels, 0.0f);

    file_ = fs_utf8::fopen(path, "wb+");
    if (file_ == nullptr) {
        std::error_code ignored;
        const std::filesystem::path parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ignored);
        }
        file_ = fs_utf8::fopen(path, "wb+");
    }
    if (file_ == nullptr) {
        return Status::fail(JOC_ERR_OUTPUT_OPEN, stage::kOutput, "cannot open " + path);
    }
    std::string header;
    header.append("RF64", 4);
    put_u32(&header, 0xFFFFFFFFu);
    header.append("WAVE", 4);
    if (std::fwrite(header.data(), 1, header.size(), file_) != header.size()) {
        abort();
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "cannot write " + path);
    }
    Status status = write_chunk("ds64", std::string(28, '\0'));
    if (!status.ok()) {
        abort();
        return status;
    }
    std::string fmt;
    put_u16(&fmt, 1);
    put_u16(&fmt, static_cast<std::uint16_t>(kChannels));
    put_u32(&fmt, kRate);
    put_u32(&fmt, kRate * kChannels * 3u);
    put_u16(&fmt, static_cast<std::uint16_t>(kChannels * 3u));
    put_u16(&fmt, 24);
    status = write_chunk("fmt ", fmt);
    if (!status.ok()) {
        abort();
        return status;
    }
    status = write_chunk("data", std::string());
    if (!status.ok()) {
        abort();
        return status;
    }
    return Status::success();
}

Status AdmBwfWriter::write_chunk(const char id[4], const std::string& body) {
    std::string header;
    header.append(id, 4);
    put_u32(&header, static_cast<std::uint32_t>(body.size()));
    if (std::fwrite(header.data(), 1, header.size(), file_) != header.size()) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "chunk header write failed");
    }
    if (!body.empty() &&
        std::fwrite(body.data(), 1, body.size(), file_) != body.size()) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "chunk body write failed");
    }
    if ((body.size() & 1u) != 0u) {
        const char pad = '\0';
        if (std::fwrite(&pad, 1, 1, file_) != 1) {
            return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "chunk padding write failed");
        }
    }
    return Status::success();
}

Status AdmBwfWriter::flush() {
    if (used_ == 0) {
        return Status::success();
    }
    if (file_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kOutput, "ADM writer is not open");
    }
    packed_.clear();
    pack_int24(buffer_.data(), used_, kChannels, &packed_);
    if (std::fwrite(packed_.data(), 1, packed_.size(), file_) != packed_.size()) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "audio write failed for " + path_);
    }
    used_ = 0;
    return Status::success();
}

Status AdmBwfWriter::write_objects16(const float* planar16) {
    if (file_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kOutput, "ADM writer is not open");
    }
    if (planar16 == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput, "null frame");
    }
    std::size_t source = 0;
    while (source < JOC_FRAME_SAMPLES) {
        const std::size_t available = block_samples_ - used_;
        const std::size_t count =
            std::min(available, static_cast<std::size_t>(JOC_FRAME_SAMPLES) - source);
        float* target = buffer_.data() + used_ * kChannels;
        std::memset(target, 0, count * kChannels * sizeof(float));
        for (std::size_t sample = 0; sample < count; ++sample) {
            float* row = target + sample * kChannels;
            row[3] = planar16[0u * JOC_FRAME_SAMPLES + source + sample];
            for (std::size_t object = 0; object < 15u; ++object) {
                row[10u + object] =
                    planar16[(object + 1u) * JOC_FRAME_SAMPLES + source + sample];
            }
        }
        used_ += count;
        source += count;
        if (used_ == block_samples_) {
            const Status status = flush();
            if (!status.ok()) {
                return status;
            }
        }
    }
    frames_ += JOC_FRAME_SAMPLES;
    return Status::success();
}

Status AdmBwfWriter::finalize(const std::string& axml, const std::string& chna,
                              const std::string& dbmd) {
    if (file_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kOutput, "ADM writer is not open");
    }
    if (finalized_) {
        return Status::fail(JOC_ERR_STATE, stage::kOutput, "ADM writer already finalized");
    }
    Status status = flush();
    if (!status.ok()) {
        return status;
    }
    status = write_chunk("axml", axml);
    if (!status.ok()) {
        return status;
    }
    status = write_chunk("chna", chna);
    if (!status.ok()) {
        return status;
    }
    status = write_chunk("dbmd", dbmd);
    if (!status.ok()) {
        return status;
    }

    if (std::fseek(file_, 0, SEEK_END) != 0) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "seek failed for " + path_);
    }
    const long long total = std::ftell(file_);
    if (total < 0) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "tell failed for " + path_);
    }
    const std::uint64_t data_len = frames_ * kChannels * 3u;
    const std::uint32_t data_field =
        data_len <= 0xFFFFFFFFull ? static_cast<std::uint32_t>(data_len) : 0xFFFFFFFFu;

    if (std::fseek(file_, kDataSizeOffset, SEEK_SET) != 0) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "seek failed for " + path_);
    }
    char buffer[4];
    std::memcpy(buffer, &data_field, 4);
    if (std::fwrite(buffer, 1, 4, file_) != 4) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "data size patch failed");
    }

    std::string ds64;
    put_u64(&ds64, static_cast<std::uint64_t>(total) - 8u);
    put_u64(&ds64, data_len);
    put_u64(&ds64, frames_);
    put_u32(&ds64, 0);
    if (std::fseek(file_, kDs64BodyOffset, SEEK_SET) != 0) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "seek failed for " + path_);
    }
    if (std::fwrite(ds64.data(), 1, ds64.size(), file_) != ds64.size()) {
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "ds64 patch failed");
    }
    finalized_ = true;
    if (std::fclose(file_) != 0) {
        file_ = nullptr;
        return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "close failed for " + path_);
    }
    file_ = nullptr;
    return Status::success();
}

void AdmBwfWriter::abort() {
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
    if (!finalized_ && !path_.empty()) {
        fs_utf8::remove(path_);
    }
}

}  // namespace joc::io
