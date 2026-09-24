#include "eac3_transport/eac3_reader.h"

#include <utility>

#include "foundation/status.h"

namespace joc::eac3 {

namespace {
constexpr std::size_t kHeaderBytes = 4;
}  // namespace

void FrameReader::push(const std::uint8_t* data, std::size_t size) {
    if (failed_ || data == nullptr || size == 0) {
        return;
    }
    if (consumed_ > 0) {
        compact();
    }
    buffer_.insert(buffer_.end(), data, data + size);
}

void FrameReader::compact() {
    if (consumed_ == 0) {
        return;
    }
    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(consumed_));
    base_offset_ += consumed_;
    consumed_ = 0;
}

void FrameReader::fail(joc_error code, std::string message) {
    failed_ = true;
    error_ = code;
    message_ = std::move(message);
}

FrameReader::Next FrameReader::next(Frame* out) {
    if (failed_) {
        return Next::Fail;
    }
    const std::size_t available = buffer_.size() - consumed_;
    if (available == 0) {
        return Next::End;
    }
    const std::uint8_t* p = buffer_.data() + consumed_;

    // The reference implementation rejects a frame whose header does not fit,
    // rather than silently resynchronising on the next 0x0B77.
    if (available < kHeaderBytes) {
        if (finished_) {
            fail(JOC_ERR_EAC3_SYNCFRAME, "E-AC-3 syncframe header truncated at end of input");
            return Next::Fail;
        }
        return Next::End;
    }

    const std::uint16_t syncword = static_cast<std::uint16_t>((static_cast<std::uint16_t>(p[0]) << 8) | p[1]);
    if (syncword != kSyncword) {
        fail(JOC_ERR_EAC3_SYNCFRAME, "invalid E-AC-3 syncword (silent resynchronisation is not allowed)");
        return Next::Fail;
    }

    // frmsiz: 11 bits spread over the low 3 bits of byte 2 and all of byte 3,
    const std::size_t words =
        static_cast<std::size_t>(((p[2] & 0x07u) << 8) | p[3]) + 1u;
    const std::size_t frame_bytes = words * 2u;

    if (frame_bytes > available) {
        if (!finished_) {
            return Next::End;
        }
        fail(JOC_ERR_BITSTREAM_TRUNCATED,
             "last E-AC-3 syncframe extends past end of input (declared " +
                 std::to_string(frame_bytes) + " bytes, remaining " +
                 std::to_string(available) + ")");
        return Next::Fail;
    }

    if (out != nullptr) {
        out->data = p;
        out->size = frame_bytes;
        out->offset = base_offset_ + consumed_;
    }
    consumed_ += frame_bytes;
    stream_offset_ = base_offset_ + consumed_;
    ++frames_emitted_;
    return Next::Ok;
}

joc_error FrameReader::frame_bytes(const std::uint8_t* data, std::size_t size, std::size_t offset,
                                   std::size_t* out_frame_bytes) {
    if (data == nullptr || out_frame_bytes == nullptr) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    if (offset + kHeaderBytes > size) {
        return JOC_ERR_EAC3_SYNCFRAME;
    }
    if (static_cast<std::uint16_t>((static_cast<std::uint16_t>(data[offset]) << 8) | data[offset + 1]) !=
        kSyncword) {
        return JOC_ERR_EAC3_SYNCFRAME;
    }
    const std::size_t words =
        static_cast<std::size_t>(((data[offset + 2] & 0x07u) << 8) | data[offset + 3]) + 1u;
    const std::size_t frame_bytes = words * 2u;
    if (offset + frame_bytes > size) {
        return JOC_ERR_BITSTREAM_TRUNCATED;
    }
    *out_frame_bytes = frame_bytes;
    return JOC_OK;
}

}  // namespace joc::eac3
