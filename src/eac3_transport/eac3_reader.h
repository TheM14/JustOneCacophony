
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "joc_core.h"

namespace joc::eac3 {

struct Frame {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t offset = 0;  // byte offset of the frame start in the fed stream
};

class FrameReader {
public:
    enum class Next {
        Ok,
        End,
        Fail
    };

    FrameReader() = default;
    FrameReader(const std::uint8_t* data, std::size_t size) {
        push(data, size);
        finish();
    }

    // Appends bytes to the internal buffer (used in incremental mode).
    void push(const std::uint8_t* data, std::size_t size);

    // Declares that no further bytes will arrive; a frame that is still
    void finish() { finished_ = true; }

    Next next(Frame* out);

    joc_error error() const { return error_; }
    const std::string& error_message() const { return message_; }

    std::size_t frames_emitted() const { return frames_emitted_; }
    std::size_t stream_offset() const { return stream_offset_; }

    // report its declared byte length.
    static joc_error frame_bytes(const std::uint8_t* data, std::size_t size, std::size_t offset,
                                 std::size_t* out_frame_bytes);

    static constexpr std::uint16_t kSyncword = 0x0B77;

private:
    void compact();
    void fail(joc_error code, std::string message);

    std::vector<std::uint8_t> buffer_;
    std::size_t consumed_ = 0;      // bytes of buffer_ already turned into frames
    std::size_t base_offset_ = 0;
    bool finished_ = false;
    bool failed_ = false;
    joc_error error_ = JOC_OK;
    std::string message_;
    std::size_t frames_emitted_ = 0;
    std::size_t stream_offset_ = 0;
};

}  // namespace joc::eac3
