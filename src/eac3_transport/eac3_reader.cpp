#include "eac3_transport/eac3_reader.h"

#include <utility>

#include "foundation/status.h"

namespace joc::eac3 {

namespace {

constexpr std::size_t kHeaderBytes = 4;
constexpr std::size_t kHeaderBytesAc3 = 6;

// AC-3 syncframe length in 16-bit words, indexed by fscod (0 = 48 kHz,
// 1 = 44.1 kHz, 2 = 32 kHz) and then frmsizecod (A/52 5.4.1.2).
constexpr std::uint16_t kAc3FrameWords[3][38] = {
    {64, 64, 80, 80, 96, 96, 112, 112, 128, 128, 160, 160, 192, 192, 224, 224,
     256, 256, 320, 320, 384, 384, 448, 448, 512, 512, 640, 640, 768, 768,
     896, 896, 1024, 1024, 1152, 1152, 1280, 1280},
    {69, 70, 87, 88, 104, 105, 121, 122, 139, 140, 174, 175, 208, 209, 243, 244,
     278, 278, 348, 348, 417, 418, 487, 487, 557, 557, 696, 696, 835, 835,
     975, 975, 1114, 1114, 1253, 1253, 1393, 1393},
    {96, 96, 120, 120, 144, 144, 168, 168, 192, 192, 240, 240, 288, 288, 336, 336,
     384, 384, 480, 480, 576, 576, 672, 672, 768, 768, 960, 960, 1152, 1152,
     1344, 1344, 1536, 1536, 1728, 1728, 1920, 1920},
};

std::uint32_t read_bits(const std::uint8_t* data, std::size_t start, std::size_t count) {
    std::uint32_t value = 0;
    for (std::size_t index = start; index < start + count; ++index) {
        const std::uint32_t bit = (data[index >> 3] >> (7u - (index & 7u))) & 1u;
        value = (value << 1) | bit;
    }
    return value;
}

}  // namespace

bool parse_syncframe_header(const std::uint8_t* data, std::size_t size, std::size_t offset,
                            Frame* out) {
    if (data == nullptr || offset + kHeaderBytesAc3 > size) {
        return false;
    }
    if (static_cast<std::uint16_t>((static_cast<std::uint16_t>(data[offset]) << 8) |
                                   data[offset + 1]) != FrameReader::kSyncword) {
        return false;
    }
    Frame frame;
    frame.data = data + offset;
    frame.offset = offset;
    const std::size_t base = offset * 8u;
    const std::uint32_t bsid = read_bits(data, base + 40u, 5u);
    frame.bsid = static_cast<std::uint8_t>(bsid);

    if (bsid <= 8u) {
        frame.codec = 0;
        const std::uint32_t fscod = read_bits(data, base + 32u, 2u);
        const std::uint32_t frmsizecod = read_bits(data, base + 34u, 6u);
        if (fscod > 2u || frmsizecod >= 38u) {
            return false;
        }
        frame.size = static_cast<std::size_t>(kAc3FrameWords[fscod][frmsizecod]) * 2u;
        frame.acmod = static_cast<std::uint8_t>(read_bits(data, base + 48u, 3u));
        std::size_t bit = 51u;
        if ((frame.acmod & 0x1u) != 0u && frame.acmod != 0x1u) {
            bit += 2u;  // cmixlev
        }
        if ((frame.acmod & 0x4u) != 0u) {
            bit += 2u;  // surmixlev
        }
        if (frame.acmod == 0x2u) {
            bit += 2u;  // dsurmod
        }
        frame.lfeon = static_cast<std::uint8_t>(read_bits(data, base + bit, 1u));
        if (out != nullptr) {
            *out = frame;
        }
        return true;
    }

    frame.codec = 1;
    frame.strmtyp = static_cast<std::uint8_t>(read_bits(data, base + 16u, 2u));
    frame.substreamid = static_cast<std::uint8_t>(read_bits(data, base + 18u, 3u));
    const std::uint32_t frmsiz = read_bits(data, base + 21u, 11u);
    frame.size = (static_cast<std::size_t>(frmsiz) + 1u) * 2u;
    // fscod==3 also consumes two bits (fscod2), so acmod always starts at bit 36.
    frame.acmod = static_cast<std::uint8_t>(read_bits(data, base + 36u, 3u));
    frame.lfeon = static_cast<std::uint8_t>(read_bits(data, base + 39u, 1u));
    std::size_t bit = 45u + 5u;  // bsid, then dialnorm
    const std::uint32_t compre = read_bits(data, base + bit, 1u);
    ++bit;
    if (compre != 0u) {
        bit += 8u;  // compr
    }
    if (frame.acmod == 0u) {
        bit += 5u;  // dialnorm2
        const std::uint32_t compr2 = read_bits(data, base + bit, 1u);
        ++bit;
        if (compr2 != 0u) {
            bit += 8u;
        }
    }
    if (frame.strmtyp == 1u) {
        if (read_bits(data, base + bit, 1u) != 0u) {
            frame.has_chanmap = 1;
            frame.chanmap = static_cast<std::uint16_t>(read_bits(data, base + bit + 1u, 16u));
        }
    }
    if (out != nullptr) {
        *out = frame;
    }
    return true;
}

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

    // A frame whose header does not fit is rejected rather than silently
    // resynchronised on the next 0x0B77.
    if (available < kHeaderBytes) {
        if (finished_) {
            fail(JOC_ERR_EAC3_SYNCFRAME, "syncframe header truncated at end of input");
            return Next::Fail;
        }
        return Next::End;
    }

    Frame frame;
    if (!parse_syncframe_header(p, available, 0, &frame) || frame.size == 0u) {
        if (available < kHeaderBytesAc3 && !finished_) {
            return Next::End;
        }
        fail(JOC_ERR_EAC3_SYNCFRAME,
             "invalid syncframe header (silent resynchronisation is not allowed)");
        return Next::Fail;
    }

    if (frame.size > available) {
        if (!finished_) {
            return Next::End;
        }
        fail(JOC_ERR_BITSTREAM_TRUNCATED,
             "last syncframe extends past end of input (declared " +
                 std::to_string(frame.size) + " bytes, remaining " +
                 std::to_string(available) + ")");
        return Next::Fail;
    }

    if (out != nullptr) {
        frame.data = p;
        frame.offset = base_offset_ + consumed_;
        *out = frame;
    }
    consumed_ += frame.size;
    stream_offset_ = base_offset_ + consumed_;
    ++frames_emitted_;
    return Next::Ok;
}

joc_error FrameReader::frame_bytes(const std::uint8_t* data, std::size_t size, std::size_t offset,
                                   std::size_t* out_frame_bytes) {
    if (data == nullptr || out_frame_bytes == nullptr) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    Frame frame;
    if (!parse_syncframe_header(data, size, offset, &frame)) {
        return offset + kHeaderBytes > size ? JOC_ERR_EAC3_SYNCFRAME
                                            : JOC_ERR_EAC3_SYNCFRAME;
    }
    if (offset + frame.size > size) {
        return JOC_ERR_BITSTREAM_TRUNCATED;
    }
    *out_frame_bytes = frame.size;
    return JOC_OK;
}

}  // namespace joc::eac3
