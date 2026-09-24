
#pragma once

#include <cstddef>
#include <cstdint>

#include "joc_core.h"

namespace joc::bits {

class BitReader {
public:
    BitReader() = default;
    BitReader(const std::uint8_t* data, std::size_t size) { reset(data, size); }

    void reset(const std::uint8_t* data, std::size_t size, std::size_t start_bit = 0) {
        data_ = data;
        size_bits_ = size * 8u;
        pos_ = start_bit;
        limit_ = size_bits_;
        error_ = JOC_OK;
        message_ = "";
    }

    void set_limit_bits(std::size_t limit_bits) {
        limit_ = limit_bits < size_bits_ ? limit_bits : size_bits_;
    }

    std::size_t position() const { return pos_; }
    std::size_t limit() const { return limit_; }
    std::size_t remaining_bits() const { return pos_ <= limit_ ? limit_ - pos_ : 0; }
    const std::uint8_t* data() const { return data_; }

    bool failed() const { return error_ != JOC_OK; }
    joc_error error() const { return error_; }
    const char* error_message() const { return message_; }

    std::uint32_t read(unsigned count) {
        if (count == 0) {
            return 0;
        }
        if (!can_read(count)) {
            fail_truncated(count);
            return 0;
        }
        std::uint32_t value = 0;
        if ((pos_ & 7u) == 0u && count >= 8u) {
            while (count >= 8u) {
                value = (value << 8) | data_[pos_ >> 3];
                pos_ += 8u;
                count -= 8u;
            }
        }
        while (count-- > 0u) {
            const std::uint32_t bit = (data_[pos_ >> 3] >> (7u - (pos_ & 7u))) & 1u;
            value = (value << 1) | bit;
            ++pos_;
        }
        return value;
    }

    std::uint64_t read64(unsigned count) {
        if (count <= 32u) {
            return static_cast<std::uint64_t>(read(count));
        }
        const std::uint64_t high = static_cast<std::uint64_t>(read(count - 32u));
        const std::uint64_t low = static_cast<std::uint64_t>(read(32u));
        return (high << 32) | low;
    }

    bool skip(std::size_t count) {
        if (!can_read(count)) {
            fail_truncated(count);
            return false;
        }
        pos_ += count;
        return true;
    }

    bool read_bytes(std::uint8_t* out, std::size_t count) {
        if (count == 0) {
            return true;
        }
        if (!can_read(count * 8u)) {
            fail_truncated(count * 8u);
            return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<std::uint8_t>(read(8u));
        }
        return true;
    }

    bool can_read(std::size_t count) const {
        return !failed() && count <= limit_ && pos_ <= limit_ - count;
    }

    // semantic check fails, so the reader never continues past it).
    void fail(joc_error code, const char* message) {
        if (!failed()) {
            error_ = code;
            message_ = message;
        }
    }

private:
    void fail_truncated(std::size_t count) {
        fail(JOC_ERR_BITSTREAM_TRUNCATED, "bit read past end of buffer");
        last_request_ = count;
    }

    const std::uint8_t* data_ = nullptr;
    std::size_t size_bits_ = 0;
    std::size_t pos_ = 0;
    std::size_t limit_ = 0;
    std::size_t last_request_ = 0;
    joc_error error_ = JOC_OK;
    const char* message_ = "";
};

// followed by a continuation bit.  Mirrors src/emdf.py:variable_bits().
bool variable_bits(BitReader& reader, unsigned width, unsigned max_groups,
                   std::uint32_t* out_value);

}  // namespace joc::bits
