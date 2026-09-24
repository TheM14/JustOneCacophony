#include "io/npy_writer.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "foundation/fs_utf8.h"
#include "io/zip_reader.h"

namespace joc::io {

namespace {

constexpr std::size_t kNpyHeaderAlignment = 64;

void append_u16(std::vector<std::uint8_t>* out, std::uint16_t value) {
    out->push_back(static_cast<std::uint8_t>(value & 0xFFu));
    out->push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFu));
}

void append_u32(std::vector<std::uint8_t>* out, std::uint32_t value) {
    for (int index = 0; index < 4; ++index) {
        out->push_back(static_cast<std::uint8_t>((value >> (8 * index)) & 0xFFu));
    }
}

void append_bytes(std::vector<std::uint8_t>* out, const void* data, std::size_t size) {
    const std::uint8_t* bytes = static_cast<const std::uint8_t*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

std::string shape_literal(const std::vector<std::uint64_t>& shape) {
    if (shape.empty()) {
        return "()";
    }
    std::string text = "(";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0u) {
            text += ", ";
        }
        text += std::to_string(shape[index]);
    }
    if (shape.size() == 1u) {
        text += ",";
    }
    text += ")";
    return text;
}

}  // namespace

std::vector<std::uint8_t> npy_image(const std::string& descr,
                                    const std::vector<std::uint64_t>& shape,
                                    const std::vector<std::uint8_t>& data) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                         shape_literal(shape) + ", }";
    // NumPy pads the header so that the payload starts on a 64-byte boundary.
    const std::size_t preamble = 10u;  // magic, version, two byte header length
    std::size_t total = preamble + header.size() + 1u;
    const std::size_t padding = (kNpyHeaderAlignment - (total % kNpyHeaderAlignment)) %
                                kNpyHeaderAlignment;
    header.append(padding, ' ');
    header.push_back('\n');

    std::vector<std::uint8_t> out;
    out.reserve(preamble + header.size() + data.size());
    static const std::uint8_t kMagic[6] = {0x93u, 'N', 'U', 'M', 'P', 'Y'};
    append_bytes(&out, kMagic, sizeof(kMagic));
    out.push_back(1u);  // major
    out.push_back(0u);  // minor
    append_u16(&out, static_cast<std::uint16_t>(header.size()));
    append_bytes(&out, header.data(), header.size());
    append_bytes(&out, data.data(), data.size());
    return out;
}

std::vector<std::uint8_t> zip_bytes(const std::vector<NpyMember>& members) {
    std::vector<std::uint8_t> out;
    struct Entry {
        std::string name;
        std::uint32_t crc = 0;
        std::uint32_t size = 0;
        std::uint32_t offset = 0;
    };
    std::vector<Entry> entries;
    entries.reserve(members.size());

    for (const NpyMember& member : members) {
        const std::string name = member.name + ".npy";
        const std::vector<std::uint8_t> payload = npy_image(member.descr, member.shape, member.data);
        Entry entry;
        entry.name = name;
        entry.crc = crc32_of(payload.data(), payload.size());
        entry.size = static_cast<std::uint32_t>(payload.size());
        entry.offset = static_cast<std::uint32_t>(out.size());
        entries.push_back(entry);

        append_u32(&out, 0x04034B50u);  // local file header
        append_u16(&out, 20u);          // version needed
        append_u16(&out, 0u);           // flags
        append_u16(&out, 0u);           // method: stored
        append_u16(&out, 0u);           // time
        append_u16(&out, 0x2821u);      // date: 2000-01-01, fixed for reproducibility
        append_u32(&out, entry.crc);
        append_u32(&out, entry.size);
        append_u32(&out, entry.size);
        append_u16(&out, static_cast<std::uint16_t>(name.size()));
        append_u16(&out, 0u);  // extra length
        append_bytes(&out, name.data(), name.size());
        append_bytes(&out, payload.data(), payload.size());
    }

    const std::uint32_t directory_offset = static_cast<std::uint32_t>(out.size());
    for (const Entry& entry : entries) {
        append_u32(&out, 0x02014B50u);  // central directory header
        append_u16(&out, 20u);          // version made by
        append_u16(&out, 20u);          // version needed
        append_u16(&out, 0u);           // flags
        append_u16(&out, 0u);           // method: stored
        append_u16(&out, 0u);           // time
        append_u16(&out, 0x2821u);      // date
        append_u32(&out, entry.crc);
        append_u32(&out, entry.size);
        append_u32(&out, entry.size);
        append_u16(&out, static_cast<std::uint16_t>(entry.name.size()));
        append_u16(&out, 0u);  // extra
        append_u16(&out, 0u);  // comment
        append_u16(&out, 0u);  // disk
        append_u16(&out, 0u);  // internal attributes
        append_u32(&out, 0u);  // external attributes
        append_u32(&out, entry.offset);
        append_bytes(&out, entry.name.data(), entry.name.size());
    }
    const std::uint32_t directory_size = static_cast<std::uint32_t>(out.size()) - directory_offset;

    append_u32(&out, 0x06054B50u);  // end of central directory
    append_u16(&out, 0u);
    append_u16(&out, 0u);
    append_u16(&out, static_cast<std::uint16_t>(entries.size()));
    append_u16(&out, static_cast<std::uint16_t>(entries.size()));
    append_u32(&out, directory_size);
    append_u32(&out, directory_offset);
    append_u16(&out, 0u);
    return out;
}

bool write_zip(const std::string& path, const std::vector<NpyMember>& members,
               std::string* error) {
    const std::vector<std::uint8_t> bytes = zip_bytes(members);
    std::FILE* stream = fs_utf8::fopen(path, "wb");
    if (stream == nullptr) {
        if (error != nullptr) {
            *error = "cannot open " + path + " for writing";
        }
        return false;
    }
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), stream);
    const bool flushed = std::fclose(stream) == 0;
    if (written != bytes.size() || !flushed) {
        if (error != nullptr) {
            *error = "short write to " + path;
        }
        return false;
    }
    return true;
}

std::vector<std::uint8_t> utf8_to_utf32le(const std::string& text) {
    std::vector<std::uint8_t> out;
    out.reserve(text.size() * 4u);
    std::size_t index = 0;
    while (index < text.size()) {
        const std::uint8_t lead = static_cast<std::uint8_t>(text[index]);
        std::uint32_t code = 0;
        std::size_t extra = 0;
        if (lead < 0x80u) {
            code = lead;
        } else if ((lead & 0xE0u) == 0xC0u) {
            code = lead & 0x1Fu;
            extra = 1;
        } else if ((lead & 0xF0u) == 0xE0u) {
            code = lead & 0x0Fu;
            extra = 2;
        } else if ((lead & 0xF8u) == 0xF0u) {
            code = lead & 0x07u;
            extra = 3;
        } else {
            code = 0xFFFDu;  // invalid lead byte: substitute rather than fail
            extra = 0;
        }
        ++index;
        for (std::size_t count = 0; count < extra && index < text.size(); ++count) {
            code = (code << 6) | (static_cast<std::uint8_t>(text[index]) & 0x3Fu);
            ++index;
        }
        for (int byte = 0; byte < 4; ++byte) {
            out.push_back(static_cast<std::uint8_t>((code >> (8 * byte)) & 0xFFu));
        }
    }
    return out;
}

std::string python_float_repr(double value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "Infinity" : "-Infinity";
    }
    // to_chars gives the shortest round-trip digits; Python's repr uses the same
    // digits but its own notation, so the digits are re-laid-out here.
    std::array<char, 64> buffer{};
    const std::to_chars_result converted =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    std::string text(buffer.data(), converted.ptr);
    const bool negative = !text.empty() && text[0] == '-';
    const std::string body = negative ? text.substr(1) : text;
    const std::size_t exponent_at = body.find_first_of("eE");
    std::string digits = body;
    int exponent = 0;
    if (exponent_at != std::string::npos) {
        digits = body.substr(0, exponent_at);
        exponent = std::atoi(body.c_str() + exponent_at + 1);
    }
    const std::size_t point = digits.find('.');
    std::string mantissa = digits;
    if (point != std::string::npos) {
        mantissa = digits.substr(0, point) + digits.substr(point + 1);
        exponent += static_cast<int>(point) - 1;
    } else {
        exponent += static_cast<int>(digits.size()) - 1;
    }
    while (mantissa.size() > 1u && mantissa.back() == '0') {
        mantissa.pop_back();
    }
    // Python switches to exponent notation below 1e-4 and at 1e16 and above.
    std::string result;
    if (exponent < -4 || exponent >= 16) {
        result = mantissa.substr(0, 1);
        if (mantissa.size() > 1u) {
            result += "." + mantissa.substr(1);
        }
        char tail[16];
        std::snprintf(tail, sizeof(tail), "e%+03d", exponent);
        result += tail;
    } else if (exponent >= 0) {
        if (static_cast<std::size_t>(exponent) + 1u >= mantissa.size()) {
            result = mantissa + std::string(static_cast<std::size_t>(exponent) + 1u - mantissa.size(), '0');
            result += ".0";
        } else {
            result = mantissa.substr(0, static_cast<std::size_t>(exponent) + 1u) + "." +
                     mantissa.substr(static_cast<std::size_t>(exponent) + 1u);
        }
    } else {
        result = "0." + std::string(static_cast<std::size_t>(-exponent - 1), '0') + mantissa;
    }
    return negative ? "-" + result : result;
}

}  // namespace joc::io
