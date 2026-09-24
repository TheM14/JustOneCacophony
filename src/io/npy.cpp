#include "io/npy.h"

#include <cstring>

namespace joc::io {

namespace {

std::uint16_t read_u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
std::uint32_t read_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

NpyType classify(const std::string& descr) {
    if (descr == "<f8" || descr == "=f8" || descr == "|f8") { return NpyType::Float64; }
    if (descr == "<f4" || descr == "=f4") { return NpyType::Float32; }
    if (descr == "<i8" || descr == "=i8") { return NpyType::Int64; }
    if (descr == "<i4" || descr == "=i4") { return NpyType::Int32; }
    if (descr == "<i2" || descr == "=i2") { return NpyType::Int16; }
    if (descr == "|u1" || descr == "<u1") { return NpyType::UInt8; }
    if (descr == "<c16" || descr == "=c16") { return NpyType::Complex128; }
    if (descr.size() > 2 && descr[0] == '<' && descr[1] == 'U') {
        return NpyType::Unicode;
    }
    if (descr.size() > 2 && descr[0] == '=' && descr[1] == 'U') {
        return NpyType::Unicode;
    }
    return NpyType::Unknown;
}

std::size_t unicode_length(const std::string& descr) {
    std::size_t index = 0;
    while (index < descr.size() && (descr[index] == '<' || descr[index] == '=')) {
        ++index;
    }
    if (index >= descr.size() || descr[index] != 'U') {
        return 0;
    }
    ++index;
    std::size_t value = 0;
    bool any = false;
    while (index < descr.size() && descr[index] >= '0' && descr[index] <= '9') {
        value = value * 10 + static_cast<std::size_t>(descr[index] - '0');
        ++index;
        any = true;
    }
    return any ? value : 0;
}

bool is_big_endian(const std::string& descr) { return !descr.empty() && descr[0] == '>'; }

bool header_value(const std::string& header, const std::string& key, std::string* out) {
    const std::string needle = "'" + key + "'";
    const std::size_t position = header.find(needle);
    if (position == std::string::npos) {
        return false;
    }
    const std::size_t colon = header.find(':', position + needle.size());
    if (colon == std::string::npos) {
        return false;
    }
    std::size_t start = colon + 1;
    while (start < header.size() && (header[start] == ' ' || header[start] == '\t')) {
        ++start;
    }
    *out = header.substr(start);
    return true;
}

}  // namespace

std::size_t NpyArray::element_count() const {
    std::size_t count = 1;
    for (const std::int64_t dimension : shape) {
        count *= static_cast<std::size_t>(dimension < 0 ? 0 : dimension);
    }
    return count;
}

std::size_t NpyArray::element_size() const {
    switch (type) {
        case NpyType::Float64: return 8;
        case NpyType::Float32: return 4;
        case NpyType::Int64: return 8;
        case NpyType::Int32: return 4;
        case NpyType::Int16: return 2;
        case NpyType::UInt8: return 1;
        case NpyType::Complex128: return 16;
        case NpyType::Unicode: return item_bytes;
        default: return 0;
    }
}

bool parse_npy(const std::uint8_t* data, std::size_t size, NpyArray* out, std::string* error) {
    if (data == nullptr || out == nullptr) {
        return false;
    }
    const std::uint8_t magic[6] = {0x93u, 'N', 'U', 'M', 'P', 'Y'};
    if (size < 10u || std::memcmp(data, magic, 6) != 0) {
        if (error != nullptr) { *error = "not a .npy image"; }
        return false;
    }
    const std::uint8_t major = data[6];
    std::size_t header_length = 0;
    std::size_t header_offset = 0;
    if (major == 1u) {
        header_length = read_u16(data + 8);
        header_offset = 10;
    } else if (major == 2u || major == 3u) {
        if (size < 12u) {
            if (error != nullptr) { *error = "truncated .npy v2 header"; }
            return false;
        }
        header_length = read_u32(data + 8);
        header_offset = 12;
    } else {
        if (error != nullptr) { *error = "unsupported .npy version " + std::to_string(major); }
        return false;
    }
    if (header_offset + header_length > size) {
        if (error != nullptr) { *error = "truncated .npy header"; }
        return false;
    }
    const std::string header(reinterpret_cast<const char*>(data + header_offset), header_length);

    out->descr.clear();
    std::string value;
    if (!header_value(header, "descr", &value)) {
        if (error != nullptr) { *error = ".npy header without descr"; }
        return false;
    }
    const std::size_t first_quote = value.find('\'');
    const std::size_t second_quote =
        first_quote == std::string::npos ? std::string::npos : value.find('\'', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        if (error != nullptr) { *error = ".npy descr is not a quoted string"; }
        return false;
    }
    out->descr = value.substr(first_quote + 1, second_quote - first_quote - 1);
    out->type = classify(out->descr);
    if (out->type == NpyType::Unknown) {
        if (error != nullptr) { *error = "unsupported .npy dtype " + out->descr; }
        return false;
    }
    out->item_bytes = 0;
    if (out->type == NpyType::Unicode) {
        const std::size_t length = unicode_length(out->descr);
        if (length == 0) {
            if (error != nullptr) { *error = "malformed unicode .npy dtype " + out->descr; }
            return false;
        }
        out->item_bytes = length * 4u;
    }

    out->fortran_order = header.find("'fortran_order': True") != std::string::npos;

    if (!header_value(header, "shape", &value)) {
        if (error != nullptr) { *error = ".npy header without shape"; }
        return false;
    }
    out->shape.clear();
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] >= '0' && value[i] <= '9') {
            long long dimension = 0;
            while (i < value.size() && value[i] >= '0' && value[i] <= '9') {
                dimension = dimension * 10 + (value[i] - '0');
                ++i;
            }
            out->shape.push_back(dimension);
        } else if (value[i] == ')') {
            break;
        }
    }

    const std::size_t expected = out->element_count() * out->element_size();
    if (header_offset + header_length + expected > size) {
        if (error != nullptr) {
            *error = ".npy payload truncated (need " + std::to_string(expected) + " bytes)";
        }
        return false;
    }
    out->data = data + header_offset + header_length;
    out->data_bytes = expected;
    return true;
}

bool npy_shape_is(const NpyArray& array, const std::vector<std::int64_t>& expected) {
    return array.shape == expected;
}

namespace {

template <typename T>
void load_le(const std::uint8_t* source, std::size_t count, bool swap, std::vector<T>* out) {
    out->resize(count);
    std::memcpy(out->data(), source, count * sizeof(T));
    if (swap) {
        std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(out->data());
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t b = 0; b < sizeof(T) / 2; ++b) {
                const std::uint8_t temporary = bytes[i * sizeof(T) + b];
                bytes[i * sizeof(T) + b] = bytes[i * sizeof(T) + sizeof(T) - 1 - b];
                bytes[i * sizeof(T) + sizeof(T) - 1 - b] = temporary;
            }
        }
    }
}

}  // namespace

bool npy_to_double(const NpyArray& array, std::vector<double>* out, std::string* error) {
    const bool swap = is_big_endian(array.descr);
    const std::size_t count = array.element_count();
    switch (array.type) {
        case NpyType::Float64:
            load_le(array.data, count, swap, out);
            return true;
        case NpyType::Complex128:
            load_le(array.data, count * 2u, swap, out);
            return true;
        case NpyType::Float32: {
            std::vector<float> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<double>(values[i]);
            }
            return true;
        }
        case NpyType::Int64: {
            std::vector<std::int64_t> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<double>(values[i]);
            }
            return true;
        }
        case NpyType::Int32: {
            std::vector<std::int32_t> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<double>(values[i]);
            }
            return true;
        }
        case NpyType::Int16: {
            std::vector<std::int16_t> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<double>(values[i]);
            }
            return true;
        }
        case NpyType::UInt8: {
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<double>(array.data[i]);
            }
            return true;
        }
        default:
            if (error != nullptr) { *error = "cannot convert " + array.descr + " to double"; }
            return false;
    }
}

bool npy_to_int16(const NpyArray& array, std::vector<std::int16_t>* out, std::string* error) {
    const bool swap = is_big_endian(array.descr);
    const std::size_t count = array.element_count();
    switch (array.type) {
        case NpyType::Int16:
            load_le(array.data, count, swap, out);
            return true;
        case NpyType::Int32: {
            std::vector<std::int32_t> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<std::int16_t>(values[i]);
            }
            return true;
        }
        case NpyType::Int64: {
            std::vector<std::int64_t> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<std::int16_t>(values[i]);
            }
            return true;
        }
        default:
            if (error != nullptr) { *error = "cannot convert " + array.descr + " to int16"; }
            return false;
    }
}

bool npy_to_int32(const NpyArray& array, std::vector<std::int32_t>* out, std::string* error) {
    const bool swap = is_big_endian(array.descr);
    const std::size_t count = array.element_count();
    switch (array.type) {
        case NpyType::Int32:
            load_le(array.data, count, swap, out);
            return true;
        case NpyType::Int64: {
            std::vector<std::int64_t> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<std::int32_t>(values[i]);
            }
            return true;
        }
        case NpyType::Int16: {
            std::vector<std::int16_t> values;
            load_le(array.data, count, swap, &values);
            out->resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                (*out)[i] = static_cast<std::int32_t>(values[i]);
            }
            return true;
        }
        default:
            if (error != nullptr) { *error = "cannot convert " + array.descr + " to int32"; }
            return false;
    }
}

bool npy_to_uint8(const NpyArray& array, std::vector<std::uint8_t>* out, std::string* error) {
    if (array.type != NpyType::UInt8) {
        if (error != nullptr) { *error = "cannot convert " + array.descr + " to uint8"; }
        return false;
    }
    out->assign(array.data, array.data + array.element_count());
    return true;
}

bool npy_unicode_to_utf8(const NpyArray& array, std::string* out, std::string* error) {
    if (array.type != NpyType::Unicode) {
        if (error != nullptr) { *error = "not a unicode .npy member: " + array.descr; }
        return false;
    }
    if (array.shape.size() != 0) {
        if (error != nullptr) { *error = "unicode .npy member must be a scalar"; }
        return false;
    }
    out->clear();
    const std::size_t count = array.item_bytes / 4u;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* p = array.data + i * 4u;
        const std::uint32_t code = static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
                              (static_cast<std::uint32_t>(p[2]) << 16) |
                              (static_cast<std::uint32_t>(p[3]) << 24);
        if (code == 0u) {
            break;
        }
        if (code < 0x80u) {
            out->push_back(static_cast<char>(code));
        } else if (code < 0x800u) {
            out->push_back(static_cast<char>(0xC0u | (code >> 6)));
            out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
        } else if (code < 0x10000u) {
            out->push_back(static_cast<char>(0xE0u | (code >> 12)));
            out->push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
            out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
        } else {
            out->push_back(static_cast<char>(0xF0u | (code >> 18)));
            out->push_back(static_cast<char>(0x80u | ((code >> 12) & 0x3Fu)));
            out->push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
            out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
        }
    }
    return true;
}

bool npy_to_c_order(const NpyArray& array, std::vector<std::uint8_t>* out, std::string* error) {
    const std::size_t element = array.element_size();
    if (element == 0) {
        if (error != nullptr) { *error = "unsupported element size for " + array.descr; }
        return false;
    }
    if (!array.fortran_order) {
        out->assign(array.data, array.data + array.data_bytes);
        return true;
    }
    const std::size_t dimensions = array.shape.size();
    if (dimensions == 0) {
        out->assign(array.data, array.data + element);
        return true;
    }
    // Source (Fortran) strides in elements; destination is C order.
    std::vector<std::size_t> source_stride(dimensions, 1);
    std::size_t running = 1;
    for (std::size_t d = 0; d < dimensions; ++d) {
        source_stride[d] = running;
        running *= static_cast<std::size_t>(array.shape[d]);
    }
    out->assign(array.data_bytes, 0);
    std::vector<std::size_t> index(dimensions, 0);
    const std::size_t total = array.element_count();
    for (std::size_t linear = 0; linear < total; ++linear) {
        std::size_t remainder = linear;
        for (std::size_t d = dimensions; d-- > 0;) {
            index[d] = remainder % static_cast<std::size_t>(array.shape[d]);
            remainder /= static_cast<std::size_t>(array.shape[d]);
        }
        std::size_t source = 0;
        for (std::size_t d = 0; d < dimensions; ++d) {
            source += index[d] * source_stride[d];
        }
        std::memcpy(out->data() + linear * element, array.data + source * element, element);
    }
    return true;
}

}  // namespace joc::io
