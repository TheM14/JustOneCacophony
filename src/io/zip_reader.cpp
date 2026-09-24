#include "io/zip_reader.h"
#include "foundation/fs_utf8.h"

#include <cstdio>
#include <cstring>

#include "io/inflate.h"

namespace joc::io {

namespace {

constexpr std::uint32_t kLocalHeaderSignature = 0x04034b50u;
constexpr std::uint32_t kCentralHeaderSignature = 0x02014b50u;
constexpr std::uint32_t kEndOfCentralDirectory = 0x06054b50u;

std::uint16_t read_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t read_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

}  // namespace

std::uint32_t crc32_of(const std::uint8_t* data, std::size_t size) {
    static std::uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value & 1u) ? (0xEDB88320u ^ (value >> 1)) : (value >> 1);
            }
            table[i] = value;
        }
        ready = true;
    }
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

bool ZipArchive::open(const std::string& path, std::string* error) {
    entries_.clear();
    data_.clear();
    std::FILE* file = fs_utf8::fopen(path, "rb");
    if (file == nullptr) {
        if (error != nullptr) {
            *error = "cannot open " + path;
        }
        return false;
    }
    std::fseek(file, 0, SEEK_END);
    const long long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(file);
        if (error != nullptr) {
            *error = "empty file " + path;
        }
        return false;
    }
    data_.resize(static_cast<std::size_t>(size));
    const std::size_t got = std::fread(data_.data(), 1, data_.size(), file);
    std::fclose(file);
    if (got != data_.size()) {
        if (error != nullptr) {
            *error = "short read on " + path;
        }
        return false;
    }

    std::size_t eocd = std::string::npos;
    const std::size_t scan_start = data_.size() > 65557u ? data_.size() - 65557u : 0u;
    for (std::size_t i = data_.size(); i-- > scan_start;) {
        if (i + 4u <= data_.size() && read_u32(&data_[i]) == kEndOfCentralDirectory) {
            eocd = i;
            break;
        }
        if (i == 0) {
            break;
        }
    }
    if (eocd == std::string::npos || eocd + 22u > data_.size()) {
        if (error != nullptr) {
            *error = "not a zip archive (no end-of-central-directory)";
        }
        return false;
    }
    const std::uint16_t entry_count = read_u16(&data_[eocd + 10]);
    const std::uint32_t directory_offset = read_u32(&data_[eocd + 16]);
    if (directory_offset >= data_.size()) {
        if (error != nullptr) {
            *error = "central directory offset out of range";
        }
        return false;
    }
    std::size_t cursor = directory_offset;
    for (std::uint16_t index = 0; index < entry_count; ++index) {
        if (cursor + 46u > data_.size() || read_u32(&data_[cursor]) != kCentralHeaderSignature) {
            if (error != nullptr) {
                *error = "malformed central directory entry " + std::to_string(index);
            }
            return false;
        }
        ZipEntry entry;
        entry.method = read_u16(&data_[cursor + 10]);
        entry.crc32 = read_u32(&data_[cursor + 16]);
        entry.compressed_size = read_u32(&data_[cursor + 20]);
        entry.uncompressed_size = read_u32(&data_[cursor + 24]);
        const std::uint16_t name_length = read_u16(&data_[cursor + 28]);
        const std::uint16_t extra_length = read_u16(&data_[cursor + 30]);
        const std::uint16_t comment_length = read_u16(&data_[cursor + 32]);
        entry.local_header_offset = read_u32(&data_[cursor + 42]);
        if (entry.compressed_size == 0xFFFFFFFFu || entry.uncompressed_size == 0xFFFFFFFFu ||
            entry.local_header_offset == 0xFFFFFFFFu) {
            if (error != nullptr) {
                *error = "zip64 archives are not supported";
            }
            return false;
        }
        if (cursor + 46u + name_length > data_.size()) {
            if (error != nullptr) {
                *error = "member name out of range";
            }
            return false;
        }
        entry.name.assign(reinterpret_cast<const char*>(&data_[cursor + 46]), name_length);
        entries_.push_back(std::move(entry));
        cursor += 46u + name_length + extra_length + comment_length;
    }
    return true;
}

const ZipEntry* ZipArchive::find(const std::string& name) const {
    for (const ZipEntry& entry : entries_) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

bool ZipArchive::extract(const ZipEntry& entry, std::vector<std::uint8_t>* out,
                         std::string* error) const {
    if (out == nullptr) {
        return false;
    }
    const std::size_t offset = entry.local_header_offset;
    if (offset + 30u > data_.size() || read_u32(&data_[offset]) != kLocalHeaderSignature) {
        if (error != nullptr) {
            *error = "bad local header for " + entry.name;
        }
        return false;
    }
    const std::uint16_t name_length = read_u16(&data_[offset + 26]);
    const std::uint16_t extra_length = read_u16(&data_[offset + 28]);
    const std::size_t start = offset + 30u + name_length + extra_length;
    if (start + entry.compressed_size > data_.size()) {
        if (error != nullptr) {
            *error = "member data out of range for " + entry.name;
        }
        return false;
    }
    if (entry.method == 0u) {
        out->assign(data_.begin() + static_cast<std::ptrdiff_t>(start),
                    data_.begin() + static_cast<std::ptrdiff_t>(start + entry.compressed_size));
    } else if (entry.method == 8u) {
        if (!inflate_raw(&data_[start], entry.compressed_size, out)) {
            if (error != nullptr) {
                *error = "deflate error in " + entry.name;
            }
            return false;
        }
    } else {
        if (error != nullptr) {
            *error = "unsupported compression method " + std::to_string(entry.method) + " for " +
                     entry.name;
        }
        return false;
    }
    if (entry.uncompressed_size != 0u && out->size() != entry.uncompressed_size) {
        if (error != nullptr) {
            *error = "size mismatch for " + entry.name + " (" + std::to_string(out->size()) +
                     " vs " + std::to_string(entry.uncompressed_size) + ")";
        }
        return false;
    }
    if (entry.crc32 != 0u && crc32_of(out->data(), out->size()) != entry.crc32) {
        if (error != nullptr) {
            *error = "CRC mismatch for " + entry.name;
        }
        return false;
    }
    return true;
}

bool ZipArchive::read_member(const std::string& name, std::vector<std::uint8_t>* out,
                             std::string* error) const {
    const ZipEntry* entry = find(name);
    if (entry == nullptr) {
        if (error != nullptr) {
            *error = "member not found: " + name;
        }
        return false;
    }
    return extract(*entry, out, error);
}

}  // namespace joc::io
