
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace joc::io {

struct ZipEntry {
    std::string name;
    std::uint16_t method = 0;
    std::uint32_t crc32 = 0;
    std::uint32_t compressed_size = 0;
    std::uint32_t uncompressed_size = 0;
    std::uint32_t local_header_offset = 0;
};

class ZipArchive {
public:
    bool open(const std::string& path, std::string* error);

    const std::vector<ZipEntry>& entries() const { return entries_; }

    const ZipEntry* find(const std::string& name) const;

    bool extract(const ZipEntry& entry, std::vector<std::uint8_t>* out, std::string* error) const;

    bool read_member(const std::string& name, std::vector<std::uint8_t>* out, std::string* error) const;

private:
    std::vector<std::uint8_t> data_;
    std::vector<ZipEntry> entries_;
};

std::uint32_t crc32_of(const std::uint8_t* data, std::size_t size);

}  // namespace joc::io
