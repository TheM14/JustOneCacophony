#include "io/hdf5.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <map>
#include <utility>

#include "foundation/fs_utf8.h"

namespace joc::io {

namespace {

// Object header message ids of the HDF5 file format specification (version 3):
// 0x02 points at the dense link heap, 0x06 is one link, 0x10 continues the
// header in another block and 0x15 points at the dense attribute heap.
constexpr std::uint8_t kMsgNil = 0x00;
constexpr std::uint8_t kMsgDataspace = 0x01;
constexpr std::uint8_t kMsgLinkInfo = 0x02;
constexpr std::uint8_t kMsgDatatype = 0x03;
constexpr std::uint8_t kMsgLink = 0x06;
constexpr std::uint8_t kMsgLayout = 0x08;
constexpr std::uint8_t kMsgAttribute = 0x0C;
constexpr std::uint8_t kMsgContinuation = 0x10;
constexpr std::uint8_t kMsgAttributeInfo = 0x15;

constexpr std::uint64_t kUndefinedAddress = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kMaxReadSize = 1ull << 32;  // a dataset larger than this is refused
constexpr int kMaxNesting = 32;

Status fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kRender, message);
}

std::uint64_t read_le(const std::uint8_t* data, std::size_t size) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
        value |= static_cast<std::uint64_t>(data[i]) << (8u * i);
    }
    return value;
}

std::uint64_t align_to_8(std::uint64_t value) { return (value + 7u) & ~std::uint64_t{7u}; }

void swap_element_bytes(std::uint8_t* data, std::uint64_t size) {
    for (std::uint64_t i = 0; i < size / 2; ++i) {
        const std::uint8_t head = data[i];
        data[i] = data[size - 1 - i];
        data[size - 1 - i] = head;
    }
}

struct Message {
    std::uint8_t type = kMsgNil;
    std::uint64_t body = 0;  // absolute file offset of the message body
    std::uint64_t size = 0;
};

struct LinkEntry {
    std::string name;
    std::uint64_t address = kUndefinedAddress;
};

struct AttributeEntry {
    std::string name;
    std::uint64_t body = 0;
    std::uint64_t size = 0;
};

// Fractal heap header fields this reader needs; the free-space accounting is
// deliberately not modelled because objects are enumerated from the blocks.
struct FractalHeap {
    std::uint64_t id_length = 0;
    std::uint32_t max_object = 0;
    std::uint64_t object_count = 0;
    std::uint64_t huge_count = 0;
    std::uint64_t tiny_count = 0;
    std::uint16_t width = 0;
    std::uint64_t start_block = 0;
    std::uint64_t max_direct_block = 0;
    std::uint8_t offset_width = 0;
    std::uint16_t current_rows = 0;
    std::uint64_t root_block = kUndefinedAddress;
};

struct Layout {
    bool compact = false;
    std::uint64_t address = kUndefinedAddress;
    std::uint64_t size = 0;
    std::uint64_t inline_offset = 0;
};

enum class HeapObjectKind { Link, Attribute };

bool is_string_type(Hdf5Type type) { return type == Hdf5Type::String; }

std::string trim_nul(std::string text) {
    while (!text.empty() && text.back() == '\0') {
        text.pop_back();
    }
    return text;
}

// Attribute names are stored with their NUL terminator included in the name size.
std::string attribute_name(const std::uint8_t* data, std::size_t size) {
    const std::size_t length = std::string(reinterpret_cast<const char*>(data), size).find('\0');
    return std::string(reinterpret_cast<const char*>(data),
                       length == std::string::npos ? size : length);
}

}  // namespace

std::uint64_t Hdf5DatasetInfo::element_count() const {
    std::uint64_t count = 1;
    for (const std::uint64_t dimension : shape) {
        count *= dimension;
    }
    return count;
}

struct Hdf5File::Impl {
    std::string file_path;
    mutable std::ifstream stream;
    bool opened = false;
    std::uint64_t file_size = 0;
    std::uint8_t offset_size = 8;
    std::uint8_t length_size = 8;
    std::uint64_t root_object = kUndefinedAddress;

    // Parsed metadata is cached by file address / path: the file itself is only
    // read again when a dataset's payload is requested.
    mutable std::map<std::uint64_t, std::vector<Message>> headers;
    mutable std::map<std::string, std::vector<LinkEntry>> link_cache;
    mutable std::map<std::string, std::uint64_t> object_cache;

    Status read_at(std::uint64_t offset, std::uint64_t size, std::vector<std::uint8_t>* out) const {
        if (out == nullptr) {
            return fail(JOC_ERR_INVALID_ARGUMENT, "HDF5 read without a destination buffer");
        }
        out->clear();
        if (size == 0) {
            return Status::success();
        }
        if (offset > file_size || size > file_size - offset) {
            return fail(JOC_ERR_INPUT_FORMAT, "HDF5 structure runs past the end of " + file_path +
                                                  " (offset " + std::to_string(offset) + ", size " +
                                                  std::to_string(size) + ")");
        }
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!stream.good()) {
            return fail(JOC_ERR_IO, "cannot seek in " + file_path);
        }
        out->resize(static_cast<std::size_t>(size));
        stream.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(size));
        if (stream.gcount() != static_cast<std::streamsize>(size)) {
            return fail(JOC_ERR_IO, "short read in " + file_path);
        }
        return Status::success();
    }

    Status read_u64(std::uint64_t offset, std::uint8_t size, std::uint64_t* out) const {
        std::vector<std::uint8_t> buffer;
        const Status status = read_at(offset, size, &buffer);
        if (!status.ok()) {
            return status;
        }
        *out = read_le(buffer.data(), buffer.size());
        return Status::success();
    }

    Status parse_messages(const std::uint8_t* data, std::size_t size, std::uint64_t base,
                          bool creation_order, std::vector<Message>* out, int depth) const {
        std::size_t position = 0;
        while (position + 4u <= size) {
            const std::uint8_t type = data[position];
            const std::uint64_t body_size = read_le(data + position + 1, 2);
            const std::size_t header_size = 4u + (creation_order ? 2u : 0u);
            if (position + header_size > size) {
                // A chunk may end with fewer bytes than a message header needs:
                // that leftover is padding, not a message.
                break;
            }
            const std::uint64_t body = position + header_size;
            if (body + body_size > size) {
                return fail(JOC_ERR_INPUT_FORMAT, 
                            "HDF5 object header message overruns its chunk in " + file_path);
            }
            if (type == kMsgNil) {
                if (body_size == 0) {
                    break;
                }
                position = static_cast<std::size_t>(body + body_size);
                continue;
            }
            if (type == kMsgContinuation) {
                if (depth >= kMaxNesting) {
                    return fail(JOC_ERR_INPUT_FORMAT, 
                                "HDF5 object header continuation nesting in " + file_path);
                }
                if (body_size < offset_size + length_size) {
                    return fail(JOC_ERR_INPUT_FORMAT, 
                                "malformed HDF5 continuation message in " + file_path);
                }
                const std::uint64_t next = read_le(data + body, offset_size);
                const std::uint64_t length = read_le(data + body + offset_size, length_size);
                std::vector<std::uint8_t> block;
                const Status status = read_at(next, length, &block);
                if (!status.ok()) {
                    return status;
                }
                std::size_t start = 0;
                std::size_t stop = block.size();
                if (block.size() >= 8u && std::memcmp(block.data(), "OCHK", 4) == 0) {
                    // A continuation block is "OCHK", the messages, then its own
                    // checksum, which must not be walked as a message header.
                    start = 4;
                    stop = block.size() - 4;
                }
                const Status nested = parse_messages(block.data() + start, stop - start,
                                                     next + start, creation_order, out, depth + 1);
                if (!nested.ok()) {
                    return nested;
                }
            } else {
                Message message;
                message.type = type;
                message.body = base + body;
                message.size = body_size;
                out->push_back(message);
            }
            position = static_cast<std::size_t>(body + body_size);
        }
        return Status::success();
    }

    Status object_messages(std::uint64_t address, std::vector<Message>* out) const {
        const auto cached = headers.find(address);
        if (cached != headers.end()) {
            *out = cached->second;
            return Status::success();
        }
        std::vector<std::uint8_t> prefix;
        Status status = read_at(address, 8, &prefix);
        if (!status.ok()) {
            return status;
        }
        if (std::memcmp(prefix.data(), "OHDR", 4) != 0) {
            return fail(JOC_ERR_INPUT_FORMAT, "HDF5 object header signature missing in " + file_path);
        }
        if (prefix[4] != 2u) {
            return fail(JOC_ERR_NOT_SUPPORTED,
                        "HDF5 object header version " + std::to_string(prefix[4]) +
                            " is not supported");
        }
        const std::uint8_t flags = prefix[5];
        std::uint64_t position = address + 6;
        if ((flags & 0x20u) != 0u) {
            position += 16;  // access/modification/change/birth times
        }
        if ((flags & 0x10u) != 0u) {
            position += 4;  // attribute storage phase change values
        }
        const std::size_t width = std::size_t{1} << (flags & 0x03u);
        std::vector<std::uint8_t> size_field;
        status = read_at(position, width, &size_field);
        if (!status.ok()) {
            return status;
        }
        const std::uint64_t chunk_size = read_le(size_field.data(), width);
        position += width;
        std::vector<std::uint8_t> chunk;
        status = read_at(position, chunk_size, &chunk);
        if (!status.ok()) {
            return status;
        }
        std::vector<Message> messages;
        // A header that tracks attribute creation order stores a two byte
        // creation order in every message header.
        status = parse_messages(chunk.data(), chunk.size(), position, (flags & 0x04u) != 0u,
                                &messages, 0);
        if (!status.ok()) {
            return status;
        }
        headers[address] = messages;
        *out = messages;
        return Status::success();
    }

    Status parse_datatype(const std::uint8_t* data, std::size_t size, Hdf5TypeInfo* out) const {
        if (size < 8u) {
            return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 datatype message in " + file_path);
        }
        const std::uint8_t version_class = data[0];
        const std::uint8_t datatype_class = static_cast<std::uint8_t>(version_class & 0x0Fu);
        const std::uint8_t bits = data[1];
        const std::uint32_t element_size = static_cast<std::uint32_t>(read_le(data + 4, 4));
        Hdf5TypeInfo info;
        info.size = element_size;
        info.big_endian = (bits & 0x01u) != 0u;
        switch (datatype_class) {
            case 0:  // fixed point
                info.is_signed = (bits & 0x08u) != 0u;
                switch (element_size) {
                    case 1: info.type = info.is_signed ? Hdf5Type::Int8 : Hdf5Type::UInt8; break;
                    case 2: info.type = info.is_signed ? Hdf5Type::Int16 : Hdf5Type::UInt16; break;
                    case 4: info.type = info.is_signed ? Hdf5Type::Int32 : Hdf5Type::UInt32; break;
                    case 8: info.type = info.is_signed ? Hdf5Type::Int64 : Hdf5Type::UInt64; break;
                    default:
                        return fail(JOC_ERR_NOT_SUPPORTED,
                                    "HDF5 integer of " + std::to_string(element_size) +
                                        " bytes is not supported");
                }
                break;
            case 1:  // floating point
                if (element_size == 4u) {
                    info.type = Hdf5Type::Float32;
                } else if (element_size == 8u) {
                    info.type = Hdf5Type::Float64;
                } else {
                    return fail(JOC_ERR_NOT_SUPPORTED,
                                "HDF5 float of " + std::to_string(element_size) +
                                    " bytes is not supported");
                }
                break;
            case 3:  // fixed length string
                info.type = Hdf5Type::String;
                break;
            default:
                return fail(JOC_ERR_NOT_SUPPORTED,
                            "HDF5 datatype class " + std::to_string(datatype_class) +
                                " (compound, variable length or reference) is not supported");
        }
        *out = info;
        return Status::success();
    }

    Status parse_dataspace(const std::uint8_t* data, std::size_t size,
                           std::vector<std::uint64_t>* out) const {
        if (size < 4u) {
            return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 dataspace message in " + file_path);
        }
        const std::uint8_t version = data[0];
        const std::uint8_t rank = data[1];
        const std::uint8_t flags = data[2];
        if (version != 1u && version != 2u) {
            return fail(JOC_ERR_NOT_SUPPORTED,
                        "HDF5 dataspace version " + std::to_string(version) + " is not supported");
        }
        // Version 1 aligns the dimension sizes on an eight byte boundary, so its
        // four byte preamble is padded; version 2 writes them immediately.
        const std::uint64_t start = (version == 1u) ? 8u : 4u;
        const std::uint64_t needed = start + static_cast<std::uint64_t>(rank) * length_size;
        if (needed > size) {
            return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 dataspace dimensions in " + file_path);
        }
        std::vector<std::uint64_t> shape;
        shape.reserve(rank);
        for (std::uint8_t index = 0; index < rank; ++index) {
            shape.push_back(read_le(data + start + static_cast<std::uint64_t>(index) * length_size,
                                    length_size));
        }
        (void)flags;  // maximum dimensions and permutation indexes are not needed
        *out = std::move(shape);
        return Status::success();
    }

    Status parse_layout(const std::uint8_t* data, std::size_t size, Layout* out) const {
        if (size < 2u) {
            return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 data layout message in " + file_path);
        }
        const std::uint8_t version = data[0];
        const std::uint8_t storage_class = data[1];
        if (version == 3u || version == 4u) {
            switch (storage_class) {
                case 0: {  // compact: the data follows the two byte size field
                    if (size < 4u) {
                        return fail(JOC_ERR_INPUT_FORMAT, 
                                    "truncated HDF5 compact layout in " + file_path);
                    }
                    out->compact = true;
                    out->size = read_le(data + 2, 2);
                    out->inline_offset = 4;
                    return Status::success();
                }
                case 1: {  // contiguous: address then allocated size
                    if (size < 2u + offset_size + length_size) {
                        return fail(JOC_ERR_INPUT_FORMAT, 
                                    "truncated HDF5 contiguous layout in " + file_path);
                    }
                    out->compact = false;
                    out->address = read_le(data + 2, offset_size);
                    out->size = read_le(data + 2 + offset_size, length_size);
                    return Status::success();
                }
                case 2:
                    return fail(JOC_ERR_NOT_SUPPORTED, "chunked datasets are not supported");
                default:
                    return fail(JOC_ERR_NOT_SUPPORTED,
                                "HDF5 data layout class " + std::to_string(storage_class) +
                                    " is not supported");
            }
        }
        if (version == 1u || version == 2u) {
            // The older layouts only describe contiguous storage in a form this
            // reader can follow.
            if (storage_class != 1u || size < 2u + offset_size) {
                return fail(JOC_ERR_NOT_SUPPORTED,
                            "HDF5 data layout version " + std::to_string(version) +
                                " is not supported");
            }
            out->compact = false;
            out->address = read_le(data + 2, offset_size);
            out->size = 0;  // derived from the dataspace and datatype
            return Status::success();
        }
        return fail(JOC_ERR_NOT_SUPPORTED,
                    "HDF5 data layout version " + std::to_string(version) + " is not supported");
    }

    Status read_fractal_heap(std::uint64_t address, FractalHeap* out) const {
        std::vector<std::uint8_t> data;
        // The header ends with the root block address and the current row count,
        // 142 bytes in total for eight byte offsets and lengths.
        Status status = read_at(address, 144, &data);
        if (!status.ok()) {
            return status;
        }
        if (std::memcmp(data.data(), "FRHP", 4) != 0) {
            return fail(JOC_ERR_INPUT_FORMAT, "HDF5 fractal heap header missing in " + file_path);
        }
        std::size_t position = 4;
        const std::uint8_t version = data[position++];
        if (version != 0u) {
            return fail(JOC_ERR_NOT_SUPPORTED,
                        "HDF5 fractal heap version " + std::to_string(version) + " is not supported");
        }
        out->id_length = read_le(data.data() + position, 2);
        position += 2;
        const std::uint64_t filter_length = read_le(data.data() + position, 2);
        position += 2;
        if (filter_length != 0u) {
            return fail(JOC_ERR_NOT_SUPPORTED, "filtered HDF5 fractal heaps are not supported");
        }
        position += 1;  // flags
        out->max_object = static_cast<std::uint32_t>(read_le(data.data() + position, 4));
        position += 4;
        position += static_cast<std::size_t>(length_size);  // next huge object id
        position += static_cast<std::size_t>(offset_size);  // huge object B-tree
        position += static_cast<std::size_t>(length_size);  // free space in managed blocks
        position += static_cast<std::size_t>(offset_size);  // free space B-tree
        position += static_cast<std::size_t>(length_size);  // managed space
        position += static_cast<std::size_t>(length_size);  // allocated managed space
        position += static_cast<std::size_t>(length_size);  // direct block iterator offset
        out->object_count = read_le(data.data() + position, length_size);
        position += static_cast<std::size_t>(length_size);
        position += static_cast<std::size_t>(length_size);  // size of huge objects
        out->huge_count = read_le(data.data() + position, length_size);
        position += static_cast<std::size_t>(length_size);
        position += static_cast<std::size_t>(length_size);  // size of tiny objects
        out->tiny_count = read_le(data.data() + position, length_size);
        position += static_cast<std::size_t>(length_size);
        out->width = static_cast<std::uint16_t>(read_le(data.data() + position, 2));
        position += 2;
        out->start_block = read_le(data.data() + position, length_size);
        position += static_cast<std::size_t>(length_size);
        out->max_direct_block = read_le(data.data() + position, length_size);
        position += static_cast<std::size_t>(length_size);
        const std::uint64_t max_heap_bits = read_le(data.data() + position, 2);
        position += 2;
        position += 2;  // starting number of rows in the root indirect block
        out->root_block = read_le(data.data() + position, offset_size);
        position += static_cast<std::size_t>(offset_size);
        out->current_rows = static_cast<std::uint16_t>(read_le(data.data() + position, 2));
        out->offset_width = static_cast<std::uint8_t>((max_heap_bits + 7u) / 8u);
        if (out->offset_width == 0u) {
            out->offset_width = 1;
        }
        if (out->id_length == 0u || out->start_block == 0u) {
            return fail(JOC_ERR_INPUT_FORMAT, "malformed HDF5 fractal heap header in " + file_path);
        }
        return Status::success();
    }

    // Size of one heap object at `offset`, used to step from object to object.
    // Free space is zero filled, so the first byte that does not describe a
    // well formed message ends the enumeration.
    Status object_extent(HeapObjectKind kind, std::uint64_t offset, std::uint64_t limit,
                         std::uint64_t* size, bool* valid) const {
        *valid = false;
        *size = 0;
        std::vector<std::uint8_t> data;
        const std::uint64_t available = (limit > offset) ? (limit - offset) : 0;
        const std::uint64_t wanted = std::min<std::uint64_t>(available, 8);
        if (wanted < 4u) {
            return Status::success();
        }
        const Status status = read_at(offset, wanted, &data);
        if (!status.ok()) {
            return status;
        }
        if (kind == HeapObjectKind::Link) {
            if (data[0] != 1u) {
                return Status::success();
            }
            const std::uint8_t flags = data[1];
            std::uint64_t position = 2;
            if ((flags & 0x08u) != 0u) {
                position += 1;
            }
            if ((flags & 0x04u) != 0u) {
                position += 8;
            }
            if ((flags & 0x10u) != 0u) {
                position += 1;
            }
            const std::uint64_t name_size_field = 1ull << (flags & 0x03u);
            std::vector<std::uint8_t> header;
            const Status head_status = read_at(offset, position + name_size_field, &header);
            if (!head_status.ok()) {
                return head_status;
            }
            const std::uint64_t name_length = read_le(header.data() + position, name_size_field);
            const std::uint64_t total = position + name_size_field + name_length + offset_size;
            if (name_length == 0u || name_length > 4096u || total > available) {
                return Status::success();
            }
            *size = total;
            *valid = true;
            return Status::success();
        }
        if (data[0] < 1u || data[0] > 3u) {
            return Status::success();
        }
        std::vector<std::uint8_t> header;
        const Status head_status = read_at(offset, std::min<std::uint64_t>(available, 8), &header);
        if (!head_status.ok()) {
            return head_status;
        }
        if (header.size() < 8u) {
            return Status::success();
        }
        const std::uint64_t name_size = read_le(header.data() + 2, 2);
        const std::uint64_t datatype_size = read_le(header.data() + 4, 2);
        const std::uint64_t dataspace_size = read_le(header.data() + 6, 2);
        // Version 1 pads the attribute name to eight bytes, versions 2 and 3
        // write it (and, for version 3, a character set byte) unpadded.  The
        // datatype and dataspace messages are always padded to eight bytes.
        const std::uint64_t name_offset = (data[0] == 3u) ? 9u : 8u;
        const std::uint64_t name_field =
            (data[0] == 1u) ? align_to_8(name_size) : name_size;
        const std::uint64_t datatype_offset = name_offset + name_field;
        const std::uint64_t dataspace_offset = datatype_offset + align_to_8(datatype_size);
        const std::uint64_t data_offset = dataspace_offset + align_to_8(dataspace_size);
        if (name_size == 0u || name_size > 4096u || data_offset > available ||
            datatype_size < 8u || dataspace_size < 4u) {
            return Status::success();
        }
        std::vector<std::uint8_t> tail;
        const Status tail_status = read_at(offset + datatype_offset, datatype_size, &tail);
        if (!tail_status.ok()) {
            return tail_status;
        }
        const std::uint64_t element_size = read_le(tail.data() + 4, 4);
        std::vector<std::uint8_t> space;
        const Status space_status = read_at(offset + dataspace_offset, dataspace_size, &space);
        if (!space_status.ok()) {
            return space_status;
        }
        std::vector<std::uint64_t> shape;
        const Status shape_status = parse_dataspace(space.data(), space.size(), &shape);
        if (!shape_status.ok()) {
            return Status::success();
        }
        std::uint64_t count = 1;
        for (const std::uint64_t dimension : shape) {
            if (dimension != 0u && count > kMaxReadSize / dimension) {
                return Status::success();
            }
            count *= dimension;
        }
        if (count != 0u && element_size > kMaxReadSize / count) {
            return Status::success();
        }
        const std::uint64_t total = data_offset + count * element_size;
        if (total > available) {
            return Status::success();
        }
        *size = total;
        *valid = true;
        return Status::success();
    }

    Status scan_direct_block(const FractalHeap& heap, std::uint64_t address, int row,
                             HeapObjectKind kind, std::vector<Message>* out) const {
        std::vector<std::uint8_t> prefix;
        Status status = read_at(address, 4 + 1 + offset_size, &prefix);
        if (!status.ok()) {
            return status;
        }
        if (std::memcmp(prefix.data(), "FHDB", 4) != 0) {
            return fail(JOC_ERR_INPUT_FORMAT, "HDF5 fractal heap direct block missing in " + file_path);
        }
        if (prefix[4] != 0u) {
            return fail(JOC_ERR_NOT_SUPPORTED, "unsupported HDF5 fractal heap block version");
        }
        const std::uint64_t block_size =
            std::min<std::uint64_t>(heap.start_block << row, heap.max_direct_block);
        if (block_size == 0u || address + block_size < address) {
            return fail(JOC_ERR_INPUT_FORMAT, "malformed HDF5 fractal heap block in " + file_path);
        }
        const std::uint64_t end =
            std::min<std::uint64_t>(address + block_size, file_size);
        std::uint64_t position = address + 4 + 1 + offset_size + heap.offset_width + 4;
        while (position < end) {
            std::uint64_t object_size = 0;
            bool valid = false;
            status = object_extent(kind, position, end, &object_size, &valid);
            if (!status.ok()) {
                return status;
            }
            if (!valid || object_size == 0u) {
                break;
            }
            Message object;
            object.type = kMsgNil;  // heap objects carry no message type
            object.body = position;
            object.size = object_size;
            out->push_back(object);
            position += object_size;
        }
        return Status::success();
    }

    Status collect_blocks(const FractalHeap& heap, std::uint64_t block, int row, int depth,
                          std::vector<std::pair<std::uint64_t, int>>* out) const {
        if (block == kUndefinedAddress || depth > kMaxNesting) {
            return Status::success();
        }
        std::vector<std::uint8_t> signature;
        Status status = read_at(block, 4, &signature);
        if (!status.ok()) {
            return status;
        }
        if (std::memcmp(signature.data(), "FHDB", 4) == 0) {
            out->emplace_back(block, row);
            return Status::success();
        }
        if (std::memcmp(signature.data(), "FHIB", 4) != 0) {
            return fail(JOC_ERR_INPUT_FORMAT, "unknown HDF5 fractal heap block in " + file_path);
        }
        // Indirect block: version, heap header address, block offset, then one
        // address per child block, two per row more than the row above.
        const std::uint64_t header = 4 + 1 + offset_size + heap.offset_width;
        const int rows = (depth == 0) ? static_cast<int>(heap.current_rows) : kMaxNesting - 1;
        std::uint64_t position = block + header;
        for (int current = 0; current <= rows; ++current) {
            const int children = 1 << std::min(current, 20);
            bool any = false;
            for (int index = 0; index < children; ++index) {
                std::uint64_t child = kUndefinedAddress;
                status = read_u64(position, offset_size, &child);
                if (!status.ok()) {
                    return status;
                }
                position += offset_size;
                if (child != kUndefinedAddress) {
                    any = true;
                    status = collect_blocks(heap, child, current, depth + 1, out);
                    if (!status.ok()) {
                        return status;
                    }
                }
            }
            if (!any) {
                break;
            }
        }
        return Status::success();
    }

    Status heap_objects(std::uint64_t heap_address, HeapObjectKind kind,
                        std::vector<Message>* out) const {
        FractalHeap heap;
        Status status = read_fractal_heap(heap_address, &heap);
        if (!status.ok()) {
            return status;
        }
        if (heap.huge_count != 0u) {
            return fail(JOC_ERR_NOT_SUPPORTED,
                        "HDF5 heap object is stored as a huge object and is not supported");
        }
        std::vector<std::pair<std::uint64_t, int>> blocks;
        status = collect_blocks(heap, heap.root_block, 0, 0, &blocks);
        if (!status.ok()) {
            return status;
        }
        for (const std::pair<std::uint64_t, int>& block : blocks) {
            status = scan_direct_block(heap, block.first, block.second, kind, out);
            if (!status.ok()) {
                return status;
            }
        }
        return Status::success();
    }

    Status parse_link(const std::uint8_t* data, std::size_t size, LinkEntry* out) const {
        if (size < 2u || data[0] != 1u) {
            return fail(JOC_ERR_INPUT_FORMAT, "malformed HDF5 link message in " + file_path);
        }
        const std::uint8_t flags = data[1];
        std::uint64_t position = 2;
        std::uint8_t link_type = 0;
        if ((flags & 0x08u) != 0u) {
            if (position + 1u > size) {
                return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 link message in " + file_path);
            }
            link_type = data[position];
            position += 1;
        }
        if ((flags & 0x04u) != 0u) {
            position += 8;
        }
        if ((flags & 0x10u) != 0u) {
            position += 1;
        }
        const std::uint64_t name_size_field = 1ull << (flags & 0x03u);
        if (position + name_size_field > size) {
            return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 link name in " + file_path);
        }
        const std::uint64_t name_length = read_le(data + position, name_size_field);
        position += name_size_field;
        if (position + name_length + offset_size > size) {
            return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 link message in " + file_path);
        }
        const std::string name(reinterpret_cast<const char*>(data + position),
                               static_cast<std::size_t>(name_length));
        position += name_length;
        if (link_type != 0u) {
            return fail(JOC_ERR_NOT_SUPPORTED,
                        std::string(link_type == 1u ? "soft" : "external") +
                            " HDF5 link '" + name + "' is not supported");
        }
        out->name = name;
        out->address = read_le(data + position, offset_size);
        return Status::success();
    }

    Status links_of(const std::string& group_path, std::vector<LinkEntry>* out) const {
        const auto cached = link_cache.find(group_path);
        if (cached != link_cache.end()) {
            *out = cached->second;
            return Status::success();
        }
        std::uint64_t address = kUndefinedAddress;
        Status status = object_address(group_path, &address);
        if (!status.ok()) {
            return status;
        }
        std::vector<Message> messages;
        status = object_messages(address, &messages);
        if (!status.ok()) {
            return status;
        }
        std::vector<LinkEntry> links;
        for (const Message& message : messages) {
            if (message.type == kMsgLink) {
                std::vector<std::uint8_t> body;
                status = read_at(message.body, message.size, &body);
                if (!status.ok()) {
                    return status;
                }
                LinkEntry entry;
                status = parse_link(body.data(), body.size(), &entry);
                if (!status.ok()) {
                    return status;
                }
                links.push_back(std::move(entry));
            } else if (message.type == kMsgLinkInfo) {
                std::vector<std::uint8_t> body;
                status = read_at(message.body, message.size, &body);
                if (!status.ok()) {
                    return status;
                }
                if (body.size() < 2u) {
                    return fail(JOC_ERR_INPUT_FORMAT, 
                                "malformed HDF5 link info message in " + file_path);
                }
                const std::uint8_t flags = body[1];
                std::uint64_t position = 2;
                if ((flags & 0x01u) != 0u) {
                    position += static_cast<std::size_t>(length_size);  // maximum creation index
                }
                if (position + offset_size > body.size()) {
                    return fail(JOC_ERR_INPUT_FORMAT, 
                                "malformed HDF5 link info message in " + file_path);
                }
                const std::uint64_t heap_address = read_le(body.data() + position, offset_size);
                if (heap_address == kUndefinedAddress) {
                    // No dense storage: this group keeps its links compact.
                    continue;
                }
                std::vector<Message> objects;
                status = heap_objects(heap_address, HeapObjectKind::Link, &objects);
                if (!status.ok()) {
                    return status;
                }
                for (const Message& object : objects) {
                    std::vector<std::uint8_t> link_body;
                    status = read_at(object.body, object.size, &link_body);
                    if (!status.ok()) {
                        return status;
                    }
                    LinkEntry entry;
                    status = parse_link(link_body.data(), link_body.size(), &entry);
                    if (!status.ok()) {
                        return status;
                    }
                    links.push_back(std::move(entry));
                }
            }
        }
        link_cache[group_path] = links;
        *out = links;
        return Status::success();
    }

    Status object_address(const std::string& path, std::uint64_t* out) const {
        if (path.empty() || path == "/") {
            *out = root_object;
            return Status::success();
        }
        const auto cached = object_cache.find(path);
        if (cached != object_cache.end()) {
            *out = cached->second;
            return Status::success();
        }
        std::uint64_t current = root_object;
        std::string prefix;
        // A leading slash is the HDF5 spelling of "from the root group".
        std::size_t start = path[0] == '/' ? 1u : 0u;
        while (true) {
            const std::size_t slash = path.find('/', start);
            const std::string component =
                path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
            if (component.empty()) {
                return fail(JOC_ERR_INVALID_ARGUMENT, "empty link name in HDF5 path " + path);
            }
            std::vector<LinkEntry> entries;
            const Status status = links_of(prefix, &entries);
            if (!status.ok()) {
                return status;
            }
            const LinkEntry* found = nullptr;
            for (const LinkEntry& entry : entries) {
                if (entry.name == component) {
                    found = &entry;
                    break;
                }
            }
            if (found == nullptr) {
                return fail(JOC_ERR_INPUT_FORMAT, "HDF5 object not found: " + path);
            }
            current = found->address;
            if (slash == std::string::npos) {
                break;
            }
            prefix = prefix.empty() ? component : prefix + "/" + component;
            start = slash + 1;
        }
        object_cache[path] = current;
        *out = current;
        return Status::success();
    }

    Status attribute_entries(const std::string& object_path,
                             std::vector<AttributeEntry>* out) const {
        std::uint64_t address = kUndefinedAddress;
        Status status = object_address(object_path, &address);
        if (!status.ok()) {
            return status;
        }
        std::vector<Message> messages;
        status = object_messages(address, &messages);
        if (!status.ok()) {
            return status;
        }
        std::vector<AttributeEntry> entries;
        for (const Message& message : messages) {
            if (message.type == kMsgAttribute) {
                std::vector<std::uint8_t> body;
                status = read_at(message.body, std::min<std::uint64_t>(message.size, 8), &body);
                if (!status.ok()) {
                    return status;
                }
                if (body.size() < 8u || body[0] < 1u || body[0] > 3u) {
                    return fail(JOC_ERR_INPUT_FORMAT, 
                                "malformed HDF5 attribute message in " + file_path);
                }
                const std::uint64_t name_size = read_le(body.data() + 2, 2);
                const std::uint64_t name_offset = (body[0] == 3u) ? 9u : 8u;
                if (name_size == 0u || name_offset + name_size > message.size) {
                    return fail(JOC_ERR_INPUT_FORMAT, "malformed HDF5 attribute name in " + file_path);
                }
                std::vector<std::uint8_t> name;
                status = read_at(message.body + name_offset, name_size, &name);
                if (!status.ok()) {
                    return status;
                }
                AttributeEntry entry;
                entry.name = attribute_name(name.data(), name.size());
                entry.body = message.body;
                entry.size = message.size;
                entries.push_back(std::move(entry));
            } else if (message.type == kMsgAttributeInfo) {
                std::vector<std::uint8_t> body;
                status = read_at(message.body, message.size, &body);
                if (!status.ok()) {
                    return status;
                }
                if (body.size() < 2u) {
                    return fail(JOC_ERR_INPUT_FORMAT, 
                                "malformed HDF5 attribute info message in " + file_path);
                }
                const std::uint8_t flags = body[1];
                std::uint64_t position = 2;
                if ((flags & 0x01u) != 0u) {
                    position += 2;  // maximum creation index
                }
                if (position + offset_size > body.size()) {
                    return fail(JOC_ERR_INPUT_FORMAT, 
                                "malformed HDF5 attribute info message in " + file_path);
                }
                const std::uint64_t heap_address = read_le(body.data() + position, offset_size);
                if (heap_address == kUndefinedAddress) {
                    // No dense storage: this object keeps its attributes compact.
                    continue;
                }
                std::vector<Message> objects;
                status = heap_objects(heap_address, HeapObjectKind::Attribute, &objects);
                if (!status.ok()) {
                    return status;
                }
                for (const Message& object : objects) {
                    std::vector<std::uint8_t> head;
                    status = read_at(object.body, std::min<std::uint64_t>(object.size, 8), &head);
                    if (!status.ok()) {
                        return status;
                    }
                    if (head.size() < 8u || head[0] < 1u || head[0] > 3u) {
                        return fail(JOC_ERR_INPUT_FORMAT,
                                    "malformed HDF5 attribute in the dense storage of " + file_path);
                    }
                    const std::uint64_t name_size = read_le(head.data() + 2, 2);
                    const std::uint64_t name_offset = (head[0] == 3u) ? 9u : 8u;
                    if (name_size == 0u || name_offset + name_size > object.size) {
                        return fail(JOC_ERR_INPUT_FORMAT, 
                                    "malformed HDF5 attribute name in " + file_path);
                    }
                    std::vector<std::uint8_t> name;
                    status = read_at(object.body + name_offset, name_size, &name);
                    if (!status.ok()) {
                        return status;
                    }
                    AttributeEntry entry;
                    entry.name = attribute_name(name.data(), name.size());
                    entry.body = object.body;
                    entry.size = object.size;
                    entries.push_back(std::move(entry));
                }
            }
        }
        *out = std::move(entries);
        return Status::success();
    }

    Status decode_attribute(std::uint64_t body_offset, std::uint64_t body_size,
                            Hdf5Attribute* out) const {
        if (body_size > kMaxReadSize) {
            return fail(JOC_ERR_NOT_SUPPORTED, "HDF5 attribute is too large to read");
        }
        std::vector<std::uint8_t> body;
        Status status = read_at(body_offset, body_size, &body);
        if (!status.ok()) {
            return status;
        }
        if (body.size() < 8u || body[0] < 1u || body[0] > 3u) {
            return fail(JOC_ERR_INPUT_FORMAT, "malformed HDF5 attribute message in " + file_path);
        }
        const std::uint8_t version = body[0];
        const std::uint64_t name_size = read_le(body.data() + 2, 2);
        const std::uint64_t datatype_size = read_le(body.data() + 4, 2);
        const std::uint64_t dataspace_size = read_le(body.data() + 6, 2);
        const std::uint64_t name_offset = (version == 3u) ? 9u : 8u;
        const std::uint64_t name_field = (version == 1u) ? align_to_8(name_size) : name_size;
        const std::uint64_t datatype_offset = name_offset + name_field;
        const std::uint64_t dataspace_offset = datatype_offset + align_to_8(datatype_size);
        const std::uint64_t data_offset = dataspace_offset + align_to_8(dataspace_size);
        if (name_size == 0u || datatype_size < 8u || dataspace_size < 4u ||
            data_offset > body.size()) {
            return fail(JOC_ERR_INPUT_FORMAT, "malformed HDF5 attribute message in " + file_path);
        }
        Hdf5Attribute attribute;
        status = parse_datatype(body.data() + datatype_offset,
                                static_cast<std::size_t>(datatype_size), &attribute.type);
        if (!status.ok()) {
            return status;
        }
        status = parse_dataspace(body.data() + dataspace_offset,
                                 static_cast<std::size_t>(dataspace_size), &attribute.shape);
        if (!status.ok()) {
            return status;
        }
        std::uint64_t count = 1;
        for (const std::uint64_t dimension : attribute.shape) {
            count *= dimension;
        }
        const std::uint64_t bytes = count * attribute.type.size;
        if (bytes > body.size() - data_offset) {
            return fail(JOC_ERR_INPUT_FORMAT, "truncated HDF5 attribute data in " + file_path);
        }
        attribute.raw.assign(body.begin() + static_cast<std::ptrdiff_t>(data_offset),
                             body.begin() + static_cast<std::ptrdiff_t>(data_offset + bytes));
        if (attribute.type.big_endian && !is_string_type(attribute.type.type)) {
            for (std::uint64_t index = 0; index < count; ++index) {
                swap_element_bytes(attribute.raw.data() + index * attribute.type.size,
                                   attribute.type.size);
            }
            attribute.type.big_endian = false;
        }
        if (is_string_type(attribute.type.type)) {
            attribute.text.assign(reinterpret_cast<const char*>(attribute.raw.data()),
                                  attribute.raw.size());
            attribute.text = trim_nul(std::move(attribute.text));
        }
        *out = std::move(attribute);
        return Status::success();
    }

    Status dataset_parts(const std::string& path, Hdf5DatasetInfo* info, Layout* layout) const {
        std::uint64_t address = kUndefinedAddress;
        Status status = object_address(path, &address);
        if (!status.ok()) {
            return status;
        }
        std::vector<Message> messages;
        status = object_messages(address, &messages);
        if (!status.ok()) {
            return status;
        }
        bool have_dataspace = false;
        bool have_datatype = false;
        bool have_layout = false;
        for (const Message& message : messages) {
            if (message.type == kMsgDataspace || message.type == kMsgDatatype ||
                message.type == kMsgLayout) {
                std::vector<std::uint8_t> body;
                status = read_at(message.body, message.size, &body);
                if (!status.ok()) {
                    return status;
                }
                if (message.type == kMsgDataspace) {
                    status = parse_dataspace(body.data(), body.size(), &info->shape);
                    have_dataspace = status.ok();
                } else if (message.type == kMsgDatatype) {
                    status = parse_datatype(body.data(), body.size(), &info->type);
                    have_datatype = status.ok();
                } else {
                    status = parse_layout(body.data(), body.size(), layout);
                    if (status.ok() && layout->compact) {
                        layout->inline_offset += message.body;
                    }
                    have_layout = status.ok();
                }
                if (!status.ok()) {
                    return status;
                }
            }
        }
        if (!have_dataspace || !have_datatype) {
            return fail(JOC_ERR_INPUT_FORMAT, "HDF5 object is not a dataset: " + path);
        }
        if (!have_layout) {
            return fail(JOC_ERR_INPUT_FORMAT, "HDF5 dataset has no data layout message: " + path);
        }
        return Status::success();
    }

    // Reads exactly one dataset: the payload of every other object stays on disk.
    Status read_dataset(const std::string& path, Hdf5DatasetInfo* info,
                        std::vector<std::uint8_t>* raw) const {
        Layout layout;
        Status status = dataset_parts(path, info, &layout);
        if (!status.ok()) {
            return status;
        }
        const std::uint64_t count = info->element_count();
        if (count > kMaxReadSize || info->type.size == 0u ||
            (count != 0u && info->type.size > kMaxReadSize / count)) {
            return fail(JOC_ERR_NOT_SUPPORTED, "HDF5 dataset is too large to read: " + path);
        }
        const std::uint64_t bytes = count * info->type.size;
        std::vector<std::uint8_t> data;
        if (bytes == 0u) {
            // An empty dataset has no payload to fetch, allocated or not.
            *raw = std::move(data);
            return Status::success();
        }
        if (layout.compact) {
            if (layout.size < bytes) {
                return fail(JOC_ERR_INPUT_FORMAT, "truncated compact HDF5 dataset: " + path);
            }
            status = read_at(layout.inline_offset, bytes, &data);
        } else {
            if (layout.address == kUndefinedAddress) {
                return fail(JOC_ERR_INPUT_FORMAT, "HDF5 dataset storage is not allocated: " + path);
            }
            if (layout.size != 0u && layout.size < bytes) {
                return fail(JOC_ERR_INPUT_FORMAT,
                            "HDF5 dataset storage is shorter than its shape: " + path);
            }
            status = read_at(layout.address, bytes, &data);
        }
        if (!status.ok()) {
            return status;
        }
        if (info->type.big_endian && !is_string_type(info->type.type)) {
            for (std::uint64_t index = 0; index < count; ++index) {
                swap_element_bytes(data.data() + index * info->type.size, info->type.size);
            }
            info->type.big_endian = false;
        }
        *raw = std::move(data);
        return Status::success();
    }
};

Hdf5File::Hdf5File() : impl_(std::make_unique<Impl>()) {}
Hdf5File::~Hdf5File() = default;
Hdf5File::Hdf5File(Hdf5File&&) noexcept = default;
Hdf5File& Hdf5File::operator=(Hdf5File&&) noexcept = default;

Status Hdf5File::open(const std::string& path) {
    if (impl_ == nullptr) {
        return fail(JOC_ERR_STATE, "HDF5 reader is not constructed");
    }
    if (!fs_utf8::exists(path)) {
        return fail(JOC_ERR_INPUT_NOT_FOUND, "HDF5 file not found: " + path);
    }
    std::error_code error;
    const std::uintmax_t size = fs_utf8::file_size(path, error);
    if (error) {
        return fail(JOC_ERR_IO, "cannot query the size of " + path);
    }
    impl_->stream = fs_utf8::open_input(path);
    if (!impl_->stream.good()) {
        return fail(JOC_ERR_IO, "cannot open " + path);
    }
    impl_->file_path = path;
    impl_->file_size = static_cast<std::uint64_t>(size);
    impl_->opened = false;
    impl_->headers.clear();
    impl_->link_cache.clear();
    impl_->object_cache.clear();

    std::vector<std::uint8_t> signature;
    Status status = impl_->read_at(0, 8, &signature);
    if (!status.ok()) {
        return status;
    }
    static const std::uint8_t kMagic[8] = {0x89u, 'H', 'D', 'F', '\r', '\n', 0x1Au, '\n'};
    if (std::memcmp(signature.data(), kMagic, 8) != 0) {
        return fail(JOC_ERR_INPUT_FORMAT, "not an HDF5 file: " + path);
    }
    std::vector<std::uint8_t> superblock;
    status = impl_->read_at(0, 64, &superblock);
    if (!status.ok()) {
        return status;
    }
    const std::uint8_t version = superblock[8];
    if (version > 1u) {
        return fail(JOC_ERR_NOT_SUPPORTED,
                    "HDF5 superblock version " + std::to_string(version) + " is not supported");
    }
    impl_->offset_size = superblock[13];
    impl_->length_size = superblock[14];
    if (impl_->offset_size == 0u || impl_->offset_size > 8u || impl_->length_size == 0u ||
        impl_->length_size > 8u) {
        return fail(JOC_ERR_INPUT_FORMAT, "malformed HDF5 superblock in " + path);
    }
    const std::uint64_t root_entry = 56;
    status = impl_->read_u64(root_entry + impl_->offset_size, impl_->offset_size,
                             &impl_->root_object);
    if (!status.ok()) {
        return status;
    }
    if (impl_->root_object == kUndefinedAddress) {
        return fail(JOC_ERR_INPUT_FORMAT, "HDF5 root group address is undefined in " + path);
    }
    std::vector<std::uint8_t> root_prefix;
    status = impl_->read_at(impl_->root_object, 4, &root_prefix);
    if (!status.ok()) {
        return status;
    }
    if (std::memcmp(root_prefix.data(), "OHDR", 4) != 0) {
        return fail(JOC_ERR_NOT_SUPPORTED,
                    "HDF5 root group does not use a version 2 object header in " + path);
    }
    impl_->opened = true;
    return Status::success();
}

bool Hdf5File::is_open() const { return impl_ != nullptr && impl_->opened; }

Status Hdf5File::links(const std::string& group_path, std::vector<std::string>* names) const {
    if (names == nullptr) {
        return fail(JOC_ERR_INVALID_ARGUMENT, "null name list");
    }
    if (!is_open()) {
        return fail(JOC_ERR_STATE, "HDF5 reader is not open");
    }
    std::vector<LinkEntry> entries;
    const Status status = impl_->links_of(group_path, &entries);
    if (!status.ok()) {
        return status;
    }
    names->clear();
    names->reserve(entries.size());
    for (const LinkEntry& entry : entries) {
        names->push_back(entry.name);
    }
    return Status::success();
}

bool Hdf5File::has_dataset(const std::string& path) const {
    if (!is_open()) {
        return false;
    }
    std::uint64_t address = kUndefinedAddress;
    if (!impl_->object_address(path, &address).ok()) {
        return false;
    }
    std::vector<Message> messages;
    if (!impl_->object_messages(address, &messages).ok()) {
        return false;
    }
    // A dataset is the object that carries both a dataspace and a datatype,
    // whatever its storage class turns out to be.
    bool dataspace = false;
    bool datatype = false;
    for (const Message& message : messages) {
        dataspace = dataspace || message.type == kMsgDataspace;
        datatype = datatype || message.type == kMsgDatatype;
    }
    return dataspace && datatype;
}

Status Hdf5File::dataset_info(const std::string& path, Hdf5DatasetInfo* out) const {
    if (out == nullptr) {
        return fail(JOC_ERR_INVALID_ARGUMENT, "null dataset info");
    }
    if (!is_open()) {
        return fail(JOC_ERR_STATE, "HDF5 reader is not open");
    }
    Layout layout;
    Hdf5DatasetInfo info;
    const Status status = impl_->dataset_parts(path, &info, &layout);
    if (!status.ok()) {
        return status;
    }
    *out = std::move(info);
    return Status::success();
}

Status Hdf5File::read_dataset_raw(const std::string& path, std::vector<std::uint8_t>* out) const {
    if (out == nullptr) {
        return fail(JOC_ERR_INVALID_ARGUMENT, "null dataset buffer");
    }
    if (!is_open()) {
        return fail(JOC_ERR_STATE, "HDF5 reader is not open");
    }
    Hdf5DatasetInfo info;
    return impl_->read_dataset(path, &info, out);
}

Status Hdf5File::read_dataset_double(const std::string& path, std::vector<double>* out) const {
    if (out == nullptr) {
        return fail(JOC_ERR_INVALID_ARGUMENT, "null dataset buffer");
    }
    if (!is_open()) {
        return fail(JOC_ERR_STATE, "HDF5 reader is not open");
    }
    Hdf5DatasetInfo info;
    std::vector<std::uint8_t> raw;
    const Status status = impl_->read_dataset(path, &info, &raw);
    if (!status.ok()) {
        return status;
    }
    const std::uint64_t count = info.element_count();
    out->clear();
    out->resize(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        const std::uint8_t* element = raw.data() + index * info.type.size;
        double value = 0.0;
        switch (info.type.type) {
            case Hdf5Type::Float64: {
                std::memcpy(&value, element, sizeof(double));
                break;
            }
            case Hdf5Type::Float32: {
                float narrow = 0.0f;
                std::memcpy(&narrow, element, sizeof(float));
                value = static_cast<double>(narrow);
                break;
            }
            case Hdf5Type::Int8: {
                value = static_cast<double>(static_cast<std::int8_t>(element[0]));
                break;
            }
            case Hdf5Type::UInt8: {
                value = static_cast<double>(element[0]);
                break;
            }
            case Hdf5Type::Int16: {
                std::int16_t narrow = 0;
                std::memcpy(&narrow, element, sizeof(narrow));
                value = static_cast<double>(narrow);
                break;
            }
            case Hdf5Type::UInt16: {
                std::uint16_t narrow = 0;
                std::memcpy(&narrow, element, sizeof(narrow));
                value = static_cast<double>(narrow);
                break;
            }
            case Hdf5Type::Int32: {
                std::int32_t narrow = 0;
                std::memcpy(&narrow, element, sizeof(narrow));
                value = static_cast<double>(narrow);
                break;
            }
            case Hdf5Type::UInt32: {
                std::uint32_t narrow = 0;
                std::memcpy(&narrow, element, sizeof(narrow));
                value = static_cast<double>(narrow);
                break;
            }
            case Hdf5Type::Int64: {
                std::int64_t narrow = 0;
                std::memcpy(&narrow, element, sizeof(narrow));
                value = static_cast<double>(narrow);
                break;
            }
            case Hdf5Type::UInt64: {
                std::uint64_t narrow = 0;
                std::memcpy(&narrow, element, sizeof(narrow));
                value = static_cast<double>(narrow);
                break;
            }
            default:
                return fail(JOC_ERR_NOT_SUPPORTED,
                            "HDF5 dataset is not numeric and cannot be read as double: " + path);
        }
        (*out)[static_cast<std::size_t>(index)] = value;
    }
    return Status::success();
}

Status Hdf5File::attribute_names(const std::string& object_path,
                                 std::vector<std::string>* names) const {
    if (names == nullptr) {
        return fail(JOC_ERR_INVALID_ARGUMENT, "null name list");
    }
    if (!is_open()) {
        return fail(JOC_ERR_STATE, "HDF5 reader is not open");
    }
    std::vector<AttributeEntry> entries;
    const Status status = impl_->attribute_entries(object_path, &entries);
    if (!status.ok()) {
        return status;
    }
    names->clear();
    names->reserve(entries.size());
    for (const AttributeEntry& entry : entries) {
        names->push_back(entry.name);
    }
    return Status::success();
}

Status Hdf5File::attribute(const std::string& object_path, const std::string& name,
                           Hdf5Attribute* out) const {
    if (out == nullptr) {
        return fail(JOC_ERR_INVALID_ARGUMENT, "null attribute");
    }
    if (!is_open()) {
        return fail(JOC_ERR_STATE, "HDF5 reader is not open");
    }
    std::vector<AttributeEntry> entries;
    Status status = impl_->attribute_entries(object_path, &entries);
    if (!status.ok()) {
        return status;
    }
    for (const AttributeEntry& entry : entries) {
        if (entry.name == name) {
            return impl_->decode_attribute(entry.body, entry.size, out);
        }
    }
    return fail(JOC_ERR_INPUT_FORMAT, "HDF5 attribute not found: " + name);
}

Status Hdf5File::attribute_text(const std::string& object_path, const std::string& name,
                                std::string* out) const {
    if (out == nullptr) {
        return fail(JOC_ERR_INVALID_ARGUMENT, "null text");
    }
    Hdf5Attribute attribute;
    Status status = this->attribute(object_path, name, &attribute);
    if (!status.ok()) {
        return status;
    }
    if (!is_string_type(attribute.type.type)) {
        return fail(JOC_ERR_INPUT_FORMAT, "HDF5 attribute is not a string: " + name);
    }
    *out = attribute.text;
    return Status::success();
}

}  // namespace joc::io
