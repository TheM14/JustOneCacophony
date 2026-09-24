#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "foundation/status.h"

// Read-only subset of the HDF5 file format, sized for the SOFA files this
// project consumes: superblock v0, version 2 object headers, fractal-heap link
// and attribute storage, compact and contiguous datasets.  The file is opened
// lazily: only the requested dataset's bytes are read into memory, everything
// else (superblock, object headers, heap blocks) is fetched on demand and the
// metadata that was parsed is cached by file address.
//
// Paths are HDF5 link paths ("Data.IR" is a single link name here, "Group/Set"
// walks two links); the empty path names the root group.  Byte order is
// normalized on read, so callers never see the file's own endianness.
namespace joc::io {

enum class Hdf5Type {
    Unknown,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    String,
};

struct Hdf5TypeInfo {
    Hdf5Type type = Hdf5Type::Unknown;
    std::uint32_t size = 0;  // bytes per element as stored in the file
    bool big_endian = false;
    bool is_signed = false;
};

struct Hdf5DatasetInfo {
    std::vector<std::uint64_t> shape;
    Hdf5TypeInfo type;

    std::uint64_t element_count() const;
};

struct Hdf5Attribute {
    Hdf5TypeInfo type;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> raw;  // C order, host byte order
    std::string text;               // decoded for fixed-length string attributes
};

class Hdf5File {
public:
    Hdf5File();
    ~Hdf5File();
    Hdf5File(Hdf5File&&) noexcept;
    Hdf5File& operator=(Hdf5File&&) noexcept;
    Hdf5File(const Hdf5File&) = delete;
    Hdf5File& operator=(const Hdf5File&) = delete;

    Status open(const std::string& path);
    bool is_open() const;

    // Names of the links of a group ("" is the root group).
    Status links(const std::string& group_path, std::vector<std::string>* names) const;

    bool has_dataset(const std::string& path) const;
    Status dataset_info(const std::string& path, Hdf5DatasetInfo* out) const;
    Status read_dataset_raw(const std::string& path, std::vector<std::uint8_t>* out) const;
    Status read_dataset_double(const std::string& path, std::vector<double>* out) const;

    Status attribute_names(const std::string& object_path, std::vector<std::string>* names) const;
    Status attribute(const std::string& object_path, const std::string& name, Hdf5Attribute* out) const;
    Status attribute_text(const std::string& object_path, const std::string& name, std::string* out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joc::io
