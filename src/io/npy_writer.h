#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace joc::io {

// NPY 1.0 images and a minimal ZIP container, used to write the compiled HRTF
// cache in exactly the layout the reader (and NumPy) expects.  Only what the
// cache needs is implemented: little-endian C-order arrays and stored members.
struct NpyMember {
    std::string name;   // archive member name, without the .npy suffix
    std::string descr;  // NumPy dtype string, e.g. "<f8", "<c16", "<U123"
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> data;  // C order payload in the dtype's byte order
};

// Serializes one array as an NPY 1.0 image (magic, header, 64-byte aligned).
std::vector<std::uint8_t> npy_image(const std::string& descr,
                                    const std::vector<std::uint64_t>& shape,
                                    const std::vector<std::uint8_t>& data);

// Writes a ZIP archive with stored (uncompressed) members.  The upstream reader
// accepts stored members, and compression would need a deflate encoder.
bool write_zip(const std::string& path, const std::vector<NpyMember>& members,
               std::string* error);

// Serializes the archive in memory (same layout as write_zip).
std::vector<std::uint8_t> zip_bytes(const std::vector<NpyMember>& members);

// UTF-8 text as the payload of a NumPy Unicode scalar string ('<U<n>').
std::vector<std::uint8_t> utf8_to_utf32le(const std::string& text);

// Python's repr() for a double: shortest round-trip digits with Python's
// exponent rules, which is what json.dumps emits for the cache metadata.
std::string python_float_repr(double value);

}  // namespace joc::io
