
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace joc::io {

enum class NpyType { Unknown, Float64, Float32, Int64, Int32, Int16, UInt8, Complex128, Unicode };

struct NpyArray {
    std::string descr;
    NpyType type = NpyType::Unknown;
    bool fortran_order = false;
    std::vector<std::int64_t> shape;
    const std::uint8_t* data = nullptr;
    std::size_t data_bytes = 0;
    std::size_t item_bytes = 0;        // bytes per element as stored

    std::size_t element_count() const;
    std::size_t element_size() const;  // bytes per element in the file
};

// Parses the header of one `.npy` image.  `data` must outlive the NpyArray.
bool parse_npy(const std::uint8_t* data, std::size_t size, NpyArray* out, std::string* error);

bool npy_to_double(const NpyArray& array, std::vector<double>* out, std::string* error);
bool npy_to_int16(const NpyArray& array, std::vector<std::int16_t>* out, std::string* error);
bool npy_to_int32(const NpyArray& array, std::vector<std::int32_t>* out, std::string* error);
bool npy_to_uint8(const NpyArray& array, std::vector<std::uint8_t>* out, std::string* error);

bool npy_unicode_to_utf8(const NpyArray& array, std::string* out, std::string* error);

// Materializes the array in C order as raw element bytes.  Fortran-order members
// hybrid synthesis table as [count][4] row-major while the shipped table stores it
// Fortran-order, so passing the file bytes straight through would transpose it.
bool npy_to_c_order(const NpyArray& array, std::vector<std::uint8_t>* out, std::string* error);

bool npy_shape_is(const NpyArray& array, const std::vector<std::int64_t>& expected);

}  // namespace joc::io
