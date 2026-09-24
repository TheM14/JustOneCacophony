
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace joc::crypto {

class Sha256 {
public:
    Sha256() { reset(); }

    void reset();
    void update(const void* data, std::size_t size);
    void finish(std::uint8_t out[32]);
    std::string finish_hex();

private:
    void transform(const std::uint8_t block[64]);

    std::uint32_t state_[8] = {};
    std::uint64_t bit_count_ = 0;
    std::uint8_t buffer_[64] = {};
    std::size_t buffer_used_ = 0;
};

std::string sha256_hex(const void* data, std::size_t size);
bool sha256_hex_matches(const void* data, std::size_t size, const std::string& expected_hex);

}  // namespace joc::crypto
