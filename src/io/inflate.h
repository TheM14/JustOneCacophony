
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace joc::io {

bool inflate_raw(const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>* out);

}  // namespace joc::io
