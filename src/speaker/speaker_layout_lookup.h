
#pragma once

#include <cstdint>

#include "eac3joc_core.h"
#include "joc_core.h"

namespace joc::speaker {

inline constexpr int kLayoutCount = 10;

struct LayoutInfo {
    const char* name = "";
    std::uint32_t out_ch_config = 0;
    std::uint32_t speaker_bitfield = 0;
    std::uint32_t channel_count = 0;
};

// All ten layouts in the reference's order (2.0, 3.1, 5.1, 7.1, 5.1.2, 5.1.4,
bool layout_at(int index, LayoutInfo* out);

// False when the name is unknown (case-sensitive, exactly as the reference CLI).
bool layout_by_name(const char* name, LayoutInfo* out);

bool layout_by_bitfield(std::uint32_t bitfield, LayoutInfo* out);

}  // namespace joc::speaker
