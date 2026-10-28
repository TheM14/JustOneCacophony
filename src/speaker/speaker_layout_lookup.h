
#pragma once

#include <cstdint>

#include "eac3joc_core.h"
#include "joc_core.h"

namespace joc::speaker {

inline constexpr int kLayoutCount = 11;

struct LayoutInfo {
    const char* name = "";
    std::uint32_t out_ch_config = 0;
    std::uint32_t speaker_bitfield = 0;
    std::uint32_t channel_count = 0;
};

// All eleven layouts in the reference's order (2.0, 3.1, 4.0, 5.1, 7.1, 5.1.2,
// 5.1.4, 7.1.2, 7.1.4, 9.1.4, 9.1.6).
bool layout_at(int index, LayoutInfo* out);

// False when the name is unknown (case-sensitive, as the reference CLI).  "4.0"
// is the quad layout (L R Lrs Rrs); a stored value is normalized by the caller
// before the name is looked up here.
bool layout_by_name(const char* name, LayoutInfo* out);

bool layout_by_bitfield(std::uint32_t bitfield, LayoutInfo* out);

}  // namespace joc::speaker
