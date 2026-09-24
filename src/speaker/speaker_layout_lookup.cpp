#include "speaker/speaker_layout_lookup.h"

#include <cstring>

namespace joc::speaker {

namespace {

struct FrozenLayout {
    const char* name;
    std::uint32_t out_ch_config;
    std::uint32_t speaker_bitfield;
};

constexpr FrozenLayout kLayouts[kLayoutCount] = {
    {"2.0", 0, 1},
    {"3.1", 3, 7},
    {"5.1", 7, 15},
    {"7.1", 11, 31},
    {"5.1.2", 13, 1039},
    {"5.1.4", 14, 2575},
    {"7.1.2", 15, 1055},
    {"7.1.4", 16, 2591},
    {"9.1.4", 19, 2719},
    {"9.1.6", 20, 3743},
};

void fill(const FrozenLayout& source, LayoutInfo* out) {
    out->name = source.name;
    out->out_ch_config = source.out_ch_config;
    out->speaker_bitfield = source.speaker_bitfield;
    out->channel_count = ejoc_speaker_layout_channel_count(source.speaker_bitfield);
}

}  // namespace

bool layout_at(int index, LayoutInfo* out) {
    if (out == nullptr || index < 0 || index >= kLayoutCount) {
        return false;
    }
    fill(kLayouts[index], out);
    return true;
}

bool layout_by_name(const char* name, LayoutInfo* out) {
    if (name == nullptr || out == nullptr) {
        return false;
    }
    for (int index = 0; index < kLayoutCount; ++index) {
        if (std::strcmp(name, kLayouts[index].name) == 0) {
            fill(kLayouts[index], out);
            return true;
        }
    }
    return false;
}

bool layout_by_bitfield(std::uint32_t bitfield, LayoutInfo* out) {
    if (out == nullptr) {
        return false;
    }
    for (int index = 0; index < kLayoutCount; ++index) {
        if (kLayouts[index].speaker_bitfield == bitfield) {
            fill(kLayouts[index], out);
            return true;
        }
    }
    return false;
}

}  // namespace joc::speaker
