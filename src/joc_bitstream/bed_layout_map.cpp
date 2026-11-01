
#include "joc_bitstream/bed_layout_map.h"

#include <cstring>

#include "speaker/speaker_layouts.h"

namespace joc::joc {
namespace {

constexpr std::uint32_t bit(BedChannel channel) {
    return 1u << static_cast<unsigned>(channel);
}

// Table 12 每个 bit 展开的声道（LSB 在前）。
struct StandardBit {
    BedChannel first;
    BedChannel second;  // kCount 表示单声道
};

constexpr StandardBit kStandardBits[10] = {
    {BedChannel::kLeft, BedChannel::kRight},
    {BedChannel::kCenter, BedChannel::kCount},
    {BedChannel::kLfe, BedChannel::kCount},
    {BedChannel::kLeftSurround, BedChannel::kRightSurround},
    {BedChannel::kLeftRearSurround, BedChannel::kRightRearSurround},
    {BedChannel::kTopFrontLeft, BedChannel::kTopFrontRight},
    {BedChannel::kTopSideLeft, BedChannel::kTopSideRight},
    {BedChannel::kTopBackLeft, BedChannel::kTopBackRight},
    {BedChannel::kLeftWide, BedChannel::kRightWide},
    {BedChannel::kLfe2, BedChannel::kCount},
};

// Table 13 的 17 bit 非标准掩码（LSB 在前）。
constexpr BedChannel kNonStandardBits[17] = {
    BedChannel::kLfe2, BedChannel::kRightWide, BedChannel::kLeftWide,
    BedChannel::kTopBackRight, BedChannel::kTopBackLeft,
    BedChannel::kTopSideRight, BedChannel::kTopSideLeft,
    BedChannel::kTopFrontRight, BedChannel::kTopFrontLeft,
    BedChannel::kRightRearSurround, BedChannel::kLeftRearSurround,
    BedChannel::kRightSurround, BedChannel::kLeftSurround,
    BedChannel::kLfe, BedChannel::kCenter, BedChannel::kRight,
    BedChannel::kLeft,
};

struct ChannelSpec {
    BedChannel channel;
    const char* short_name;
    const char* label;
    double xyz[3];
};

constexpr ChannelSpec kChannels[] = {
    {BedChannel::kLeft, "L", "RC_L", {-1.0, 1.0, 0.0}},
    {BedChannel::kRight, "R", "RC_R", {1.0, 1.0, 0.0}},
    {BedChannel::kCenter, "C", "RC_C", {0.0, 1.0, 0.0}},
    {BedChannel::kLfe, "LFE", "RC_LFE", {0.0, 0.0, 0.0}},
    {BedChannel::kLeftSurround, "Ls", "RC_LS", {-1.0, 0.0, 0.0}},
    {BedChannel::kRightSurround, "Rs", "RC_RS", {1.0, 0.0, 0.0}},
    {BedChannel::kLeftRearSurround, "Lrs", "RC_LB", {-1.0, -1.0, 0.0}},
    {BedChannel::kRightRearSurround, "Rrs", "RC_RB", {1.0, -1.0, 0.0}},
    {BedChannel::kTopFrontLeft, "Ltf", "RC_TFL", {-0.5, 0.5, 1.0}},
    {BedChannel::kTopFrontRight, "Rtf", "RC_TFR", {0.5, 0.5, 1.0}},
    {BedChannel::kTopSideLeft, "Ltm", "RC_TSL", {-1.0, 0.0, 1.0}},
    {BedChannel::kTopSideRight, "Rtm", "RC_TSR", {1.0, 0.0, 1.0}},
    {BedChannel::kTopBackLeft, "Ltr", "RC_TBL", {-0.5, -0.5, 1.0}},
    {BedChannel::kTopBackRight, "Rtr", "RC_TBR", {0.5, -0.5, 1.0}},
    {BedChannel::kLeftWide, "Lw", "RC_LW", {-1.0, 0.5, 0.0}},
    {BedChannel::kRightWide, "Rw", "RC_RW", {1.0, 0.5, 0.0}},
    {BedChannel::kLfe2, "LFE2", "RC_LFE2", {0.0, 0.0, 0.0}},
};

const ChannelSpec* spec_of(BedChannel channel) {
    for (const ChannelSpec& spec : kChannels) {
        if (spec.channel == channel) {
            return &spec;
        }
    }
    return nullptr;
}

// 标准 bed 布局名，键是声明声道的集合。
struct LayoutName {
    std::uint32_t mask;
    const char* name;
};

constexpr std::uint32_t mask_of() { return 0u; }

constexpr LayoutName kLayoutNames[] = {
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight), "2.0"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) | bit(BedChannel::kCenter), "3.0"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) | bit(BedChannel::kCenter) |
         bit(BedChannel::kLfe), "3.1"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround), "4.0"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter), "5.0"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe), "5.1"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround), "7.1"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround) |
         bit(BedChannel::kLeftWide) | bit(BedChannel::kRightWide), "9.1"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kTopSideLeft) | bit(BedChannel::kTopSideRight), "5.1.2"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kTopFrontLeft) | bit(BedChannel::kTopFrontRight) |
         bit(BedChannel::kTopBackLeft) | bit(BedChannel::kTopBackRight), "5.1.4"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround) |
         bit(BedChannel::kTopSideLeft) | bit(BedChannel::kTopSideRight), "7.1.2"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround) |
         bit(BedChannel::kTopFrontLeft) | bit(BedChannel::kTopFrontRight) |
         bit(BedChannel::kTopBackLeft) | bit(BedChannel::kTopBackRight), "7.1.4"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround) |
         bit(BedChannel::kTopFrontLeft) | bit(BedChannel::kTopFrontRight) |
         bit(BedChannel::kTopSideLeft) | bit(BedChannel::kTopSideRight) |
         bit(BedChannel::kTopBackLeft) | bit(BedChannel::kTopBackRight), "7.1.6"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround) |
         bit(BedChannel::kLeftWide) | bit(BedChannel::kRightWide) |
         bit(BedChannel::kTopSideLeft) | bit(BedChannel::kTopSideRight), "9.1.2"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround) |
         bit(BedChannel::kLeftWide) | bit(BedChannel::kRightWide) |
         bit(BedChannel::kTopFrontLeft) | bit(BedChannel::kTopFrontRight) |
         bit(BedChannel::kTopBackLeft) | bit(BedChannel::kTopBackRight), "9.1.4"},
    {bit(BedChannel::kLeft) | bit(BedChannel::kRight) |
         bit(BedChannel::kLeftSurround) | bit(BedChannel::kRightSurround) |
         bit(BedChannel::kCenter) | bit(BedChannel::kLfe) |
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround) |
         bit(BedChannel::kLeftWide) | bit(BedChannel::kRightWide) |
         bit(BedChannel::kTopFrontLeft) | bit(BedChannel::kTopFrontRight) |
         bit(BedChannel::kTopSideLeft) | bit(BedChannel::kTopSideRight) |
         bit(BedChannel::kTopBackLeft) | bit(BedChannel::kTopBackRight), "9.1.6"},
};

const char* layout_name_for(std::uint32_t mask) {
    for (const LayoutName& entry : kLayoutNames) {
        if (entry.mask == mask) {
            return entry.name;
        }
    }
    return "";
}

// 每个扬声器布局的标准声道名（按该布局的标准声道顺序排列）。
struct LayoutChannels {
    std::uint32_t out_ch_config;
    std::uint32_t count;
    const char* names[16];
};

constexpr LayoutChannels kLayoutChannels[] = {
    {0, 2, {"L", "R"}},
    {3, 4, {"L", "R", "C", "LFE"}},
    {255, 4, {"L", "R", "Lrs", "Rrs"}},
    {7, 6, {"L", "R", "C", "LFE", "Ls", "Rs"}},
    {11, 8, {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs"}},
    {13, 8, {"L", "R", "C", "LFE", "Ls", "Rs", "Ltm", "Rtm"}},
    {14, 10, {"L", "R", "C", "LFE", "Ls", "Rs", "Ltf", "Rtf", "Ltr", "Rtr"}},
    {15, 10, {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltm", "Rtm"}},
    {16, 12, {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf", "Ltr", "Rtr"}},
    {19, 14, {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Lw", "Rw",
              "Ltf", "Rtf", "Ltr", "Rtr"}},
    {20, 16, {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Lw", "Rw",
              "Ltf", "Rtf", "Ltm", "Rtm", "Ltr", "Rtr"}},
};

const ejoc::speaker_tables::LayoutGeometry* geometry_for(std::uint32_t out_ch_config) {
    for (const ejoc::speaker_tables::LayoutGeometry& layout : ejoc::speaker_tables::kLayouts) {
        if (layout.out_ch_config == out_ch_config) {
            return &layout;
        }
    }
    return nullptr;
}

}  // namespace

BedLayout resolve_bed_layout(const std::vector<oamd::BedAssignment>& assignments,
                             bool bed_only) {
    BedLayout layout;
    layout.bed_only = bed_only;
    std::uint32_t mask = 0;
    for (const oamd::BedAssignment& assignment : assignments) {
        if (assignment.lfe_only) {
            mask |= bit(BedChannel::kLfe);
            layout.complete = false;
            continue;
        }
        if (assignment.standard) {
            for (int index = 0; index < 10; ++index) {
                if (((assignment.mask >> index) & 1u) == 0u) {
                    continue;
                }
                mask |= bit(kStandardBits[index].first);
                if (kStandardBits[index].second != BedChannel::kCount) {
                    mask |= bit(kStandardBits[index].second);
                }
            }
            continue;
        }
        for (int index = 0; index < 17; ++index) {
            if (((assignment.mask >> index) & 1u) != 0u) {
                mask |= bit(kNonStandardBits[index]);
            }
        }
        layout.complete = false;  // 非标准掩码的声道名不在本标准表里
    }
    if (mask == 0u) {
        return layout;
    }
    layout.present = true;
    layout.mask = mask;
    layout.name = layout_name_for(mask);
    for (const ChannelSpec& spec : kChannels) {
        const BedChannel channel = spec.channel;
        if ((mask & bit(channel)) == 0u) {
            continue;
        }
        if (channel == BedChannel::kLfe || channel == BedChannel::kLfe2) {
            continue;
        }
        layout.channels.push_back(channel);
    }
    return layout;
}

const char* bed_channel_short_name(BedChannel channel) {
    const ChannelSpec* spec = spec_of(channel);
    return spec != nullptr ? spec->short_name : "";
}

const char* bed_channel_label(BedChannel channel) {
    const ChannelSpec* spec = spec_of(channel);
    return spec != nullptr ? spec->label : "";
}

bool bed_channel_adm_position(BedChannel channel, double xyz[3]) {
    const ChannelSpec* spec = spec_of(channel);
    if (spec == nullptr || xyz == nullptr) {
        return false;
    }
    std::memcpy(xyz, spec->xyz, sizeof(spec->xyz));
    return true;
}

bool bed_channel_q15(BedChannel channel, std::uint16_t q15[3]) {
    double xyz[3] = {0.0, 0.0, 0.0};
    if (!bed_channel_adm_position(channel, xyz) || q15 == nullptr) {
        return false;
    }
    const double q1 = (xyz[0] + 1.0) * 0.5 * 32767.0;
    const double q2 = (1.0 - xyz[1]) * 0.5 * 32767.0;
    const double q3 = xyz[2] * 32767.0;
    q15[0] = static_cast<std::uint16_t>(q1 < 0.0 ? 0.0 : (q1 > 32767.0 ? 32767.0 : q1));
    q15[1] = static_cast<std::uint16_t>(q2 < 0.0 ? 0.0 : (q2 > 32767.0 ? 32767.0 : q2));
    q15[2] = static_cast<std::uint16_t>(q3 < 0.0 ? 0.0 : (q3 > 32767.0 ? 32767.0 : q3));
    return true;
}

bool layout_bed_anchor(std::uint32_t out_ch_config, BedChannel channel,
                       std::uint16_t q15[3]) {
    if (q15 == nullptr) {
        return false;
    }
    const char* name = bed_channel_short_name(channel);
    if (name == nullptr || name[0] == '\0' || std::strcmp(name, "LFE") == 0) {
        return false;
    }
    const LayoutChannels* table = nullptr;
    for (const LayoutChannels& entry : kLayoutChannels) {
        if (entry.out_ch_config == out_ch_config) {
            table = &entry;
            break;
        }
    }
    const ejoc::speaker_tables::LayoutGeometry* geometry = geometry_for(out_ch_config);
    if (table == nullptr || geometry == nullptr) {
        return false;
    }
    int standard_index = -1;
    for (std::uint32_t index = 0; index < table->count; ++index) {
        if (std::strcmp(table->names[index], name) == 0) {
            standard_index = static_cast<int>(index);
            break;
        }
    }
    if (standard_index < 0) {
        return false;
    }
    const std::uint8_t internal = geometry->standard_from_internal[standard_index];
    const ejoc::speaker_tables::RegionGeometry& region = geometry->regions[0];
    for (std::uint32_t point = 0; point < region.point_count; ++point) {
        if (region.points[point].speaker_id != internal) {
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            q15[axis] = region.points[point].coordinate_q15[axis];
        }
        return true;
    }
    return false;
}

}  // namespace joc::joc
