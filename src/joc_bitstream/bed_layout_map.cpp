
#include "joc_bitstream/bed_layout_map.h"

#include <cstring>

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

// Table 13 的 17 bit 非标准掩码，按 channel order 排列：RC_L 是最先的声道
// （array position 16），落在 LSB；RC_LFE2 最后（array position 0）。
constexpr BedChannel kNonStandardBits[17] = {
    BedChannel::kLeft, BedChannel::kRight, BedChannel::kCenter,
    BedChannel::kLfe, BedChannel::kLeftSurround, BedChannel::kRightSurround,
    BedChannel::kLeftRearSurround, BedChannel::kRightRearSurround,
    BedChannel::kTopFrontLeft, BedChannel::kTopFrontRight,
    BedChannel::kTopSideLeft, BedChannel::kTopSideRight,
    BedChannel::kTopBackLeft, BedChannel::kTopBackRight,
    BedChannel::kLeftWide, BedChannel::kRightWide, BedChannel::kLfe2,
};

struct ChannelSpec {
    BedChannel channel;
    const char* short_name;
    const char* label;
    double xyz[3];
};

// 固定方位：ETSI TS 103 420 V1.2.1 clause B.2.1.4 Table B.10 的 ADM 笛卡尔坐标
// （x 右、y 前、z 上）。bed 声道的位置只由声道标签决定，与目标输出布局无关。
// Table B.10 把 Lw/Rw 的 Y 记作 0,677419，即 ADM 62 级网格上的 42/62 = 21/31。
constexpr ChannelSpec kChannels[] = {
    {BedChannel::kLeft, "L", "RC_L", {-1.0, 1.0, 0.0}},
    {BedChannel::kRight, "R", "RC_R", {1.0, 1.0, 0.0}},
    {BedChannel::kCenter, "C", "RC_C", {0.0, 1.0, 0.0}},
    {BedChannel::kLfe, "LFE", "RC_LFE", {-1.0, 1.0, -1.0}},
    {BedChannel::kLeftSurround, "Ls", "RC_LS", {-1.0, 0.0, 0.0}},
    {BedChannel::kRightSurround, "Rs", "RC_RS", {1.0, 0.0, 0.0}},
    {BedChannel::kLeftRearSurround, "Lrs", "RC_LB", {-1.0, -1.0, 0.0}},
    {BedChannel::kRightRearSurround, "Rrs", "RC_RB", {1.0, -1.0, 0.0}},
    {BedChannel::kTopFrontLeft, "Ltf", "RC_TFL", {-1.0, 1.0, 1.0}},
    {BedChannel::kTopFrontRight, "Rtf", "RC_TFR", {1.0, 1.0, 1.0}},
    {BedChannel::kTopSideLeft, "Ltm", "RC_TSL", {-1.0, 0.0, 1.0}},
    {BedChannel::kTopSideRight, "Rtm", "RC_TSR", {1.0, 0.0, 1.0}},
    {BedChannel::kTopBackLeft, "Ltr", "RC_TBL", {-1.0, -1.0, 1.0}},
    {BedChannel::kTopBackRight, "Rtr", "RC_TBR", {1.0, -1.0, 1.0}},
    {BedChannel::kLeftWide, "Lw", "RC_LW", {-1.0, 21.0 / 31.0, 0.0}},
    {BedChannel::kRightWide, "Rw", "RC_RW", {1.0, 21.0 / 31.0, 0.0}},
    {BedChannel::kLfe2, "LFE2", "RC_LFE2", {1.0, 1.0, -1.0}},
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

}  // namespace joc::joc
