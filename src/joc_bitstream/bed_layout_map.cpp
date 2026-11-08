
#include "joc_bitstream/bed_layout_map.h"

#include <cmath>
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
         bit(BedChannel::kLeftRearSurround) | bit(BedChannel::kRightRearSurround), "6.0"},
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


namespace {

// ETSI TS 103 420 Table B.10: the fixed ADM position of every Table 12 bed
// channel.  bed_output_description() matches these points against the target
// layout; the kChannels table above keeps the ADM/binaural path's coordinates.
struct TableB10Position {
    BedChannel channel;
    double xyz[3];
};

constexpr TableB10Position kTableB10[oamd::kBedLabelCount] = {
    {BedChannel::kLeft, {-1.0, 1.0, 0.0}},
    {BedChannel::kRight, {1.0, 1.0, 0.0}},
    {BedChannel::kCenter, {0.0, 1.0, 0.0}},
    {BedChannel::kLfe, {-1.0, 1.0, -1.0}},
    {BedChannel::kLeftSurround, {-1.0, 0.0, 0.0}},
    {BedChannel::kRightSurround, {1.0, 0.0, 0.0}},
    {BedChannel::kLeftRearSurround, {-1.0, -1.0, 0.0}},
    {BedChannel::kRightRearSurround, {1.0, -1.0, 0.0}},
    {BedChannel::kTopFrontLeft, {-1.0, 1.0, 1.0}},
    {BedChannel::kTopFrontRight, {1.0, 1.0, 1.0}},
    {BedChannel::kTopSideLeft, {-1.0, 0.0, 1.0}},
    {BedChannel::kTopSideRight, {1.0, 0.0, 1.0}},
    {BedChannel::kTopBackLeft, {-1.0, -1.0, 1.0}},
    {BedChannel::kTopBackRight, {1.0, -1.0, 1.0}},
    {BedChannel::kLeftWide, {-1.0, 21.0 / 31.0, 0.0}},
    {BedChannel::kRightWide, {1.0, 21.0 / 31.0, 0.0}},
    {BedChannel::kLfe2, {1.0, 1.0, -1.0}},
};

// bed_layout.BED_REFERENCE_POSITIONS_Q15; index 3 is the non-spatial sentinel.
constexpr std::int32_t kReferencePositions[kBedReferencePositionCount][3] = {
    {0, 0, 0},          {32767, 0, 0},      {16384, 0, 0},      {-1, -1, -1},
    {0, 16384, 0},      {32767, 16384, 0},  {0, 32767, 0},      {32767, 32767, 0},
    {16384, 32767, 0},  {5285, 0, 0},       {27483, 0, 0},      {0, 5285, 0},
    {32767, 5285, 0},   {7928, 0, 32767},   {24840, 0, 32767},  {7928, 7928, 32767},
    {24840, 7928, 32767}, {7928, 16384, 32767}, {24840, 16384, 32767},
    {7928, 24840, 32767}, {24840, 24840, 32767}, {7928, 32767, 32767},
    {24840, 32767, 32767}, {11099, 0, 0},    {21669, 0, 0},      {0, 11099, 0},
    {32767, 11099, 0},  {0, 21669, 0},      {32767, 21669, 0},  {0, 27483, 0},
    {32767, 27483, 0},  {5285, 32767, 0},   {27483, 32767, 0},  {11099, 32767, 0},
    {21669, 32767, 0},  {16384, 0, 32767},  {16384, 16384, 32767},
    {16384, 32767, 32767},
};

// 自动 trim 的每个布局基准衰减，单位 dB。
struct AttenuationEntry {
    const char* compact_name;
    double db;
};

constexpr AttenuationEntry kAttenuation[] = {
    {"20", -4.5},  {"31", -4.5},  {"40", -3.0},  {"51", -3.0},
    {"60", -1.5},  {"71", -1.5},  {"512", -2.25}, {"514", -1.5},
    {"712", -0.75}, {"714", 0.0}, {"914", 0.0},  {"916", 0.0},
};

constexpr int kMaxChannels = static_cast<int>(ejoc::speaker_tables::kMaxChannels);

void set_sentinel(std::int32_t coordinate[3]) {
    coordinate[0] = -1;
    coordinate[1] = -1;
    coordinate[2] = -1;
}

bool is_sentinel(const std::int32_t coordinate[3]) {
    return coordinate[0] == -1 && coordinate[1] == -1 && coordinate[2] == -1;
}

bool is_lfe_label(BedChannel channel) {
    return channel == BedChannel::kLfe || channel == BedChannel::kLfe2;
}

const TableB10Position* table_b10(BedChannel channel) {
    for (const TableB10Position& entry : kTableB10) {
        if (entry.channel == channel) {
            return &entry;
        }
    }
    return nullptr;
}

// ADM 笛卡尔坐标 -> 渲染器方位 (u 右, v 后, w 上)。
void render_coordinates(const double xyz[3], double out[3]) {
    out[0] = (xyz[0] + 1.0) * 0.5;
    out[1] = (1.0 - xyz[1]) * 0.5;
    out[2] = xyz[2];
}

// 渲染器方位的欧氏距离，比较的是归一化到 0..1 的坐标。
double render_distance(const double left[3], const double right[3]) {
    const double du = left[0] - right[0];
    const double dv = left[1] - right[1];
    const double dw = left[2] - right[2];
    return std::sqrt(du * du + dv * dv + dw * dw);
}

// Reference-position matching: the non-spatial sentinel resolves to index 3.
int nearest_reference(const std::int32_t coordinate[3]) {
    if (is_sentinel(coordinate)) {
        return 3;
    }
    const double position[3] = {
        static_cast<double>(coordinate[0]) / 32768.0,
        static_cast<double>(coordinate[1]) / 32768.0,
        static_cast<double>(coordinate[2]) / 32768.0,
    };
    int best = 3;
    double best_distance = 0.0;
    bool have_best = false;
    for (int index = 0; index < kBedReferencePositionCount; ++index) {
        const std::int32_t* candidate = kReferencePositions[index];
        if (candidate[0] == -1 && candidate[1] == -1 && candidate[2] == -1) {
            continue;
        }
        const double point[3] = {
            static_cast<double>(candidate[0]) / 32768.0,
            static_cast<double>(candidate[1]) / 32768.0,
            static_cast<double>(candidate[2]) / 32768.0,
        };
        const double distance = render_distance(position, point);
        if (!have_best || distance < best_distance) {
            best = index;
            best_distance = distance;
            have_best = true;
        }
    }
    return best;
}

// 目标布局里离 desired 最近、且在 0.05 容差内的输出声道；没有返回 -1。
int nearest_channel(const double desired[3], const std::int32_t positions[][3],
                    std::uint32_t channel_count) {
    int best = -1;
    double best_distance = 0.0;
    bool have_best = false;
    for (std::uint32_t index = 0; index < channel_count; ++index) {
        if (is_sentinel(positions[index])) {
            continue;
        }
        const double point[3] = {
            static_cast<double>(positions[index][0]) / 32768.0,
            static_cast<double>(positions[index][1]) / 32768.0,
            static_cast<double>(positions[index][2]) / 32768.0,
        };
        const double distance = render_distance(desired, point);
        if (!have_best || distance < best_distance) {
            best = static_cast<int>(index);
            best_distance = distance;
            have_best = true;
        }
    }
    if (have_best && best_distance <= 0.05) {
        return best;
    }
    return -1;
}

// 输出声道 index -> 该声道的 Q15 方位；LFE 与非空间声道给 sentinel。
void channel_positions(const speaker::LayoutInfo& layout,
                       std::int32_t positions[][3]) {
    for (int index = 0; index < kMaxChannels; ++index) {
        set_sentinel(positions[index]);
    }
    const auto* geometry = ejoc::speaker_tables::find_layout(layout.speaker_bitfield);
    if (geometry != nullptr) {
        bool known[kMaxChannels] = {};
        std::int32_t found[kMaxChannels][3];
        for (int index = 0; index < kMaxChannels; ++index) {
            set_sentinel(found[index]);
        }
        for (const auto& region : geometry->regions) {
            for (std::size_t point = 0; point < region.point_count; ++point) {
                const auto speaker = static_cast<int>(region.points[point].speaker_id);
                if (speaker < 0 || speaker >= kMaxChannels || known[speaker]) {
                    continue;
                }
                for (int axis = 0; axis < 3; ++axis) {
                    found[speaker][axis] =
                        static_cast<std::int32_t>(region.points[point].coordinate_q15[axis]);
                }
                known[speaker] = true;
            }
        }
        const std::uint32_t count =
            layout.channel_count < static_cast<std::uint32_t>(kMaxChannels)
                ? layout.channel_count
                : static_cast<std::uint32_t>(kMaxChannels);
        for (std::uint32_t index = 0; index < count; ++index) {
            const char* name = layout.channel_names != nullptr ? layout.channel_names[index]
                                                               : nullptr;
            if (name != nullptr && std::strcmp(name, "LFE") == 0) {
                continue;
            }
            if (known[index]) {
                for (int axis = 0; axis < 3; ++axis) {
                    positions[index][axis] = found[index][axis];
                }
            }
        }
    }
}

int channel_index(const speaker::LayoutInfo& layout, const char* name) {
    if (name == nullptr || layout.channel_names == nullptr) {
        return -1;
    }
    for (std::uint32_t index = 0; index < layout.channel_count; ++index) {
        if (std::strcmp(layout.channel_names[index], name) == 0) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

double attenuation_db(const speaker::LayoutInfo& layout) {
    for (const AttenuationEntry& entry : kAttenuation) {
        if (std::strcmp(entry.compact_name, layout.compact_name) == 0) {
            return entry.db;
        }
    }
    return 0.0;
}

}  // namespace

const std::int32_t (*bed_reference_positions_q15())[3] {
    return kReferencePositions;
}

BedOutputDescription bed_output_description(const speaker::LayoutInfo& layout) {
    BedOutputDescription description;
    std::int32_t positions[kMaxChannels][3];
    channel_positions(layout, positions);
    const std::uint32_t output_count =
        layout.channel_count < static_cast<std::uint32_t>(kMaxChannels)
            ? layout.channel_count
            : static_cast<std::uint32_t>(kMaxChannels);
    description.output_labels.reserve(output_count);
    for (std::uint32_t index = 0; index < output_count; ++index) {
        description.output_labels.push_back(nearest_reference(positions[index]));
    }
    for (int label = 0; label < oamd::kBedLabelCount; ++label) {
        const BedChannel channel = static_cast<BedChannel>(label);
        const char* name = channel == BedChannel::kLfe2 ? nullptr
                                                        : bed_channel_short_name(channel);
        int slot = -1;
        if (name != nullptr && name[0] != '\0') {
            slot = channel_index(layout, name);
        }
        if (slot < 0 && channel == BedChannel::kLfe2) {
            slot = channel_index(layout, "LFE");
        }
        const TableB10Position* position = table_b10(channel);
        double desired[3] = {0.0, 0.0, 0.0};
        if (position != nullptr) {
            render_coordinates(position->xyz, desired);
        }
        if (slot < 0 && position != nullptr) {
            slot = nearest_channel(desired, positions, output_count);
        }
        if (slot < 0) {
            description.label_to_slot[label] = is_lfe_label(channel) ? -1 : -2;
        } else {
            description.label_to_slot[label] = slot;
        }
        if (is_lfe_label(channel)) {
            description.direct_output_labels[label] = 3;
        } else {
            int best = 3;
            double best_distance = 0.0;
            bool have_best = false;
            for (int index = 0; index < kBedReferencePositionCount; ++index) {
                const std::int32_t* candidate = kReferencePositions[index];
                if (candidate[0] == -1 && candidate[1] == -1 && candidate[2] == -1) {
                    continue;
                }
                const double point[3] = {
                    static_cast<double>(candidate[0]) / 32768.0,
                    static_cast<double>(candidate[1]) / 32768.0,
                    static_cast<double>(candidate[2]) / 32768.0,
                };
                const double distance = render_distance(desired, point);
                if (!have_best || distance < best_distance) {
                    best = index;
                    best_distance = distance;
                    have_best = true;
                }
            }
            description.direct_output_labels[label] = best;
        }
    }
    description.automatic_attenuation_db = attenuation_db(layout);
    return description;
}

}  // namespace joc::joc
