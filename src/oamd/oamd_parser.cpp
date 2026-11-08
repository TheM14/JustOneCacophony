#include "oamd/oamd_parser.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "foundation/bit_reader.h"

namespace joc::oamd {

namespace {

constexpr int kSampleOffsetIndex[4] = {8, 16, 18, 24};
constexpr int kRampDurations[3] = {0, 512, 1536};
constexpr int kRampDurationIndex[16] = {32,  64,   128,  256,  320,  480,
                                        1000, 1001, 1024, 1600, 1601, 1602,
                                        1920, 2000, 2002, 2048};
constexpr int kIsfObjectCounts[6] = {4, 8, 10, 14, 15, 30};
// ETSI TS 103 420 Table 12 trim levels, 4-bit index, dB; index 15 is the
// minimum (-36 dB), not silence.
constexpr double kTrimLut[16] = {6.0,   3.0,   1.5,   0.75, -0.75, -1.5,
                                 -3.0,  -4.5,  -6.0,  -7.5, -9.0,  -10.5,
                                 -12.0, -13.5, -15.0, -36.0};
// bal3d_y is coded as (code + 1) / 16, so the magnitude never reaches 0.
constexpr double kBalanceStep = 16.0;
// 5.6.1.1.4 的 10 bit 标准 bed 掩码 -> 17 bit 非标准 bed 掩码。
constexpr std::uint32_t kStandardBedToWide[10] = {
    0x3u,   0x4u,   0x8u,    0x30u,   0xC0u,
    0x300u, 0xC00u, 0x3000u, 0xC000u, 0x10000u};
// 5.5.9 位置差分码表。
constexpr int kPositionDelta[8] = {0, 1, 2, 3, -4, -3, -2, -1};
// 位置细化码表。
constexpr int kRefineDelta[4] = {1, 2, -1, -2};
// 5.5.3/5.5.5 divergence 码表（公开实现固定值），Q15 量纲。
constexpr int kDivergenceQuickIndex[4] = {26, 29, 32, 63};
constexpr int kDivergenceTable[64] = {
    0,     0,     66,    117,   209,   331,   467,   659,   829,   1042,  1309,
    1643,  2059,  2302,  2573,  2872,  3202,  3564,  3960,  4391,  4856,  5354,
    5882,  6437,  7014,  7605,  8204,  8803,  9394,  9970,  10524, 11051, 11548,
    12011, 12441, 12835, 13197, 13525, 13823, 14092, 14335, 14553, 14748, 14924,
    15081, 15221, 15347, 15459, 15559, 15649, 15728, 15799, 15863, 15970, 16055,
    16123, 16176, 16219, 16267, 16301, 16332, 16355, 16368, 16384,
};

// 5.6.1.1.4 的 10 bit 掩码位展开（LSB 在前，逐位给出 1..2 个声道）。
struct StandardBedBit {
    std::uint8_t count;
    ::joc::joc::BedChannel channels[2];
};

constexpr StandardBedBit kStandardBedBits[10] = {
    {2, {::joc::joc::BedChannel::kLeft, ::joc::joc::BedChannel::kRight}},
    {1, {::joc::joc::BedChannel::kCenter, ::joc::joc::BedChannel::kLeft}},
    {1, {::joc::joc::BedChannel::kLfe, ::joc::joc::BedChannel::kLeft}},
    {2,
     {::joc::joc::BedChannel::kLeftSurround,
      ::joc::joc::BedChannel::kRightSurround}},
    {2,
     {::joc::joc::BedChannel::kLeftRearSurround,
      ::joc::joc::BedChannel::kRightRearSurround}},
    {2,
     {::joc::joc::BedChannel::kTopFrontLeft,
      ::joc::joc::BedChannel::kTopFrontRight}},
    {2,
     {::joc::joc::BedChannel::kTopSideLeft,
      ::joc::joc::BedChannel::kTopSideRight}},
    {2,
     {::joc::joc::BedChannel::kTopBackLeft,
      ::joc::joc::BedChannel::kTopBackRight}},
    {2, {::joc::joc::BedChannel::kLeftWide, ::joc::joc::BedChannel::kRightWide}},
    {1, {::joc::joc::BedChannel::kLfe2, ::joc::joc::BedChannel::kLeft}},
};

// 5.6.1.1.5 的 17 bit 非标准掩码位（LSB 在前）。
constexpr ::joc::joc::BedChannel kNonStandardBedBits[kBedLabelCount] = {
    ::joc::joc::BedChannel::kLeft,
    ::joc::joc::BedChannel::kRight,
    ::joc::joc::BedChannel::kCenter,
    ::joc::joc::BedChannel::kLfe,
    ::joc::joc::BedChannel::kLeftSurround,
    ::joc::joc::BedChannel::kRightSurround,
    ::joc::joc::BedChannel::kLeftRearSurround,
    ::joc::joc::BedChannel::kRightRearSurround,
    ::joc::joc::BedChannel::kTopFrontLeft,
    ::joc::joc::BedChannel::kTopFrontRight,
    ::joc::joc::BedChannel::kTopSideLeft,
    ::joc::joc::BedChannel::kTopSideRight,
    ::joc::joc::BedChannel::kTopBackLeft,
    ::joc::joc::BedChannel::kTopBackRight,
    ::joc::joc::BedChannel::kLeftWide,
    ::joc::joc::BedChannel::kRightWide,
    ::joc::joc::BedChannel::kLfe2,
};

Status variant(const std::string& name, const std::string& message) {
    return Status::fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT, stage::kOamd,
                        name + ": " + message);
}

Status syntax(const std::string& message) {
    return Status::fail(JOC_ERR_OAMD_SYNTAX, stage::kOamd, message);
}

bool variable_bits_max(bits::BitReader& reader, unsigned width, unsigned max_groups,
                       std::uint32_t* out_value) {
    std::uint32_t value = reader.read(width);
    if (reader.failed()) {
        return false;
    }
    std::uint32_t more = reader.read(1);
    if (reader.failed()) {
        return false;
    }
    unsigned num_group = 1;
    if (max_groups > num_group) {
        if (more != 0u) {
            value = (value + 1u) << width;
        }
        while (more != 0u) {
            value += reader.read(width);
            more = reader.read(1);
            if (reader.failed()) {
                return false;
            }
            if (num_group >= max_groups) {
                break;
            }
            if (more != 0u) {
                value = (value + 1u) << width;
                num_group += 1;
            }
        }
    }
    *out_value = value;
    return true;
}

// q15_ratio(n, d) = min(32767, floor(32768 n / d + 0.5))，参考实现的取整规则。
std::int32_t q15_ratio(int numerator, int denominator) {
    const double value = std::floor(
        32768.0 * static_cast<double>(numerator) / static_cast<double>(denominator) + 0.5);
    const std::int32_t quantised = static_cast<std::int32_t>(value);
    return quantised > 32767 ? 32767 : quantised;
}

// gain_idx/gain_bits -> 整数增益 Qg（单位增益 4096）。
std::int32_t gain_value(std::uint32_t gain_idx, std::uint32_t gain_bits) {
    switch (gain_idx) {
        case 0u:
            return 4096;
        case 1u:
            return 0;
        case 2u: {
            const int bits = static_cast<int>(gain_bits);
            const int db = bits < 15 ? 15 - bits : 14 - bits;
            return static_cast<std::int32_t>(
                std::floor(4096.0 * std::pow(10.0, static_cast<double>(db) / 20.0)));
        }
        default:
            return 4096;
    }
}

// size_idx 与 5 bit 数值 -> (sx, sy, sz) Q15。
void size_value(std::uint32_t mode, const std::uint32_t* values, std::uint16_t out[3]) {
    if (mode == 1u) {
        const auto value = static_cast<std::uint16_t>(q15_ratio(static_cast<int>(values[0]), 31));
        out[0] = value;
        out[1] = value;
        out[2] = value;
        return;
    }
    if (mode == 2u) {
        for (int axis = 0; axis < 3; ++axis) {
            out[axis] =
                static_cast<std::uint16_t>(q15_ratio(static_cast<int>(values[axis]), 31));
        }
        return;
    }
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
}

// coarse (x, y, z) -> Q15 位置，z 带符号。
void coarse_to_q15(const int coarse[3], std::int32_t* qx, std::int32_t* qy,
                   std::int32_t* qz) {
    *qx = q15_ratio(coarse[0], kNQ12);
    *qy = q15_ratio(coarse[1], kNQ12);
    const std::int32_t magnitude = q15_ratio(std::abs(coarse[2]), kNQ3);
    *qz = coarse[2] < 0 ? -magnitude : magnitude;
}

void clamp_coarse(int coarse[3]) {
    coarse[0] = coarse[0] < 0 ? 0 : (coarse[0] > kNQ12 ? kNQ12 : coarse[0]);
    coarse[1] = coarse[1] < 0 ? 0 : (coarse[1] > kNQ12 ? kNQ12 : coarse[1]);
    coarse[2] = coarse[2] < -kNQ3 ? -kNQ3 : (coarse[2] > kNQ3 ? kNQ3 : coarse[2]);
}

// bed/ISF 与未激活对象的默认位置：中心 16384/16384/0。
ObjectRenderState render_defaults() {
    ObjectRenderState state;
    state.qx = 16384;
    state.qy = 16384;
    state.qz = 0;
    state.zone_idx = 0;
    state.enable_elevation = true;
    state.size_mode = 0;
    state.size_q[0] = 0;
    state.size_q[1] = 0;
    state.size_q[2] = 0;
    state.snap = false;
    state.divergence_q = 0;
    return state;
}

// render_status 2/3 沿用上一块的可渲染字段（增益与优先级不在这里）。
void copy_render_fields(const ObjectRenderState& source, ObjectRenderState* target) {
    target->qx = source.qx;
    target->qy = source.qy;
    target->qz = source.qz;
    target->zone_idx = source.zone_idx;
    target->enable_elevation = source.enable_elevation;
    target->size_mode = source.size_mode;
    for (int axis = 0; axis < 3; ++axis) {
        target->size_q[axis] = source.size_q[axis];
    }
    target->snap = source.snap;
    target->divergence_q = source.divergence_q;
}

std::vector<::joc::joc::BedChannel> standard_bed_channels(std::uint32_t bed_mask) {
    std::vector<::joc::joc::BedChannel> channels;
    for (int bit = 0; bit < 10; ++bit) {
        if (((bed_mask >> bit) & 1u) == 0u) {
            continue;
        }
        for (std::uint8_t index = 0; index < kStandardBedBits[bit].count; ++index) {
            channels.push_back(kStandardBedBits[bit].channels[index]);
        }
    }
    return channels;
}

std::vector<::joc::joc::BedChannel> nonstandard_bed_channels(std::uint32_t bed_mask) {
    std::vector<::joc::joc::BedChannel> channels;
    for (int bit = 0; bit < kBedLabelCount; ++bit) {
        if (((bed_mask >> bit) & 1u) != 0u) {
            channels.push_back(kNonStandardBedBits[bit]);
        }
    }
    return channels;
}

struct RawProgram {
    bool dynamic_object_only = false;
    bool lfe_present = false;
    bool content_reserved = false;
    bool content_bed = false;
    bool content_isf = false;
    bool content_dynamic = false;
    bool b_bed_chan_distribute = false;
    std::uint32_t num_bed_objects = 0;
    std::uint32_t num_isf_objects = 0;
    std::int32_t num_dynamic_objects = -1;
    std::uint32_t bed_mask = 0;
    std::int32_t isf_idx = -1;
    std::vector<BedAssignment> bed_assignments;
};

// 5.5.3 program_assignment()：bed / ISF / dynamic 对象数量与 bed 声明。
Status parse_program_assignment(bits::BitReader& reader, RawProgram* program) {
    program->dynamic_object_only = reader.read(1) != 0u;
    if (reader.failed()) {
        return syntax("program_assignment truncated");
    }
    if (program->dynamic_object_only) {
        program->lfe_present = reader.read(1) != 0u;
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        program->num_bed_objects = program->lfe_present ? 1u : 0u;
        if (program->lfe_present) {
            BedAssignment assignment;
            assignment.lfe_only = true;
            assignment.channels.push_back(::joc::joc::BedChannel::kLfe);
            program->bed_assignments.push_back(assignment);
            program->bed_mask = 0x8u;
        }
        return Status::success();
    }

    const std::uint32_t mask = reader.read(4);
    if (reader.failed()) {
        return syntax("program_assignment truncated");
    }
    program->content_reserved = (mask & 0x8u) != 0u;
    program->content_dynamic = (mask & 0x4u) != 0u;
    program->content_isf = (mask & 0x2u) != 0u;
    program->content_bed = (mask & 0x1u) != 0u;
    if (program->content_bed) {
        program->b_bed_chan_distribute = reader.read(1) != 0u;
        const bool multi = reader.read(1) != 0u;
        std::uint32_t instances = 1;
        if (multi) {
            instances = reader.read(3) + 2u;
        }
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        for (std::uint32_t instance = 0; instance < instances; ++instance) {
            BedAssignment assignment;
            (void)instance;
            if (reader.read(1) != 0u) {  // b_lfe_only
                assignment.lfe_only = true;
                assignment.channels.push_back(::joc::joc::BedChannel::kLfe);
                if (reader.failed()) {
                    return syntax("program_assignment truncated");
                }
                program->bed_assignments.push_back(assignment);
                program->num_bed_objects += 1;
                continue;
            }
            if (reader.read(1) != 0u) {  // b_standard_chan_assign
                assignment.standard = true;
                assignment.mask = reader.read(10);
                if (reader.failed()) {
                    return syntax("program_assignment truncated");
                }
                assignment.channels = standard_bed_channels(assignment.mask);
                program->bed_mask |= standard_bed_mask(assignment.mask);
            } else {
                assignment.standard = false;
                assignment.mask = reader.read(17);
                if (reader.failed()) {
                    return syntax("program_assignment truncated");
                }
                assignment.channels = nonstandard_bed_channels(assignment.mask);
                program->bed_mask |= assignment.mask;
            }
            program->num_bed_objects +=
                static_cast<std::uint32_t>(assignment.channels.size());
            program->bed_assignments.push_back(assignment);
        }
    }
    if (program->content_isf) {
        const std::uint32_t isf_idx = reader.read(3);
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        program->isf_idx = static_cast<std::int32_t>(isf_idx);
        program->num_isf_objects =
            (isf_idx < 6u) ? static_cast<std::uint32_t>(kIsfObjectCounts[isf_idx]) : 0u;
    }
    if (program->content_dynamic) {
        std::uint32_t count = reader.read(5);
        if (count == 0x1Fu) {
            count += reader.read(7);
        }
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        program->num_dynamic_objects = static_cast<std::int32_t>(count + 1u);
    }
    if (program->content_reserved) {
        const std::uint32_t reserved_bytes = reader.read(4) + 1u;
        if (!reader.skip(static_cast<std::size_t>(reserved_bytes) * 8u)) {
            return syntax("program_assignment reserved data truncated");
        }
    }
    return Status::success();
}

// 5.5.9 object_info_block：一个对象一个块的完整状态。
struct RenderWork {
    ObjectRenderState render;
    int coarse[3] = {31, 31, 0};
};

Status parse_object_info_block(bits::BitReader& reader, int object_index, int block_index,
                               ObjectKind kind, const RenderWork* previous,
                               const std::vector<ObjectElementState>* earlier_objects,
                               RenderWork* out) {
    *out = RenderWork{};
    const bool not_active = reader.read(1) != 0u;
    if (reader.failed()) {
        return syntax("object_info_block truncated");
    }
    out->render.not_active = not_active;
    std::int32_t gain_q = 0;
    std::int32_t priority_q = 0;
    ObjectRenderState render;
    std::uint32_t render_status = 0;

    if (not_active) {
        render = render_defaults();
        render.not_active = true;
        coarse_to_q15(out->coarse, &render.qx, &render.qy, &render.qz);
    } else {
        const std::uint32_t basic_status = block_index == 0 ? 1u : reader.read(2);
        if (reader.failed()) {
            return syntax("object_info_block truncated");
        }
        // gain_idx=3 沿用同块前一对象的增益；对象 0 用单位增益，前一对象未激活
        // 时为 0。
        const auto read_gain = [&](std::int32_t* value) -> bool {
            const std::uint32_t gain_idx = reader.read(2);
            std::uint32_t gain_bits = 0;
            if (gain_idx == 2u) {
                gain_bits = reader.read(6);
            }
            if (reader.failed()) {
                return false;
            }
            if (gain_idx == 3u) {
                if (object_index == 0) {
                    *value = 4096;
                } else {
                    const ObjectRenderState& earlier =
                        (*earlier_objects)[static_cast<std::size_t>(object_index) - 1]
                            .blocks[static_cast<std::size_t>(block_index)]
                            .render;
                    *value = earlier.not_active ? 0 : earlier.gain_q;
                }
            } else {
                *value = gain_value(gain_idx, gain_bits);
            }
            return true;
        };
        const auto read_priority = [&](std::int32_t* value) -> bool {
            const bool default_priority = reader.read(1) != 0u;
            if (reader.failed()) {
                return false;
            }
            if (default_priority) {
                *value = 32767;
            } else {
                const std::uint32_t bits = reader.read(5);
                if (reader.failed()) {
                    return false;
                }
                *value = static_cast<std::int32_t>(1024u * bits);
            }
            return true;
        };

        switch (basic_status) {
            case 0u:
                gain_q = 0;
                priority_q = 0;
                break;
            case 1u:
                if (!read_gain(&gain_q) || !read_priority(&priority_q)) {
                    return syntax("object_info_block truncated");
                }
                break;
            case 2u:
                if (previous == nullptr) {
                    return syntax("object_info_block status 2 without a previous block");
                }
                gain_q = previous->render.gain_q;
                priority_q = previous->render.priority_q;
                break;
            default: {
                const std::uint32_t flags = reader.read(2);
                if (reader.failed()) {
                    return syntax("object_info_block truncated");
                }
                if (previous == nullptr) {
                    return syntax("object_info_block mask without a previous block");
                }
                gain_q = previous->render.gain_q;
                priority_q = previous->render.priority_q;
                if ((flags & 1u) != 0u) {
                    if (!read_gain(&gain_q)) {
                        return syntax("object_info_block truncated");
                    }
                }
                if ((flags & 2u) != 0u) {
                    if (!read_priority(&priority_q)) {
                        return syntax("object_info_block truncated");
                    }
                }
                break;
            }
        }

        render_status = (kind == ObjectKind::kBed || kind == ObjectKind::kIsf)
                            ? 0u
                            : (block_index == 0 ? 1u : reader.read(2));
        if (reader.failed()) {
            return syntax("object_info_block truncated");
        }
        if (render_status == 1u || render_status == 3u) {
            std::uint32_t flags = 0xFu;
            if (render_status == 3u) {
                if (previous == nullptr) {
                    return syntax("object_info_block differential without a previous block");
                }
                flags = reader.read(4);
                if (reader.failed()) {
                    return syntax("object_info_block truncated");
                }
                render = render_defaults();
                copy_render_fields(previous->render, &render);
                out->coarse[0] = previous->coarse[0];
                out->coarse[1] = previous->coarse[1];
                out->coarse[2] = previous->coarse[2];
            } else {
                render = render_defaults();
            }
            const bool position_present = (flags & 0x1u) != 0u;
            const bool zone_present = (flags & 0x2u) != 0u;
            const bool size_present = (flags & 0x4u) != 0u;
            const bool screen_present = (flags & 0x8u) != 0u;
            if (position_present) {
                bool differential = false;
                if (block_index > 0) {
                    differential = reader.read(1) != 0u;
                }
                if (reader.failed()) {
                    return syntax("object_info_block truncated");
                }
                if (differential) {
                    for (int axis = 0; axis < 3; ++axis) {
                        out->coarse[axis] += kPositionDelta[reader.read(3) & 0x7u];
                        if (reader.failed()) {
                            return syntax("object_info_block truncated");
                        }
                    }
                } else {
                    const int x = static_cast<int>(reader.read(6));
                    const int y = static_cast<int>(reader.read(6));
                    const std::uint32_t z_sign = reader.read(1);
                    const int z = static_cast<int>(reader.read(4));
                    if (reader.failed()) {
                        return syntax("object_info_block truncated");
                    }
                    out->coarse[0] = x;
                    out->coarse[1] = y;
                    out->coarse[2] = z_sign != 0u ? z : -z;
                }
                clamp_coarse(out->coarse);
                const bool outside = reader.read(1) != 0u;
                if (outside) {
                    const bool infinite = reader.read(1) != 0u;
                    if (!infinite) {
                        reader.read(4);
                    }
                }
                if (reader.failed()) {
                    return syntax("object_info_block truncated");
                }
            }
            if (zone_present) {
                render.zone_idx = static_cast<std::uint8_t>(reader.read(3));
                render.enable_elevation = reader.read(1) != 0u;
            }
            if (size_present) {
                const std::uint32_t size_mode = reader.read(2);
                render.size_mode = static_cast<std::uint8_t>(size_mode);
                std::uint32_t values[3] = {0, 0, 0};
                if (size_mode == 1u) {
                    values[0] = reader.read(5);
                } else if (size_mode == 2u) {
                    values[0] = reader.read(5);
                    values[1] = reader.read(5);
                    values[2] = reader.read(5);
                }
                size_value(size_mode, values, render.size_q);
            }
            if (screen_present) {
                const bool screen_ref = reader.read(1) != 0u;
                if (screen_ref) {
                    reader.read(3);
                    reader.read(2);
                }
            }
            render.snap = reader.read(1) != 0u;
            if (reader.failed()) {
                return syntax("object_info_block truncated");
            }
            if (position_present) {
                coarse_to_q15(out->coarse, &render.qx, &render.qy, &render.qz);
            }
        } else if (render_status == 2u) {
            if (previous == nullptr) {
                return syntax("object_info_block hold without a previous block");
            }
            render = render_defaults();
            copy_render_fields(previous->render, &render);
            out->coarse[0] = previous->coarse[0];
            out->coarse[1] = previous->coarse[1];
            out->coarse[2] = previous->coarse[2];
        } else {
            render = render_defaults();
            coarse_to_q15(out->coarse, &render.qx, &render.qy, &render.qz);
        }
    }

    if (reader.read(1) != 0u) {  // b_additional_table_data_exists
        const std::uint32_t additional_bytes = reader.read(4) + 1u;
        if (!reader.skip(static_cast<std::size_t>(additional_bytes) * 8u)) {
            return syntax("object_info_block additional table truncated");
        }
    }
    if (reader.failed()) {
        return syntax("object_info_block truncated");
    }
    render.not_active = not_active;
    render.gain_q = gain_q;
    render.priority_q = priority_q;
    out->render = render;
    return Status::success();
}

// 5.5.5 object_element()：时间头 + 对象外/块内顺序的完整块状态。
Status parse_object_element(bits::BitReader& reader, std::uint32_t object_count,
                            const ObjectKind* kinds, FrameState* frame,
                            std::uint32_t* sample_offset_code_out,
                            bool* reserved_data_not_present_out) {
    const std::uint32_t sample_offset_code = reader.read(2);
    if (reader.failed()) {
        return syntax("object_element truncated");
    }
    *sample_offset_code_out = sample_offset_code;
    std::uint32_t sample_offset = 0;
    switch (sample_offset_code) {
        case 0u:
            sample_offset = 0;
            break;
        case 1u:
            sample_offset = static_cast<std::uint32_t>(
                kSampleOffsetIndex[reader.read(2) & 0x3u]);
            break;
        case 2u:
            sample_offset = reader.read(5);
            break;
        default:
            return variant("md_sample_offset_mode",
                           "MD sample-offset mode " + std::to_string(sample_offset_code) +
                               " is not covered");
    }
    if (reader.failed()) {
        return syntax("object_element truncated");
    }
    const std::uint32_t block_count = reader.read(3) + 1u;
    if (reader.failed()) {
        return syntax("object_element truncated");
    }
    frame->block_count = block_count;
    frame->sample_offset = sample_offset;
    frame->blocks.clear();
    frame->blocks.reserve(block_count);
    for (std::uint32_t block = 0; block < block_count; ++block) {
        const std::uint32_t block_offset_factor = reader.read(6);
        const std::uint32_t ramp_code = reader.read(2);
        std::uint32_t ramp_duration = 0;
        if (ramp_code == 3u) {
            if (reader.read(1) != 0u) {
                ramp_duration = static_cast<std::uint32_t>(
                    kRampDurationIndex[reader.read(4) & 0xFu]);
            } else {
                ramp_duration = reader.read(11);
            }
        } else {
            ramp_duration = static_cast<std::uint32_t>(kRampDurations[ramp_code]);
        }
        if (reader.failed()) {
            return syntax("object_element truncated");
        }
        BlockTiming timing;
        timing.block_offset_samples = sample_offset + block_offset_factor * 32u;
        timing.ramp_duration_samples = ramp_duration;
        frame->blocks.push_back(timing);
    }
    const bool reserved_data_not_present = reader.read(1) != 0u;
    if (!reserved_data_not_present) {
        reader.read(5);
    }
    if (reader.failed()) {
        return syntax("object_element truncated");
    }
    *reserved_data_not_present_out = reserved_data_not_present;

    frame->objects.clear();
    frame->objects.reserve(object_count);
    std::vector<std::vector<RenderWork>> work(
        object_count, std::vector<RenderWork>(block_count));
    for (std::uint32_t object_index = 0; object_index < object_count; ++object_index) {
        ObjectElementState element;
        element.kind = kinds[object_index];
        element.blocks.resize(block_count);
        for (std::uint32_t block_index = 0; block_index < block_count; ++block_index) {
            RenderWork& current =
                work[object_index][static_cast<std::size_t>(block_index)];
            const RenderWork* previous =
                block_index > 0
                    ? &work[object_index][static_cast<std::size_t>(block_index) - 1]
                    : nullptr;
            const Status status = parse_object_info_block(
                reader, static_cast<int>(object_index), static_cast<int>(block_index),
                element.kind, previous, &frame->objects, &current);
            if (!status.ok()) {
                return status;
            }
            element.blocks[block_index].render = current.render;
        }
        frame->objects.push_back(std::move(element));
    }
    return Status::success();
}

// Q15 位置索引细化：refined index = 5 * coarse + 2 + delta。
std::int32_t refine_xy(int index) {
    const double value =
        std::floor((static_cast<double>(index) - 2.0) * 32768.0 / 310.0 + 0.5);
    std::int32_t quantised = static_cast<std::int32_t>(value);
    if (quantised < 0) {
        quantised = 0;
    }
    if (quantised > 32767) {
        quantised = 32767;
    }
    return quantised;
}

std::int32_t refine_z(int index) {
    const double value =
        std::floor((static_cast<double>(index) - 2.0) * 32768.0 / 75.0 + 0.5);
    std::int32_t quantised = static_cast<std::int32_t>(value);
    if (quantised < -32767) {
        quantised = -32767;
    }
    if (quantised > 32767) {
        quantised = 32767;
    }
    return quantised;
}

int sign_of(int value) {
    return value < 0 ? -1 : 1;
}

int coarse_xy_from_q15(int value) {
    const int offset = value - 16384;
    return 31 + sign_of(offset) * ((62 * std::abs(offset) + 16384) >> 15);
}

int coarse_z_from_q15(int value) {
    value = value > -32767 ? value : -32767;
    return sign_of(value) * ((15 * std::abs(value) + 16384) >> 15);
}

// 扩展对象元素：divergence 与高精度位置补充。
Status parse_extended_element(bits::BitReader& reader,
                              std::vector<ObjectElementState>* objects,
                              std::uint32_t block_count) {
    const bool divergence_present = reader.read(1) != 0u;
    if (reader.failed()) {
        return syntax("extended object element truncated");
    }
    if (divergence_present) {
        for (std::size_t object_index = 0; object_index < objects->size(); ++object_index) {
            ObjectElementState& element = (*objects)[object_index];
            for (std::uint32_t block_index = 0; block_index < block_count; ++block_index) {
                ObjectRenderState& state = element.blocks[block_index].render;
                if (state.not_active) {
                    state.divergence_q = 0;
                    continue;
                }
                if (element.kind != ObjectKind::kDynamic) {
                    continue;
                }
                const bool present = reader.read(1) != 0u;
                if (reader.failed()) {
                    return syntax("extended object element truncated");
                }
                if (!present) {
                    state.divergence_q = 0;
                    continue;
                }
                const std::uint32_t mode = reader.read(2);
                if (reader.failed()) {
                    return syntax("extended object element truncated");
                }
                switch (mode) {
                    case 0u:
                        state.divergence_q = static_cast<std::uint16_t>(
                            kDivergenceTable[kDivergenceQuickIndex[reader.read(2) & 0x3u]]);
                        break;
                    case 1u:
                        state.divergence_q =
                            block_index == 0
                                ? 0
                                : element.blocks[block_index - 1].render.divergence_q;
                        break;
                    case 2u:
                        state.divergence_q = static_cast<std::uint16_t>(
                            kDivergenceTable[reader.read(6) & 0x3Fu]);
                        break;
                    default:
                        reader.read(6);
                        state.divergence_q = 0;
                        break;
                }
                if (reader.failed()) {
                    return syntax("extended object element truncated");
                }
            }
        }
    }
    const bool refinement_present = reader.read(1) != 0u;
    if (reader.failed()) {
        return syntax("extended object element truncated");
    }
    if (refinement_present) {
        for (std::size_t object_index = 0; object_index < objects->size(); ++object_index) {
            ObjectElementState& element = (*objects)[object_index];
            for (std::uint32_t block_index = 0; block_index < block_count; ++block_index) {
                ObjectRenderState& state = element.blocks[block_index].render;
                if (state.not_active || element.kind != ObjectKind::kDynamic) {
                    continue;
                }
                const bool present = reader.read(1) != 0u;
                if (reader.failed()) {
                    return syntax("extended object element truncated");
                }
                if (!present) {
                    continue;
                }
                const std::uint32_t mask = reader.read(3);
                if (reader.failed()) {
                    return syntax("extended object element truncated");
                }
                if (mask == 0u) {
                    continue;
                }
                int deltas[3] = {0, 0, 0};
                for (int axis = 0; axis < 3; ++axis) {
                    if ((mask & (1u << axis)) != 0u) {
                        deltas[axis] = kRefineDelta[reader.read(2) & 0x3u];
                        if (reader.failed()) {
                            return syntax("extended object element truncated");
                        }
                    }
                }
                const int ix = coarse_xy_from_q15(state.qx);
                const int iy = coarse_xy_from_q15(state.qy);
                const int iz = coarse_z_from_q15(state.qz);
                const int z_sign = sign_of(iz);
                state.qx = refine_xy(5 * ix + 2 + deltas[0]);
                state.qy = refine_xy(5 * iy + 2 + deltas[1]);
                state.qz = z_sign * refine_z(z_sign * (5 * iz + deltas[2]) + 2);
            }
        }
    }
    return Status::success();
}

// trim_element(): one programme-wide record per payload.  The first bit of each
// configuration is b_default_trim; b_disable_trim and the value record follow
// only when that bit is clear.
Status parse_trim_element(bits::BitReader& reader, std::uint32_t object_count, OamdTrim* out) {
    OamdTrim trim;
    trim.warp_mode = reader.read(2);
    reader.read(2);  // reserved
    trim.global_trim_mode = reader.read(2);
    if (reader.failed()) {
        return syntax("OAMD trim element header truncated");
    }
    if (trim.global_trim_mode == 2u) {
        for (int config = 0; config < kTrimConfigs; ++config) {
            TrimConfig& record = trim.configs[config];
            if (reader.read(1) != 0u) {
                record.kind = TrimConfigKind::kDefault;
                continue;
            }
            if (reader.read(1) != 0u) {
                record.kind = TrimConfigKind::kDisabled;
                continue;
            }
            record.kind = TrimConfigKind::kExplicit;
            const std::uint32_t presence = reader.read(5);
            record.presence = static_cast<std::uint8_t>(presence);
            if ((presence & 0x01u) != 0u) {
                record.centre = kTrimLut[reader.read(4) & 0x0Fu];
            }
            if ((presence & 0x02u) != 0u) {
                record.surround = kTrimLut[reader.read(4) & 0x0Fu];
            }
            if ((presence & 0x04u) != 0u) {
                record.height = kTrimLut[reader.read(4) & 0x0Fu];
            }
            if ((presence & 0x08u) != 0u) {
                const double sign = reader.read(1) != 0u ? 1.0 : -1.0;
                record.balance_overhead =
                    (static_cast<double>(reader.read(4)) + 1.0) / kBalanceStep * sign;
            }
            if ((presence & 0x10u) != 0u) {
                const double sign = reader.read(1) != 0u ? 1.0 : -1.0;
                record.balance_listener =
                    (static_cast<double>(reader.read(4)) + 1.0) / kBalanceStep * sign;
            }
            if (reader.failed()) {
                return syntax("OAMD trim configuration " + std::to_string(config) +
                              " truncated");
            }
        }
    }
    if (reader.read(1) != 0u) {
        for (std::uint32_t index = 0; index < object_count; ++index) {
            const bool bypass = reader.read(1) != 0u;
            if (index < static_cast<std::uint32_t>(kMaxSlots)) {
                trim.disable_per_object[index] = bypass;
            }
        }
    }
    if (reader.failed()) {
        return syntax("OAMD trim bypass table truncated");
    }
    trim.object_count = object_count;
    *out = trim;
    return Status::success();
}

// 一个 element 的目录信息，供 frame_state 的 size_mismatch 记录。
struct ElementRecord {
    std::uint32_t ordinal = 0;
    std::uint32_t element_id = 0;
    std::uint32_t size_bytes = 0;
    std::size_t body_end_bit = 0;
    std::size_t parsed_end_bit = 0;
    bool has_trim = false;
    OamdTrim trim;
};

std::vector<ObjectKind> object_kinds(const RawProgram& program,
                                     std::uint32_t object_count) {
    std::vector<ObjectKind> kinds(object_count, ObjectKind::kDynamic);
    const std::uint32_t bed_end = program.num_bed_objects;
    const std::uint32_t isf_end = bed_end + program.num_isf_objects;
    for (std::uint32_t index = 0; index < object_count; ++index) {
        if (index < bed_end) {
            kinds[index] = ObjectKind::kBed;
        } else if (index < isf_end) {
            kinds[index] = ObjectKind::kIsf;
        }
    }
    return kinds;
}

// 尾部 padding 必须全零。
Status check_padding(const bits::BitReader& reader) {
    const std::size_t start = reader.position();
    bits::BitReader tail;
    tail.reset(reader.data(), (reader.limit() + 7u) / 8u, start);
    while (tail.remaining_bits() > 0) {
        const std::size_t chunk = tail.remaining_bits() > 32u ? 32u : tail.remaining_bits();
        if (tail.read(static_cast<unsigned>(chunk)) != 0u || tail.failed()) {
            return Status::fail(JOC_ERR_BITSTREAM_PADDING, stage::kOamd,
                                "OAMD payload padding is not zero (from bit " +
                                    std::to_string(start) + ")");
        }
    }
    return Status::success();
}

void fill_program_state(const RawProgram& program, ProgramState* out) {
    out->dynamic_object_only = program.dynamic_object_only;
    out->lfe_present = program.lfe_present;
    out->b_bed_chan_distribute = program.b_bed_chan_distribute;
    out->content_reserved = program.content_reserved;
    out->content_bed = program.content_bed;
    out->content_isf = program.content_isf;
    out->content_dynamic = program.content_dynamic;
    out->num_bed_objects = program.num_bed_objects;
    out->num_isf_objects = program.num_isf_objects;
    out->num_dynamic_objects = program.num_dynamic_objects;
    out->bed_mask = program.bed_mask;
    out->isf_idx = program.isf_idx;
    out->bed_labels.clear();
    for (const BedAssignment& assignment : program.bed_assignments) {
        for (const ::joc::joc::BedChannel channel : assignment.channels) {
            out->bed_labels.push_back(channel);
        }
    }
}

// 参考 frame_update() 的 16 槽位置视图：bed/ISF 与未激活对象保持上一帧位置。
void fill_legacy_slots(const FrameState& state, OamdUpdate* out) {
    const std::uint32_t count =
        state.object_count < static_cast<std::uint32_t>(kMaxSlots)
            ? state.object_count
            : static_cast<std::uint32_t>(kMaxSlots);
    for (std::uint32_t index = 0; index < count; ++index) {
        const ObjectElementState& element = state.objects[index];
        if (element.blocks.empty()) {
            continue;
        }
        const ObjectRenderState& render = element.blocks[0].render;
        if (render.not_active) {
            continue;
        }
        if (element.kind != ObjectKind::kDynamic) {
            continue;
        }
        SlotUpdate& slot = out->slots[index];
        slot.q1 = static_cast<std::int16_t>(render.qx);
        slot.q2 = static_cast<std::int16_t>(render.qy);
        const std::int32_t elevation = render.enable_elevation ? render.qz : 0;
        slot.q3 = static_cast<std::int16_t>(elevation > 0 ? elevation : 0);
    }
}

// 完整解析：header + program + element 目录 + frame state。
Status parse_full(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out,
                  ParseTrace* trace, bool* have_object_element_out) {
    *out = OamdUpdate{};
    bits::BitReader reader(payload, payload_size);
    std::uint32_t version = reader.read(2);
    if (reader.failed()) {
        return variant("header_truncated", "OAMD payload cannot hold the 2-bit version");
    }
    if (version == 3u) {
        version += reader.read(3);
        if (reader.failed()) {
            return syntax("OAMD version extension truncated");
        }
    }
    std::uint32_t object_count_bits = reader.read(5);
    if (object_count_bits == 0x1Fu) {
        object_count_bits += reader.read(7);
    }
    if (reader.failed()) {
        return syntax("OAMD object count truncated");
    }
    const std::uint32_t object_count = object_count_bits + 1u;
    RawProgram program;
    Status status = parse_program_assignment(reader, &program);
    if (!status.ok()) {
        return status;
    }
    const bool alternate_present = reader.read(1) != 0u;
    if (reader.failed()) {
        return syntax("OAMD alternate data flag truncated");
    }

    FrameState state;
    state.object_count = object_count;
    std::vector<ElementRecord> records;
    records.reserve(16);
    bool have_object_element = false;
    std::uint32_t element_count = 0;
    std::uint32_t sample_offset_code = 0;
    bool reserved_data_not_present = false;
    std::size_t object_parsed_end = 0;
    (void)reserved_data_not_present;
    const std::vector<ObjectKind> kinds = object_kinds(program, object_count);

    const std::uint32_t declared_elements_base = reader.read(4);
    std::uint32_t declared_elements = declared_elements_base;
    if (declared_elements_base == 0xFu) {
        declared_elements += reader.read(5);
    }
    if (reader.failed()) {
        return syntax("OAMD element count truncated");
    }
    if (declared_elements == 0u) {
        return variant("missing_object_element", "OAMD declares no element");
    }
    element_count = declared_elements;

    bool seen_object = false;
    bool seen_trim = false;
    bool seen_extended = false;
    for (std::uint32_t ordinal = 0; ordinal < element_count; ++ordinal) {
        const std::size_t header_start = reader.position();
        const std::uint32_t element_id = reader.read(4);
        std::uint32_t size_field = 0;
        if (!variable_bits_max(reader, 4, 4, &size_field)) {
            return syntax("OAMD element header truncated at ordinal " +
                          std::to_string(ordinal) + " (header at bit " +
                          std::to_string(header_start) + ")");
        }
        const std::uint32_t size_bytes = size_field + 1u;
        const std::size_t region_start = reader.position();
        const std::size_t region_end =
            region_start + static_cast<std::size_t>(size_bytes) * 8u;
        if (region_end > reader.limit()) {
            return variant("element_bounds",
                           "element " + std::to_string(ordinal) + " (id " +
                               std::to_string(element_id) + ") declares " +
                               std::to_string(size_bytes) + " bytes, beyond the payload");
        }
        const std::uint32_t alternate_data_id = alternate_present ? reader.read(4) : 0u;
        reader.read(1);  // discard_unknown
        if (reader.failed()) {
            return syntax("OAMD element control fields truncated at ordinal " +
                          std::to_string(ordinal));
        }
        ElementRecord record;
        record.ordinal = ordinal;
        record.element_id = element_id;
        record.size_bytes = size_bytes;
        record.body_end_bit = region_end;
        record.parsed_end_bit = region_end;
        const bool selected =
            (element_id == kObjectElementId || element_id == kTrimElementId ||
             element_id == kExtendedElementId) &&
            alternate_data_id == 0u &&
            !((element_id == kObjectElementId && seen_object) ||
              (element_id == kTrimElementId && seen_trim) ||
              (element_id == kExtendedElementId && seen_extended));
        if (!selected) {
            if (!reader.skip(region_end - reader.position())) {
                return syntax("OAMD element skip past payload end");
            }
        } else if (element_id == kObjectElementId) {
            const Status parsed = parse_object_element(
                reader, object_count, kinds.data(), &state, &sample_offset_code,
                &reserved_data_not_present);
            if (!parsed.ok()) {
                return parsed;
            }
            record.parsed_end_bit = reader.position();
            object_parsed_end = reader.position();
            seen_object = true;
            have_object_element = true;
            if (reader.position() < region_end) {
                if (!reader.skip(region_end - reader.position())) {
                    return syntax("OAMD element skip past payload end");
                }
            }
        } else if (element_id == kExtendedElementId) {
            if (!have_object_element) {
                return variant("extended_before_object_element",
                               "OAMD extended object element precedes the default object "
                               "element");
            }
            const Status parsed =
                parse_extended_element(reader, &state.objects, state.block_count);
            if (!parsed.ok()) {
                return parsed;
            }
            record.parsed_end_bit = reader.position();
            seen_extended = true;
            if (reader.position() < region_end) {
                if (!reader.skip(region_end - reader.position())) {
                    return syntax("OAMD element skip past payload end");
                }
            }
        } else {
            OamdTrim trim;
            const Status parsed = parse_trim_element(reader, object_count, &trim);
            if (!parsed.ok()) {
                return parsed;
            }
            record.parsed_end_bit = reader.position();
            record.has_trim = true;
            record.trim = trim;
            seen_trim = true;
            if (reader.position() < region_end) {
                if (!reader.skip(region_end - reader.position())) {
                    return syntax("OAMD element skip past payload end");
                }
            }
        }
        if (reader.failed()) {
            return syntax("OAMD element parse failed at ordinal " + std::to_string(ordinal));
        }
        records.push_back(record);
    }
    if (!have_object_element) {
        return variant("missing_object_element", "OAMD has no object element");
    }

    status = check_padding(reader);
    if (!status.ok()) {
        return status;
    }
    if (version != 0u) {
        return variant("oamd_version",
                       "OAMD syntax version " + std::to_string(version) + " is not covered");
    }

    // 对外状态：节目分区、trim、元素长度不一致记录。
    fill_program_state(program, &state.program);
    state.has_trim = false;
    for (const ElementRecord& record : records) {
        if (record.parsed_end_bit > record.body_end_bit) {
            ElementSizeMismatch mismatch;
            mismatch.element_ordinal = record.ordinal;
            mismatch.element_id = record.element_id;
            mismatch.declared_size_bytes = record.size_bytes;
            mismatch.declared_end_bit = static_cast<std::uint32_t>(record.body_end_bit);
            mismatch.parsed_end_bit = static_cast<std::uint32_t>(record.parsed_end_bit);
            state.size_mismatch.push_back(mismatch);
        }
        if (record.has_trim && !state.has_trim) {
            state.has_trim = true;
            state.trim = record.trim;
        }
    }

    // 旧接口字段：16 槽位置视图、block 0 的时间参数与节目声明。
    if (!state.blocks.empty()) {
        out->block_offset_samples = state.blocks[0].block_offset_samples;
        out->ramp_duration_samples = state.blocks[0].ramp_duration_samples;
    }
    out->object_count = object_count;
    out->dynamic_object_only = program.dynamic_object_only;
    out->lfe_present = program.lfe_present;
    out->num_bed_objects = program.num_bed_objects;
    out->num_isf_objects = program.num_isf_objects;
    out->num_dynamic_objects = program.num_dynamic_objects;
    out->content_bed = program.content_bed;
    out->content_isf = program.content_isf;
    out->content_dynamic = program.content_dynamic;
    out->has_trim = state.has_trim;
    out->trim = state.trim;
    fill_legacy_slots(state, out);
    out->has_frame_state = true;
    out->frame_state = std::move(state);

    if (trace != nullptr) {
        trace->bed_assignments = program.bed_assignments;
        trace->element_count = element_count;
        trace->object_element_count = have_object_element ? 1u : 0u;
        trace->sample_offset_code = sample_offset_code;
        trace->sample_offset = out->frame_state.sample_offset;
        trace->block_count = out->frame_state.block_count;
        trace->parsed_end_bit = static_cast<std::uint32_t>(object_parsed_end);
        trace->reserved_data_not_present = reserved_data_not_present;
    }
    *have_object_element_out = have_object_element;
    return Status::success();
}

Status parse_payload(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out,
                     ParseTrace* trace, bool require_legacy_shape) {
    if (payload == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd, "null payload or output");
    }
    if (payload_size == 0) {
        return variant("header_truncated", "OAMD payload is empty");
    }
    bool have_object_element = false;
    const Status status = parse_full(payload, payload_size, out, trace, &have_object_element);
    if (!status.ok()) {
        return status;
    }
    if (!require_legacy_shape) {
        return Status::success();
    }
    if (out->object_count > static_cast<std::uint32_t>(kMaxSlots)) {
        return variant("object_count", "OAMD object count " +
                                           std::to_string(out->object_count) +
                                           " exceeds the 16-slot model");
    }
    if (out->frame_state.block_count != 1u) {
        return variant("multiple_position_blocks",
                       "OAMD frame carries " +
                           std::to_string(out->frame_state.block_count) +
                           " position blocks; a single frame_update cannot express that");
    }
    return Status::success();
}

// 只解析头部与 program_assignment()，供跨层 bed 掩码校验使用。
Status parse_header_program(const std::uint8_t* payload, std::size_t payload_size,
                            std::uint32_t* object_count_out, RawProgram* program_out) {
    bits::BitReader reader(payload, payload_size);
    std::uint32_t version = reader.read(2);
    if (reader.failed()) {
        return variant("header_truncated", "OAMD payload cannot hold the 2-bit version");
    }
    if (version == 3u) {
        version += reader.read(3);
        if (reader.failed()) {
            return syntax("OAMD version extension truncated");
        }
    }
    std::uint32_t object_count_bits = reader.read(5);
    if (object_count_bits == 0x1Fu) {
        object_count_bits += reader.read(7);
    }
    if (reader.failed()) {
        return syntax("OAMD object count truncated");
    }
    *object_count_out = object_count_bits + 1u;
    return parse_program_assignment(reader, program_out);
}

}  // namespace

int q_of(int k, int n) {
    const double value =
        std::floor(32768.0 * static_cast<double>(k) / static_cast<double>(n) + 0.5);
    const int quantised = static_cast<int>(value);
    return quantised > 32767 ? 32767 : quantised;
}

std::uint32_t standard_bed_mask(std::uint32_t bed_mask) {
    std::uint32_t wide = 0;
    for (int bit = 0; bit < 10; ++bit) {
        if (((bed_mask >> bit) & 1u) != 0u) {
            wide |= kStandardBedToWide[bit];
        }
    }
    return wide;
}

Status parse_id11(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out) {
    return parse_id11_verbose(payload, payload_size, out, nullptr);
}

Status parse_id11_state(const std::uint8_t* payload, std::size_t payload_size,
                        OamdUpdate* out) {
    if (payload == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd, "null payload or output");
    }
    if (payload_size == 0) {
        return variant("header_truncated", "OAMD payload is empty");
    }
    return parse_payload(payload, payload_size, out, nullptr, false);
}

Status parse_id11_verbose(const std::uint8_t* payload, std::size_t payload_size,
                          OamdUpdate* out, ParseTrace* trace) {
    if (payload == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd, "null payload or output");
    }
    if (payload_size == 0) {
        return variant("header_truncated", "OAMD payload is empty");
    }
    return parse_payload(payload, payload_size, out, trace, true);
}

Status parse_id11_bed_assignments(const std::uint8_t* payload, std::size_t payload_size,
                                  std::vector<BedAssignment>* out) {
    if (payload == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd, "null payload or output");
    }
    if (payload_size == 0) {
        return variant("header_truncated", "OAMD payload is empty");
    }
    out->clear();
    std::uint32_t object_count = 0;
    RawProgram program;
    const Status status = parse_header_program(payload, payload_size, &object_count, &program);
    if (!status.ok()) {
        return status;
    }
    (void)object_count;
    *out = program.bed_assignments;
    return Status::success();
}

Status parse_trim(const std::uint8_t* payload, std::size_t payload_size, OamdTrim* out,
                  bool* present) {
    if (payload == nullptr || out == nullptr || present == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd, "null payload or output");
    }
    *present = false;
    if (payload_size == 0) {
        return variant("header_truncated", "OAMD payload is empty");
    }
    bits::BitReader reader(payload, payload_size);
    std::uint32_t version = reader.read(2);
    if (reader.failed()) {
        return variant("header_truncated", "OAMD payload cannot hold the 2-bit version");
    }
    if (version == 3u) {
        version += reader.read(3);
        if (reader.failed()) {
            return syntax("OAMD version extension truncated");
        }
    }
    std::uint32_t object_count_bits = reader.read(5);
    if (object_count_bits == 0x1Fu) {
        object_count_bits += reader.read(7);
    }
    if (reader.failed()) {
        return syntax("OAMD object count truncated");
    }
    const std::uint32_t object_count = object_count_bits + 1u;
    RawProgram program;
    const Status program_status = parse_program_assignment(reader, &program);
    if (!program_status.ok()) {
        return program_status;
    }
    const std::uint32_t alternate_present = reader.read(1);
    std::uint32_t element_count = reader.read(4);
    if (reader.failed()) {
        return syntax("OAMD element count truncated");
    }
    for (std::uint32_t ordinal = 0; ordinal < element_count; ++ordinal) {
        const std::uint32_t element_id = reader.read(4);
        std::uint32_t size_field = 0;
        if (!variable_bits_max(reader, 4, 4, &size_field)) {
            return syntax("OAMD element header truncated at ordinal " +
                          std::to_string(ordinal));
        }
        const std::uint32_t size_bytes = size_field + 1u;
        const std::size_t region_start = reader.position();
        const std::size_t region_end =
            region_start + static_cast<std::size_t>(size_bytes) * 8u;
        if (region_end > reader.limit()) {
            return variant("element_bounds",
                           "element " + std::to_string(ordinal) + " (id " +
                               std::to_string(element_id) + ") declares " +
                               std::to_string(size_bytes) + " bytes, beyond the payload");
        }
        if (alternate_present != 0u) {
            reader.read(4);
        }
        reader.read(1);
        if (reader.failed()) {
            return syntax("OAMD element control fields truncated at ordinal " +
                          std::to_string(ordinal));
        }
        if (element_id == kTrimElementId) {
            const Status parsed = parse_trim_element(reader, object_count, out);
            if (!parsed.ok()) {
                return parsed;
            }
            *present = true;
            return Status::success();
        }
        if (!reader.skip(region_end - reader.position())) {
            return syntax("OAMD element skip past payload end");
        }
    }
    return Status::success();
}

}  // namespace joc::oamd
