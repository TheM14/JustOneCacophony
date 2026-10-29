
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "oamd/oamd_parser.h"

namespace joc::joc {

// OAMD Table 12 的标准 bed 声道。LSB 顺序与声明顺序一致：L/R、C、LFE、
// Ls/Rs、Lrs/Rrs、TFL/TFR、TSL/TSR、TBL/TBR、LW/RW、LFE2。
enum class BedChannel : std::uint8_t {
    kLeft = 0,
    kRight,
    kCenter,
    kLfe,
    kLeftSurround,
    kRightSurround,
    kLeftRearSurround,
    kRightRearSurround,
    kTopFrontLeft,
    kTopFrontRight,
    kTopSideLeft,
    kTopSideRight,
    kTopBackLeft,
    kTopBackRight,
    kLeftWide,
    kRightWide,
    kLfe2,
    kCount,
};

// 内容声明的 bed 布局。bed 对象在 OAMD 里没有位置字段，位置由这里的布局决定。
struct BedLayout {
    bool present = false;             // 有 bed_assignment() 声明
    bool bed_only = false;            // program 只由 bed 组成（无 ISF、无动态对象）
    bool complete = true;             // 所有声明声道都在本标准表里
    const char* name = "";            // "9.1.6" 等；未知集合为空串
    std::uint32_t mask = 0;           // 声明声道的位集合（1 << BedChannel）
    std::vector<BedChannel> channels; // 去掉 LFE/LFE2，按 JOC 对象槽顺序
};

// 声明里是否包含某个 bed 声道。
inline bool bed_layout_has(const BedLayout& layout, BedChannel channel) {
    return (layout.mask & (1u << static_cast<unsigned>(channel))) != 0u;
}

// 展开 bed_assignment() 声明；``bed_only`` 来自 program_assignment() 的
// content_description。
BedLayout resolve_bed_layout(const std::vector<oamd::BedAssignment>& assignments,
                             bool bed_only);

// 扬声器布局里的短名（"Lw"）与 Table 12 的标签（"RC_LW"）。
const char* bed_channel_short_name(BedChannel channel);
const char* bed_channel_label(BedChannel channel);

// 固定方位：ADM 笛卡尔坐标（x 右、y 前、z 上）与渲染器使用的 Q15 方位。
bool bed_channel_adm_position(BedChannel channel, double xyz[3]);
bool bed_channel_q15(BedChannel channel, std::uint16_t q15[3]);

// 目标布局里同名扬声器的 Q15 方位：bed 声道直连到该扬声器，而不是参与声像。
// 目标布局没有这个声道时返回 false。
bool layout_bed_anchor(std::uint32_t out_ch_config, BedChannel channel,
                       std::uint16_t q15[3]);

}  // namespace joc::joc
