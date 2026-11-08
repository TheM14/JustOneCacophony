
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "oamd/oamd_parser.h"
#include "speaker/speaker_layout_lookup.h"

namespace joc::joc {

// BedChannel lives in oamd/oamd_parser.h: the enum is the OAMD Table 12
// label order and OAMD frame state carries it directly.

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
// 目标布局没有这个声道时返回 false。4.0 与 6.0 共用未分配的 out_ch_config
// 0xFF，因此这里按 4.0 的声道集合解释 0xFF。
bool layout_bed_anchor(std::uint32_t out_ch_config, BedChannel channel,
                       std::uint16_t q15[3]);

// 参考位置网格 ``BED_REFERENCE_POSITIONS_Q15``：bed 标签与输出 slots 的最近邻
// 匹配都用这一组固定 Q15 点。索引 3 是 (-1,-1,-1)，表示 LFE 那样的非空间声源。
inline constexpr int kBedReferencePositionCount = 38;
const std::int32_t (*bed_reference_positions_q15())[3];

// 一个目标扬声器布局的 bed 路由描述（bed_layout.bed_output_description）：
// 按标准 bed 标签序（Table 12 的 17 个标签）给出目标 layout 的输出参考位置、
// 每个标签的输出 slot 与直连参考位置，以及该布局的自动 trim 基准衰减。
struct BedOutputDescription {
    std::vector<int> output_labels;  // 每个输出声道最近邻的参考位置索引
    int label_to_slot[oamd::kBedLabelCount] = {};  // -2 不可达，-1 LFE 通路
    int direct_output_labels[oamd::kBedLabelCount] = {};
    double automatic_attenuation_db = 0.0;
};

// 由布局名/几何派生 bed 路由描述；``layout.channel_names`` 必须按标准声道
// 顺序给出 ``layout.channel_count`` 个名字。
BedOutputDescription bed_output_description(const speaker::LayoutInfo& layout);

}  // namespace joc::joc
