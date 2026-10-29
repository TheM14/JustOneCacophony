
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "eac3joc_core.h"
#include "foundation/status.h"
#include "oamd/oamd_parser.h"
#include "speaker/speaker_layout_lookup.h"

namespace joc::speaker {

struct SpeakerStep {
    ejoc_speaker_renderer_handle handle = nullptr;
    LayoutInfo layout{};
    oamd::OamdState state;
    std::uint32_t metadata_offset = 1473;
    std::vector<float> interleaved;
    std::vector<double> output;
    std::uint32_t last_block_offset = 0;
    std::uint32_t last_ramp_duration = 0;
    std::uint32_t last_object_count = 0;
    bool last_had_payload = false;
    // bed 声道是固定通道：OAMD 不给位置，方位由声明的 bed 布局给出。这里存
    // 对象槽 → Q15 方位；bed_override 为 false 的槽仍按 OAMD 坐标声像。
    std::uint16_t bed_positions[oamd::kObjects][oamd::kAxes] = {};
    bool bed_override[oamd::kObjects] = {};
};

Status step(SpeakerStep* context, const std::vector<float>& objects16_planar,
            const oamd::OamdUpdate* update, std::string* error);

}  // namespace joc::speaker
