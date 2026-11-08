
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "eac3joc_core.h"
#include "foundation/status.h"
#include "oamd/oamd_parser.h"
#include "speaker/speaker_layout_lookup.h"
#include "speaker/speaker_meta.h"
#include "speaker/speaker_trim.h"

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
    // OAMD speaker controls.  A payload carrying the trim element replaces
    // them; a payload without it keeps the previous set.
    ejoc_speaker_controls controls{};
    bool controls_ready = false;
    // bed 声道是固定通道：OAMD 不给位置，方位由声明的 bed 布局给出。这里存
    // 对象槽 → Q15 方位；bed_override 为 false 的槽仍按 OAMD 坐标声像。
    std::uint16_t bed_positions[oamd::kObjects][oamd::kAxes] = {};
    bool bed_override[oamd::kObjects] = {};
    // Metadata path: a frame that carries the full per-object block state
    // switches the speaker output to the metadata renderer.  The programme trim
    // element and its raw global mode stay here because the metadata path needs
    // the per-object bypass flags and the ISF compensation mode.
    MetaRenderer meta;
    bool meta_configured = false;
    bool meta_engaged = false;
    oamd::OamdTrim trim{};
    bool has_trim = false;
    std::uint32_t raw_trim_mode = 0;
};

Status step(SpeakerStep* context, const std::vector<float>& objects16_planar,
            const oamd::OamdUpdate* update, std::string* error);

}  // namespace joc::speaker
