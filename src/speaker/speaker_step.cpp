#include "speaker/speaker_step.h"

#include <cstdio>

namespace joc::speaker {

Status step(SpeakerStep* context, const std::vector<float>& objects16_planar,
            const oamd::OamdUpdate* update, std::string* error) {
    if (context == nullptr || context->handle == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "speaker renderer is not open");
    }
    if (objects16_planar.size() !=
        static_cast<std::size_t>(JOC_OUTPUT_CHANNELS) * JOC_FRAME_SAMPLES) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "objects16 must be [16][1536]");
    }
    context->interleaved.assign(static_cast<std::size_t>(JOC_FRAME_SAMPLES) * JOC_OUTPUT_CHANNELS,
                                0.0f);
    for (std::size_t n = 0; n < JOC_FRAME_SAMPLES; ++n) {
        for (std::size_t c = 0; c < JOC_OUTPUT_CHANNELS; ++c) {
            context->interleaved[n * JOC_OUTPUT_CHANNELS + c] =
                objects16_planar[c * JOC_FRAME_SAMPLES + n];
        }
    }
    context->output.assign(
        static_cast<std::size_t>(JOC_FRAME_SAMPLES) * context->layout.channel_count, 0.0);

    context->last_had_payload = update != nullptr;
    std::uint32_t ramp = 0;
    if (update != nullptr) {
        context->state.apply(*update);
        ramp = update->ramp_duration_samples;
        context->last_block_offset = update->block_offset_samples;
        context->last_ramp_duration = update->ramp_duration_samples;
        context->last_object_count = update->object_count;
    }

    std::uint16_t positions[oamd::kObjects][3] = {};
    context->state.object_positions_q15(positions);
    // A frame without OAMD must not touch the gains at all (no event), otherwise a
    // pending ramp would snap - matching the reference exactly.
    const std::uint32_t event_count = context->last_had_payload ? 1u : 0u;
    const std::uint32_t offset = context->metadata_offset;
    const int result = ejoc_speaker_renderer_process(
        context->handle, context->interleaved.data(), JOC_FRAME_SAMPLES, event_count,
        event_count != 0u ? &offset : nullptr, event_count != 0u ? &ramp : nullptr,
        event_count != 0u ? &positions[0][0] : nullptr, nullptr, nullptr, nullptr,
        context->output.data());
    if (result != 0) {
        const char* message = ejoc_speaker_renderer_last_error(context->handle);
        const std::string text =
            "ejoc_speaker_renderer_process failed (" + std::to_string(result) + "): " +
            (message != nullptr ? message : "unknown");
        if (error != nullptr) {
            *error = text;
        }
        return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender, text);
    }
    return Status::success();
}

}  // namespace joc::speaker
