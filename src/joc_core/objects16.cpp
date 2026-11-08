#include "joc_core/objects16.h"

#include <cstdint>
#include <cstdio>

namespace joc::joc {

Status rebuild_objects16(ejoc_renderer_handle handle, const joc_frame_params& params,
                         const float* bed_planar, const float* lfe,
                         std::uint32_t core_channel_count, bool has_metadata, float gain,
                         std::vector<float>* out16, std::string* error) {
    if (handle == nullptr || bed_planar == nullptr || out16 == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kDsp, "null argument");
    }
    if (core_channel_count != JOC_CORE_CHANNELS &&
        core_channel_count != JOC_MAX_CORE_CHANNELS) {
        if (error != nullptr) {
            *error = "the reused JOC kernel requires 5 or 7 core channels, the frame has " +
                     std::to_string(core_channel_count);
        }
        return Status::fail(JOC_ERR_JOC_UNSUPPORTED_VARIANT, stage::kDsp, *error);
    }

    std::vector<double> dq(static_cast<std::size_t>(JOC_MAX_OBJECTS) * JOC_MAX_DPOINTS *
                               JOC_MAX_CORE_CHANNELS * JOC_MAX_PARAMETER_BANDS,
                           0.0);
    std::uint8_t n_bands[JOC_MAX_OBJECTS] = {};
    std::uint8_t n_dpoints[JOC_MAX_OBJECTS] = {};
    std::uint8_t slope_idx[JOC_MAX_OBJECTS] = {};
    std::uint8_t offset_ts[JOC_MAX_OBJECTS * JOC_MAX_DPOINTS] = {};
    for (unsigned obj = 0; obj < JOC_MAX_OBJECTS; ++obj) {
        const joc_object_params& object = params.objects[obj];
        if (object.present == 0) {
            continue;
        }
        n_bands[obj] = object.n_bands;
        n_dpoints[obj] = object.n_dpoints;
        slope_idx[obj] = object.slope_idx;
        for (unsigned dp = 0; dp < JOC_MAX_DPOINTS; ++dp) {
            offset_ts[obj * JOC_MAX_DPOINTS + dp] = object.offset_ts[dp];
        }
        for (unsigned dp = 0; dp < object.n_dpoints; ++dp) {
            for (unsigned ch = 0; ch < core_channel_count; ++ch) {
                for (unsigned pb = 0; pb < object.n_bands; ++pb) {
                    const std::size_t index =
                        ((static_cast<std::size_t>(obj) * JOC_MAX_DPOINTS + dp) *
                             JOC_MAX_CORE_CHANNELS +
                         ch) *
                            JOC_MAX_PARAMETER_BANDS +
                        pb;
                    dq[index] = object.dq[dp][ch][pb];
                }
            }
        }
    }

    ejoc_frame_input input{};
    input.struct_size = sizeof(input);
    input.struct_version = EJOC_FRAME_INPUT_VERSION;
    input.bed_planar = bed_planar;
    input.lfe = lfe;
    input.core_channel_count = core_channel_count;
    // A frame without metadata has no declared channel count; the renderer's
    // status-9 path is gated on has_metadata, so the value is not consulted.
    input.declared_core_channels =
        has_metadata ? static_cast<std::uint32_t>(params.n_channels) : 0u;
    // sigma(k) = (k in {3,4}) gates the Ls/Rs -j rotation and the band-0 21-tap
    // DC compensation together; other configurations keep the common processing.
    // An unsupported downmix configuration (status 2) has no usable switch, so
    // the renderer keeps the previous one.
    input.phase_shift =
        (params.dmx_config_idx == 3u || params.dmx_config_idx == 4u) ? 1u : 0u;
    input.phase_shift_valid = params.status == JOC_FRAME_STATUS_UNSUPPORTED_DMX ? 0u : 1u;
    input.frame_status = has_metadata ? params.status : EJOC_FRAME_STATUS_NO_METADATA;
    input.seq = params.seq_count;
    input.has_metadata = has_metadata ? 1u : 0u;
    input.object_mask = params.present_mask;
    input.n_bands = n_bands;
    input.n_dpoints = n_dpoints;
    input.slope_idx = slope_idx;
    input.offset_ts = offset_ts;
    input.dq = dq.data();
    input.clipgain = params.clipgain;
    input.output_scale = gain;
    out16->assign(static_cast<std::size_t>(JOC_OUTPUT_CHANNELS) * JOC_FRAME_SAMPLES, 0.0f);
    input.output16_planar = out16->data();

    const int result = ejoc_renderer_render_frame(handle, &input);
    if (result != 0) {
        const char* detail = ejoc_renderer_last_error(handle);
        const std::string message =
            "ejoc_renderer_render_frame failed (" + std::to_string(result) + "): " +
            (detail != nullptr ? detail : "unknown");
        if (error != nullptr) {
            *error = message;
        }
        return Status::fail(JOC_ERR_RENDER_FAILED, stage::kDsp, message);
    }
    return Status::success();
}

}  // namespace joc::joc
