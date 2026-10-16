#include "joc_core/objects16.h"

#include <cstdint>
#include <cstdio>

namespace joc::joc {

Status rebuild_objects16(ejoc_renderer_handle handle, const joc_frame_params& params,
                         const float* bed_planar, const float* lfe, float gain,
                         std::vector<float>* out16, std::string* error) {
    if (handle == nullptr || bed_planar == nullptr || out16 == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kDsp, "null argument");
    }
    if (params.n_channels != JOC_CORE_CHANNELS &&
        params.n_channels != JOC_MAX_CORE_CHANNELS) {
        if (error != nullptr) {
            *error = "the reused JOC kernel requires 5 or 7 core channels, frame declares " +
                     std::to_string(static_cast<unsigned>(params.n_channels));
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
            for (unsigned ch = 0; ch < params.n_channels; ++ch) {
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

    out16->assign(static_cast<std::size_t>(JOC_OUTPUT_CHANNELS) * JOC_FRAME_SAMPLES, 0.0f);
    // sigma(k) = (k in {3,4}) gates the Ls/Rs -j rotation and the band-0 21-tap
    // DC compensation together; other configurations keep the common processing.
    const std::uint32_t phase_shift =
        (params.dmx_config_idx == 3 || params.dmx_config_idx == 4) ? 1u : 0u;
    const int result = ejoc_renderer_process(
        handle, bed_planar, lfe, params.n_channels, phase_shift, params.present_mask, n_bands,
        n_dpoints, slope_idx, offset_ts, dq.data(), params.clipgain, 0.0625f, gain,
        out16->data());
    if (result != 0) {
        const char* detail = ejoc_renderer_last_error(handle);
        const std::string message =
            "ejoc_renderer_process failed (" + std::to_string(result) + "): " +
            (detail != nullptr ? detail : "unknown");
        if (error != nullptr) {
            *error = message;
        }
        return Status::fail(JOC_ERR_RENDER_FAILED, stage::kDsp, message);
    }
    return Status::success();
}

}  // namespace joc::joc
