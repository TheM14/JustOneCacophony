
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "eac3joc_core.h"
#include "foundation/status.h"
#include "joc_core.h"

namespace joc::joc {

// Rebuilds the 16-channel object frame ([16][1536] planar float32) for one
// audio frame through the renderer's frame-level state machine.  bed_planar is
// [core_channel_count][1536] in JOC lane order (L,R,C,Ls,Rs,(extra pair)), where
// core_channel_count is the actual bed lane count (5 or 7) and lfe is the
// separate core LFE plane (NULL = zeros; the LFE ring still advances).
//
// has_metadata is false for an audio frame without a JOC EMDF container: such a
// frame always passes the 8 core planes through to output slots 0..7.  For a
// frame with metadata, params.status selects normal rendering or the invalid
// frame path (hold / hard cut), and params.seq_count drives the state machine.
// gain is the user output gain (not joc_clipgain).
Status rebuild_objects16(ejoc_renderer_handle handle, const joc_frame_params& params,
                         const float* bed_planar, const float* lfe,
                         std::uint32_t core_channel_count, bool has_metadata, float gain,
                         std::vector<float>* out16, std::string* error);

}  // namespace joc::joc
