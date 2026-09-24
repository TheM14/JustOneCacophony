
#pragma once

#include <string>
#include <vector>

#include "eac3joc_core.h"
#include "foundation/status.h"
#include "joc_core.h"

namespace joc::joc {

// [16][1536] planar float32.  The kernel requires exactly 5 core channels.
Status rebuild_objects16(ejoc_renderer_handle handle, const joc_frame_params& params,
                         const float* bed5_planar, const float* lfe, float gain,
                         std::vector<float>* out16, std::string* error);

}  // namespace joc::joc
