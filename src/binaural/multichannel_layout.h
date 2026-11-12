// Channel order and fixed ADM directions for the multichannel binaural path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "foundation/status.h"

namespace joc::multichannel {

struct ChannelSpec {
    const char* label;
    bool lfe;
    double position[3];
};

struct Layout {
    const char* name;
    int channels;
    const ChannelSpec* specs;
};

// WAV channel order follows the ffmpeg definitions:
//   2.0 = FL FR
//   5.1 = FL FR FC LFE BL BR
//   7.1 = FL FR FC LFE BL BR SL SR
// BL/BR are the rear pair, SL/SR the side pair.
const Layout* find_layout(const std::string& name);
const Layout* detect_layout(int channels);

Status object_positions(const Layout& layout, double positions[15][3]);

// Maps interleaved layout PCM to the JOC source layout: channel 0 is LFE,
// channels 1..15 are objects.  Unused object channels stay silent.
Status pcm_to_objects16(const float* interleaved, std::size_t frames, const Layout& layout,
                        float* objects16_planar);

}  // namespace joc::multichannel
