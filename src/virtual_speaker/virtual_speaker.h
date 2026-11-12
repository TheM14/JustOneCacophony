// Virtual speaker input: ordinary multichannel PCM as fixed-direction JOC
// objects.  It owns only the layout mapping and the PCM-to-objects16 shape;
// decoding, speaker rendering and binaural rendering stay in their own modules.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"
#include "oamd/oamd_parser.h"

namespace joc::virtual_speaker {

struct ChannelSpec {
    bool lfe;
    double x;
    double y;
    double z;
};

struct Layout {
    const char* name;
    int channels;
    ChannelSpec channel_specs[8];
};

const Layout* find_layout(const std::string& name);
const Layout* detect_layout(int channels);

class Input {
public:
    // gain_db is the module's master attenuation; it is applied to every
    // mapped channel once, before the objects16 frame leaves the module.
    Status configure(const std::string& layout_name, int channels, double gain_db = 0.0);
    void reset();

    const oamd::OamdUpdate& fixed_update() const { return fixed_update_; }
    int channels() const { return channels_; }
    const Layout* layout() const { return layout_; }

    // Appends interleaved frames; take_frame() emits whole 1536-frame blocks.
    Status append(const float* interleaved, std::size_t frames);
    bool frame_ready() const;
    Status take_frame(std::vector<float>* objects16);

private:
    const Layout* layout_ = nullptr;
    int channels_ = 0;
    double gain_ = 1.0;
    oamd::OamdUpdate fixed_update_{};
    std::vector<float> pending_;
    std::size_t read_offset_ = 0;
};

}  // namespace joc::virtual_speaker
