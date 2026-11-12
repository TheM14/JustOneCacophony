#include "binaural/multichannel_layout.h"

#include <cstring>

namespace joc::multichannel {

namespace {

constexpr ChannelSpec kStereo[] = {
    {"L", false, {-1.0, 1.0, 0.0}},
    {"R", false, {1.0, 1.0, 0.0}},
};

constexpr ChannelSpec kSurround51[] = {
    {"L", false, {-1.0, 1.0, 0.0}},
    {"R", false, {1.0, 1.0, 0.0}},
    {"C", false, {0.0, 1.0, 0.0}},
    {"LFE", true, {0.0, 1.0, 0.0}},
    {"BL", false, {-1.0, -1.0, 0.0}},
    {"BR", false, {1.0, -1.0, 0.0}},
};

constexpr ChannelSpec kSurround71[] = {
    {"L", false, {-1.0, 1.0, 0.0}},
    {"R", false, {1.0, 1.0, 0.0}},
    {"C", false, {0.0, 1.0, 0.0}},
    {"LFE", true, {0.0, 1.0, 0.0}},
    {"BL", false, {-1.0, -1.0, 0.0}},
    {"BR", false, {1.0, -1.0, 0.0}},
    {"SL", false, {-1.0, 0.0, 0.0}},
    {"SR", false, {1.0, 0.0, 0.0}},
};

constexpr Layout kLayouts[] = {
    {"2.0", 2, kStereo},
    {"5.1", 6, kSurround51},
    {"7.1", 8, kSurround71},
};

}  // namespace

const Layout* find_layout(const std::string& name) {
    for (const Layout& layout : kLayouts) {
        if (name == layout.name) {
            return &layout;
        }
    }
    return nullptr;
}

const Layout* detect_layout(int channels) {
    for (const Layout& layout : kLayouts) {
        if (layout.channels == channels) {
            return &layout;
        }
    }
    return nullptr;
}

Status object_positions(const Layout& layout, double positions[15][3]) {
    if (positions == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kFoundation,
                            "object position output is null");
    }
    std::memset(positions, 0, sizeof(double) * 15 * 3);
    int slot = 0;
    for (int channel = 0; channel < layout.channels; ++channel) {
        const ChannelSpec& spec = layout.specs[channel];
        if (spec.lfe) {
            continue;
        }
        if (slot >= 15) {
            return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kFoundation,
                                "layout needs more than 15 object channels");
        }
        for (int axis = 0; axis < 3; ++axis) {
            positions[slot][axis] = spec.position[axis];
        }
        ++slot;
    }
    return Status::success();
}

Status pcm_to_objects16(const float* interleaved, std::size_t frames, const Layout& layout,
                        float* objects16_planar) {
    if (interleaved == nullptr || objects16_planar == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kFoundation,
                            "multichannel PCM buffers are null");
    }
    std::memset(objects16_planar, 0, sizeof(float) * frames * 16u);
    int slot = 0;
    for (int channel = 0; channel < layout.channels; ++channel) {
        const ChannelSpec& spec = layout.specs[channel];
        const int destination = spec.lfe ? 0 : 1 + slot;
        for (std::size_t frame = 0; frame < frames; ++frame) {
            objects16_planar[static_cast<std::size_t>(destination) * frames + frame] =
                interleaved[frame * static_cast<std::size_t>(layout.channels) + channel];
        }
        if (!spec.lfe) {
            ++slot;
        }
    }
    return Status::success();
}

}  // namespace joc::multichannel
