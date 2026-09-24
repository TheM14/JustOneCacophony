#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "foundation/status.h"
#include "hrtf/rosella_model.h"
#include "oamd/oamd_parser.h"
#include "timeline/position_timeline.h"

// Rosella ".personalized_headphone" binaural renderer (upstream rosella_core.py,
// rosella_direct.py, rosella_room.py and rosella_binaural_renderer.py).  It takes
// the same pipeline slot as the SOFA runtime: sixteen object channels per frame in,
// interleaved stereo out, with the OAMD timeline driving the per-block parameters.
namespace joc::hrtf {

// rosella_direct.BINAURAL_PROFILE_NAMES.
enum class RosellaProfile : std::int32_t { Near = 1, Far = 2, Mid = 3 };

struct RosellaRenderOptions {
    RosellaProfile profile = RosellaProfile::Mid;
    std::int64_t object_delay_samples = 1473;
    double tail_seconds = 5.0;
    double output_gain = 1.0;
    int chunk_frames = 64;
    int room_impulse_slots = 4096;
};

class RosellaRuntime {
public:
    RosellaRuntime();
    ~RosellaRuntime();
    RosellaRuntime(const RosellaRuntime&) = delete;
    RosellaRuntime& operator=(const RosellaRuntime&) = delete;

    Status open(const RosellaModel& model, const RosellaRenderOptions& options);

    // objects16_planar is channel-major: channel * 1536 + sample.
    Status submit_frame(const float* objects16_planar, const oamd::OamdUpdate* update,
                        std::int64_t frame_index, std::int64_t outer_sample_offset,
                        std::int64_t object_delay_samples);

    // Drains the flush tail: the pending partial chunk plus flush_samples of silence.
    Status finish(std::uint32_t flush_samples, std::vector<double>* out);
    std::uint32_t finish_capacity(double tail_seconds) const;

    Status reset();

    const std::vector<double>& output() const;
    void take_output(std::vector<double>* out);

    std::uint64_t input_samples() const;
    std::uint64_t processed_input_samples() const;
    std::uint64_t metadata_block_updates() const;
    const timeline::OamdPositionTimeline& timeline() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joc::hrtf
