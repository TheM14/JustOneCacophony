// Port of src/binaural_metadata.py.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "foundation/status.h"
#include "oamd/oamd_parser.h"

namespace joc::timeline {

// The JOC binaural renderers carry one LFE plus fifteen object sources, but the
// timeline is not tied to that product: it keeps one track per declared OAMD
// object (bounded only by oamd::kMaxObjects) and the renderer reads as many as
// its source layout consumes.  A binaural output with a different source count
// (for example one fed by a TrueHD decode) can ask for more tracks.
inline constexpr int kTimelineObjects = 15;

struct PositionTransition {
    std::int64_t start_sample = 0;
    std::int64_t duration_samples = 0;
    double origin[3] = {};
    double target[3] = {};
    std::int64_t end_sample() const { return start_sample + duration_samples; }
};

class ObjectPositionTrack {
public:
    void set_initial(const double position[3]);
    Status append(std::int64_t start_sample, std::int64_t duration_samples, const double target[3],
                  int object_index);
    // Monotonic queries only; out receives the interpolated position.
    Status position_at(std::int64_t sample, double out[3]);

    const std::vector<PositionTransition>& transitions() const { return transitions_; }

private:
    double initial_[3] = {};
    double last_target_[3] = {};
    std::vector<PositionTransition> transitions_;
    std::size_t cursor_ = 0;
    std::int64_t last_query_sample_ = -1;
};

class OamdPositionTimeline {
public:
    explicit OamdPositionTimeline(int object_count = kTimelineObjects);

    // Schedules one OAMD payload.  A payload parsed by parse_id11_state() carries
    // the full frame state: every position block is applied at its own offset and
    // ramp, and the track list grows to cover every declared object.  The legacy
    // single-block slot view is still accepted for direct callers.
    Status submit_update(const oamd::OamdUpdate& update, std::int64_t frame_start_sample,
                         std::int64_t outer_sample_offset, std::int64_t object_delay_samples,
                         std::int64_t processed_sample);

    // positions[count][3]; queries must be monotonically increasing.
    Status positions_at(std::int64_t sample, double* positions, int count);

    // bed 声道是固定通道：OAMD 不给位置，方位由声明的 bed 布局给出。
    void set_fixed_position(int object_index, const double position[3]);

    // Drops the OAMD state and every transition but keeps the fixed bed positions:
    // they describe the content, not the playback position.
    void reset();

    int object_count() const { return object_count_; }
    std::uint64_t payload_count() const { return payload_count_; }
    std::uint64_t transition_count() const { return transition_count_; }
    bool initialized() const { return initialized_; }

    const ObjectPositionTrack& track(int index) const { return tracks_[static_cast<std::size_t>(index)]; }

private:
    // Grows the track list so every declared OAMD object has one (the JOC
    // renderers use the first kTimelineObjects).
    void ensure_tracks(int count);
    // OAMD position state after applying one frame-state block.
    void apply_block(const oamd::FrameState& state, std::uint32_t block_index);
    // Current ADM cartesian target of every track.
    void targets(std::vector<std::array<double, 3>>* out) const;
    // Initialises from the first update or appends the changed transitions.
    Status schedule(const std::vector<std::array<double, 3>>& targets,
                    std::int64_t coded_event, std::int64_t ramp_samples,
                    std::int64_t object_delay_samples, std::int64_t processed_sample);

    int object_count_;
    oamd::OamdState state_;
    std::vector<ObjectPositionTrack> tracks_;
    bool initialized_ = false;
    std::vector<std::array<double, 3>> previous_targets_;
    std::vector<bool> has_previous_;
    std::vector<bool> fixed_;
    std::vector<std::array<double, 3>> fixed_xyz_;
    std::vector<std::array<double, 3>> targets_;
    std::uint64_t payload_count_ = 0;
    std::uint64_t transition_count_ = 0;
    std::int64_t last_coded_event_sample_ = -1;
};

}  // namespace joc::timeline
