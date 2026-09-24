// Port of src/binaural_metadata.py.

#pragma once

#include <cstdint>
#include <vector>

#include "foundation/status.h"
#include "oamd/oamd_parser.h"

namespace joc::timeline {

inline constexpr std::int64_t kOamdUpdateQuantumSamples = 64;
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

    Status submit_update(const oamd::OamdUpdate& update, std::int64_t frame_start_sample,
                         std::int64_t outer_sample_offset, std::int64_t object_delay_samples,
                         std::int64_t processed_sample);

    // positions[15][3]; queries must be monotonically increasing.
    Status positions_at(std::int64_t sample, double positions[kTimelineObjects][3]);

    std::uint64_t payload_count() const { return payload_count_; }
    std::uint64_t transition_count() const { return transition_count_; }
    bool initialized() const { return initialized_; }

    const ObjectPositionTrack& track(int index) const { return tracks_[index]; }

private:
    int object_count_;
    oamd::OamdState state_;
    ObjectPositionTrack tracks_[kTimelineObjects];
    bool initialized_ = false;
    double previous_targets_[kTimelineObjects][3] = {};
    bool has_previous_[kTimelineObjects] = {};
    std::uint64_t payload_count_ = 0;
    std::uint64_t transition_count_ = 0;
    std::int64_t last_coded_event_sample_ = -1;
};

}  // namespace joc::timeline
