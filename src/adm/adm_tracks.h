// Port of src/oamd_tracks.py.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "adm/adm_metadata.h"
#include "foundation/status.h"
#include "oamd/oamd_parser.h"

namespace joc::adm {

struct OamdEvent {
    std::int64_t sample = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::int64_t ramp_samples = 0;
};

enum class TrajectoryMode { Compact, Dense64 };

// Feeds the same per-frame OAMD state machine the reference's build_adm_tracks
// runs, and records one event per object whenever its coordinates change.
class TrajectoryBuilder {
public:
    TrajectoryBuilder(std::uint32_t rate = 48000, std::int64_t update_quantum_samples = 64,
                      std::int64_t object_delay_samples = 1473)
        : rate_(rate),
          quantum_(update_quantum_samples),
          object_delay_(object_delay_samples) {}

    void submit_frame(std::int64_t frame_index, const oamd::OamdUpdate* update, std::int64_t outer_offset);

    Status build(std::int64_t total_samples, TrajectoryMode mode, std::vector<Track>* out) const;

    const std::vector<OamdEvent>& events(int object_index) const { return events_[object_index]; }
    std::uint32_t rate() const { return rate_; }
    std::int64_t object_delay_samples() const { return object_delay_; }

private:
    std::uint32_t rate_;
    std::int64_t quantum_;
    std::int64_t object_delay_;
    oamd::OamdState state_;
    std::vector<OamdEvent> events_[kObjectCount];
    bool has_previous_[kObjectCount] = {};
    double previous_[kObjectCount][3] = {};
};

Status expand_compact(const std::vector<OamdEvent>& events, std::int64_t total_samples, std::uint32_t rate,
                      std::int64_t update_quantum_samples, std::int64_t object_delay_samples,
                      int object_index, std::vector<Keyframe>* out);

Status expand_dense64(const std::vector<OamdEvent>& events, std::int64_t total_samples, std::uint32_t rate,
                      std::int64_t update_quantum_samples, std::int64_t object_delay_samples,
                      int object_index, std::vector<Keyframe>* out);

}  // namespace joc::adm
