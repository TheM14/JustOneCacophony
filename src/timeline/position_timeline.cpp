#include "timeline/position_timeline.h"

#include <cstring>

#include "foundation/geometry.h"

namespace joc::timeline {

namespace {

Status timeline_fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kOamd, message);
}

}  // namespace

void ObjectPositionTrack::set_initial(const double position[3]) {
    for (int i = 0; i < 3; ++i) {
        initial_[i] = position[i];
        last_target_[i] = position[i];
    }
}

Status ObjectPositionTrack::append(std::int64_t start_sample, std::int64_t duration_samples,
                                   const double target[3], int object_index) {
    if (start_sample < 0 || duration_samples < 0) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd,
                            "position transition timing must be non-negative");
    }
    if (!transitions_.empty()) {
        const PositionTransition& previous = transitions_.back();
        if (start_sample < previous.end_sample()) {
            return timeline_fail(
                JOC_ERR_OAMD_UNSUPPORTED_VARIANT,
                "overlapping_binaural_position_ramps: object " + std::to_string(object_index) +
                    " has a new position update at sample " + std::to_string(start_sample) +
                    " before the previous ramp ends at " + std::to_string(previous.end_sample()));
        }
        if (start_sample == previous.start_sample && previous.duration_samples == 0) {
            PositionTransition replacement;
            replacement.start_sample = start_sample;
            replacement.duration_samples = duration_samples;
            std::memcpy(replacement.origin, previous.origin, sizeof(replacement.origin));
            std::memcpy(replacement.target, target, sizeof(replacement.target));
            transitions_.back() = replacement;
            std::memcpy(last_target_, target, sizeof(last_target_));
            return Status::success();
        }
    }
    PositionTransition transition;
    transition.start_sample = start_sample;
    transition.duration_samples = duration_samples;
    std::memcpy(transition.origin, last_target_, sizeof(transition.origin));
    std::memcpy(transition.target, target, sizeof(transition.target));
    transitions_.push_back(transition);
    std::memcpy(last_target_, target, sizeof(last_target_));
    return Status::success();
}

Status ObjectPositionTrack::position_at(std::int64_t sample, double out[3]) {
    if (sample < last_query_sample_) {
        return Status::fail(JOC_ERR_STATE, stage::kOamd,
                            "binaural metadata positions must be queried monotonically");
    }
    last_query_sample_ = sample;
    while (cursor_ < transitions_.size()) {
        const PositionTransition& transition = transitions_[cursor_];
        if (sample < transition.end_sample()) {
            break;
        }
        std::memcpy(initial_, transition.target, sizeof(initial_));
        ++cursor_;
    }
    if (cursor_ >= transitions_.size()) {
        std::memcpy(out, initial_, sizeof(initial_));
        return Status::success();
    }
    const PositionTransition& transition = transitions_[cursor_];
    if (sample < transition.start_sample) {
        std::memcpy(out, initial_, sizeof(initial_));
        return Status::success();
    }
    if (transition.duration_samples == 0) {
        std::memcpy(out, transition.target, sizeof(transition.target));
        return Status::success();
    }
    const double amount =
        static_cast<double>(sample - transition.start_sample) /
        static_cast<double>(transition.duration_samples);
    for (int i = 0; i < 3; ++i) {
        out[i] = transition.origin[i] + (transition.target[i] - transition.origin[i]) * amount;
    }
    return Status::success();
}

OamdPositionTimeline::OamdPositionTimeline(int object_count) : object_count_(object_count) {
    if (object_count != kTimelineObjects) {
        object_count_ = kTimelineObjects;
    }
}

Status OamdPositionTimeline::submit_update(const oamd::OamdUpdate& update,
                                           std::int64_t frame_start_sample,
                                           std::int64_t outer_sample_offset,
                                           std::int64_t object_delay_samples,
                                           std::int64_t processed_sample) {
    if (frame_start_sample < 0 || outer_sample_offset < 0 || object_delay_samples < 0) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd,
                            "OAMD frame, outer offset and object delay must be non-negative");
    }
    state_.apply(update);

    double targets[kTimelineObjects][3] = {};
    for (int index = 0; index < kTimelineObjects; ++index) {
        geometry::q_to_adm_xyz(state_.q(index + 1, 0), state_.q(index + 1, 1),
                               state_.q(index + 1, 2), &targets[index][0], &targets[index][1],
                               &targets[index][2]);
    }

    const std::int64_t coded_event =
        frame_start_sample + outer_sample_offset +
        static_cast<std::int64_t>(update.block_offset_samples);
    if (coded_event < last_coded_event_sample_) {
        return timeline_fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT,
                             "non_monotonic_binaural_updates: event sample " +
                                 std::to_string(coded_event) + " follows " +
                                 std::to_string(last_coded_event_sample_));
    }
    last_coded_event_sample_ = coded_event;

    if (!initialized_) {
        if (processed_sample > 0) {
            return timeline_fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT,
                                 "late_initial_binaural_state: the first OAMD state arrived after "
                                 "sample " + std::to_string(processed_sample) +
                                     " had been processed, so sample 0 cannot be backfilled");
        }
        for (int index = 0; index < kTimelineObjects; ++index) {
            tracks_[index].set_initial(targets[index]);
            std::memcpy(previous_targets_[index], targets[index], sizeof(targets[index]));
            has_previous_[index] = true;
        }
        initialized_ = true;
        ++payload_count_;
        return Status::success();
    }

    const std::int64_t ramp_duration = static_cast<std::int64_t>(update.ramp_duration_samples);
    const std::int64_t effective_ramp =
        ramp_duration - kOamdUpdateQuantumSamples > 0 ? ramp_duration - kOamdUpdateQuantumSamples
                                                      : 0;
    std::int64_t transition_start = coded_event + object_delay_samples;
    if (effective_ramp != 0) {
        transition_start += kOamdUpdateQuantumSamples;
    }
    for (int index = 0; index < kTimelineObjects; ++index) {
        const bool changed = !has_previous_[index] ||
                             std::memcmp(previous_targets_[index], targets[index],
                                         sizeof(targets[index])) != 0;
        if (!changed) {
            continue;
        }
        const Status status = tracks_[index].append(transition_start, effective_ramp,
                                                    targets[index], index + 1);
        if (!status.ok()) {
            return status;
        }
        std::memcpy(previous_targets_[index], targets[index], sizeof(targets[index]));
        has_previous_[index] = true;
        ++transition_count_;
    }
    ++payload_count_;
    return Status::success();
}

Status OamdPositionTimeline::positions_at(std::int64_t sample,
                                          double positions[kTimelineObjects][3]) {
    for (int index = 0; index < kTimelineObjects; ++index) {
        const Status status = tracks_[index].position_at(sample, positions[index]);
        if (!status.ok()) {
            return status;
        }
    }
    return Status::success();
}

}  // namespace joc::timeline
