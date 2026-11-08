#include "timeline/position_timeline.h"

#include <algorithm>
#include <cstring>

#include "foundation/geometry.h"
#include "foundation/metadata_timing.h"

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

OamdPositionTimeline::OamdPositionTimeline(int object_count)
    : object_count_(std::clamp(object_count, 1, oamd::kMaxObjects - 1)) {
    const std::size_t count = static_cast<std::size_t>(object_count_);
    tracks_.resize(count);
    previous_targets_.resize(count);
    has_previous_.assign(count, false);
    fixed_.assign(count, false);
    fixed_xyz_.resize(count);
}

void OamdPositionTimeline::ensure_tracks(int count) {
    const int capped = std::clamp(count, 1, oamd::kMaxObjects - 1);
    if (capped <= object_count_) {
        return;
    }
    const std::size_t size = static_cast<std::size_t>(capped);
    object_count_ = capped;
    tracks_.resize(size);
    previous_targets_.resize(size);
    has_previous_.resize(size, false);
    fixed_.resize(size, false);
    fixed_xyz_.resize(size);
}

void OamdPositionTimeline::set_fixed_position(int object_index, const double position[3]) {
    if (object_index < 1 || object_index > object_count_) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(object_index - 1);
    fixed_[index] = true;
    for (int axis = 0; axis < 3; ++axis) {
        fixed_xyz_[index][static_cast<std::size_t>(axis)] = position[axis];
    }
}

void OamdPositionTimeline::reset() {
    state_.reset();
    for (int index = 0; index < object_count_; ++index) {
        tracks_[static_cast<std::size_t>(index)] = ObjectPositionTrack();
        has_previous_[static_cast<std::size_t>(index)] = false;
        previous_targets_[static_cast<std::size_t>(index)] = {0.0, 0.0, 0.0};
    }
    initialized_ = false;
    payload_count_ = 0;
    transition_count_ = 0;
    last_coded_event_sample_ = -1;
}

void OamdPositionTimeline::apply_block(const oamd::FrameState& state,
                                       std::uint32_t block_index) {
    // Track i reads OAMD object i + 1: object 0 is the LFE/bed slot the JOC
    // renderers do not position.  Bed and ISF objects carry no position field,
    // and an inactive block keeps the previous position, exactly like the
    // reference frame_update() value construction.
    for (int track = 0; track < object_count_; ++track) {
        const int object_index = track + 1;
        if (object_index >= static_cast<int>(state.objects.size())) {
            break;
        }
        const oamd::ObjectElementState& element =
            state.objects[static_cast<std::size_t>(object_index)];
        if (block_index >= element.blocks.size()) {
            continue;
        }
        const oamd::ObjectRenderState& render =
            element.blocks[static_cast<std::size_t>(block_index)].render;
        if (render.not_active || element.kind != oamd::ObjectKind::kDynamic) {
            continue;
        }
        state_.apply_render_state(object_index, render);
    }
}

void OamdPositionTimeline::targets(std::vector<std::array<double, 3>>* out) const {
    out->resize(static_cast<std::size_t>(object_count_));
    for (int index = 0; index < object_count_; ++index) {
        const std::size_t slot = static_cast<std::size_t>(index);
        if (fixed_[slot]) {
            (*out)[slot] = fixed_xyz_[slot];
            continue;
        }
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        geometry::q_to_adm_xyz(state_.q(index + 1, 0), state_.q(index + 1, 1),
                               state_.q(index + 1, 2), &x, &y, &z);
        (*out)[slot] = {x, y, z};
    }
}

Status OamdPositionTimeline::schedule(const std::vector<std::array<double, 3>>& targets,
                                      std::int64_t coded_event, std::int64_t ramp_samples,
                                      std::int64_t object_delay_samples,
                                      std::int64_t processed_sample) {
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
        for (int index = 0; index < object_count_; ++index) {
            const std::size_t slot = static_cast<std::size_t>(index);
            tracks_[slot].set_initial(targets[slot].data());
            previous_targets_[slot] = targets[slot];
            has_previous_[slot] = true;
        }
        initialized_ = true;
        return Status::success();
    }

    const std::int64_t effective_ramp = std::max<std::int64_t>(0, ramp_samples);
    const std::int64_t transition_start =
        timing::align_metadata_sample(coded_event + object_delay_samples);
    for (int index = 0; index < object_count_; ++index) {
        const std::size_t slot = static_cast<std::size_t>(index);
        const bool changed =
            !has_previous_[slot] ||
            std::memcmp(previous_targets_[slot].data(), targets[slot].data(),
                        sizeof(targets[slot])) != 0;
        if (!changed) {
            continue;
        }
        const Status status =
            tracks_[slot].append(transition_start, effective_ramp, targets[slot].data(),
                                 index + 1);
        if (!status.ok()) {
            return status;
        }
        previous_targets_[slot] = targets[slot];
        has_previous_[slot] = true;
        ++transition_count_;
    }
    return Status::success();
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
    if (!update.has_frame_state) {
        state_.apply(update);
        targets(&targets_);
        const std::int64_t coded_event =
            frame_start_sample + outer_sample_offset +
            static_cast<std::int64_t>(update.block_offset_samples);
        const Status status =
            schedule(targets_, coded_event, update.ramp_duration_samples, object_delay_samples,
                     processed_sample);
        if (status.ok()) {
            ++payload_count_;
        }
        return status;
    }

    const oamd::FrameState& state = update.frame_state;
    ensure_tracks(state.object_count > 0u ? static_cast<int>(state.object_count) - 1 : 0);
    const std::uint32_t block_count =
        state.blocks.empty() ? 1u : static_cast<std::uint32_t>(state.blocks.size());
    for (std::uint32_t block = 0; block < block_count; ++block) {
        apply_block(state, block);
        targets(&targets_);
        std::int64_t block_offset = 0;
        std::int64_t ramp_samples = 0;
        if (block < state.blocks.size()) {
            block_offset = static_cast<std::int64_t>(state.blocks[block].block_offset_samples);
            ramp_samples = static_cast<std::int64_t>(state.blocks[block].ramp_duration_samples);
        }
        const std::int64_t coded_event =
            frame_start_sample + outer_sample_offset + block_offset;
        const Status status = schedule(targets_, coded_event, ramp_samples,
                                       object_delay_samples, processed_sample);
        if (!status.ok()) {
            return status;
        }
    }
    ++payload_count_;
    return Status::success();
}

Status OamdPositionTimeline::positions_at(std::int64_t sample, double* positions,
                                          int count) {
    if (positions == nullptr || count < 0) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd,
                            "binaural positions need a non-negative count and a buffer");
    }
    if (count > object_count_) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd,
                            "binaural positions requested " + std::to_string(count) +
                                " objects but the timeline tracks " +
                                std::to_string(object_count_));
    }
    for (int index = 0; index < count; ++index) {
        const Status status = tracks_[static_cast<std::size_t>(index)].position_at(
            sample, positions + static_cast<std::size_t>(index) * 3u);
        if (!status.ok()) {
            return status;
        }
    }
    return Status::success();
}

}  // namespace joc::timeline
