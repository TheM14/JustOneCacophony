#include "adm/adm_tracks.h"

#include <algorithm>
#include <cmath>

#include "foundation/geometry.h"
#include "foundation/metadata_timing.h"

namespace joc::adm {

namespace {

struct Point {
    std::int64_t sample = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::int64_t interpolation_samples = 0;
};

// otherwise (the reference drops silently).
void append_point(std::vector<Point>* points, std::int64_t sample, double x, double y, double z,
                  std::int64_t interpolation_samples) {
    if (!points->empty() && sample == points->back().sample) {
        points->back() = Point{sample, x, y, z, interpolation_samples};
    } else if (points->empty() || sample > points->back().sample) {
        points->push_back(Point{sample, x, y, z, interpolation_samples});
    }
}

void lerp(double ax, double ay, double az, double bx, double by, double bz, double amount,
          double* x, double* y, double* z) {
    *x = ax + (bx - ax) * amount;
    *y = ay + (by - ay) * amount;
    *z = az + (bz - az) * amount;
}

void points_to_blocks(const std::vector<Point>& points, std::int64_t total_samples,
                      std::vector<Keyframe>* out) {
    for (std::size_t index = 0; index < points.size(); ++index) {
        const Point& point = points[index];
        const std::int64_t end =
            (index + 1 < points.size()) ? points[index + 1].sample : total_samples;
        const std::int64_t duration = std::max<std::int64_t>(0, end - point.sample);
        if (duration == 0) {
            continue;
        }
        Keyframe keyframe;
        keyframe.rtime_samples = point.sample;
        keyframe.x = point.x;
        keyframe.y = point.y;
        keyframe.z = point.z;
        keyframe.duration_samples = duration;
        keyframe.interpolation_samples = std::min(point.interpolation_samples, duration);
        out->push_back(keyframe);
    }
}

Status non_monotonic(const char* name, const char* message, int object_index, std::int64_t sample,
                     std::int64_t previous_sample) {
    return Status::fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT, stage::kOamd,
                        std::string(name) + ": " + message + " (object " +
                            std::to_string(object_index) + ", sample " + std::to_string(sample) +
                            ", previous " + std::to_string(previous_sample) + ")");
}

}  // namespace

Status expand_compact(const std::vector<OamdEvent>& events, std::int64_t total_samples, std::uint32_t rate,
                      std::int64_t update_block_samples, std::int64_t object_delay_samples,
                      int object_index, std::vector<Keyframe>* out) {
    out->clear();
    const double scale = static_cast<double>(rate);
    if (events.empty()) {
        Keyframe keyframe;
        keyframe.rtime_samples = 0;
        keyframe.duration_samples = total_samples;
        keyframe.interpolation_samples = 0;
        keyframe.x = 0.0;
        keyframe.y = 0.0;
        keyframe.z = 0.0;
        out->push_back(keyframe);
        return Status::success();
    }

    std::vector<Point> points;
    double current[3] = {events[0].x, events[0].y, events[0].z};
    append_point(&points, 0, current[0], current[1], current[2], 0);

    for (std::size_t index = 1; index < events.size(); ++index) {
        const OamdEvent& event = events[index];
        const std::int64_t event_start = timing::align_metadata_sample(
            event.sample + object_delay_samples, update_block_samples);
        if (event_start >= total_samples) {
            break;
        }
        const std::int64_t ramp = std::max<std::int64_t>(0, event.ramp_samples);
        const std::int64_t block_start = event_start;
        if (block_start >= total_samples) {
            break;
        }
        const std::int64_t ramp_end = block_start + ramp;

        if (block_start < points.back().sample) {
            return non_monotonic("non_monotonic_compact_position_updates",
                                 "compact object position update moved backwards", object_index,
                                 block_start, points.back().sample);
        }
        if (index + 1 < events.size()) {
            const std::int64_t next_block_start = timing::align_metadata_sample(
                events[index + 1].sample + object_delay_samples, update_block_samples);
            if (next_block_start < ramp_end) {
                return non_monotonic("overlapping_compact_position_ramps",
                                     "a new position update arrived before the previous compact "
                                     "ramp finished",
                                     object_index, block_start, ramp_end);
            }
        }

        double target[3] = {event.x, event.y, event.z};
        std::int64_t interpolation = ramp;
        const std::int64_t available = total_samples - block_start;
        if (ramp > available) {
            const double amount = static_cast<double>(available) / static_cast<double>(ramp);
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            lerp(current[0], current[1], current[2], event.x, event.y, event.z, amount, &x, &y, &z);
            target[0] = x;
            target[1] = y;
            target[2] = z;
            interpolation = available;
        }
        append_point(&points, block_start, target[0], target[1], target[2], interpolation);
        current[0] = event.x;
        current[1] = event.y;
        current[2] = event.z;
    }

    points_to_blocks(points, total_samples, out);
    (void)scale;
    return Status::success();
}

Status expand_dense64(const std::vector<OamdEvent>& events, std::int64_t total_samples, std::uint32_t rate,
                      std::int64_t update_block_samples, std::int64_t object_delay_samples,
                      int object_index, std::vector<Keyframe>* out) {
    out->clear();
    (void)rate;
    if (events.empty()) {
        Keyframe keyframe;
        keyframe.duration_samples = total_samples;
        out->push_back(keyframe);
        return Status::success();
    }

    std::vector<Point> points;
    double current[3] = {events[0].x, events[0].y, events[0].z};
    append_point(&points, 0, current[0], current[1], current[2], 0);

    for (std::size_t index = 1; index < events.size(); ++index) {
        const OamdEvent& event = events[index];
        const std::int64_t start =
            timing::align_metadata_sample(event.sample + object_delay_samples, update_block_samples);
        if (start >= total_samples) {
            break;
        }
        if (start < points.back().sample) {
            return non_monotonic("non_monotonic_position_updates",
                                 "object position update moved backwards", object_index, start,
                                 points.back().sample);
        }
        if (start > points.back().sample) {
            append_point(&points, start, current[0], current[1], current[2], 0);
        }
        const std::int64_t ramp = std::max<std::int64_t>(0, event.ramp_samples);
        if (ramp == 0) {
            append_point(&points, start, event.x, event.y, event.z, 0);
            current[0] = event.x;
            current[1] = event.y;
            current[2] = event.z;
            continue;
        }
        const std::int64_t steps = (ramp + update_block_samples - 1) / update_block_samples;
        const std::int64_t end = start + steps * update_block_samples;
        if (index + 1 < events.size()) {
            const std::int64_t next_start = timing::align_metadata_sample(
                events[index + 1].sample + object_delay_samples, update_block_samples);
            if (next_start < end) {
                return non_monotonic("overlapping_position_ramps",
                                     "a new position update arrived before the previous ramp "
                                     "finished",
                                     object_index, start, end);
            }
        }
        std::int64_t future = ramp;
        std::int64_t elapsed = 0;
        double position[3] = {current[0], current[1], current[2]};
        while (future > 0) {
            const double amount =
                std::min(static_cast<double>(update_block_samples) / static_cast<double>(future),
                         1.0);
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            lerp(position[0], position[1], position[2], event.x, event.y, event.z, amount, &x, &y,
                 &z);
            position[0] = x;
            position[1] = y;
            position[2] = z;
            elapsed += update_block_samples;
            const std::int64_t sample = start + elapsed;
            if (sample >= total_samples) {
                break;
            }
            append_point(&points, sample, position[0], position[1], position[2],
                         update_block_samples);
            future -= update_block_samples;
        }
        current[0] = event.x;
        current[1] = event.y;
        current[2] = event.z;
    }

    points_to_blocks(points, total_samples, out);
    return Status::success();
}

void TrajectoryBuilder::submit_frame(std::int64_t frame_index, const oamd::OamdUpdate* update,
                                     std::int64_t outer_offset) {
    std::int64_t event_sample = frame_index * 1536;
    std::int64_t ramp_samples = 0;
    if (update != nullptr) {
        state_.apply(*update);
        event_sample += outer_offset + static_cast<std::int64_t>(update->block_offset_samples);
        ramp_samples = static_cast<std::int64_t>(update->ramp_duration_samples);
    }
    for (int object = 1; object <= kObjectCount; ++object) {
        if (fixed_[object - 1]) {
            continue;  // 固定 bed 声道：位置由布局给出，不跟 OAMD 更新
        }
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        geometry::q_to_adm_xyz(state_.q(object, 0), state_.q(object, 1), state_.q(object, 2), &x,
                               &y, &z);
        const int slot = object - 1;
        if (!has_previous_[slot] || previous_[slot][0] != x || previous_[slot][1] != y ||
            previous_[slot][2] != z) {
            events_[slot].push_back(OamdEvent{event_sample, x, y, z, ramp_samples});
            previous_[slot][0] = x;
            previous_[slot][1] = y;
            previous_[slot][2] = z;
            has_previous_[slot] = true;
        }
    }
}

Status TrajectoryBuilder::build(std::int64_t total_samples, TrajectoryMode mode,
                                std::vector<Track>* out) const {
    out->clear();
    out->reserve(kObjectCount);
    for (int object = 1; object <= kObjectCount; ++object) {
        Track track;
        track.name = "JOC_Object_" + std::to_string(object);
        if (fixed_[object - 1]) {
            Keyframe block;
            block.rtime_samples = 0;
            block.x = fixed_xyz_[object - 1][0];
            block.y = fixed_xyz_[object - 1][1];
            block.z = fixed_xyz_[object - 1][2];
            block.duration_samples = total_samples;
            block.interpolation_samples = 0;
            track.blocks.push_back(block);
            out->push_back(std::move(track));
            continue;
        }
        const Status status =
            (mode == TrajectoryMode::Compact)
                ? expand_compact(events_[object - 1], total_samples, rate_, block_,
                                 object_delay_, object, &track.blocks)
                : expand_dense64(events_[object - 1], total_samples, rate_, block_,
                                 object_delay_, object, &track.blocks);
        if (!status.ok()) {
            return status;
        }
        out->push_back(std::move(track));
    }
    return Status::success();
}

}  // namespace joc::adm
