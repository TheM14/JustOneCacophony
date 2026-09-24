#include "telemetry/event_bus.h"

#include <cstring>
#include <type_traits>

namespace joc::telemetry {

// The event must stay a trivially copyable POD with no pointers: that is what
// makes "an event can never carry audio" a compile-time property.
static_assert(std::is_trivially_copyable_v<joc_event>, "joc_event must be a POD");
static_assert(sizeof(joc_event) <= 512, "joc_event must stay small");

const char* stage_name(joc_stage stage) {
    switch (stage) {
        case JOC_STAGE_IDLE: return "idle";
        case JOC_STAGE_INPUT: return "input";
        case JOC_STAGE_METADATA: return "metadata";
        case JOC_STAGE_DECODE: return "decode";
        case JOC_STAGE_JOC: return "joc";
        case JOC_STAGE_RENDER: return "render";
        case JOC_STAGE_OUTPUT: return "output";
        case JOC_STAGE_DONE: return "done";
        default: return "unknown";
    }
}

EventBus::EventBus(const joc_event_sink* sink) {
    if (sink != nullptr && sink->callback != nullptr) {
        sink_ = *sink;
        has_sink_ = true;
    }
}

void EventBus::set_totals(std::uint64_t total_frames, std::uint64_t total_samples) {
    total_frames_ = total_frames;
    total_samples_ = total_samples;
}

double EventBus::elapsed_seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
}

void EventBus::publish(joc_event* event) {
    if (!has_sink_) {
        return;
    }
    if (sink_.min_type != 0u && event->type < sink_.min_type) {
        return;
    }
    if (sink_.max_type != 0u && event->type > sink_.max_type) {
        return;
    }
    sink_.callback(sink_.user, event);
}

void EventBus::emit(std::uint32_t type, joc_stage stage, std::uint32_t log_level,
                    const std::string& message, joc_error code) {
    joc_event event{};
    event.struct_size = sizeof(joc_event);
    event.type = type;
    event.sequence = ++sequence_;
    event.timestamp_us = static_cast<std::uint64_t>(elapsed_seconds() * 1e6);
    event.total_frames = total_frames_;
    event.total_samples = total_samples_;
    event.stage = static_cast<std::uint32_t>(stage);
    event.backend = backend_;
    event.progress = total_frames_ != 0u
                         ? static_cast<double>(event.current_frame) /
                               static_cast<double>(total_frames_)
                         : -1.0;
    event.elapsed_seconds = elapsed_seconds();
    event.error_code = code;
    event.log_level = log_level;
    std::snprintf(event.stage_name, sizeof(event.stage_name), "%s", stage_name(stage));
    std::snprintf(event.message, sizeof(event.message), "%s", message.c_str());
    publish(&event);
}

void EventBus::progress(std::uint64_t frame, std::uint64_t sample, std::uint64_t output_samples,
                        std::uint64_t output_bytes, double output_seconds) {
    joc_event event{};
    event.struct_size = sizeof(joc_event);
    event.type = JOC_EV_PROGRESS;
    event.sequence = ++sequence_;
    event.timestamp_us = static_cast<std::uint64_t>(elapsed_seconds() * 1e6);
    event.current_frame = frame;
    event.total_frames = total_frames_;
    event.current_sample = sample;
    event.total_samples = total_samples_;
    event.stage = static_cast<std::uint32_t>(stage_);
    event.backend = backend_;
    event.progress = total_frames_ != 0u
                         ? static_cast<double>(frame) / static_cast<double>(total_frames_)
                         : -1.0;
    event.elapsed_seconds = elapsed_seconds();
    const double audio_seconds = static_cast<double>(sample) / 48000.0;
    event.realtime_factor = event.elapsed_seconds > 0.0 ? audio_seconds / event.elapsed_seconds
                                                        : 0.0;
    event.output_samples = output_samples;
    event.output_bytes = output_bytes;
    event.output_duration_seconds = output_seconds;
    std::snprintf(event.stage_name, sizeof(event.stage_name), "%s", stage_name(stage_));
    publish(&event);
}

void EventBus::stage(joc_stage stage, const std::string& message) {
    stage_ = stage;
    emit(JOC_EV_STAGE_CHANGED, stage, JOC_LOG_INFO, message);
}

}  // namespace joc::telemetry
