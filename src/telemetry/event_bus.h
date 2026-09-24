
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "joc_core.h"

namespace joc::telemetry {

const char* stage_name(joc_stage stage);

class EventBus {
public:
    EventBus() = default;
    explicit EventBus(const joc_event_sink* sink);

    void set_totals(std::uint64_t total_frames, std::uint64_t total_samples);
    void set_backend(std::uint32_t backend) { backend_ = backend; }

    // Generic emit; `message` is truncated into the 256-byte field.
    void emit(std::uint32_t type, joc_stage stage, std::uint32_t log_level, const std::string& message,
              joc_error code = JOC_OK);

    void progress(std::uint64_t frame, std::uint64_t sample, std::uint64_t output_samples, std::uint64_t output_bytes,
                  double output_seconds);
    void stage(joc_stage stage, const std::string& message = std::string());
    void info(const std::string& message) { emit(JOC_EV_LOG, stage_, JOC_LOG_INFO, message); }
    void warning(const std::string& message)
    {
        ++warning_count_;
        emit(JOC_EV_WARNING, stage_, JOC_LOG_WARNING, message);
    }
    void error(joc_error code, const std::string& stage_text, const std::string& message)
    {
        ++error_count_;
        emit(JOC_EV_ERROR, stage_, JOC_LOG_ERROR,
             (stage_text.empty() ? message : stage_text + ": " + message), code);
    }

    std::uint64_t sequence() const { return sequence_; }
    std::uint32_t warning_count() const { return warning_count_; }
    std::uint32_t error_count() const { return error_count_; }
    double elapsed_seconds() const;

private:
    void publish(joc_event* event);

    joc_event_sink sink_{};
    bool has_sink_ = false;
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    std::uint64_t sequence_ = 0;
    std::uint64_t total_frames_ = 0;
    std::uint64_t total_samples_ = 0;
    std::uint32_t backend_ = 0;
    std::uint32_t warning_count_ = 0;
    std::uint32_t error_count_ = 0;
    joc_stage stage_ = JOC_STAGE_IDLE;
};

}  // namespace joc::telemetry
