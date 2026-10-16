
#pragma once

#include <string>
#include <utility>

#include "joc_core.h"

namespace joc {

class Status {
public:
    Status() = default;

    static Status success() { return Status(); }

    static Status fail(joc_error code, std::string stage, std::string message) {
        Status s;
        s.code_ = code;
        s.stage_ = std::move(stage);
        s.message_ = std::move(message);
        return s;
    }

    bool ok() const { return code_ == JOC_OK; }
    joc_error code() const { return code_; }
    const std::string& stage() const { return stage_; }
    const std::string& message() const { return message_; }

private:
    joc_error code_ = JOC_OK;
    std::string stage_ = "none";
    std::string message_;
};

// Structured detail of the most recent failing call, as returned by
// joc_last_error_detail().  It is one instance per thread across the whole
// library, so every C entry point -- the task API and the streaming API alike --
// reports through the same string.
inline thread_local std::string last_error_detail;

inline void record_error_detail(const Status& status) {
    if (status.ok()) {
        last_error_detail.clear();
        return;
    }
    last_error_detail.assign(status.stage());
    last_error_detail.append(": ");
    last_error_detail.append(status.message());
}

// Stage names are kept as plain literals so that C++ and the Python frontend
namespace stage {
inline constexpr const char* kFoundation = "foundation";
inline constexpr const char* kEac3 = "eac3_transport";
inline constexpr const char* kEmdf = "emdf";
inline constexpr const char* kJoc = "joc";
inline constexpr const char* kOamd = "oamd";
inline constexpr const char* kDsp = "dsp";
inline constexpr const char* kRender = "render";
inline constexpr const char* kOutput = "output";
}  // namespace stage

}  // namespace joc
