
#include <atomic>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "foundation/status.h"
#include "joc_core.h"
#include "task/task.h"

// task and whichever frontend wants to stop it (plan 29.1/29.2).
struct joc_cancel_token {
    std::atomic<std::uint32_t> requested{0u};
};

namespace {

thread_local std::string g_task_detail;

joc_error finish_task(const joc::Status& status) {
    if (status.ok()) {
        g_task_detail.clear();
        return JOC_OK;
    }
    g_task_detail.assign(status.stage());
    g_task_detail.append(": ");
    g_task_detail.append(status.message());
    return status.code();
}

}  // namespace

extern "C" {

joc_cancel_token* JOC_CALL joc_cancel_token_create(void) {
    return new (std::nothrow) joc_cancel_token();
}

void JOC_CALL joc_cancel_token_request(joc_cancel_token* token) {
    if (token != nullptr) {
        token->requested.store(1u, std::memory_order_relaxed);
    }
}

std::int32_t JOC_CALL joc_cancel_token_is_requested(const joc_cancel_token* token) {
    return (token != nullptr && token->requested.load(std::memory_order_relaxed) != 0u) ? 1 : 0;
}

void JOC_CALL joc_cancel_token_destroy(joc_cancel_token* token) { delete token; }

joc_error JOC_CALL joc_task_validate(const joc_task_config* config,
                                     joc_validation_issue* issues, std::uint32_t capacity,
                                     std::uint32_t* count) {
    if (config == nullptr) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    if (config->struct_size != sizeof(joc_task_config)) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    std::vector<joc_validation_issue> found;
    std::uint32_t errors = 0;
    const joc::Status status = joc::task::validate(*config, &found, &errors);
    if (count != nullptr) {
        *count = static_cast<std::uint32_t>(found.size());
    }
    if (issues != nullptr) {
        for (std::uint32_t i = 0; i < capacity && i < found.size(); ++i) {
            issues[i] = found[i];
        }
    }
    return status.ok() ? JOC_OK : JOC_ERR_INVALID_CONFIG;
}

joc_error JOC_CALL joc_task_execute(const joc_task_config* config, const joc_event_sink* sink,
                                    joc_task_result* out) {
    if (config == nullptr || config->struct_size != sizeof(joc_task_config)) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    if (out != nullptr) {
        std::memset(out, 0, sizeof(*out));
        out->struct_size = sizeof(joc_task_result);
        out->struct_version = JOC_TASK_RESULT_VERSION;
    }
    const joc::Status status = joc::task::run(*config, sink, out);
    if (!status.ok() && out != nullptr && out->status == 0u) {
        out->status = JOC_TASK_FAILED;
        out->error_code = static_cast<std::uint32_t>(status.code());
        std::snprintf(out->error_stage, sizeof(out->error_stage), "%s", status.stage().c_str());
        std::snprintf(out->error_message, sizeof(out->error_message), "%s",
                      status.message().c_str());
    }
    g_task_detail = status.ok() ? std::string() : status.message();
    return status.code();
}

joc_error JOC_CALL joc_task_result_to_json(const joc_task_result* result, char* buffer,
                                           std::size_t capacity, std::size_t* needed) {
    if (result == nullptr) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    std::string json;
    const joc::Status status = joc::task::result_to_json(*result, &json);
    if (!status.ok()) {
        return status.code();
    }
    if (needed != nullptr) {
        *needed = json.size() + 1u;
    }
    if (buffer == nullptr) {
        return JOC_OK;
    }
    if (capacity < json.size() + 1u) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    std::memcpy(buffer, json.c_str(), json.size() + 1u);
    return JOC_OK;
}

}  // extern "C"
