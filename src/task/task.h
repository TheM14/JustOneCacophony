
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"
#include "joc_core.h"

namespace joc::task {

Status validate(const joc_task_config& raw, std::vector<joc_validation_issue>* issues,
                std::uint32_t* error_count);

// Runs the task on the calling thread.  Never throws; failures come back as the
Status run(const joc_task_config& raw, const joc_event_sink* sink, joc_task_result* out);

// Serialises the stable subset of the result (plan 22.2 / 33.5).
Status result_to_json(const joc_task_result& result, std::string* out);

}  // namespace joc::task
