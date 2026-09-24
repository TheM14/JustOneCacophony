
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"

namespace joc::io {

struct ProcessResult {
    std::uint32_t exit_code = 0;
    std::string output;
};

Status run_process(const std::vector<std::string>& argv, ProcessResult* out);

}  // namespace joc::io
