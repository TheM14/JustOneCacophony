// The same headers under C++: aggregate initialisation, enum values and the
// function declarations must all be usable from a C++ translation unit.
#include "joc_core.h"
#include "joc_stream.h"

namespace {

int probe() {
    joc_task_config task{};
    task.struct_size = sizeof(task);
    task.operation = JOC_OP_BINAURAL;
    task.binaural_mode = JOC_BINAURAL_MID;
    joc_stream_config stream{};
    stream.struct_size = sizeof(stream);
    stream.input = JOC_STREAM_IN_CORE_PCM;
    stream.output = JOC_STREAM_OUT_SPEAKER;
    joc_stream_buffer buffer{};
    buffer.struct_size = sizeof(buffer);
    joc_stream_status_info status{};
    status.struct_size = sizeof(status);
    joc_error (*create)(const joc_stream_config*, joc_stream**) = &joc_stream_create;
    joc_error (*execute)(const joc_task_config*, const joc_event_sink*, joc_task_result*) =
        &joc_task_execute;
    return static_cast<int>(task.operation) + static_cast<int>(stream.output) +
           static_cast<int>(JOC_TIMESLOTS) + (create != nullptr ? 1 : 0) +
           (execute != nullptr ? 1 : 0);
}

}  // namespace

int joc_header_cpp_probe() { return probe(); }
