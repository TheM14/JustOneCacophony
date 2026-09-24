/* The public headers must stay valid C: this translation unit is compiled as C
 * and never linked, so it fails the build if a header stops being C-compatible. */
#include "joc_core.h"
#include "joc_stream.h"

int joc_header_c_probe(void) {
    joc_task_config task = {0};
    joc_stream_config stream = {0};
    joc_event_sink sink = {0};
    joc_event event = {0};
    joc_task_result result = {0};
    joc_emdf_info emdf = {0};
    joc_frame_params params = {0};
    (void)task;
    (void)stream;
    (void)sink;
    (void)event;
    (void)result;
    (void)emdf;
    (void)params;
    return (int)JOC_FRAME_SAMPLES + (int)JOC_STREAM_IN_CORE_PCM;
}
