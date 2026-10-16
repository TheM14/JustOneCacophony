#include "foundation/status.h"

namespace joc {

// The single definition of the per-thread detail string.  It cannot live in the
// header: a thread_local with dynamic initialisation that is included by several
// translation units makes each of them emit its own TLS initialisation routine,
// and ld64 rejects those as duplicate symbols (GNU ld and MSVC fold them).
thread_local std::string last_error_detail;

}  // namespace joc
