#pragma once

#include <string>

#include "foundation/status.h"
#include "hrtf/jochrtf.h"
#include "hrtf/sofa_field.h"

// Compiled-field cache: the .jochrtf is an internal artifact, so the caller only
// names the SOFA file and the policy.  "memory" keeps the compiled field in this
// process, "disk" additionally reuses (and writes) <cache_dir>/<name>.<key>.jochrtf.
namespace joc::hrtf {

enum class CachePolicy { None, Memory, Disk };

struct SofaFieldRequest {
    std::string sofa_path;
    CompileOptions options;
    CachePolicy policy = CachePolicy::Memory;
    std::string cache_dir;  // required for the disk policy
};

// Parses "none"/"memory"/"disk"; anything else is rejected.
Status parse_cache_policy(const std::string& text, CachePolicy* out);

// Returns the compiled field, reusing a valid cache when the policy allows it.
// `cache_path` (optional) receives the cache file that was read or written.
Status load_or_compile_sofa_field(const SofaFieldRequest& request, Field* out,
                                  std::string* cache_path);

// Writes the field to `path` through a temporary file and an atomic rename.
Status save_jochrtf_atomic(const Field& field, const std::string& path);

// Verifies that a cache file belongs to `source_sha256` and `cache_key`.
Status validate_jochrtf(const std::string& path, const std::string& source_sha256,
                        const std::string& cache_key, Field* out);

}  // namespace joc::hrtf
