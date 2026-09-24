#include "hrtf/sofa_cache.h"

#include <algorithm>
#include <filesystem>
#include <list>
#include <mutex>
#include <utility>
#include <vector>

#include "foundation/fs_utf8.h"
#include "hrtf/sofa.h"

namespace joc::hrtf {

namespace {

namespace fs = std::filesystem;

// Small process-local cache: the reference keeps the last eight compiled fields.
constexpr std::size_t kMemoryCacheEntries = 8;

struct MemoryEntry {
    std::string key;
    Field field;
};

std::mutex& memory_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::list<MemoryEntry>& memory_cache() {
    static std::list<MemoryEntry> cache;
    return cache;
}

bool memory_cache_get(const std::string& key, Field* out) {
    std::lock_guard<std::mutex> lock(memory_mutex());
    std::list<MemoryEntry>& cache = memory_cache();
    for (auto entry = cache.begin(); entry != cache.end(); ++entry) {
        if (entry->key == key) {
            *out = entry->field;
            cache.splice(cache.begin(), cache, entry);
            return true;
        }
    }
    return false;
}

void memory_cache_put(const std::string& key, const Field& field) {
    std::lock_guard<std::mutex> lock(memory_mutex());
    std::list<MemoryEntry>& cache = memory_cache();
    for (auto entry = cache.begin(); entry != cache.end(); ++entry) {
        if (entry->key == key) {
            entry->field = field;
            cache.splice(cache.begin(), cache, entry);
            return;
        }
    }
    cache.push_front(MemoryEntry{key, field});
    while (cache.size() > kMemoryCacheEntries) {
        cache.pop_back();
    }
}

Status cache_fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kRender, message);
}

std::string upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return text;
}

}  // namespace

Status parse_cache_policy(const std::string& text, CachePolicy* out) {
    if (out == nullptr) {
        return cache_fail(JOC_ERR_INVALID_ARGUMENT, "null cache policy");
    }
    if (text == "none") {
        *out = CachePolicy::None;
        return Status::success();
    }
    if (text == "memory") {
        *out = CachePolicy::Memory;
        return Status::success();
    }
    if (text == "disk") {
        *out = CachePolicy::Disk;
        return Status::success();
    }
    return cache_fail(JOC_ERR_INVALID_CONFIG, "cache_policy must be none, memory, or disk");
}

Status validate_jochrtf(const std::string& path, const std::string& source_sha256,
                        const std::string& cache_key, Field* out) {
    Field field;
    const Status status = load_jochrtf(path, &field);
    if (!status.ok()) {
        return status;
    }
    if (!source_sha256.empty() && field.source_sha256 != upper(source_sha256)) {
        return cache_fail(JOC_ERR_HRTF_HASH, "compiled HRTF source hash mismatch");
    }
    if (!cache_key.empty() && field.cache_key != upper(cache_key)) {
        return cache_fail(JOC_ERR_HRTF_HASH, "compiled HRTF configuration hash mismatch");
    }
    if (out != nullptr) {
        *out = std::move(field);
    }
    return Status::success();
}

Status save_jochrtf_atomic(const Field& field, const std::string& path) {
    if (path.empty()) {
        return cache_fail(JOC_ERR_INVALID_ARGUMENT, "empty compiled HRTF cache path");
    }
    const fs::path target = fs_utf8::to_path(path);
    std::error_code error;
    if (target.has_parent_path()) {
        fs::create_directories(target.parent_path(), error);
        if (error) {
            return cache_fail(JOC_ERR_OUTPUT_OPEN,
                              "cannot create " + fs_utf8::from_path(target.parent_path()));
        }
    }
    const std::string temporary = path + ".tmp";
    Status status = write_jochrtf(field, temporary);
    if (!status.ok()) {
        return status;
    }
    // The rename is what makes a half-written cache impossible to observe.
    fs::rename(fs_utf8::to_path(temporary), target, error);
    if (error) {
        fs::remove(fs_utf8::to_path(temporary), error);
        return cache_fail(JOC_ERR_OUTPUT_WRITE, "cannot replace " + path);
    }
    return Status::success();
}

Status load_or_compile_sofa_field(const SofaFieldRequest& request, Field* out,
                                  std::string* cache_path) {
    if (out == nullptr) {
        return cache_fail(JOC_ERR_INVALID_ARGUMENT, "null compiled HRTF destination");
    }
    if (request.sofa_path.empty()) {
        return cache_fail(JOC_ERR_INVALID_ARGUMENT, "no SOFA path for the compiled HRTF field");
    }
    if (request.policy == CachePolicy::Disk && request.cache_dir.empty()) {
        return cache_fail(JOC_ERR_INVALID_CONFIG, "the disk cache policy needs a cache directory");
    }
    SofaHrir sofa;
    Status status = load_sofa(request.sofa_path, &sofa);
    if (!status.ok()) {
        return status;
    }
    CanonicalHrtf canonical;
    status = canonicalize_sofa(sofa, &canonical);
    if (!status.ok()) {
        return status;
    }
    // The key depends on the shell that the radius selects, exactly as upstream.
    double actual_radius = request.options.shell_radius_m;
    (void)canonical_shell_indices(canonical, request.options.shell_radius_m, &actual_radius);
    const std::string key = compiled_hrtf_cache_key(
        canonical.source_sha256, canonical.sample_rate_hz, actual_radius, request.options.order,
        request.options.projection_ridge, request.options.sh_ridge);
    if (cache_path != nullptr) {
        cache_path->clear();
    }

    std::string target;
    if (request.policy == CachePolicy::Disk) {
        target = request.cache_dir;
        if (!target.empty() && target.back() != '/' && target.back() != '\\') {
            target += "/";
        }
        target += cache_file_name(canonical.source_path.empty()
                                      ? std::string()
                                      : canonical.source_path,
                                  key);
        if (fs_utf8::exists(target)) {
            Field cached_field;
            const Status cached =
                validate_jochrtf(target, canonical.source_sha256, key, &cached_field);
            if (cached.ok()) {
                memory_cache_put(key, cached_field);
                *out = std::move(cached_field);
                if (cache_path != nullptr) {
                    *cache_path = target;
                }
                return Status::success();
            }
        }
    }
    Field field;
    if (request.policy != CachePolicy::None && memory_cache_get(key, &field)) {
        // A memory hit still materialises the disk cache the caller asked for.
        if (request.policy == CachePolicy::Disk) {
            status = save_jochrtf_atomic(field, target);
            if (!status.ok()) {
                return status;
            }
            if (cache_path != nullptr) {
                *cache_path = target;
            }
        }
        *out = std::move(field);
        return Status::success();
    }
    status = compile_canonical_field(canonical, request.options, &field);
    if (!status.ok()) {
        return status;
    }
    if (field.cache_key != key) {
        return cache_fail(JOC_ERR_INTERNAL, "internal compiled HRTF cache-key mismatch");
    }
    if (request.policy == CachePolicy::Disk) {
        status = save_jochrtf_atomic(field, target);
        if (!status.ok()) {
            return status;
        }
        if (cache_path != nullptr) {
            *cache_path = target;
        }
    }
    if (request.policy != CachePolicy::None) {
        memory_cache_put(key, field);
    }
    *out = std::move(field);
    return Status::success();
}

}  // namespace joc::hrtf
