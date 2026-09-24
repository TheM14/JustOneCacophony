#include "hrtf/jochrtf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "foundation/mini_json.h"
#include "foundation/sha256.h"
#include "io/npy.h"
#include "io/zip_reader.h"

namespace joc::hrtf {

namespace {

std::string to_upper(std::string text) {
    for (char& c : text) {
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 'a' + 'A');
        }
    }
    return text;
}

bool is_sha256_hex(const std::string& text) {
    if (text.size() != 64) {
        return false;
    }
    for (const char c : text) {
        const bool digit = c >= '0' && c <= '9';
        const bool upper = c >= 'A' && c <= 'F';
        if (!digit && !upper) {
            return false;
        }
    }
    return true;
}

// json.dumps(list(shape)) as the reference writes it, e.g. "[36, 2, 77]".
std::string shape_json(const std::vector<std::int64_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        text += (i == 0 ? "" : ", ");
        text += std::to_string(shape[i]);
    }
    text += "]";
    return text;
}

Status hrtf_fail(const std::string& message) {
    return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender, message);
}

std::string payload_sha256(const std::vector<double>& centers,
                           const std::vector<double>& coefficients,
                           const std::vector<double>& delay_coefficients,
                           const std::vector<double>& delay_bounds) {
    crypto::Sha256 hash;
    const char prefix[] = "JOC-HRTF-CACHE-PAYLOAD-V1";
    hash.update(prefix, sizeof(prefix) - 1);
    const std::uint8_t zero = 0;
    hash.update(&zero, 1);

    struct Entry {
        const char* name;
        const char* dtype;
        const std::vector<double>* values;
        std::vector<std::int64_t> shape;
    };
    const Entry entries[4] = {
        {"band_center_frequencies_hz", "<f8", &centers, {kHybridBands}},
        {"coefficients", "<c16", &coefficients, {kShTerms, kEars, kHybridBands}},
        {"delay_coefficients", "<f8", &delay_coefficients, {kShTerms, kEars}},
        {"delay_bounds", "<f8", &delay_bounds, {2, 2}},
    };
    for (const Entry& entry : entries) {
        const std::string name(entry.name);
        const std::string dtype(entry.dtype);
        const std::string shape = shape_json(entry.shape);
        hash.update(name.data(), name.size());
        hash.update(&zero, 1);
        hash.update(dtype.data(), dtype.size());
        hash.update(&zero, 1);
        hash.update(shape.data(), shape.size());
        hash.update(&zero, 1);
        hash.update(entry.values->data(), entry.values->size() * sizeof(double));
    }
    return hash.finish_hex();
}

}  // namespace

Status load_jochrtf(const std::string& path, Field* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender, "null field");
    }
    io::ZipArchive archive;
    std::string error;
    if (!archive.open(path, &error)) {
        return Status::fail(JOC_ERR_HRTF_NOT_FOUND, stage::kRender,
                            "cannot read compiled HRTF " + path + ": " + error);
    }

    // Member set must be exactly the five expected names.
    static const char* kMembers[5] = {"metadata_json.npy", "band_center_frequencies_hz.npy",
                                      "coefficients.npy", "delay_coefficients.npy",
                                      "delay_bounds.npy"};
    if (archive.entries().size() != 5u) {
        return hrtf_fail("compiled HRTF cache has an invalid member set (" +
                         std::to_string(archive.entries().size()) + " members)");
    }
    for (const char* name : kMembers) {
        if (archive.find(name) == nullptr) {
            return hrtf_fail(std::string("compiled HRTF cache is missing ") + name);
        }
    }

    auto read_member = [&](const char* name, std::vector<std::uint8_t>* raw,
                           io::NpyArray* array) -> Status {
        if (!archive.read_member(name, raw, &error)) {
            return hrtf_fail(std::string("compiled HRTF member ") + name + ": " + error);
        }
        if (!io::parse_npy(raw->data(), raw->size(), array, &error)) {
            return hrtf_fail(std::string("compiled HRTF member ") + name + ": " + error);
        }
        if (array->fortran_order) {
            return hrtf_fail(std::string("compiled HRTF member must be C-contiguous: ") + name);
        }
        return Status::success();
    };

    std::vector<std::uint8_t> raw;
    io::NpyArray array;

    Status status = read_member("metadata_json.npy", &raw, &array);
    if (!status.ok()) {
        return status;
    }
    std::string metadata_text;
    if (!io::npy_unicode_to_utf8(array, &metadata_text, &error)) {
        return hrtf_fail("compiled HRTF metadata: " + error);
    }
    if (metadata_text.size() > 64u * 1024u) {
        return hrtf_fail("compiled HRTF metadata is too large");
    }

    status = read_member("band_center_frequencies_hz.npy", &raw, &array);
    if (!status.ok()) {
        return status;
    }
    if (array.descr != "<f8" || !io::npy_shape_is(array, {kHybridBands})) {
        return hrtf_fail("band_center_frequencies_hz must be <f8(77,)");
    }
    std::vector<double> centers;
    io::npy_to_double(array, &centers, &error);

    status = read_member("coefficients.npy", &raw, &array);
    if (!status.ok()) {
        return status;
    }
    if (array.descr != "<c16" || !io::npy_shape_is(array, {kShTerms, kEars, kHybridBands})) {
        return hrtf_fail("coefficients must be <c16(36, 2, 77)");
    }
    std::vector<double> coefficients;
    if (!io::npy_to_double(array, &coefficients, &error)) {
        return hrtf_fail("coefficients: " + error);
    }

    status = read_member("delay_coefficients.npy", &raw, &array);
    if (!status.ok()) {
        return status;
    }
    if (array.descr != "<f8" || !io::npy_shape_is(array, {kShTerms, kEars})) {
        return hrtf_fail("delay_coefficients must be <f8(36, 2)");
    }
    std::vector<double> delay_coefficients;
    io::npy_to_double(array, &delay_coefficients, &error);

    status = read_member("delay_bounds.npy", &raw, &array);
    if (!status.ok()) {
        return status;
    }
    if (array.descr != "<f8" || !io::npy_shape_is(array, {2, 2})) {
        return hrtf_fail("delay_bounds must be <f8(2, 2)");
    }
    std::vector<double> delay_bounds;
    io::npy_to_double(array, &delay_bounds, &error);

    std::vector<json::Member> members;
    if (!json::parse_object(metadata_text, &members, &error)) {
        return hrtf_fail("compiled HRTF metadata: " + error);
    }
    auto require_string = [&](const char* key, std::string* value) -> Status {
        const json::Member* member = json::find(members, key);
        if (member == nullptr || !json::as_string(*member, value)) {
            return hrtf_fail(std::string("compiled HRTF metadata is missing ") + key);
        }
        return Status::success();
    };
    std::string magic;
    std::string schema;
    std::string source_sha256;
    std::string cache_key;
    std::string payload_hash;
    std::string delay_source;
    status = require_string("magic", &magic);
    if (!status.ok()) { return status; }
    status = require_string("cache_schema", &schema);
    if (!status.ok()) { return status; }
    status = require_string("source_sha256", &source_sha256);
    if (!status.ok()) { return status; }
    status = require_string("cache_key", &cache_key);
    if (!status.ok()) { return status; }
    status = require_string("payload_sha256", &payload_hash);
    if (!status.ok()) { return status; }
    status = require_string("delay_source", &delay_source);
    if (!status.ok()) { return status; }

    if (magic != kMagic) {
        return hrtf_fail("compiled HRTF magic mismatch: " + magic);
    }
    if (schema != kCacheSchema) {
        return hrtf_fail("compiled HRTF cache schema mismatch: " + schema);
    }
    const json::Member* version_member = json::find(members, "format_version");
    long long version = -1;
    if (version_member == nullptr || !json::as_integer(*version_member, &version)) {
        return hrtf_fail("compiled HRTF metadata is missing format_version");
    }
    if (version != kFormatVersion) {
        return Status::fail(JOC_ERR_HRTF_VERSION, stage::kRender,
                            "unsupported .jochrtf version " + std::to_string(version) +
                                "; rebuild it from the source SOFA");
    }
    out->source_sha256 = to_upper(source_sha256);
    out->cache_key = to_upper(cache_key);
    if (!is_sha256_hex(out->source_sha256)) {
        return hrtf_fail("compiled HRTF source_sha256 is not a 64-digit digest");
    }
    if (!is_sha256_hex(out->cache_key)) {
        return hrtf_fail("compiled HRTF cache_key is not a 64-digit digest");
    }

    const std::string expected = payload_sha256(centers, coefficients, delay_coefficients,
                                                delay_bounds);
    if (to_upper(payload_hash) != to_upper(expected)) {
        return Status::fail(JOC_ERR_HRTF_HASH, stage::kRender,
                            "compiled HRTF payload hash mismatch");
    }
    out->payload_sha256 = to_upper(payload_hash);

    const json::Member* radius_member = json::find(members, "measurement_radius_m");
    double radius = 0.0;
    if (radius_member == nullptr || !json::as_number(*radius_member, &radius) || radius <= 0.0) {
        return hrtf_fail("compiled HRTF measurement_radius_m must be a positive number");
    }
    out->measurement_radius_m = radius;
    const json::Member* order_member = json::find(members, "order");
    long long order = 0;
    if (order_member == nullptr || !json::as_integer(*order_member, &order) || order <= 0 ||
        order * order > kShTerms) {
        return hrtf_fail("compiled HRTF order is out of range");
    }
    out->order = order;

    for (const double value : coefficients) {
        if (!std::isfinite(value)) {
            return hrtf_fail("compiled HRTF coefficients contain non-finite values");
        }
    }
    for (const double value : delay_coefficients) {
        if (!std::isfinite(value) || std::abs(value) > 48000.0 * 64.0) {
            return hrtf_fail("compiled HRTF delay coefficients are out of range");
        }
    }
    for (const double value : delay_bounds) {
        if (!std::isfinite(value)) {
            return hrtf_fail("compiled HRTF delay bounds contain non-finite values");
        }
    }
    if (delay_bounds.size() == 4u && delay_bounds[0] > delay_bounds[1]) {
        return hrtf_fail("compiled HRTF delay bounds are inverted");
    }

    if (const json::Member* member = json::find(members, "compiler_version")) {
        json::as_string(*member, &out->compiler_version);
    }
    if (const json::Member* member = json::find(members, "phase_policy_version")) {
        json::as_string(*member, &out->phase_policy_version);
    }
    if (const json::Member* member = json::find(members, "sh_convention")) {
        json::as_string(*member, &out->sh_convention);
    }
    if (const json::Member* member = json::find(members, "source_display_name")) {
        json::as_string(*member, &out->source_display_name);
    }
    if (const json::Member* member = json::find(members, "projection_ridge")) {
        json::as_number(*member, &out->projection_ridge);
    }
    if (const json::Member* member = json::find(members, "spherical_harmonic_ridge")) {
        json::as_number(*member, &out->spherical_harmonic_ridge);
    }
    if (const json::Member* member = json::find(members, "fit_report")) {
        out->fit_report_json = member->raw;
    }
    if (const json::Member* member = json::find(members, "filterbank")) {
        out->filterbank_json = member->raw;
    }
    out->delay_source = delay_source;
    out->metadata_json = metadata_text;
    out->coefficients = std::move(coefficients);
    out->delay_coefficients = std::move(delay_coefficients);
    out->delay_bounds = std::move(delay_bounds);
    out->band_centers_hz = std::move(centers);
    return Status::success();
}

Status load_kernels(const std::string& npz_path, Kernels* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender, "null kernels");
    }
    io::ZipArchive archive;
    std::string error;
    if (!archive.open(npz_path, &error)) {
        return Status::fail(JOC_ERR_HRTF_NOT_FOUND, stage::kRender,
                            "cannot read kernel tables " + npz_path + ": " + error);
    }

    struct Request {
        const char* member;
        const char* shape_text;
        std::vector<std::int64_t> shape;
    };
    const Request requests[6] = {
        {"qmf_analysis_coefficients.npy", "<f4", {64, 10}},
        {"hybrid_analysis_low_kernel.npy", "<f4", {3, 2, 13, 16, 2}},
        {"hybrid_synthesis_indices.npy", "<i2", {154, 4}},
        {"hybrid_synthesis_values.npy", "<f4", {154}},
        {"qmf_synthesis_basis.npy", "<f8", {64, 4, 128}},
        {"qmf_synthesis_taps.npy", "<f8", {64, 10, 4}},
    };

    std::vector<std::uint8_t> raw;
    std::vector<std::uint8_t> ordered;
    for (const Request& request : requests) {
        const std::string name = request.member;
        if (!archive.read_member(name, &raw, &error)) {
            return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                                "kernel table member " + name + ": " + error);
        }
        io::NpyArray array;
        if (!io::parse_npy(raw.data(), raw.size(), &array, &error)) {
            return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                                "kernel table member " + name + ": " + error);
        }
        if (array.descr != request.shape_text || !io::npy_shape_is(array, request.shape)) {
            return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                                "kernel table member " + name + " has an unexpected dtype/shape");
        }
        // Logical C order: required because the reused kernel indexes the hybrid
        // synthesis table row-major while the shipped member is Fortran-order.
        if (!io::npy_to_c_order(array, &ordered, &error)) {
            return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                                "kernel table member " + name + ": " + error);
        }
        const std::size_t count = array.element_count();
        if (std::strcmp(request.member, "qmf_analysis_coefficients.npy") == 0) {
            std::vector<float> values(count);
            std::memcpy(values.data(), ordered.data(), count * sizeof(float));
            out->qmf_analysis.assign(values.begin(), values.end());
        } else if (std::strcmp(request.member, "hybrid_analysis_low_kernel.npy") == 0) {
            std::vector<float> values(count);
            std::memcpy(values.data(), ordered.data(), count * sizeof(float));
            out->hybrid_low.assign(values.begin(), values.end());
        } else if (std::strcmp(request.member, "hybrid_synthesis_indices.npy") == 0) {
            out->hybrid_indices.resize(count);
            std::memcpy(out->hybrid_indices.data(), ordered.data(), count * sizeof(std::int16_t));
        } else if (std::strcmp(request.member, "hybrid_synthesis_values.npy") == 0) {
            std::vector<float> values(count);
            std::memcpy(values.data(), ordered.data(), count * sizeof(float));
            out->hybrid_values.assign(values.begin(), values.end());
        } else if (std::strcmp(request.member, "qmf_synthesis_basis.npy") == 0) {
            std::memcpy(out->qmf_basis.empty() ? (out->qmf_basis.resize(count), out->qmf_basis.data())
                                               : out->qmf_basis.data(),
                        ordered.data(), count * sizeof(double));
            out->qmf_basis.resize(count);
        } else {
            out->qmf_taps.resize(count);
            std::memcpy(out->qmf_taps.data(), ordered.data(), count * sizeof(double));
        }
    }
    out->hybrid_count = static_cast<std::uint32_t>(out->hybrid_values.size());
    if (out->hybrid_count == 0u) {
        return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                            "kernel tables contain no hybrid synthesis entries");
    }
    return Status::success();
}

}  // namespace joc::hrtf
