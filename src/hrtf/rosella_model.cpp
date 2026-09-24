#include "hrtf/rosella_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "foundation/fs_utf8.h"
#include "foundation/mini_json.h"
#include "foundation/sha256.h"

namespace joc::hrtf {

namespace {

// The model's fixed-point lane scale: every stored value is a Q15 integer.
constexpr float kQ15 = 1.0f / 32768.0f;

Status model_fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kRender, message);
}

float q15(std::int32_t value) { return static_cast<float>(value) * kQ15; }

float q15_exp(std::int32_t value, int exponent) {
    return q15(value) * static_cast<float>(std::ldexp(1.0, exponent));
}

std::uint16_t low16(std::int32_t value) {
    return static_cast<std::uint16_t>(static_cast<std::uint32_t>(value) & 0xFFFFu);
}

std::string trim(const std::string& text) {
    const std::size_t begin = text.find_first_not_of(" \t\r\n");
    const std::size_t end = text.find_last_not_of(" \t\r\n");
    return begin == std::string::npos ? std::string() : text.substr(begin, end - begin + 1u);
}

// The lane array is read straight out of the JSON text: it is one flat list of
// integers, and building a 15691-node DOM for it would only cost time.
bool parse_int_array(const std::string& raw, std::vector<std::int32_t>* out, std::string* error) {
    out->clear();
    const char* cursor = raw.c_str();
    const char* end = cursor + raw.size();
    while (cursor < end && *cursor != '[') {
        ++cursor;
    }
    if (cursor == end) {
        *error = "rosella_coefficients must be a JSON array";
        return false;
    }
    ++cursor;
    while (cursor < end) {
        while (cursor < end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
                                *cursor == '\n' || *cursor == ',')) {
            ++cursor;
        }
        if (cursor >= end) {
            break;
        }
        if (*cursor == ']') {
            return true;
        }
        const bool negative = *cursor == '-';
        if (negative) {
            ++cursor;
        }
        if (cursor >= end || *cursor < '0' || *cursor > '9') {
            *error = "rosella_coefficients contains a non-integer value";
            return false;
        }
        long long value = 0;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') {
            value = value * 10 + (*cursor - '0');
            if (value > (1ll << 40)) {
                *error = "rosella_coefficients value is out of range";
                return false;
            }
            ++cursor;
        }
        // A fractional part or an exponent means the value is not an exact integer.
        if (cursor < end && (*cursor == '.' || *cursor == 'e' || *cursor == 'E')) {
            *error = "rosella_coefficients contains a non-integer value";
            return false;
        }
        if (negative) {
            value = -value;
        }
        if (value < -(1ll << 31) || value > (1ll << 31) - 1) {
            *error = "rosella_coefficients value is outside signed int32";
            return false;
        }
        out->push_back(static_cast<std::int32_t>(value));
    }
    *error = "rosella_coefficients array is truncated";
    return false;
}

struct RpHeader {
    std::uint16_t stored_checksum = 0;
    std::uint16_t computed_checksum = 0;
    bool checksum_valid = false;
    bool table_a_present = false;
    bool table_b_present = false;
    bool table_c_present = false;
    int table_a_dimension = 0;
    int table_a_option = 0;
    int table_a_extra = 0;
    int table_b_dimension = 0;
    int table_b_extra = 0;
    int table_b_groups = 0;
    int table_c_dimension = 0;
    std::size_t active_lanes = 0;
};

Status inspect_rp(const std::vector<std::int32_t>& lanes, RpHeader* out) {
    if (lanes.size() < 5u) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "Rosella rp must contain whole int32 lanes");
    }
    if (low16(lanes[0]) != 0x7072u) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "bad Rosella rp magic");
    }
    out->stored_checksum = low16(lanes[1]);
    out->table_a_present = low16(lanes[2]) != 0u;
    out->table_b_present = low16(lanes[3]) != 0u;
    out->table_c_present = low16(lanes[4]) != 0u;
    std::size_t index = 5u;
    if (out->table_a_present) {
        out->table_a_dimension = low16(lanes[index]);
        out->table_a_option = low16(lanes[index + 1u]);
        out->table_a_extra = low16(lanes[index + 2u]);
        index += 5u;
    } else {
        out->table_a_dimension = 77;
    }
    if (out->table_b_present) {
        if (!out->table_a_present) {
            return model_fail(JOC_ERR_HRTF_FORMAT,
                              "Rosella rp table B cannot be present without table A");
        }
        out->table_b_dimension = low16(lanes[index]);
        out->table_b_extra = low16(lanes[index + 1u]);
        out->table_b_groups = low16(lanes[index + 2u]);
        index += 3u;
    }
    if (out->table_c_present) {
        out->table_c_dimension = low16(lanes[index]);
        index += 1u;
    }
    const long long payload_words =
        static_cast<long long>(index) - 2 +
        (out->table_b_present ? (out->table_b_dimension + 380 * out->table_b_groups +
                                 out->table_b_extra + 79)
                              : 0) +
        (out->table_a_present ? (171 * out->table_a_extra + 79 +
                                 2 * (out->table_a_option + 14 * out->table_a_dimension))
                              : 0) +
        11 + (out->table_c_present ? (314 * out->table_c_dimension + 1) : 0);
    if (payload_words < 0) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "malformed Rosella rp header");
    }
    out->active_lanes = static_cast<std::size_t>(2 + payload_words);
    if (lanes.size() < out->active_lanes) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "Rosella rp is truncated");
    }
    std::uint32_t computed = 0xA569u;
    for (std::size_t lane = 2u; lane < out->active_lanes; ++lane) {
        computed ^= low16(lanes[lane]);
    }
    out->computed_checksum = static_cast<std::uint16_t>(computed & 0xFFFFu);
    out->checksum_valid = out->computed_checksum == out->stored_checksum;
    return Status::success();
}

// _unpack_field: the serialized 154-per-direction field lanes to the padded grid.
void unpack_field(const std::int32_t* serialized, int directions, int exponent,
                  std::vector<float>* padded) {
    padded->assign(static_cast<std::size_t>(160 * directions), 0.0f);
    const int stride8 = 8 * directions;
    const int stride2 = 2 * directions;
    for (int source = 0; source < 154 * directions; ++source) {
        const int group4 = (source % stride8) / stride2;
        const int destination = (group4 & 3) + 4 * (source % stride2 +
                                                    2 * directions * (source / stride8 +
                                                                      (group4 >> 2)));
        (*padded)[static_cast<std::size_t>(destination)] =
            q15_exp(serialized[source], exponent);
    }
}

// _unpack_table_a_grid: the serialized table-A rows to the padded lane grid.
void unpack_table_a_grid(const std::int32_t* serialized, int dimension, int serialized_rows,
                         int padded_rows, int lane_group, std::vector<float>* padded) {
    padded->assign(static_cast<std::size_t>(padded_rows) * static_cast<std::size_t>(dimension),
                   0.0f);
    const int group_width = lane_group * 4;
    for (int source = 0; source < serialized_rows * dimension; ++source) {
        const int remainder = source % group_width;
        const int destination = (remainder / lane_group) +
                                4 * (remainder % lane_group +
                                     group_width / 4 * (source / group_width));
        (*padded)[static_cast<std::size_t>(destination)] = q15(serialized[source]);
    }
}

void unpack_table_a_extra(const std::int32_t* serialized, std::vector<float>* padded) {
    padded->assign(160u, 0.0f);
    for (int source = 0; source < 154; ++source) {
        const int remainder = source & 7;
        const int destination = (remainder >> 1) + 4 * ((source & 1) + 2 * (source >> 3));
        (*padded)[static_cast<std::size_t>(destination)] = q15(serialized[source]);
    }
}

}  // namespace

std::string RosellaModel::summary() const {
    std::string name = capture.name.empty() ? std::string("unnamed") : capture.name;
    return "Rosella personalized_headphone '" + name + "' (" +
           (room_model.empty() ? std::string("unknown room") : room_model) + "), " +
           std::to_string(table_a_dimension) + " HQMF / 77 hybrid @ " +
           std::to_string(sample_rate) + " Hz";
}

Status load_personalized_headphone(const std::string& path, RosellaModel* out) {
    if (out == nullptr) {
        return model_fail(JOC_ERR_INVALID_ARGUMENT, "null Rosella model destination");
    }
    if (!fs_utf8::exists(path)) {
        return model_fail(JOC_ERR_HRTF_NOT_FOUND, "personalized headphone model not found: " + path);
    }
    std::ifstream stream = fs_utf8::open_input(path);
    if (!stream.good()) {
        return model_fail(JOC_ERR_IO, "cannot open " + path);
    }
    std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (text.empty()) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "empty personalized headphone model: " + path);
    }
    // The checksum is taken over the coefficient lanes, exactly as upstream hashes
    // the int32 image of the array.
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || text[first] != '{') {
        return model_fail(JOC_ERR_NOT_SUPPORTED,
                          "raw rp models are not supported; use a .personalized_headphone JSON");
    }
    std::vector<json::Member> root;
    std::string error;
    if (!json::parse_object(text, &root, &error)) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "invalid personalized headphone JSON: " + error);
    }
    const json::Member* personalized = json::find(root, "personalized_hrtf");
    if (personalized == nullptr) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "personalized_hrtf is missing");
    }
    std::vector<json::Member> inner;
    if (!json::parse_object(personalized->raw, &inner, &error)) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "invalid personalized_hrtf object: " + error);
    }
    const json::Member* virtualizer = json::find(inner, "virtualizer_parameters");
    if (virtualizer == nullptr) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "virtualizer_parameters is missing");
    }
    std::vector<json::Member> parameters;
    if (!json::parse_object(virtualizer->raw, &parameters, &error)) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "invalid virtualizer_parameters: " + error);
    }
    const json::Member* coefficient_member = json::find(parameters, "rosella_coefficients");
    if (coefficient_member == nullptr) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "rosella_coefficients is missing");
    }
    RosellaModel model;
    model.source_path = path;
    if (const json::Member* member = json::find(parameters, "rosella_coefficients_version")) {
        json::as_string(*member, &model.coefficient_version);
    }
    if (const json::Member* member = json::find(parameters, "room_model")) {
        json::as_string(*member, &model.room_model);
    }
    if (const json::Member* capture = json::find(inner, "phrtf_capture_metadata")) {
        std::vector<json::Member> fields;
        if (json::parse_object(capture->raw, &fields, &error)) {
            const std::pair<const char*, std::string*> mapping[] = {
                {"capture_submission_date", &model.capture.capture_submission_date},
                {"capture_type", &model.capture.capture_type},
                {"label", &model.capture.label},
                {"name", &model.capture.name},
                {"phrtf_algorithm_version", &model.capture.algorithm_version},
                {"phrtf_creation_date", &model.capture.creation_date},
                {"uuid", &model.capture.uuid},
                {"version", &model.capture.version},
            };
            for (const auto& entry : mapping) {
                if (const json::Member* member = json::find(fields, entry.first)) {
                    json::as_string(*member, entry.second);
                }
            }
        }
    }
    std::vector<std::int32_t> lanes;
    if (!parse_int_array(coefficient_member->raw, &lanes, &error)) {
        return model_fail(JOC_ERR_HRTF_FORMAT, error);
    }
    {
        crypto::Sha256 hash;
        hash.update(lanes.data(), lanes.size() * sizeof(std::int32_t));
        model.coefficient_sha256 = hash.finish_hex();
    }

    RpHeader header;
    Status status = inspect_rp(lanes, &header);
    if (!status.ok()) {
        return status;
    }
    if (!header.checksum_valid || header.active_lanes != lanes.size()) {
        return model_fail(JOC_ERR_HRTF_FORMAT,
                          "invalid or non-active Rosella rp coefficient sequence");
    }
    if (!header.table_a_present || !header.table_b_present || header.table_c_present) {
        return model_fail(JOC_ERR_NOT_SUPPORTED,
                          "the renderer requires table A+B and no table C");
    }
    if (header.table_a_dimension != 64 || header.table_a_option != 3) {
        return model_fail(JOC_ERR_NOT_SUPPORTED,
                          "the renderer requires the observed 64-channel HQMF layout");
    }
    if (header.table_b_dimension != 20 || header.table_b_groups != 36) {
        return model_fail(JOC_ERR_NOT_SUPPORTED,
                          "the renderer requires 20 hybrid groups and 36 direction terms");
    }

    const std::int32_t* values = lanes.data();
    const std::size_t total = lanes.size();
    std::size_t position = 13u;
    const int extra = header.table_a_extra;
    model.table_a_dimension = header.table_a_dimension;
    model.table_a_option = header.table_a_option;
    model.table_a_extra = extra;
    model.table_a_header_field = low16(values[8]);
    model.table_a_header_25 = low16(values[9]);
    model.table_a_control = low16(values[position]);
    model.field_exponent = values[position];
    position += 1u;
    const int option_count = header.table_a_option;
    model.table_a_option_ids.resize(static_cast<std::size_t>(option_count));
    for (int index = 0; index < option_count; ++index) {
        model.table_a_option_ids[static_cast<std::size_t>(index)] =
            low16(values[position + static_cast<std::size_t>(index)]);
    }
    position += static_cast<std::size_t>(option_count);
    model.table_a_option_values.resize(static_cast<std::size_t>(option_count));
    for (int index = 0; index < option_count; ++index) {
        model.table_a_option_values[static_cast<std::size_t>(index)] =
            q15(values[position + static_cast<std::size_t>(index)]);
    }
    position += static_cast<std::size_t>(option_count);
    model.table_a_scalar = q15(values[position]);
    position += 1u;
    const int dimension = header.table_a_dimension;
    unpack_table_a_grid(values + position, dimension, 16, 20, 16,
                        &model.table_a_filter_16x64_padded);
    position += static_cast<std::size_t>(16 * dimension);
    for (int index = 0; index < 4; ++index) {
        model.table_a_four_integers[static_cast<std::size_t>(index)] =
            low16(values[position + static_cast<std::size_t>(index)]);
    }
    position += 4u;
    model.table_a_integer = low16(values[position]);
    position += 1u;
    unpack_table_a_grid(values + position, dimension, 8, 10, 8,
                        &model.table_a_filter_8x64_padded);
    position += static_cast<std::size_t>(8 * dimension);
    model.table_a_vector16.resize(16u);
    for (int index = 0; index < 16; ++index) {
        model.table_a_vector16[static_cast<std::size_t>(index)] =
            q15(values[position + static_cast<std::size_t>(index)]);
    }
    position += 16u;
    unpack_table_a_grid(values + position, dimension, 4, 5, 4,
                        &model.table_a_filter_4x64_padded);
    position += static_cast<std::size_t>(4 * dimension);
    model.table_a_extra_indices.resize(static_cast<std::size_t>(extra));
    for (int index = 0; index < extra; ++index) {
        model.table_a_extra_indices[static_cast<std::size_t>(index)] =
            low16(values[position + static_cast<std::size_t>(index)]);
    }
    position += static_cast<std::size_t>(extra);
    model.table_a_extra_fields_padded.assign(static_cast<std::size_t>(extra) * 160u, 0.0f);
    std::vector<float> unpacked;
    for (int index = 0; index < extra; ++index) {
        unpack_table_a_extra(values + position, &unpacked);
        std::copy(unpacked.begin(), unpacked.end(),
                  model.table_a_extra_fields_padded.begin() + static_cast<std::ptrdiff_t>(index) * 160);
        position += 154u;
    }
    model.table_a_extra_vectors.assign(static_cast<std::size_t>(extra) * 16u, 0.0f);
    for (int index = 0; index < extra; ++index) {
        for (int lane = 0; lane < 16; ++lane) {
            model.table_a_extra_vectors[static_cast<std::size_t>(index) * 16u +
                                        static_cast<std::size_t>(lane)] =
                q15(values[position + static_cast<std::size_t>(lane)]);
        }
        position += 16u;
    }
    const std::size_t table_b_start = position;
    if (table_b_start != 13u + 1821u + static_cast<std::size_t>(171 * extra)) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "Rosella table-A parser lost its place");
    }

    model.sample_rate = 2 * low16(values[position]);
    position += 1u;
    model.matrix_exponent = values[position];
    position += 1u;
    const std::size_t matrix_count = 36u * 36u;
    model.matrix_left.resize(matrix_count);
    model.matrix_right.resize(matrix_count);
    for (std::size_t index = 0; index < matrix_count; ++index) {
        model.matrix_left[index] = q15_exp(values[position + index], model.matrix_exponent);
    }
    position += matrix_count;
    for (std::size_t index = 0; index < matrix_count; ++index) {
        model.matrix_right[index] = q15_exp(values[position + index], model.matrix_exponent);
    }
    position += matrix_count;
    model.vector_left.resize(36u);
    model.vector_right.resize(36u);
    for (int index = 0; index < 36; ++index) {
        model.vector_left[static_cast<std::size_t>(index)] =
            q15_exp(values[position + static_cast<std::size_t>(index)], model.matrix_exponent);
    }
    position += 36u;
    for (int index = 0; index < 36; ++index) {
        model.vector_right[static_cast<std::size_t>(index)] =
            q15_exp(values[position + static_cast<std::size_t>(index)], model.matrix_exponent);
    }
    position += 36u;

    const std::size_t serialized_count = 154u * 36u;
    unpack_field(values + position, 36, model.field_exponent, &model.field_left_padded);
    bool odd_zero = true;
    for (std::size_t index = 1u; index < serialized_count; index += 2u) {
        const float value = q15_exp(values[position + index], model.field_exponent);
        if (std::abs(value) > 1.0e-6f) {
            odd_zero = false;
            break;
        }
    }
    model.field_left_odd_serialized_zero = odd_zero;
    position += serialized_count;
    unpack_field(values + position, 36, model.field_exponent, &model.field_right_padded);
    position += serialized_count;

    model.hybrid_flags.resize(20u);
    int active_hybrid = 0;
    for (int index = 0; index < 20; ++index) {
        model.hybrid_flags[static_cast<std::size_t>(index)] =
            low16(values[position + static_cast<std::size_t>(index)]);
        if (model.hybrid_flags[static_cast<std::size_t>(index)] == 1) {
            ++active_hybrid;
        }
    }
    position += 20u;
    if (active_hybrid != header.table_b_extra) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "hybrid value count does not match the header");
    }
    model.hybrid_values.resize(static_cast<std::size_t>(active_hybrid));
    for (int index = 0; index < active_hybrid; ++index) {
        model.hybrid_values[static_cast<std::size_t>(index)] =
            q15(values[position + static_cast<std::size_t>(index)]);
    }
    position += static_cast<std::size_t>(active_hybrid);
    model.model_scalars.resize(5u);
    for (int index = 0; index < 5; ++index) {
        model.model_scalars[static_cast<std::size_t>(index)] =
            q15(values[position + static_cast<std::size_t>(index)]);
    }
    position += 5u;
    const std::size_t expected_tail =
        table_b_start + static_cast<std::size_t>(header.table_b_dimension +
                                                 380 * header.table_b_groups +
                                                 header.table_b_extra + 79);
    if (position != expected_tail) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "Rosella table-B parser lost its place");
    }

    model.header_float_scalars[0] = q15(values[position]);
    model.header_float_scalars[1] = q15(values[position + 1u]) * 16.0f;
    model.header_integer_fields[0] = values[position + 2u];
    model.header_integer_fields[1] = low16(values[position + 3u]);
    position += 4u;
    for (int profile = 0; profile < 4; ++profile) {
        RosellaDistanceProfile parsed;
        for (int index = 0; index < 6; ++index) {
            parsed.bounds[static_cast<std::size_t>(index)] =
                q15(values[position + static_cast<std::size_t>(index)]);
        }
        position += 6u;
        parsed.distance_scale_m =
            q15_exp(values[position], values[position + 1u]);
        position += 2u;
        parsed.inverse_distance_per_m = q15(values[position]);
        parsed.axis_scales_internal[0] = q15(values[position + 1u]);
        parsed.axis_scales_internal[1] = q15(values[position + 2u]);
        parsed.axis_scales_internal[2] = q15(values[position + 3u]);
        parsed.minimum_normalized_radius = q15(values[position + 4u]);
        position += 5u;
        model.profiles[static_cast<std::size_t>(profile)] = parsed;
    }
    model.profile_tail.resize(8u);
    for (int index = 0; index < 8; ++index) {
        model.profile_tail[static_cast<std::size_t>(index)] =
            q15(values[position + static_cast<std::size_t>(index)]);
    }
    position += 8u;
    for (int index = 0; index < 3; ++index) {
        model.post_fields[static_cast<std::size_t>(index)] =
            values[position + static_cast<std::size_t>(index)];
    }
    position += 3u;
    if (position != total) {
        return model_fail(JOC_ERR_HRTF_FORMAT, "unparsed Rosella coefficient lanes");
    }
    if (model.sample_rate != 48000) {
        return model_fail(JOC_ERR_HRTF_FORMAT,
                          "Rosella model sample rate must be 48000, got " +
                              std::to_string(model.sample_rate));
    }
    *out = std::move(model);
    return Status::success();
}

}  // namespace joc::hrtf
