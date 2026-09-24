#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"

// Parser for the Dolby ".personalized_headphone" model (upstream rosella_model.py).
// The file is JSON whose virtualizer_parameters carry the raw "rp" coefficient
// lanes; everything the renderer needs is unpacked here, in the same float32
// arithmetic the reference uses, because those values are part of the model.
namespace joc::hrtf {

struct RosellaDistanceProfile {
    std::array<float, 6> bounds{};
    float distance_scale_m = 0.0f;
    float inverse_distance_per_m = 0.0f;
    std::array<float, 3> axis_scales_internal{};
    float minimum_normalized_radius = 0.0f;
};

struct RosellaCaptureMetadata {
    std::string capture_submission_date;
    std::string capture_type;
    std::string label;
    std::string name;
    std::string algorithm_version;
    std::string creation_date;
    std::string uuid;
    std::string version;
};

struct RosellaModel {
    std::string source_path;
    std::string coefficient_sha256;
    std::string coefficient_version;
    std::string room_model;
    RosellaCaptureMetadata capture;

    int table_a_dimension = 0;
    int table_a_option = 0;
    int table_a_extra = 0;
    int table_a_header_field = 0;
    int table_a_header_25 = 0;
    int table_a_control = 0;
    std::vector<int> table_a_option_ids;
    std::vector<float> table_a_option_values;
    float table_a_scalar = 0.0f;
    std::vector<float> table_a_filter_16x64_padded;
    std::array<int, 4> table_a_four_integers{};
    int table_a_integer = 0;
    std::vector<float> table_a_filter_8x64_padded;
    std::vector<float> table_a_vector16;
    std::vector<float> table_a_filter_4x64_padded;
    std::vector<int> table_a_extra_indices;
    std::vector<float> table_a_extra_fields_padded;
    std::vector<float> table_a_extra_vectors;

    int sample_rate = 0;
    int matrix_exponent = 0;
    int field_exponent = 0;
    std::vector<float> matrix_left;
    std::vector<float> matrix_right;
    std::vector<float> vector_left;
    std::vector<float> vector_right;
    std::vector<float> field_left_padded;
    std::vector<float> field_right_padded;
    bool field_left_odd_serialized_zero = false;
    std::vector<int> hybrid_flags;
    std::vector<float> hybrid_values;
    std::vector<float> model_scalars;
    std::array<float, 2> header_float_scalars{};
    std::array<int, 2> header_integer_fields{};
    std::array<RosellaDistanceProfile, 4> profiles{};
    std::vector<float> profile_tail;
    std::array<int, 3> post_fields{};

    // One line for reports and logs: the capture name and room model are the
    // model's own strings, followed by the table layout and sample rate, e.g.
    // "Rosella personalized_headphone '<name>' (<room>), <N> HQMF / 77 hybrid @ <rate> Hz".
    std::string summary() const;
};

Status load_personalized_headphone(const std::string& path, RosellaModel* out);

}  // namespace joc::hrtf
