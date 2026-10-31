
#include "binaural/binaural_reference.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

#include "foundation/fft.h"
#include "foundation/fs_utf8.h"
#include "foundation/mini_json.h"
#include "foundation/sha256.h"
#include "hrtf/sofa_field.h"

namespace joc::binaural {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kMaximumExtension = 50.1;

Status reference_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_CONFIG, joc::stage::kRender, message);
}

bool read_text_file(const std::string& path, std::string* out) {
    std::ifstream stream(fs_utf8::to_path(path), std::ios::binary);
    if (!stream) {
        return false;
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    *out = buffer.str();
    return true;
}

bool read_binary_file(const std::string& path, std::string* out) {
    return read_text_file(path, out);
}

std::string directory_of(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

std::string join_path(const std::string& directory, const std::string& name) {
    if (directory.empty()) {
        return name;
    }
    const char last = directory.back();
    return (last == '/' || last == '\\' ? directory : directory + "/") + name;
}

std::vector<double> parse_number_list(const std::string& raw) {
    std::vector<double> values;
    std::size_t index = 0;
    while (index < raw.size()) {
        if (raw[index] == '-' || raw[index] == '+' || raw[index] == '.' ||
            (raw[index] >= '0' && raw[index] <= '9')) {
            std::size_t end = index;
            while (end < raw.size() &&
                   (raw[end] == '-' || raw[end] == '+' || raw[end] == '.' ||
                    raw[end] == 'e' || raw[end] == 'E' ||
                    (raw[end] >= '0' && raw[end] <= '9'))) {
                ++end;
            }
            values.push_back(std::strtod(raw.substr(index, end - index).c_str(), nullptr));
            index = end;
            continue;
        }
        ++index;
    }
    return values;
}

}  // namespace

std::string ReferenceEnvironment::default_reference_config_path() {
    return std::string("local/binaural_reference.json");
}

Status ReferenceEnvironment::load(const std::string& config_path, int mode_index,
                                  bool mode_specified) {
    config_path_ = config_path.empty() ? default_reference_config_path() : config_path;
    std::string text;
    if (!read_text_file(config_path_, &text)) {
        return Status::fail(JOC_ERR_HRTF_NOT_FOUND, joc::stage::kRender,
                            "reference configuration not found: " + config_path_);
    }
    configuration_sha256_ = joc::crypto::sha256_hex(text.data(), text.size());

    std::vector<json::Member> members;
    std::string error;
    if (!json::parse_object(text, &members, &error)) {
        return reference_error("reference configuration is not a JSON object");
    }
    std::string schema;
    const json::Member* schema_member = json::find(members, "schema");
    if (schema_member == nullptr || !json::as_string(*schema_member, &schema) ||
        schema != "binaural-reference-v1") {
        return reference_error("unsupported reference configuration schema");
    }
    std::string model_path;
    std::string expected_sha;
    const json::Member* model_member = json::find(members, "model_path");
    const json::Member* sha_member = json::find(members, "model_sha256");
    if (model_member == nullptr || sha_member == nullptr ||
        !json::as_string(*model_member, &model_path) ||
        !json::as_string(*sha_member, &expected_sha)) {
        return reference_error("reference configuration needs model_path and model_sha256");
    }
    long long profile_index = 0;
    const json::Member* profile_member = json::find(members, "profile_index");
    if (profile_member == nullptr || !json::as_integer(*profile_member, &profile_index)) {
        return reference_error("reference configuration needs profile_index");
    }
    calibration_profile_index_ = static_cast<int>(profile_index);
    if (calibration_profile_index_ != kProfileFar &&
        calibration_profile_index_ != kProfileMid) {
        return reference_error("unsupported reference profile");
    }
    const json::Member* profile_id_member = json::find(members, "profile_id");
    if (profile_id_member != nullptr) {
        json::as_string(*profile_id_member, &profile_id_);
    }
    long long slots = 4096;
    if (const json::Member* member = json::find(members, "room_impulse_slots")) {
        json::as_integer(*member, &slots);
    }
    long long directions = 128;
    if (const json::Member* member = json::find(members, "level_calibration_directions")) {
        json::as_integer(*member, &directions);
    }
    if (slots < 128 || slots > 16384 || directions < 16 || directions > 2048) {
        return reference_error("reference configuration exceeds supported bounds");
    }
    room_impulse_slots_ = static_cast<int>(slots);
    calibration_directions_ = static_cast<int>(directions);

    // The model path is relative to the configuration file unless absolute.
    std::string resolved = model_path;
    if (!(model_path.size() > 1 && (model_path[1] == ':' || model_path[0] == '/' ||
                                    model_path[0] == '\\'))) {
        resolved = join_path(directory_of(config_path_), model_path);
    }
    std::string model_bytes;
    if (!read_binary_file(resolved, &model_bytes)) {
        return Status::fail(JOC_ERR_HRTF_NOT_FOUND, joc::stage::kRender,
                            "reference model not found: " + resolved);
    }
    model_sha256_ = joc::crypto::sha256_hex(model_bytes.data(), model_bytes.size());
    std::string lower_expected = expected_sha;
    std::transform(lower_expected.begin(), lower_expected.end(), lower_expected.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    std::string lower_actual = model_sha256_;
    std::transform(lower_actual.begin(), lower_actual.end(), lower_actual.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (lower_actual != lower_expected) {
        return reference_error("reference data hash mismatch");
    }
    const Status status = hrtf::load_personalized_headphone(resolved, &model_);
    if (!status.ok()) {
        return status;
    }
    if (model_.sample_rate != 48000) {
        return reference_error("unsupported reference profile");
    }
    // Cartesian reference configurations only.
    if (model_.header_integer_fields[0] != 0) {
        return reference_error("reference-v1 requires a Cartesian mapping");
    }

    mode_name_ = mode_specified
                     ? (mode_index == kProfileNear ? "near"
                                                   : (mode_index == kProfileFar ? "far" : "mid"))
                     : (calibration_profile_index_ == kProfileFar ? "far" : "mid");
    profile_index_ = mode_specified ? mode_index
                                    : (calibration_profile_index_ == kProfileFar ? kProfileFar
                                                                                 : kProfileMid);
    if (profile_index_ != kProfileNear && profile_index_ != kProfileFar &&
        profile_index_ != kProfileMid) {
        return reference_error("reference mode must be near, mid, or far");
    }

    header_constant_ = model_.header_float_scalars[0];
    coupling_ = model_.header_float_scalars[1];
    if (!(header_constant_ > 0.0) || !(coupling_ > 0.0)) {
        return reference_error("invalid reference energy coupling");
    }
    const hrtf::RosellaDistanceProfile& profile =
        model_.profiles[calibration_profile_index_];
    minimum_distance_m_ = static_cast<double>(profile.minimum_normalized_radius) *
                          static_cast<double>(profile.distance_scale_m);
    const double origin[3] = {0.0, 1.0, 0.0};
    ReferencePosition position;
    const Status mapped = map_position(origin, &position);
    if (!mapped.ok()) {
        return mapped;
    }
    reference_distance_m_ = position.distance_m;

    if (const json::Member* member = json::find(members, "compiled_receiver_positions_m")) {
        compiled_receivers_m_ = parse_number_list(member->raw);
        if (compiled_receivers_m_.size() != 6u) {
            compiled_receivers_m_.clear();
        }
    }
    return Status::success();
}

Status ReferenceEnvironment::map_position(const double position[3],
                                          ReferencePosition* out) const {
    return map_position(position, profile_index_, out);
}

Status ReferenceEnvironment::map_position(const double position[3], int profile_index,
                                          ReferencePosition* out) const {
    if (out == nullptr) {
        return reference_error("reference position output is null");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(position[axis])) {
            return reference_error("reference position must contain three finite values");
        }
    }
    double maximum = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        maximum = std::max(maximum, std::fabs(position[axis]));
    }
    const double extension = std::max(1.0, maximum);
    if (extension > kMaximumExtension + 1e-9) {
        return reference_error("position exceeds the finite metadata domain");
    }
    const double scaled[3] = {position[0] / extension, position[1] / extension,
                              position[2] / extension};
    const int profile = profile_index < 0 ? profile_index_ : profile_index;
    hrtf::ProfileGeometry geometry;
    Status status = hrtf::profile_geometry(model_, scaled, profile, &geometry);
    if (!status.ok()) {
        return status;
    }
    const hrtf::RosellaDistanceProfile& distance_profile = model_.profiles[profile];
    const double scale = static_cast<double>(distance_profile.distance_scale_m);
    const double distance = scale * geometry.radius * extension;
    const double minimum =
        static_cast<double>(distance_profile.minimum_normalized_radius) * scale;
    const double regularized = std::max(distance, minimum);
    const double alpha = regularized > 0.0 ? distance / regularized : 0.0;
    const double effective = header_constant_ * distance;
    double gain = 1.0;
    double send = 0.0;
    if (profile != kProfileNear) {
        gain = 1.0 / std::sqrt(1.0 + coupling_ * effective * effective);
        send = effective * gain;
    }
    for (int axis = 0; axis < 3; ++axis) {
        out->direction[axis] = geometry.direction[axis];
    }
    out->distance_m = distance;
    out->ear_distance_m = regularized;
    out->centre_weight = alpha;
    out->direct_gain = gain;
    out->room_send = send;
    return Status::success();
}

Status ReferenceEnvironment::create_room(std::unique_ptr<hrtf::RosellaRoomFir>* out) const {
    if (out == nullptr) {
        return reference_error("room output is null");
    }
    std::unique_ptr<hrtf::RosellaRoomFir> room = std::make_unique<hrtf::RosellaRoomFir>();
    Status status = room->configure(model_, room_impulse_slots_);
    if (!status.ok()) {
        return status;
    }
    *out = std::move(room);
    return Status::success();
}

std::unique_ptr<hrtf::RosellaBank> ReferenceEnvironment::create_analysis_bank(
    std::size_t channels) const {
    return hrtf::RosellaBank::create(hrtf::RosellaBankKind::Public, channels);
}

std::unique_ptr<hrtf::RosellaBank> ReferenceEnvironment::create_synthesis_bank() const {
    return hrtf::RosellaBank::create(hrtf::RosellaBankKind::Public, 2u);
}

Status ReferenceEnvironment::calibrate(std::size_t fft_size,
                                       const std::vector<double>& frequencies,
                                       const SpectrumAt& spectrum_at, double* level) const {
    if (level == nullptr || frequencies.empty()) {
        return reference_error("calibration needs a spectrum callback and frequencies");
    }
    const std::vector<double> dictionary =
        hrtf::hybrid_gain_synthesis_dictionary_for_check(fft_size);
    if (dictionary.size() != fft_size * 154u) {
        return reference_error("hybrid gain dictionary has the wrong shape");
    }
    const std::vector<double> field_left = hrtf::rosella_logical_field(model_.field_left_padded);
    const std::vector<double> field_right = hrtf::rosella_logical_field(model_.field_right_padded);

    // Fibonacci sphere: equal-area calibration directions.
    const int count = calibration_directions_;
    std::vector<std::size_t> band_bins;
    std::vector<double> weights;
    for (std::size_t bin = 0; bin < frequencies.size(); ++bin) {
        if (frequencies[bin] >= 100.0 && frequencies[bin] <= 16000.0) {
            band_bins.push_back(bin);
            weights.push_back(1.0 / frequencies[bin]);
        }
    }
    if (band_bins.empty()) {
        return reference_error("calibration band is empty");
    }
    const std::size_t bins = frequencies.size();
    double reference_energy = 0.0;
    double new_energy = 0.0;
    std::vector<std::complex<double>> reference(2 * bins);
    std::vector<std::complex<double>> target;
    std::vector<double> parameters(2 * 154);
    for (int index = 0; index < count; ++index) {
        const double z = 1.0 - 2.0 * (static_cast<double>(index) + 0.5) / count;
        const double phi = static_cast<double>(index) * kPi * (3.0 - std::sqrt(5.0));
        const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double direction[3] = {-radial * std::sin(phi), radial * std::cos(phi), z};
        const double adm[3] = {-direction[1], direction[0], direction[2]};
        ReferencePosition state;
        Status status = map_position(adm, calibration_profile_index_, &state);
        if (!status.ok()) {
            return status;
        }
        hrtf::DirectResult result;
        status = hrtf::direct_and_room_send(model_, field_left, field_right, adm,
                                            calibration_profile_index_, &result);
        if (!status.ok()) {
            return status;
        }
        for (int ear = 0; ear < 2; ++ear) {
            for (int band = 0; band < 77; ++band) {
                parameters[static_cast<std::size_t>(ear * 154 + 2 * band)] =
                    result.gains[ear][band].real();
                parameters[static_cast<std::size_t>(ear * 154 + 2 * band + 1)] =
                    result.gains[ear][band].imag();
            }
        }
        for (int ear = 0; ear < 2; ++ear) {
            std::vector<double> time(fft_size, 0.0);
            for (std::size_t tap = 0; tap < fft_size; ++tap) {
                double sum = 0.0;
                for (int parameter = 0; parameter < 154; ++parameter) {
                    sum += dictionary[tap * 154u + static_cast<std::size_t>(parameter)] *
                           parameters[static_cast<std::size_t>(ear * 154 + parameter)];
                }
                time[tap] = sum;
            }
            std::vector<std::complex<double>> buffer(fft_size);
            for (std::size_t tap = 0; tap < fft_size; ++tap) {
                buffer[tap] = std::complex<double>(time[tap], 0.0);
            }
            dsp::fft_radix2(&buffer, false);
            for (std::size_t bin = 0; bin < bins; ++bin) {
                reference[static_cast<std::size_t>(ear) * bins + bin] = buffer[bin];
            }
        }
        status = spectrum_at(state, &target);
        if (!status.ok()) {
            return status;
        }
        if (target.size() != 2 * bins) {
            return reference_error("calibration spectrum has the wrong shape");
        }
        for (std::size_t index_band = 0; index_band < band_bins.size(); ++index_band) {
            const std::size_t bin = band_bins[index_band];
            const double weight = weights[index_band];
            for (int ear = 0; ear < 2; ++ear) {
                const std::size_t offset = static_cast<std::size_t>(ear) * bins + bin;
                reference_energy += std::norm(reference[offset]) * weight;
                new_energy +=
                    std::norm(target[offset] * state.direct_gain) * weight;
            }
        }
    }
    if (!(reference_energy > 1e-24) || !(new_energy > 1e-24) ||
        !std::isfinite(new_energy)) {
        return reference_error("reference level calibration has no finite signal energy");
    }
    *level = std::sqrt(reference_energy / new_energy);
    return Status::success();
}

}  // namespace joc::binaural
