#include "binaural/reference_distance.h"

#include <algorithm>
#include <cmath>

namespace joc::binaural {
namespace {

using ArgumentError = joc::Status;

Status argument_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kRender, message);
}

struct ModePreset {
    const char* name;
    double reference_distance_m;
    double late_send;
};

constexpr ModePreset kModes[3] = {
    {"near", 1.00000465, 0.06},
    {"mid", 2.19327927, 0.16},
    {"far", 6.40177584, 0.28},
};

const ModePreset* find_mode(const std::string& mode) {
    for (const ModePreset& preset : kModes) {
        if (mode == preset.name) {
            return &preset;
        }
    }
    return nullptr;
}

}  // namespace

Status normalize_mode_name(const std::string& mode, std::string* out) {
    if (out == nullptr) {
        return argument_error("mode output is null");
    }
    std::string lowered;
    lowered.reserve(mode.size());
    for (const char value : mode) {
        if (value == ' ' || value == '\t') {
            continue;
        }
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(value))));
    }
    if (find_mode(lowered) == nullptr) {
        return argument_error("distance profile must be near, mid, or far");
    }
    *out = lowered;
    return Status::success();
}

Status source_reference_distance(const std::string& mode, double* out) {
    std::string normalized;
    Status status = normalize_mode_name(mode, &normalized);
    if (!status.ok()) {
        return status;
    }
    if (out == nullptr) {
        return argument_error("reference distance output is null");
    }
    *out = find_mode(normalized)->reference_distance_m;
    return Status::success();
}

Status map_adm_position(const std::array<double, 3>& position, const std::string& mode,
                        DistanceState* out) {
    if (out == nullptr) {
        return argument_error("distance state output is null");
    }
    for (const double value : position) {
        if (!std::isfinite(value)) {
            return argument_error("ADM position must contain three finite Cartesian values");
        }
    }
    std::string normalized;
    Status status = normalize_mode_name(mode, &normalized);
    if (!status.ok()) {
        return status;
    }
    const double radius = std::sqrt(position[0] * position[0] + position[1] * position[1] +
                                    position[2] * position[2]);
    std::array<double, 3> direction = {0.0, 1.0, 0.0};
    if (radius > 1.0e-15) {
        direction[0] = position[0] / radius;
        direction[1] = position[1] / radius;
        direction[2] = position[2] / radius;
    }
    const double reference = find_mode(normalized)->reference_distance_m;
    out->profile = normalized;
    out->normalized_radius = radius;
    out->reference_distance_m = reference;
    out->physical_distance_m = std::max(kMinimumSourceDistanceM, radius * reference);
    out->direction_adm = direction;
    return Status::success();
}

double direct_level_gain(const DistanceState& state) {
    if (state.profile == "near") {
        return 1.0;
    }
    const double distance = state.physical_distance_m;
    return 1.0 / std::sqrt(1.0 + kRoomEnergyCouplingPerM2 * distance * distance);
}

double late_send(const DistanceState& state) {
    const ModePreset* preset = find_mode(state.profile);
    const double base = preset == nullptr ? 0.16 : preset->late_send;
    const double radial = std::sqrt(std::max(state.normalized_radius, 0.0));
    return base * std::min(std::max(radial, 0.25), 1.5);
}

Status inverse_distance_gain(double measurement_radius_m, double path_distance_m, double* out) {
    if (!std::isfinite(measurement_radius_m) || !std::isfinite(path_distance_m)) {
        return argument_error("measurement and path distances must be finite");
    }
    if (measurement_radius_m <= 0.0 || path_distance_m <= 0.0) {
        return argument_error("measurement and path distances must be positive");
    }
    if (out == nullptr) {
        return argument_error("gain output is null");
    }
    *out = measurement_radius_m / path_distance_m;
    return Status::success();
}

}  // namespace joc::binaural
