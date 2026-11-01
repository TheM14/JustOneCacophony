#pragma once

#include <array>
#include <string>

#include "foundation/status.h"

// Project-owned distance policy for the SOFA renderer, ported from the
// reference src/reference_distance.py.  These are presentation presets of this
// project, not a claim about an external standard: the reference distances are
// the nominal object-coordinate scales, the direct level is an equal-power dry
// coefficient (never a raw 1/r attenuation of authored programme gain), and the
// late send is a fixed per-mode share scaled by the object radius.
namespace joc::binaural {

inline constexpr double kMinimumSourceDistanceM = 0.10;
inline constexpr double kRoomEnergyCouplingPerM2 = 0.01318359375;
inline constexpr double kPublicRoomCalibrationGain = 1.4;

// Reference distance of one mode, in metres.
Status source_reference_distance(const std::string& mode, double* out);
// Validates a mode name ("near", "mid" or "far") and stores its canonical spelling.
Status normalize_mode_name(const std::string& mode, std::string* out);

struct DistanceState {
    std::string profile = "mid";
    double normalized_radius = 0.0;
    double reference_distance_m = 0.0;
    double physical_distance_m = 0.0;
    std::array<double, 3> direction_adm = {0.0, 1.0, 0.0};
};

// Resolves an ADM position on one reference distance profile.
Status map_adm_position(const std::array<double, 3>& position, const std::string& mode,
                        DistanceState* out);

// Equal-power dry coefficient: unity for near, 1/sqrt(1+coupling*d^2) otherwise.
double direct_level_gain(const DistanceState& state);
// Per-mode late send, scaled by the square root of the normalized radius.
double late_send(const DistanceState& state);
// measurement radius / path distance, the image-source level of one reflection.
Status inverse_distance_gain(double measurement_radius_m, double path_distance_m, double* out);

}  // namespace joc::binaural
