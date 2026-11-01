#pragma once

#include <array>
#include <string>

#include "foundation/status.h"

// Listener-centred point-source coordinates for direct HRTF rendering, ported
// from the reference src/point_source.py: the ADM position selects a direction
// in the listener frame and a distance scaled by the mode's reference distance.
// There is no distance-dependent HRTF adaptation; the measured SOFA response is
// kept as it was measured.
namespace joc::binaural {

struct PointSourcePosition {
    std::array<double, 3> direction = {1.0, 0.0, 0.0};  // listener frame, unit
    double distance_m = 0.0;
    double normalized_radius = 0.0;
};

class PointSourceReference {
public:
    // `mode` is near/mid/far (empty selects mid).  `has_reference_distance`
    // overrides the mode's reference distance, as the reference CLI option does.
    Status configure(const std::string& mode, bool has_reference_distance,
                     double reference_distance_m);

    const std::string& mode() const { return mode_; }
    double reference_distance_m() const { return reference_distance_m_; }

    // `previous_direction` resolves a coincident position (nullptr keeps front).
    Status map_position(const std::array<double, 3>& position,
                        const std::array<double, 3>* previous_direction,
                        PointSourcePosition* out) const;

private:
    std::string mode_ = "mid";
    double reference_distance_m_ = 0.0;
};

}  // namespace joc::binaural
