#pragma once

#include <array>
#include <string>
#include <vector>

#include "foundation/status.h"

// Project-owned shoebox geometry and first-order image sources, ported from the
// reference src/public_room.py.  The room is this project's own construction, not
// a measured BRIR and not an external model: one rectangular room, one listener
// position, six wall gains and a speed of sound.
namespace joc::binaural {

struct ShoeboxRoomConfig {
    std::array<double, 3> dimensions_m = {18.0, 18.0, 14.0};
    std::array<double, 3> listener_position_m = {9.0, 9.0, 7.0};
    std::array<double, 6> wall_reflection_gain = {0.62, 0.60, 0.58, 0.61, 0.52, 0.56};
    double speed_of_sound_m_s = 343.3;

    Status validate() const;
};

struct EarlyReflection {
    std::string wall;
    std::array<double, 3> direction_adm = {0.0, 1.0, 0.0};
    double path_distance_m = 0.0;
    double extra_delay_samples = 0.0;
    double reflection_gain = 0.0;
};

// Six first-order image-source paths for one object at `direction_adm` and
// `source_distance_m` metres from the listener.
Status first_order_image_sources(const std::array<double, 3>& direction_adm,
                                 double source_distance_m, double sample_rate_hz,
                                 const ShoeboxRoomConfig& config,
                                 std::vector<EarlyReflection>* out);

}  // namespace joc::binaural
