#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"

namespace joc::hrtf {

// Coordinate declaration of one SOFA variable: Type ("spherical"/"cartesian") and
// Units.  Empty when the file does not declare them (ListenerUp inherits).
struct SofaCoordinate {
    std::string type;
    std::string units;
};

// SOFA SimpleFreeFieldHRIR as this project consumes it: the impulse responses,
// the measurement geometry and the metadata needed to report what was loaded.
// All angles are degrees, all distances metres, exactly as the file stores them.
struct SofaHrir {
    double sample_rate = 0.0;
    std::uint32_t ir_count = 0;   // M: number of measurements
    std::uint32_t ir_length = 0;  // N: taps per impulse response
    std::vector<double> ir;       // C order [M][2][N]
    double delay[2] = {0.0, 0.0};
    std::vector<double> source_position;  // M*3
    double listener_position[3] = {0.0, 0.0, 0.0};
    double listener_view[3] = {1.0, 0.0, 0.0};
    double listener_up[3] = {0.0, 0.0, 1.0};
    double emitter_position[3] = {0.0, 0.0, 0.0};
    double receiver_position[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::string conventions;
    std::string sofa_conventions;
    std::string convention_version;
    std::string version;
    std::string data_type;
    std::string room_type;
    std::string title;
    std::string database_name;
    std::string listener_short_name;
    std::string comment;
    std::string sampling_rate_units;
    SofaCoordinate source_position_coordinates;
    SofaCoordinate listener_position_coordinates;
    SofaCoordinate listener_view_coordinates;
    SofaCoordinate listener_up_coordinates;
    SofaCoordinate emitter_position_coordinates;
    SofaCoordinate receiver_position_coordinates;
    // Identity of the file itself, needed for the compiled-cache key.
    std::string source_path;
    std::string source_sha256;

    // One line for reports and logs:
    // "SOFA SimpleFreeFieldHRIR, <M> IRs x <N> taps @ <rate> Hz".
    std::string summary() const;
};

Status load_sofa(const std::string& path, SofaHrir* out);

}  // namespace joc::hrtf
