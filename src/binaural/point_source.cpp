#include "binaural/point_source.h"

#include <algorithm>
#include <cmath>

#include "binaural/reference_distance.h"

namespace joc::binaural {
namespace {

Status argument_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kRender, message);
}

}  // namespace

Status PointSourceReference::configure(const std::string& mode, bool has_reference_distance,
                                       double reference_distance_m) {
    Status status = normalize_mode_name(mode.empty() ? std::string("mid") : mode, &mode_);
    if (!status.ok()) {
        return status;
    }
    if (has_reference_distance) {
        if (!std::isfinite(reference_distance_m) || reference_distance_m <= 0.0) {
            return argument_error("source reference distance must be finite and positive");
        }
        reference_distance_m_ = reference_distance_m;
        return Status::success();
    }
    return source_reference_distance(mode_, &reference_distance_m_);
}

Status PointSourceReference::map_position(const std::array<double, 3>& position,
                                          const std::array<double, 3>* previous_direction,
                                          PointSourcePosition* out) const {
    if (out == nullptr) {
        return argument_error("point source output is null");
    }
    for (const double value : position) {
        if (!std::isfinite(value)) {
            return argument_error("source position must contain three finite values");
        }
        if (std::fabs(value) > 50.1 + 1e-8) {
            return argument_error("source position exceeds the finite metadata domain");
        }
    }
    const double radius = std::sqrt(position[0] * position[0] + position[1] * position[1] +
                                    position[2] * position[2]);
    // ADM (+x right, +y front, +z up) to the listener frame (+x front, +y left).
    std::array<double, 3> direction = {position[1], -position[0], position[2]};
    if (radius > 1e-12) {
        for (double& value : direction) {
            value /= radius;
        }
    } else if (previous_direction == nullptr) {
        direction = {1.0, 0.0, 0.0};
    } else {
        direction = *previous_direction;
    }
    out->direction = direction;
    out->distance_m = radius * reference_distance_m_;
    out->normalized_radius = radius;
    return Status::success();
}

}  // namespace joc::binaural
