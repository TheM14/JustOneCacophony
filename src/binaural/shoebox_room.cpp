#include "binaural/shoebox_room.h"

#include <algorithm>
#include <cmath>

namespace joc::binaural {
namespace {

Status argument_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kRender, message);
}

const char* const kWallNames[6] = {"left", "right", "back", "front", "floor", "ceiling"};

}  // namespace

Status ShoeboxRoomConfig::validate() const {
    for (const double value : dimensions_m) {
        if (!std::isfinite(value) || value <= 0.0) {
            return argument_error("room dimensions must be three positive finite values");
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double value = listener_position_m[static_cast<std::size_t>(axis)];
        if (!std::isfinite(value) || value <= 0.0 ||
            value >= dimensions_m[static_cast<std::size_t>(axis)]) {
            return argument_error("listener must be strictly inside the shoebox");
        }
    }
    for (const double gain : wall_reflection_gain) {
        if (!std::isfinite(gain) || std::fabs(gain) >= 1.0) {
            return argument_error("six finite wall gains must have magnitude below one");
        }
    }
    if (!std::isfinite(speed_of_sound_m_s) || speed_of_sound_m_s <= 0.0) {
        return argument_error("speed of sound must be positive");
    }
    return Status::success();
}

Status first_order_image_sources(const std::array<double, 3>& direction_adm,
                                 double source_distance_m, double sample_rate_hz,
                                 const ShoeboxRoomConfig& config,
                                 std::vector<EarlyReflection>* out) {
    if (out == nullptr) {
        return argument_error("reflection output is null");
    }
    const Status validated = config.validate();
    if (!validated.ok()) {
        return validated;
    }
    for (const double value : direction_adm) {
        if (!std::isfinite(value)) {
            return argument_error(
                "reflection direction must contain three finite ADM values");
        }
    }
    const double norm = std::sqrt(direction_adm[0] * direction_adm[0] +
                                  direction_adm[1] * direction_adm[1] +
                                  direction_adm[2] * direction_adm[2]);
    const std::array<double, 3> direction = norm <= 1.0e-15
                                                ? std::array<double, 3>{0.0, 1.0, 0.0}
                                                : std::array<double, 3>{
                                                      direction_adm[0] / norm,
                                                      direction_adm[1] / norm,
                                                      direction_adm[2] / norm};
    if (!std::isfinite(source_distance_m) || source_distance_m <= 0.0 ||
        !std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0) {
        return argument_error("source distance and sample rate must be positive");
    }
    std::array<double, 3> source = {};
    for (int axis = 0; axis < 3; ++axis) {
        source[static_cast<std::size_t>(axis)] =
            config.listener_position_m[static_cast<std::size_t>(axis)] +
            direction[static_cast<std::size_t>(axis)] * source_distance_m;
        const double value = source[static_cast<std::size_t>(axis)];
        if (value <= 0.0 || value >= config.dimensions_m[static_cast<std::size_t>(axis)]) {
            return argument_error(
                "source lies outside the configured public shoebox; enlarge the room");
        }
    }
    std::array<std::array<double, 3>, 6> images = {};
    for (int axis = 0; axis < 3; ++axis) {
        std::array<double, 3> low = source;
        low[static_cast<std::size_t>(axis)] = -source[static_cast<std::size_t>(axis)];
        std::array<double, 3> high = source;
        high[static_cast<std::size_t>(axis)] =
            2.0 * config.dimensions_m[static_cast<std::size_t>(axis)] -
            source[static_cast<std::size_t>(axis)];
        images[static_cast<std::size_t>(axis * 2)] = low;
        images[static_cast<std::size_t>(axis * 2 + 1)] = high;
    }
    out->clear();
    out->reserve(6);
    for (int index = 0; index < 6; ++index) {
        const std::array<double, 3>& image = images[static_cast<std::size_t>(index)];
        std::array<double, 3> vector = {};
        for (int axis = 0; axis < 3; ++axis) {
            vector[static_cast<std::size_t>(axis)] =
                image[static_cast<std::size_t>(axis)] -
                config.listener_position_m[static_cast<std::size_t>(axis)];
        }
        const double path_distance = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] +
                                               vector[2] * vector[2]);
        if (!(path_distance > 0.0)) {
            return argument_error("image source coincides with the listener");
        }
        EarlyReflection reflection;
        reflection.wall = kWallNames[index];
        for (int axis = 0; axis < 3; ++axis) {
            reflection.direction_adm[static_cast<std::size_t>(axis)] =
                vector[static_cast<std::size_t>(axis)] / path_distance;
        }
        reflection.path_distance_m = path_distance;
        const double extra_metres = std::max(path_distance - source_distance_m, 0.0);
        reflection.extra_delay_samples = extra_metres * sample_rate_hz / config.speed_of_sound_m_s;
        const double air = std::exp(-0.002 * extra_metres);
        reflection.reflection_gain =
            config.wall_reflection_gain[static_cast<std::size_t>(index)] * air;
        out->push_back(reflection);
    }
    return Status::success();
}

}  // namespace joc::binaural
