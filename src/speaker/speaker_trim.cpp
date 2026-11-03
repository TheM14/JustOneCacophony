#include "speaker/speaker_trim.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "foundation/py_num.h"

namespace joc::speaker {

namespace {

int height_class(std::uint32_t count) {
    if (count == 0u) {
        return 0;
    }
    if (count == 2u) {
        return 1;
    }
    return 2;
}

int floor_class(std::uint32_t count) {
    if (count == 0u) {
        return 0;
    }
    if (count == 2u || count == 3u) {
        return 1;
    }
    return 2;
}

}  // namespace

int trim_config_index(const LayoutInfo& layout) {
    return floor_class(layout.surround_channels) + 3 * height_class(layout.height_channels);
}

Status parameters_from_oamd(const oamd::OamdTrim* trim, const LayoutInfo& layout,
                            ejoc_speaker_controls* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "null speaker controls");
    }
    ejoc_speaker_controls controls{};
    controls.struct_size = sizeof(controls);
    controls.struct_version = EJOC_SPEAKER_CONTROLS_VERSION;
    controls.trim_mode = EJOC_SPEAKER_TRIM_AUTOMATIC;
    if (trim != nullptr) {
        controls.warp_mode = trim->warp_mode;
        for (int slot = 0; slot < oamd::kMaxSlots; ++slot) {
            controls.trim_bypass[slot] = trim->disable_per_object[slot] ? 1u : 0u;
        }
    }
    if (controls.warp_mode > 3u) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "speaker: invalid warp mode " +
                                std::to_string(controls.warp_mode));
    }
    if (!layout.carries_trim) {
        controls.trim_mode = EJOC_SPEAKER_TRIM_DISABLED;
        *out = controls;
        return Status::success();
    }
    const std::uint32_t mode = trim != nullptr ? trim->global_trim_mode : 0u;
    if (mode == 1u) {
        controls.trim_mode = EJOC_SPEAKER_TRIM_DISABLED;
        *out = controls;
        return Status::success();
    }
    if (mode > 2u) {
        return Status::fail(JOC_ERR_NOT_SUPPORTED, stage::kRender,
                            "speaker: trim_mode_not_implemented: OAMD global trim mode " +
                                std::to_string(mode) + " is reserved");
    }
    if (mode == 2u) {
        const oamd::TrimConfig& record = trim->configs[trim_config_index(layout)];
        if (record.kind == oamd::TrimConfigKind::kDefault ||
            record.kind == oamd::TrimConfigKind::kAbsent) {
            *out = controls;
            return Status::success();
        }
        if (record.kind == oamd::TrimConfigKind::kDisabled) {
            controls.trim_mode = EJOC_SPEAKER_TRIM_DISABLED;
            *out = controls;
            return Status::success();
        }
        if (!std::isfinite(record.centre) || !std::isfinite(record.surround) ||
            !std::isfinite(record.height)) {
            return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                                "speaker: trim levels must be finite");
        }
        for (const double balance : {record.balance_listener, record.balance_overhead}) {
            if (!std::isfinite(balance) || balance < -1.0 || balance > 1.0) {
                return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                                    "speaker: balance must be finite and in [-1,1]");
            }
        }
        // Surround and height are attenuation-only; the centre level keeps its
        // sign.
        controls.trim_mode = EJOC_SPEAKER_TRIM_EXPLICIT;
        controls.centre_db = record.centre;
        controls.surround_db = std::min(record.surround, 0.0);
        controls.height_db = std::min(record.height, 0.0);
        controls.listener_balance = record.balance_listener;
        controls.overhead_balance = record.balance_overhead;
    }
    *out = controls;
    return Status::success();
}

double front_back_balance(double position, double balance) {
    const long long steps =
        pynum::py_round(std::fabs(balance) * static_cast<double>(kBalanceSteps));
    if (steps == 0) {
        return position;
    }
    const double slope = static_cast<double>(steps) / 18.0;
    if (position <= kBalanceWall || position >= 1.0 - kBalanceWall) {
        return position;
    }
    if (position < 0.5) {
        return balance > 0.0 ? (1.0 + slope) * position - kBalanceWall * slope
                             : (1.0 - slope) * position + kBalanceWall * slope;
    }
    return balance > 0.0
               ? (1.0 - slope) * (position - 0.5) + 0.5 + 0.45 * slope
               : (1.0 + slope) * (position - 0.5) + 0.5 - 0.45 * slope;
}

double apply_balance(double position, double height, double listener, double overhead) {
    if (listener == 0.0 && overhead == 0.0) {
        return position;
    }
    const double low = front_back_balance(position, listener);
    const double high = front_back_balance(position, overhead);
    return (1.0 - height) * low + height * high;
}

double explicit_position_gain(const ejoc_speaker_controls& controls, double x, double y,
                              double z) {
    const double surround = std::min(std::max(2.0 * y, 0.0), 1.0) * (1.0 - z);
    const double centre = 1.0 - std::min(
                                    std::max(10.0 * std::max(std::max(std::fabs(z) - kBalanceWall,
                                                                     y - kBalanceWall),
                                                             std::fabs(x - 0.5) - kBalanceWall),
                                             0.0),
                                    1.0);
    const double db = surround * std::min(controls.surround_db, 0.0) +
                      z * std::min(controls.height_db, 0.0) + centre * controls.centre_db;
    return std::pow(10.0, db / 20.0);
}

}  // namespace joc::speaker
