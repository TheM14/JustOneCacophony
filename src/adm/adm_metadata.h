// Port of src/adm_atmos.py.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"

namespace joc::adm {

inline constexpr int kObjectCount = 15;
inline constexpr std::uint32_t kTrackCount = 25;

struct Keyframe {
    std::int64_t rtime_samples = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::int64_t duration_samples = 0;
    std::int64_t interpolation_samples = 0;
};

struct Track {
    std::string name;
    std::vector<Keyframe> blocks;
};

// HH:MM:SS.fffff with the reference's truncation + round-half-even carry.
std::string ts(double seconds);

enum class BinauralMode : std::uint32_t { Off = 0, Near = 1, Far = 2, Mid = 3, Unspecified = 4 };

bool binaural_mode_from_name(const char* name, BinauralMode* out);

std::string build_axml(const std::vector<Track>& tracks, double duration_sec, std::uint32_t rate);

std::string build_chna();

Status build_dbmd(std::uint32_t object_count, BinauralMode mode, std::string* out);

}  // namespace joc::adm
