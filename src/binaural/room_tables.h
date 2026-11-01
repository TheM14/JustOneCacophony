#pragma once

#include <array>
#include <string>
#include <vector>

#include "binaural/shoebox_room.h"
#include "foundation/status.h"

// The project's own room, generated in code: a shared unitary FDN (four
// orthogonal Hadamard feedback lines, two Schroeder all-pass diffusers and
// low-pass damping) rendered to a two-channel float64 FIR, plus the shoebox
// geometry its first-order images are computed from.  Ported from the reference
// src/public_room.py; there is no measured BRIR and no external model behind it.
namespace joc::binaural {

struct RoomPreset {
    ShoeboxRoomConfig early;
    std::vector<double> late_fir[2];  // [2][taps]
    double rt60_seconds = 0.0;
    double size_time_scale = 1.0;
};

// "legacy" is the original project room; "compact" keeps the same algorithm with
// the Zahorik-table geometry and the late time constants scaled by V/S.
Status build_room_preset(const std::string& preset, RoomPreset* out);

// One FDN realization: the response to a unit impulse, cut to 2*RT60.
Status render_late_fir(double sample_rate_hz, double rt60_seconds, double damping,
                       double output_gain, const std::array<double, 4>& delay_seconds,
                       const std::array<double, 2>& allpass_seconds,
                       const std::array<double, 2>& allpass_gain,
                       std::size_t impulse_samples, std::vector<double>* left,
                       std::vector<double>* right);

}  // namespace joc::binaural
