// Port of src/trim_balance.py.

#pragma once

#include <cstdint>

#include "eac3joc_core.h"
#include "foundation/status.h"
#include "oamd/oamd_parser.h"
#include "speaker/speaker_layout_lookup.h"

namespace joc::speaker {

// Front-back balance keeps a 5 percent wall band at each end unmapped.
inline constexpr double kBalanceWall = 0.05;
// bal3d_y is coded on a 1/16 grid, so a balance is quantised to that grid.
inline constexpr int kBalanceSteps = 16;

// Trim configuration a layout selects, from its surround and height counts:
//
//   cfg = kappa_f(surrounds) + 3 * kappa_h(heights)
//   kappa_f: 0 -> 0, 2..3 -> 1, otherwise 2
//   kappa_h: 0 -> 0, 2 -> 1, otherwise 2
int trim_config_index(const LayoutInfo& layout);

// Converts OAMD calibration to the speaker controls of one layout.  Warp modes
// 2/3 and the reserved global trim mode are reported as unsupported.  A null
// trim selects warp 0 and automatic trim, or disabled trim for a layout that
// carries no trim semantics.
Status parameters_from_oamd(const oamd::OamdTrim* trim, const LayoutInfo& layout,
                            ejoc_speaker_controls* out);

// Reparameterises a front-back coordinate in [0, 1] by ``balance``: 0 is the
// front wall and 1 the back wall, and the two wall bands pass through
// unchanged.
double front_back_balance(double position, double balance);

// Blends the listener-plane and overhead balance by height: 0 at the listener
// plane, 1 at the top layer.
double apply_balance(double position, double height, double listener, double overhead);

// Position-dependent object trim at the balanced original Y.  Surround fades
// in with Y and out towards Z=1, height fades in with Z, and the centre trim is
// local to the front-centre region.  The scalar multiplies the object's whole
// panning vector before accumulation.
double explicit_position_gain(const ejoc_speaker_controls& controls, double x, double y,
                              double z);

}  // namespace joc::speaker
