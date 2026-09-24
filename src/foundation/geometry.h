// Port of the reference's adm_atmos.q_to_adm_xyz: OAMD Q15 coordinates to the ADM
// cartesian triple.  It lives in foundation because both the ADM writer and the
// object position timeline need it, and the timeline must not depend on output.
#pragma once

#include <algorithm>

#include "foundation/py_num.h"

namespace joc::geometry {

inline void q_to_adm_xyz(int q1, int q2, int q3, double* x, double* y, double* z) {
    const double posX = std::min(1.0, static_cast<double>(pynum::py_round(
                                          static_cast<double>(q1) * 62.0 / 32767.0)) / 62.0);
    const double posY = std::min(1.0, static_cast<double>(pynum::py_round(
                                          static_cast<double>(q2) * 62.0 / 32767.0)) / 62.0);
    double posZ = static_cast<double>(pynum::py_round(
                      static_cast<double>(q3) * 15.0 / 32767.0)) / 15.0;
    posZ = std::max(-1.0, std::min(1.0, posZ));
    *x = posX * 2.0 - 1.0;
    *y = 1.0 - posY * 2.0;
    *z = posZ;
}

}  // namespace joc::geometry
