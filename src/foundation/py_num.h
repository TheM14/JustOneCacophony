
#pragma once

#include <cfenv>
#include <cmath>
#include <cstdio>
#include <string>

namespace joc::pynum {

inline long long py_round(double value) {
    return static_cast<long long>(std::nearbyint(value));
}

inline std::string format_fixed(double value, int decimals) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return std::string(buffer);
}

inline long long trunc_to_ll(double value) {
    return static_cast<long long>(value);
}

}  // namespace joc::pynum
