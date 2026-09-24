#include "hrtf/sofa_field.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "foundation/fft.h"
#include "foundation/sha256.h"
#include "hrtf/public_filterbank.h"
#include "io/npy_writer.h"

namespace joc::hrtf {

namespace {

using Complex = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;
constexpr int kParameterCount = 2 * kHybridBands;
constexpr std::int32_t kFilterbankFormatVersion = 1;
constexpr int kQmfBands = 64;

// Hybrid-band centre frequencies of the public analysis bank at 48 kHz
// (public_filterbank._BAND_CENTER_FREQUENCIES_HZ; part of the cache contract).
const double kBandCenters[kHybridBands] = {
    53.19564095937407,   26.3876219849709,    140.9074183269806,  98.55238901464415,
    234.09258166297573,  344.8745321543293,   321.80435900819805, 401.38762185365727,
    476.19239745597804,  473.552388917001,    648.8076026837931,  719.8745321201852,
    780.1254679168173,   851.1923973714038,   1023.8076024849751, 1155.125467695814,
    1293.0325067063661,  1668.032511377253,   2043.0325074138086, 2456.967490229666,
    2831.96748495363,    3206.967488172826,   3581.9675052705525, 3918.0325089666067,
    4331.967500201844,   4706.967500350216,   5043.032510771545,  5456.9674914391635,
    5831.967490225267,   6168.0324885741875,  6543.032508972284,  6956.96748844665,
    7293.032503259869,   7668.032503551393,   8043.032504806491,  8418.032498852166,
    8793.032502508235,   9206.967488589786,   9543.032511549152,  9956.967486913867,
    10293.032513008677,  10668.032507835102,  11043.032513641429, 11456.967482937946,
    11793.032513984212,  12206.967486015788,  12543.032517062376, 12956.96748635822,
    13331.967492165066,  13706.967486991198,  14043.032513086031, 14456.967488451,
    14793.032511410214,  15206.967497492202,  15581.967501147887, 15956.967495193188,
    16331.96749644876,   16706.967496740173,  17043.03251155318,  17456.96749102773,
    17831.967511425748,  18168.032509774734,  18543.032508560515, 18956.967489228293,
    19293.032499649784,  19668.032499798002,  20081.967491033392, 20418.0324947296,
    20793.032511827063,  21168.032515046092,  21543.032509770488, 21956.967492586176,
    22331.96748862257,   22706.967493293465,  23043.03251375972,  23418.03251061199,
    23831.96749768645,
};

Status field_fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kRender, message);
}

double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void cross3(const double* a, const double* b, double* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

bool normalize3(double* value) {
    const double length = std::sqrt(dot3(value, value));
    if (!std::isfinite(length) || length <= 1.0e-15) {
        return false;
    }
    value[0] /= length;
    value[1] /= length;
    value[2] /= length;
    return true;
}

// Python's round-half-even, used for the reference's shell and direction keys.
double round_half_even(double value, int digits) {
    const double scale = std::pow(10.0, digits);
    const double scaled = value * scale;
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    double rounded = lower;
    if (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0)) {
        rounded = lower + 1.0;
    }
    return rounded / scale;
}

// numpy.percentile with linear interpolation.
double percentile(std::vector<double> values, double percent) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = (static_cast<double>(values.size()) - 1.0) * percent / 100.0;
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper || upper >= values.size()) {
        return values[std::min(lower, values.size() - 1u)];
    }
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}


// Port of sofa_canonical.shift_signal_fft: positive shift delays, negative advances.
std::vector<double> shift_signal_fft(const std::vector<double>& values, double shift) {
    const std::size_t count = values.size();
    std::vector<double> result(count, 0.0);
    if (std::abs(shift) < 1.0e-12) {
        return values;
    }
    const std::size_t guard =
        std::max<std::size_t>(128u, static_cast<std::size_t>(std::ceil(std::abs(shift))) + 64u);
    const std::size_t fft_size = dsp::next_fast_len(count + 2u * guard);
    std::vector<Complex> spectrum(fft_size, Complex(0.0, 0.0));
    for (std::size_t index = 0u; index < count; ++index) {
        spectrum[guard + index] = Complex(values[index], 0.0);
    }
    std::vector<Complex> transformed;
    dsp::fft_any(spectrum, false, &transformed);
    const std::size_t half = fft_size / 2u;
    for (std::size_t bin = 0u; bin <= half; ++bin) {
        const double angle = -2.0 * kPi * static_cast<double>(bin) * shift /
                             static_cast<double>(fft_size);
        transformed[bin] *= std::polar(1.0, angle);
    }
    for (std::size_t bin = half + 1u; bin < fft_size; ++bin) {
        const double angle = -2.0 * kPi * static_cast<double>(bin - fft_size) * shift /
                             static_cast<double>(fft_size);
        transformed[bin] *= std::polar(1.0, angle);
    }
    std::vector<Complex> restored;
    dsp::fft_any(transformed, true, &restored);
    for (std::size_t index = 0u; index < count; ++index) {
        result[index] = restored[guard + index].real();
    }
    return result;
}

// public_filterbank.hybrid_gain_synthesis_dictionary for one FIR length.
std::vector<double> hybrid_gain_synthesis_dictionary(std::size_t sample_count,
                                                     const Kernels& kernels) {
    const std::size_t total =
        ((kLatencySamples + sample_count + 512u + kQmfHop - 1u) / kQmfHop) * kQmfHop;
    const std::size_t slots = total / kQmfHop;
    std::vector<double> impulse(total, 0.0);
    impulse[0] = 1.0;
    PublicFilterbank analysis_bank(kernels, 1u);
    std::vector<Complex> base;
    analysis_bank.analyze_full_rate(impulse, slots, &base);

    std::vector<Complex> hybrid(slots * kParameterCount * kHybridBands, Complex(0.0, 0.0));
    for (std::size_t slot = 0u; slot < slots; ++slot) {
        for (int band = 0; band < kHybridBands; ++band) {
            const Complex value = base[slot * kHybridBands + static_cast<std::size_t>(band)];
            hybrid[(slot * kParameterCount + static_cast<std::size_t>(2 * band)) * kHybridBands +
                   static_cast<std::size_t>(band)] = value;
            hybrid[(slot * kParameterCount + static_cast<std::size_t>(2 * band + 1)) * kHybridBands +
                   static_cast<std::size_t>(band)] = Complex(0.0, 1.0) * value;
        }
    }
    PublicFilterbank synthesis_bank(kernels, kParameterCount);
    std::vector<double> rendered;
    synthesis_bank.synthesize_full_rate(hybrid, slots, &rendered);
    std::vector<double> dictionary(sample_count * kParameterCount, 0.0);
    for (std::size_t tap = 0u; tap < sample_count; ++tap) {
        for (int parameter = 0; parameter < kParameterCount; ++parameter) {
            dictionary[tap * kParameterCount + static_cast<std::size_t>(parameter)] =
                rendered[(kLatencySamples + tap) * kParameterCount + static_cast<std::size_t>(parameter)];
        }
    }
    return dictionary;
}


// -------------------------------------------------------- linear algebra ----

// LU with partial pivoting, complex (mirrors the LAPACK zgesv path the reference
// takes when the ridge system is cast to complex for a complex right-hand side).
bool lu_solve_complex(std::vector<Complex>* matrix, int order, std::vector<Complex>* rhs,
                      int columns) {
    std::vector<int> pivot(static_cast<std::size_t>(order));
    for (int step = 0; step < order; ++step) {
        int best = step;
        double best_magnitude = std::abs((*matrix)[static_cast<std::size_t>(step) * order + step]);
        for (int row = step + 1; row < order; ++row) {
            const double magnitude =
                std::abs((*matrix)[static_cast<std::size_t>(row) * order + step]);
            if (magnitude > best_magnitude) {
                best_magnitude = magnitude;
                best = row;
            }
        }
        pivot[static_cast<std::size_t>(step)] = best;
        if (best != step) {
            for (int column = 0; column < order; ++column) {
                std::swap((*matrix)[static_cast<std::size_t>(step) * order + column],
                          (*matrix)[static_cast<std::size_t>(best) * order + column]);
            }
        }
        const Complex diagonal = (*matrix)[static_cast<std::size_t>(step) * order + step];
        if (std::abs(diagonal) <= 1.0e-300) {
            return false;
        }
        for (int row = step + 1; row < order; ++row) {
            const Complex factor =
                (*matrix)[static_cast<std::size_t>(row) * order + step] / diagonal;
            (*matrix)[static_cast<std::size_t>(row) * order + step] = factor;
            for (int column = step + 1; column < order; ++column) {
                (*matrix)[static_cast<std::size_t>(row) * order + column] -=
                    factor * (*matrix)[static_cast<std::size_t>(step) * order + column];
            }
        }
    }
    for (int column = 0; column < columns; ++column) {
        for (int step = 0; step < order; ++step) {
            const int source = pivot[static_cast<std::size_t>(step)];
            if (source != step) {
                std::swap((*rhs)[static_cast<std::size_t>(step) * columns + column],
                          (*rhs)[static_cast<std::size_t>(source) * columns + column]);
            }
        }
        for (int row = 1; row < order; ++row) {
            Complex sum = (*rhs)[static_cast<std::size_t>(row) * columns + column];
            for (int column2 = 0; column2 < row; ++column2) {
                sum -= (*matrix)[static_cast<std::size_t>(row) * order + column2] *
                       (*rhs)[static_cast<std::size_t>(column2) * columns + column];
            }
            (*rhs)[static_cast<std::size_t>(row) * columns + column] = sum;
        }
        for (int row = order - 1; row >= 0; --row) {
            Complex sum = (*rhs)[static_cast<std::size_t>(row) * columns + column];
            for (int column2 = row + 1; column2 < order; ++column2) {
                sum -= (*matrix)[static_cast<std::size_t>(row) * order + column2] *
                       (*rhs)[static_cast<std::size_t>(column2) * columns + column];
            }
            (*rhs)[static_cast<std::size_t>(row) * columns + column] =
                sum / (*matrix)[static_cast<std::size_t>(row) * order + row];
        }
    }
    return true;
}

// LU with partial pivoting, real (the LAPACK dgesv path).
bool lu_solve_real(std::vector<double>* matrix, int order, std::vector<double>* rhs, int columns) {
    std::vector<int> pivot(static_cast<std::size_t>(order));
    for (int step = 0; step < order; ++step) {
        int best = step;
        double best_magnitude = std::abs((*matrix)[static_cast<std::size_t>(step) * order + step]);
        for (int row = step + 1; row < order; ++row) {
            const double magnitude = std::abs((*matrix)[static_cast<std::size_t>(row) * order + step]);
            if (magnitude > best_magnitude) {
                best_magnitude = magnitude;
                best = row;
            }
        }
        pivot[static_cast<std::size_t>(step)] = best;
        if (best != step) {
            for (int column = 0; column < order; ++column) {
                std::swap((*matrix)[static_cast<std::size_t>(step) * order + column],
                          (*matrix)[static_cast<std::size_t>(best) * order + column]);
            }
        }
        const double diagonal = (*matrix)[static_cast<std::size_t>(step) * order + step];
        if (std::abs(diagonal) <= 1.0e-300) {
            return false;
        }
        for (int row = step + 1; row < order; ++row) {
            const double factor = (*matrix)[static_cast<std::size_t>(row) * order + step] / diagonal;
            (*matrix)[static_cast<std::size_t>(row) * order + step] = factor;
            for (int column = step + 1; column < order; ++column) {
                (*matrix)[static_cast<std::size_t>(row) * order + column] -=
                    factor * (*matrix)[static_cast<std::size_t>(step) * order + column];
            }
        }
    }
    for (int column = 0; column < columns; ++column) {
        for (int step = 0; step < order; ++step) {
            const int source = pivot[static_cast<std::size_t>(step)];
            if (source != step) {
                std::swap((*rhs)[static_cast<std::size_t>(step) * columns + column],
                          (*rhs)[static_cast<std::size_t>(source) * columns + column]);
            }
        }
        for (int row = 1; row < order; ++row) {
            double sum = (*rhs)[static_cast<std::size_t>(row) * columns + column];
            for (int column2 = 0; column2 < row; ++column2) {
                sum -= (*matrix)[static_cast<std::size_t>(row) * order + column2] *
                       (*rhs)[static_cast<std::size_t>(column2) * columns + column];
            }
            (*rhs)[static_cast<std::size_t>(row) * columns + column] = sum;
        }
        for (int row = order - 1; row >= 0; --row) {
            double sum = (*rhs)[static_cast<std::size_t>(row) * columns + column];
            for (int column2 = row + 1; column2 < order; ++column2) {
                sum -= (*matrix)[static_cast<std::size_t>(row) * order + column2] *
                       (*rhs)[static_cast<std::size_t>(column2) * columns + column];
            }
            (*rhs)[static_cast<std::size_t>(row) * columns + column] =
                sum / (*matrix)[static_cast<std::size_t>(row) * order + row];
        }
    }
    return true;
}

// ---------------------------------------------------- spherical harmonics ---

// P_degree^order(x) with the Condon-Shortley phase (spherical_harmonics.py).
double associated_legendre(int order, int degree, double x) {
    double p_mm = 1.0;
    if (order != 0) {
        double double_factorial = 1.0;
        for (int value = 1; value < 2 * order; value += 2) {
            double_factorial *= static_cast<double>(value);
        }
        p_mm = (order % 2 == 0 ? 1.0 : -1.0) * double_factorial *
               std::pow(std::max(0.0, 1.0 - x * x), 0.5 * static_cast<double>(order));
    }
    if (degree == order) {
        return p_mm;
    }
    double previous_previous = p_mm;
    double previous = x * static_cast<double>(2 * order + 1) * p_mm;
    if (degree == order + 1) {
        return previous;
    }
    for (int current = order + 2; current <= degree; ++current) {
        const double value =
            (static_cast<double>(2 * current - 1) * x * previous -
             static_cast<double>(current + order - 1) * previous_previous) /
            static_cast<double>(current - order);
        previous_previous = previous;
        previous = value;
    }
    return previous;
}

double factorial(int value) {
    double result = 1.0;
    for (int index = 2; index <= value; ++index) {
        result *= static_cast<double>(index);
    }
    return result;
}

// [M, (order+1)^2] orthonormal real harmonics in ACN order.
void real_spherical_harmonics(const std::vector<double>& directions, std::size_t count, int order,
                              std::vector<double>* basis) {
    const std::size_t terms = static_cast<std::size_t>((order + 1) * (order + 1));
    basis->assign(count * terms, 0.0);
    for (std::size_t point = 0u; point < count; ++point) {
        double unit[3] = {directions[point * 3u], directions[point * 3u + 1u],
                          directions[point * 3u + 2u]};
        if (!normalize3(unit)) {
            unit[0] = 1.0;
            unit[1] = 0.0;
            unit[2] = 0.0;
        }
        const double azimuth = std::atan2(unit[1], unit[0]);
        const double cos_colatitude = std::min(1.0, std::max(-1.0, unit[2]));
        std::size_t column = 0u;
        for (int degree = 0; degree <= order; ++degree) {
            for (int m = -degree; m <= degree; ++m) {
                const int absolute = std::abs(m);
                const double normalization =
                    std::sqrt((2.0 * static_cast<double>(degree) + 1.0) / (4.0 * kPi) *
                              factorial(degree - absolute) / factorial(degree + absolute));
                const double legendre = associated_legendre(absolute, degree, cos_colatitude);
                double value = 0.0;
                if (m < 0) {
                    value = std::sqrt(2.0) * normalization * legendre *
                            std::sin(static_cast<double>(absolute) * azimuth);
                } else if (m > 0) {
                    value = std::sqrt(2.0) * normalization * legendre *
                            std::cos(static_cast<double>(m) * azimuth);
                } else {
                    value = normalization * legendre;
                }
                (*basis)[point * terms + column] = value;
                ++column;
            }
        }
    }
}

// Area weights of an irregular full-sphere grid (spherical_harmonics.py).
//
// The Voronoi cell of p_i is the intersection of the hemispheres
// dot(x, p_i - p_j) >= 0.  Projected gnomonicly onto the tangent plane at p_i
// that intersection is a convex polygon of half-planes, so the cell can be
// clipped directly without building a convex hull: this has no triangulation
// ambiguity on cocircular grids (latitude rings) and reproduces the reference's
// spherical Voronoi areas to roundoff.  The spherical area is the solid-angle fan
// from p_i over the cell's corners.
std::vector<double> spherical_voronoi_weights(const std::vector<double>& directions,
                                              std::size_t count) {
    std::vector<double> weights(count, 1.0 / static_cast<double>(count));
    if (count < 4u) {
        return weights;
    }
    constexpr double kBox = 4.0;  // a cap of ~76 degrees: far larger than any cell
    std::vector<double> areas(count, 0.0);
    std::vector<std::pair<double, double>> polygon;
    std::vector<std::pair<double, double>> clipped;
    for (std::size_t point = 0u; point < count; ++point) {
        double p[3] = {directions[point * 3u], directions[point * 3u + 1u],
                       directions[point * 3u + 2u]};
        if (!normalize3(p)) {
            return weights;
        }
        double helper[3] = {0.0, 0.0, 1.0};
        if (std::abs(p[2]) >= 0.9) {
            helper[0] = 1.0;
            helper[2] = 0.0;
        }
        double east[3];
        double north[3];
        cross3(helper, p, east);
        if (!normalize3(east)) {
            return weights;
        }
        cross3(p, east, north);

        polygon.clear();
        polygon.emplace_back(-kBox, -kBox);
        polygon.emplace_back(kBox, -kBox);
        polygon.emplace_back(kBox, kBox);
        polygon.emplace_back(-kBox, kBox);
        for (std::size_t other = 0u; other < count && polygon.size() >= 3u; ++other) {
            if (other == point) {
                continue;
            }
            const double* q = directions.data() + other * 3u;
            const double cosine = dot3(p, q);
            const double a = dot3(east, q);
            const double b = dot3(north, q);
            clipped.clear();
            for (std::size_t index = 0u; index < polygon.size(); ++index) {
                const std::pair<double, double>& current = polygon[index];
                const std::pair<double, double>& next = polygon[(index + 1u) % polygon.size()];
                const double value = (1.0 - cosine) - current.first * a - current.second * b;
                const double next_value = (1.0 - cosine) - next.first * a - next.second * b;
                if (value >= 0.0) {
                    clipped.push_back(current);
                }
                if ((value >= 0.0) != (next_value >= 0.0)) {
                    const double fraction = value / (value - next_value);
                    clipped.emplace_back(current.first + fraction * (next.first - current.first),
                                         current.second + fraction * (next.second - current.second));
                }
            }
            polygon.swap(clipped);
        }
        if (polygon.size() < 3u) {
            return weights;  // a cell that does not close: keep the uniform fallback
        }
        double area = 0.0;
        for (std::size_t index = 0u; index < polygon.size(); ++index) {
            const std::pair<double, double>& current = polygon[index];
            const std::pair<double, double>& next = polygon[(index + 1u) % polygon.size()];
            double first[3] = {p[0] + current.first * east[0] + current.second * north[0],
                               p[1] + current.first * east[1] + current.second * north[1],
                               p[2] + current.first * east[2] + current.second * north[2]};
            double second[3] = {p[0] + next.first * east[0] + next.second * north[0],
                                p[1] + next.first * east[1] + next.second * north[1],
                                p[2] + next.first * east[2] + next.second * north[2]};
            if (!normalize3(first) || !normalize3(second)) {
                return weights;
            }
            double cross[3];
            cross3(first, second, cross);
            const double determinant = dot3(p, cross);
            const double denominator =
                1.0 + dot3(p, first) + dot3(first, second) + dot3(second, p);
            area += std::abs(2.0 * std::atan2(determinant, denominator));
        }
        if (!std::isfinite(area) || area <= 0.0) {
            return weights;
        }
        areas[point] = area;
    }
    double sum = 0.0;
    for (const double area : areas) {
        sum += area;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return weights;
    }
    for (std::size_t point = 0u; point < count; ++point) {
        weights[point] = areas[point] / sum;
    }
    return weights;
}
// ---------------------------------------------------------- coordinates -----

double length_factor(const std::string& token, bool* found) {
    static const std::pair<const char*, double> kTable[] = {
        {"m", 1.0},          {"metre", 1.0},      {"metres", 1.0},      {"meter", 1.0},
        {"meters", 1.0},     {"cm", 1.0e-2},      {"centimetre", 1.0e-2}, {"centimetres", 1.0e-2},
        {"centimeter", 1.0e-2}, {"centimeters", 1.0e-2}, {"mm", 1.0e-3},  {"millimetre", 1.0e-3},
        {"millimetres", 1.0e-3}, {"millimeter", 1.0e-3}, {"millimeters", 1.0e-3},
    };
    for (const auto& entry : kTable) {
        if (token == entry.first) {
            *found = true;
            return entry.second;
        }
    }
    *found = false;
    return 0.0;
}

bool angle_in_radians(const std::string& token, double value, double* out) {
    if (token == "degree" || token == "degrees") {
        *out = value * (kPi / 180.0);
        return true;
    }
    if (token == "radian" || token == "radians") {
        *out = value;
        return true;
    }
    return false;
}

std::vector<std::string> split_units(const std::string& units) {
    std::vector<std::string> tokens;
    std::string current;
    for (const char character : units) {
        if (character == ',') {
            const std::size_t begin = current.find_first_not_of(" \t");
            const std::size_t end = current.find_last_not_of(" \t");
            if (begin != std::string::npos) {
                std::string token = current.substr(begin, end - begin + 1u);
                std::transform(token.begin(), token.end(), token.begin(),
                               [](unsigned char value) {
                                   return static_cast<char>(std::tolower(value));
                               });
                tokens.push_back(token);
            }
            current.clear();
        } else {
            current.push_back(character);
        }
    }
    const std::size_t begin = current.find_first_not_of(" \t");
    const std::size_t end = current.find_last_not_of(" \t");
    if (begin != std::string::npos) {
        std::string token = current.substr(begin, end - begin + 1u);
        std::transform(token.begin(), token.end(), token.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        tokens.push_back(token);
    }
    return tokens;
}

std::string lowered(const std::string& text) {
    std::string result = text;
    const std::size_t begin = result.find_first_not_of(" \t");
    const std::size_t end = result.find_last_not_of(" \t");
    result = begin == std::string::npos ? std::string() : result.substr(begin, end - begin + 1u);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return result;
}

// coordinates_to_cartesian_m: spherical or Cartesian rows to metres, in place.
Status convert_coordinates(const std::vector<double>& values, std::size_t rows,
                           const std::string& type, const std::string& units,
                           const std::string& variable, std::vector<double>* out) {
    if (values.size() != rows * 3u) {
        return field_fail(JOC_ERR_HRTF_FORMAT, variable + " must contain finite C=3 coordinates");
    }
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              variable + " must contain finite C=3 coordinates");
        }
    }
    const std::string kind = lowered(type);
    const std::vector<std::string> tokens = split_units(units);
    out->assign(values.size(), 0.0);
    if (kind == "cartesian") {
        double factors[3] = {0.0, 0.0, 0.0};
        if (tokens.size() == 1u) {
            bool found = false;
            const double factor = length_factor(tokens[0], &found);
            if (!found) {
                return field_fail(JOC_ERR_HRTF_FORMAT,
                                  "unsupported Cartesian units for " + variable + ": " + units);
            }
            factors[0] = factors[1] = factors[2] = factor;
        } else if (tokens.size() == 3u) {
            for (int axis = 0; axis < 3; ++axis) {
                bool found = false;
                factors[axis] = length_factor(tokens[static_cast<std::size_t>(axis)], &found);
                if (!found) {
                    return field_fail(JOC_ERR_HRTF_FORMAT,
                                      "unsupported Cartesian units for " + variable + ": " + units);
                }
            }
        } else {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              "unsupported Cartesian units for " + variable + ": " + units);
        }
        for (std::size_t row = 0u; row < rows; ++row) {
            for (int axis = 0; axis < 3; ++axis) {
                (*out)[row * 3u + static_cast<std::size_t>(axis)] =
                    values[row * 3u + static_cast<std::size_t>(axis)] * factors[axis];
            }
        }
        return Status::success();
    }
    if (kind != "spherical" || tokens.size() != 3u) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "unsupported coordinates for " + variable +
                                                   ": Type=" + type + ", Units=" + units);
    }
    bool radius_found = false;
    const double radius_factor = length_factor(tokens[2], &radius_found);
    if (!radius_found) {
        return field_fail(JOC_ERR_HRTF_FORMAT,
                          "unsupported spherical radius unit for " + variable + ": " + units);
    }
    for (std::size_t row = 0u; row < rows; ++row) {
        double azimuth = 0.0;
        double elevation = 0.0;
        if (!angle_in_radians(tokens[0], values[row * 3u], &azimuth) ||
            !angle_in_radians(tokens[1], values[row * 3u + 1u], &elevation)) {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              "unsupported spherical angle units for " + variable + ": " + units);
        }
        const double radius = values[row * 3u + 2u] * radius_factor;
        if (radius < 0.0) {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              variable + " contains a negative spherical radius");
        }
        const double horizontal = std::cos(elevation);
        (*out)[row * 3u] = radius * horizontal * std::cos(azimuth);
        (*out)[row * 3u + 1u] = radius * horizontal * std::sin(azimuth);
        (*out)[row * 3u + 2u] = radius * std::sin(elevation);
    }
    return Status::success();
}

// ------------------------------------------------------------- JSON ---------

std::string json_escape(const std::string& text) {
    std::string out = "\"";
    for (const char character : text) {
        const unsigned char value = static_cast<unsigned char>(character);
        switch (character) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (value < 0x20u) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", value);
                    out += buffer;
                } else {
                    out.push_back(character);
                }
                break;
        }
    }
    out += "\"";
    return out;
}

std::string json_number(double value) { return io::python_float_repr(value); }

// Python's float.hex(), the representation json.dumps uses for the cache-key
// payload, so the key matches the reference byte for byte.
std::string python_hex(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    if (value == 0.0) {
        return std::signbit(value) ? "-0x0.0p+0" : "0x0.0p+0";
    }
    std::uint64_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const bool negative = (bits >> 63u) != 0u;
    const int exponent = static_cast<int>((bits >> 52u) & 0x7FFu);
    const std::uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;
    char buffer[40];
    if (exponent == 0) {
        std::snprintf(buffer, sizeof(buffer), "%s0x0.%013llxp-1022", negative ? "-" : "",
                      static_cast<unsigned long long>(mantissa));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%s0x1.%013llxp%+d", negative ? "-" : "",
                      static_cast<unsigned long long>(mantissa), exponent - 1023);
    }
    return buffer;
}

using JsonMembers = std::vector<std::pair<std::string, std::string>>;

std::string json_object(JsonMembers members) {
    std::sort(members.begin(), members.end(),
              [](const std::pair<std::string, std::string>& left,
                 const std::pair<std::string, std::string>& right) {
                  return left.first < right.first;
              });
    std::string out = "{";
    for (std::size_t index = 0u; index < members.size(); ++index) {
        if (index != 0u) {
            out += ",";
        }
        out += json_escape(members[index].first) + ":" + members[index].second;
    }
    out += "}";
    return out;
}

std::string json_array(const std::vector<std::string>& values) {
    std::string out = "[";
    for (std::size_t index = 0u; index < values.size(); ++index) {
        if (index != 0u) {
            out += ",";
        }
        out += values[index];
    }
    out += "]";
    return out;
}

// ------------------------------------------------------- cache contract -----

std::string sha256_hex_upper(const void* data, std::size_t size) {
    crypto::Sha256 hash;
    hash.update(data, size);
    std::string digest = hash.finish_hex();
    std::transform(digest.begin(), digest.end(), digest.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return digest;
}

std::string to_upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return text;
}

// filterbank_fingerprint(): stable identifiers of the embedded public tables.
std::string filterbank_fingerprint_json() {
    const Kernels& kernels = builtin_kernels();
    auto hash_float32 = [](const std::vector<double>& values) {
        std::vector<float> narrowed(values.size());
        for (std::size_t index = 0u; index < values.size(); ++index) {
            narrowed[index] = static_cast<float>(values[index]);
        }
        return sha256_hex_upper(narrowed.data(), narrowed.size() * sizeof(float));
    };
    auto hash_float64 = [](const std::vector<double>& values) {
        return sha256_hex_upper(values.data(), values.size() * sizeof(double));
    };
    const std::string band_centers_sha =
        sha256_hex_upper(kBandCenters, static_cast<std::size_t>(kHybridBands) * sizeof(double));
    const JsonMembers arrays = {
        // The fingerprint also covers the archive's format_version member.
        {"format_version",
         json_escape(sha256_hex_upper(&kFilterbankFormatVersion, sizeof(kFilterbankFormatVersion)))},
        {"hybrid_analysis_low_kernel", json_escape(hash_float32(kernels.hybrid_low))},
        {"hybrid_synthesis_indices",
         json_escape(sha256_hex_upper(kernels.hybrid_indices.data(),
                                      kernels.hybrid_indices.size() * sizeof(std::int16_t)))},
        {"hybrid_synthesis_values", json_escape(hash_float32(kernels.hybrid_values))},
        {"qmf_analysis_coefficients", json_escape(hash_float32(kernels.qmf_analysis))},
        {"qmf_synthesis_basis", json_escape(hash_float64(kernels.qmf_basis))},
        {"qmf_synthesis_taps", json_escape(hash_float64(kernels.qmf_taps))},
    };
    return json_object({
        {"archive_sha256", json_escape(kFilterbankArchiveSha256)},
        {"array_sha256", json_object(arrays)},
        {"band_centers_sha256", json_escape(band_centers_sha)},
        {"table_version", json_escape(kFilterbankTableVersion)},
    });
}

}  // namespace

std::vector<double> hybrid_gain_synthesis_dictionary_for_check(std::size_t sample_count) {
    return hybrid_gain_synthesis_dictionary(sample_count, builtin_kernels());
}

std::vector<double> analysis_impulse_for_check(std::size_t total_samples) {
    const std::size_t slots = total_samples / kQmfHop;
    std::vector<double> impulse(total_samples, 0.0);
    impulse[0] = 1.0;
    PublicFilterbank bank(builtin_kernels(), 1u);
    std::vector<Complex> base;
    bank.analyze_full_rate(impulse, slots, &base);
    std::vector<double> out(base.size() * 2u, 0.0);
    for (std::size_t index = 0u; index < base.size(); ++index) {
        out[index * 2u] = base[index].real();
        out[index * 2u + 1u] = base[index].imag();
    }
    return out;
}

const std::vector<double>& hybrid_band_center_frequencies_hz() {
    static const std::vector<double> centers(kBandCenters, kBandCenters + kHybridBands);
    return centers;
}

std::string compiled_hrtf_cache_key(const std::string& source_sha256, double sample_rate_hz,
                                    double shell_radius_m, int order, double projection_ridge,
                                    double sh_ridge) {
    const std::string payload = json_object({
        {"compiler_version", json_escape(kCompilerVersion)},
        {"filterbank", filterbank_fingerprint_json()},
        {"order", std::to_string(order)},
        {"phase_policy_version", json_escape(kPhasePolicyVersion)},
        {"projection_ridge", json_escape(python_hex(projection_ridge))},
        {"sh_convention", json_escape(kShConvention)},
        {"shell_radius_m", json_escape(python_hex(shell_radius_m))},
        {"source_sha256", json_escape(to_upper(source_sha256))},
        {"spherical_harmonic_ridge", json_escape(python_hex(sh_ridge))},
        {"target_sample_rate_hz", json_escape(python_hex(sample_rate_hz))},
    });
    const std::string encoded = std::string("JOC-HRTF-CACHE-KEY-V1\0", 22u) + payload;
    return to_upper(sha256_hex_upper(encoded.data(), encoded.size()));
}

std::string field_payload_sha256(const Field& field) {
    crypto::Sha256 hash;
    const char prefix[] = "JOC-HRTF-CACHE-PAYLOAD-V1";
    hash.update(prefix, sizeof(prefix) - 1u);
    const std::uint8_t zero = 0u;
    hash.update(&zero, 1u);
    auto feed = [&](const char* name, const char* dtype, const std::string& shape,
                    const void* data, std::size_t size) {
        const std::string name_text(name);
        const std::string dtype_text(dtype);
        hash.update(name_text.data(), name_text.size());
        hash.update(&zero, 1u);
        hash.update(dtype_text.data(), dtype_text.size());
        hash.update(&zero, 1u);
        hash.update(shape.data(), shape.size());
        hash.update(&zero, 1u);
        hash.update(data, size);
    };
    feed("band_center_frequencies_hz", "<f8", "[77]", field.band_centers_hz.data(),
         field.band_centers_hz.size() * sizeof(double));
    feed("coefficients", "<c16", "[36, 2, 77]", field.coefficients.data(),
         field.coefficients.size() * sizeof(double));
    feed("delay_coefficients", "<f8", "[36, 2]", field.delay_coefficients.data(),
         field.delay_coefficients.size() * sizeof(double));
    feed("delay_bounds", "<f8", "[2, 2]", field.delay_bounds.data(),
         field.delay_bounds.size() * sizeof(double));
    return to_upper(hash.finish_hex());
}

std::string cache_file_name(const std::string& display_name, const std::string& cache_key) {
    std::string stem = "hrtf";
    if (!display_name.empty()) {
        const std::size_t slash = display_name.find_last_of("/\\");
        stem = slash == std::string::npos ? display_name : display_name.substr(slash + 1u);
        const std::size_t dot = stem.find_last_of('.');
        if (dot != std::string::npos && dot != 0u) {
            stem = stem.substr(0u, dot);
        }
    }
    std::string safe;
    for (const char character : stem) {
        const bool allowed = (character >= 'A' && character <= 'Z') ||
                             (character >= 'a' && character <= 'z') ||
                             (character >= '0' && character <= '9') || character == '.' ||
                             character == '_' || character == '-';
        safe.push_back(allowed ? character : '_');
    }
    const std::size_t begin = safe.find_first_not_of("._");
    const std::size_t end = safe.find_last_not_of("._");
    safe = begin == std::string::npos ? std::string() : safe.substr(begin, end - begin + 1u);
    if (safe.empty()) {
        safe = "hrtf";
    }
    return safe + "." + cache_key.substr(0u, 20u) + ".jochrtf";
}

Status write_jochrtf(const Field& field, const std::string& path) {
    if (field.metadata_json.empty() || field.payload_sha256.empty()) {
        return field_fail(JOC_ERR_INVALID_ARGUMENT,
                          "compiled HRTF field has no metadata to write");
    }
    std::vector<io::NpyMember> members;
    auto add_real = [&](const char* name, const std::vector<double>& values,
                        std::vector<std::uint64_t> shape) {
        io::NpyMember member;
        member.name = name;
        member.descr = "<f8";
        member.shape = std::move(shape);
        member.data.resize(values.size() * sizeof(double));
        std::memcpy(member.data.data(), values.data(), member.data.size());
        members.push_back(std::move(member));
    };
    add_real("band_center_frequencies_hz", field.band_centers_hz, {kHybridBands});
    {
        io::NpyMember member;
        member.name = "coefficients";
        member.descr = "<c16";
        member.shape = {kFieldTerms, 2u, kHybridBands};
        member.data.resize(field.coefficients.size() * sizeof(double));
        std::memcpy(member.data.data(), field.coefficients.data(), member.data.size());
        members.push_back(std::move(member));
    }
    add_real("delay_coefficients", field.delay_coefficients, {kFieldTerms, 2u});
    add_real("delay_bounds", field.delay_bounds, {2u, 2u});
    {
        // metadata_json is a NumPy Unicode scalar string: UTF-32LE, no terminator.
        io::NpyMember member;
        member.name = "metadata_json";
        member.descr = "<U" + std::to_string(field.metadata_json.size());
        member.shape = {};
        member.data = io::utf8_to_utf32le(field.metadata_json);
        members.push_back(std::move(member));
    }
    std::string error;
    if (!io::write_zip(path, members, &error)) {
        return field_fail(JOC_ERR_OUTPUT_WRITE, "cannot write compiled HRTF cache: " + error);
    }
    return Status::success();
}

namespace {

// -------------------------------------------------------- normal system -----

void normal_system(const std::vector<double>& basis, std::size_t count, int terms,
                   const std::vector<double>& weights, double ridge,
                   std::vector<double>* system, std::vector<double>* weighted_basis) {
    weighted_basis->assign(count * static_cast<std::size_t>(terms), 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        for (int term = 0; term < terms; ++term) {
            (*weighted_basis)[row * static_cast<std::size_t>(terms) + static_cast<std::size_t>(term)] =
                basis[row * static_cast<std::size_t>(terms) + static_cast<std::size_t>(term)] *
                weights[row];
        }
    }
    system->assign(static_cast<std::size_t>(terms) * static_cast<std::size_t>(terms), 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        for (int left = 0; left < terms; ++left) {
            const double value =
                basis[row * static_cast<std::size_t>(terms) + static_cast<std::size_t>(left)];
            if (value == 0.0) {
                continue;
            }
            for (int right = 0; right < terms; ++right) {
                (*system)[static_cast<std::size_t>(left) * static_cast<std::size_t>(terms) +
                          static_cast<std::size_t>(right)] +=
                    value * (*weighted_basis)[row * static_cast<std::size_t>(terms) +
                                              static_cast<std::size_t>(right)];
            }
        }
    }
    double trace = 0.0;
    for (int term = 0; term < terms; ++term) {
        trace += (*system)[static_cast<std::size_t>(term) * static_cast<std::size_t>(terms) +
                           static_cast<std::size_t>(term)];
    }
    const double scale = trace / static_cast<double>(terms);
    for (int term = 0; term < terms; ++term) {
        (*system)[static_cast<std::size_t>(term) * static_cast<std::size_t>(terms) +
                  static_cast<std::size_t>(term)] += ridge * scale;
    }
}

// fit_real_spherical_harmonics for complex targets: [terms, columns].
std::vector<Complex> fit_complex(const std::vector<double>& basis, std::size_t count, int terms,
                                 const std::vector<double>& weights, double ridge,
                                 const std::vector<Complex>& values, std::size_t columns) {
    std::vector<double> system;
    std::vector<double> weighted_basis;
    normal_system(basis, count, terms, weights, ridge, &system, &weighted_basis);
    std::vector<Complex> matrix(system.size());
    for (std::size_t index = 0u; index < system.size(); ++index) {
        matrix[index] = Complex(system[index], 0.0);
    }
    std::vector<Complex> right(static_cast<std::size_t>(terms) * columns, Complex(0.0, 0.0));
    for (std::size_t row = 0u; row < count; ++row) {
        for (int term = 0; term < terms; ++term) {
            const double basis_value =
                basis[row * static_cast<std::size_t>(terms) + static_cast<std::size_t>(term)];
            if (basis_value == 0.0) {
                continue;
            }
            for (std::size_t column = 0u; column < columns; ++column) {
                right[static_cast<std::size_t>(term) * columns + column] +=
                    basis_value * weights[row] * values[row * columns + column];
            }
        }
    }
    lu_solve_complex(&matrix, terms, &right, static_cast<int>(columns));
    return right;
}

// fit_real_spherical_harmonics for real targets: [terms, columns].
std::vector<double> fit_real(const std::vector<double>& basis, std::size_t count, int terms,
                             const std::vector<double>& weights, double ridge,
                             const std::vector<double>& values, std::size_t columns) {
    std::vector<double> system;
    std::vector<double> weighted_basis;
    normal_system(basis, count, terms, weights, ridge, &system, &weighted_basis);
    std::vector<double> right(static_cast<std::size_t>(terms) * columns, 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        for (int term = 0; term < terms; ++term) {
            const double basis_value =
                basis[row * static_cast<std::size_t>(terms) + static_cast<std::size_t>(term)];
            if (basis_value == 0.0) {
                continue;
            }
            for (std::size_t column = 0u; column < columns; ++column) {
                right[static_cast<std::size_t>(term) * columns + column] +=
                    basis_value * weights[row] * values[row * columns + column];
            }
        }
    }
    lu_solve_real(&system, terms, &right, static_cast<int>(columns));
    return right;
}

// --------------------------------------------------------- time alignment ---

// sofa_canonical._subsample_peak: parabolic peak of |values|.
double subsample_peak(const std::vector<double>& values) {
    std::size_t index = 0u;
    double magnitude = -1.0;
    for (std::size_t position = 0u; position < values.size(); ++position) {
        const double current = std::abs(values[position]);
        if (current > magnitude) {
            magnitude = current;
            index = position;
        }
    }
    if (index == 0u || index + 1u >= values.size()) {
        return static_cast<double>(index);
    }
    const double previous = std::abs(values[index - 1u]);
    const double current = std::abs(values[index]);
    const double next = std::abs(values[index + 1u]);
    const double denominator = previous - 2.0 * current + next;
    double correction = 0.0;
    if (std::abs(denominator) >= 1.0e-30) {
        correction = 0.5 * (previous - next) / denominator;
    }
    correction = std::min(0.5, std::max(-0.5, correction));
    return static_cast<double>(index) + correction;
}

struct AlignedHrtf {
    std::vector<double> aligned;            // [M,2,N]
    std::vector<double> runtime_delay;      // [M,2]
    std::vector<double> embedded_removed;   // [M,2]
    std::string delay_source;
};

// sofa_canonical.time_align_hrtf: separate exactly one delay representation.
AlignedHrtf time_align_hrtf(const CanonicalHrtf& canonical) {
    AlignedHrtf result;
    const std::size_t measurements = canonical.measurements;
    const std::size_t taps = canonical.taps;
    result.aligned = canonical.hrir;
    result.runtime_delay.assign(measurements * 2u, 0.0);
    result.embedded_removed.assign(measurements * 2u, 0.0);
    double maximum = 0.0;
    for (const double value : canonical.delay_samples) {
        maximum = std::max(maximum, std::abs(value));
    }
    if (maximum > 1.0e-12) {
        result.runtime_delay = canonical.delay_samples;
        result.delay_source = "Data.Delay (external; applied once at render time)";
        return result;
    }
    std::size_t used_peak = 0u;
    std::size_t retained = 0u;
    std::vector<double> ear(taps, 0.0);
    // hrir[measurement][channel] as a tap range.
    auto ear_range = [&](std::size_t measurement, int channel) {
        const std::size_t first = (measurement * 2u + static_cast<std::size_t>(channel)) * taps;
        return std::make_pair(canonical.hrir.begin() + static_cast<std::ptrdiff_t>(first),
                              canonical.hrir.begin() + static_cast<std::ptrdiff_t>(first + taps));
    };
    for (std::size_t measurement = 0u; measurement < measurements; ++measurement) {
        double peaks[2] = {0.0, 0.0};
        for (int channel = 0; channel < 2; ++channel) {
            const auto range = ear_range(measurement, channel);
            std::copy(range.first, range.second, ear.begin());
            peaks[channel] = subsample_peak(ear);
        }
        if (std::max(peaks[0], peaks[1]) > 2.0) {
            ++used_peak;
        } else {
            peaks[0] = 0.0;
            peaks[1] = 0.0;
            ++retained;
        }
        for (int channel = 0; channel < 2; ++channel) {
            const std::size_t ear_index = measurement * 2u + static_cast<std::size_t>(channel);
            result.runtime_delay[ear_index] = peaks[channel];
            result.embedded_removed[ear_index] = peaks[channel];
            if (peaks[channel] != 0.0) {
                const auto range = ear_range(measurement, channel);
                std::copy(range.first, range.second, ear.begin());
                const std::vector<double> shifted = shift_signal_fft(ear, -peaks[channel]);
                std::copy(shifted.begin(), shifted.end(),
                          result.aligned.begin() + static_cast<std::ptrdiff_t>(ear_index * taps));
            }
        }
    }
    result.delay_source = "embedded Data.IR arrival separation: peak=" +
                          std::to_string(used_peak) +
                          ", zero-origin embedded phase retained=" + std::to_string(retained) +
                          "; positive onset restored once at render time";
    return result;
}

// ------------------------------------------------------------- pipeline -----

// canonical.shell_indices: the measurements on the shell nearest to radius_m.
}  // namespace

std::vector<std::size_t> canonical_shell_indices(const CanonicalHrtf& canonical, double radius_m,
                                                 double* actual_radius_m) {
    std::vector<double> shells;
    for (const double radius : canonical.measurement_radius_m) {
        const double rounded = round_half_even(radius, 9);
        if (std::find(shells.begin(), shells.end(), rounded) == shells.end()) {
            shells.push_back(rounded);
        }
    }
    std::sort(shells.begin(), shells.end());
    double shell = shells.empty() ? radius_m : shells.front();
    double best = std::numeric_limits<double>::infinity();
    for (const double candidate : shells) {
        const double distance = std::abs(candidate - radius_m);
        if (distance < best) {
            best = distance;
            shell = candidate;
        }
    }
    std::vector<std::size_t> indices;
    double total = 0.0;
    for (std::size_t index = 0u; index < canonical.measurement_radius_m.size(); ++index) {
        if (std::abs(canonical.measurement_radius_m[index] - shell) <= 5.0e-7) {
            indices.push_back(index);
            total += canonical.measurement_radius_m[index];
        }
    }
    *actual_radius_m = indices.empty() ? shell : total / static_cast<double>(indices.size());
    return indices;
}

namespace {

// _group_coincident: merge measurements that share a direction (keys are the
// directions rounded to ten decimals, grouped in sorted key order).
void group_coincident(const std::vector<double>& directions, std::size_t count,
                      const std::vector<Complex>& gains, std::size_t columns,
                      const std::vector<double>& delays, std::vector<double>* out_directions,
                      std::vector<Complex>* out_gains, std::vector<double>* out_delays) {
    struct Key {
        double value[3];
        std::size_t first;
    };
    std::vector<Key> keys(count);
    for (std::size_t index = 0u; index < count; ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            keys[index].value[axis] =
                round_half_even(directions[index * 3u + static_cast<std::size_t>(axis)], 10);
        }
        keys[index].first = index;
    }
    std::sort(keys.begin(), keys.end(), [](const Key& left, const Key& right) {
        for (int axis = 0; axis < 3; ++axis) {
            if (left.value[axis] != right.value[axis]) {
                return left.value[axis] < right.value[axis];
            }
        }
        return left.first < right.first;
    });
    std::vector<Key> unique;
    std::vector<std::size_t> group_of(count, 0u);
    for (const Key& key : keys) {
        if (unique.empty() || unique.back().value[0] != key.value[0] ||
            unique.back().value[1] != key.value[1] || unique.back().value[2] != key.value[2]) {
            unique.push_back(key);
        } else if (key.first < unique.back().first) {
            unique.back().first = key.first;
        }
        group_of[key.first] = unique.size() - 1u;
    }
    const std::size_t groups = unique.size();
    out_directions->assign(groups * 3u, 0.0);
    out_gains->assign(groups * columns, Complex(0.0, 0.0));
    out_delays->assign(groups * 2u, 0.0);
    std::vector<double> counts(groups, 0.0);
    for (std::size_t index = 0u; index < count; ++index) {
        const std::size_t group = group_of[index];
        ++counts[group];
        for (int axis = 0; axis < 3; ++axis) {
            (*out_directions)[group * 3u + static_cast<std::size_t>(axis)] +=
                directions[index * 3u + static_cast<std::size_t>(axis)];
        }
        for (std::size_t column = 0u; column < columns; ++column) {
            (*out_gains)[group * columns + column] += gains[index * columns + column];
        }
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            (*out_delays)[group * 2u + ear] += delays[index * 2u + ear];
        }
    }
    for (std::size_t group = 0u; group < groups; ++group) {
        for (std::size_t column = 0u; column < columns; ++column) {
            (*out_gains)[group * columns + column] /= counts[group];
        }
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            (*out_delays)[group * 2u + ear] /= counts[group];
        }
    }
}

std::string basename_of(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1u);
}

// sofa_canonical._processing_label: the non-empty identifying attributes.
std::string processing_label(const SofaHrir& sofa) {
    std::vector<std::string> parts;
    const std::string candidates[4] = {sofa.database_name, sofa.title, sofa.listener_short_name,
                                       sofa.comment};
    for (const std::string& value : candidates) {
        const std::size_t begin = value.find_first_not_of(" \t");
        const std::size_t end = value.find_last_not_of(" \t");
        if (begin == std::string::npos) {
            continue;
        }
        const std::string trimmed = value.substr(begin, end - begin + 1u);
        if (std::find(parts.begin(), parts.end(), trimmed) == parts.end()) {
            parts.push_back(trimmed);
        }
    }
    std::string label;
    for (std::size_t index = 0u; index < parts.size(); ++index) {
        if (index != 0u) {
            label += " | ";
        }
        label += parts[index];
    }
    return label;
}

double minimum_of(const std::vector<double>& values) {
    double result = std::numeric_limits<double>::infinity();
    for (const double value : values) {
        result = std::min(result, value);
    }
    return values.empty() ? 0.0 : result;
}

double maximum_of(const std::vector<double>& values) {
    double result = -std::numeric_limits<double>::infinity();
    for (const double value : values) {
        result = std::max(result, value);
    }
    return values.empty() ? 0.0 : result;
}

}  // namespace

Status canonicalize_sofa(const SofaHrir& sofa, CanonicalHrtf* out) {
    if (out == nullptr) {
        return field_fail(JOC_ERR_INVALID_ARGUMENT, "null canonical HRTF destination");
    }
    if (sofa.conventions != "SOFA") {
        return field_fail(JOC_ERR_HRTF_UNSUPPORTED_CONVENTION, "Conventions must be SOFA");
    }
    if (sofa.sofa_conventions != "SimpleFreeFieldHRIR") {
        return field_fail(JOC_ERR_HRTF_UNSUPPORTED_CONVENTION,
                          "unsupported SOFAConventions=" + sofa.sofa_conventions +
                              "; convert explicitly first");
    }
    if (sofa.convention_version != "0.4" && sofa.convention_version != "1.0" &&
        sofa.convention_version != "1.1") {
        return field_fail(JOC_ERR_HRTF_UNSUPPORTED_CONVENTION,
                          "unsupported SimpleFreeFieldHRIR version " + sofa.convention_version);
    }
    if (sofa.data_type != "FIR") {
        return field_fail(JOC_ERR_HRTF_FORMAT, "DataType must be FIR");
    }
    const std::string room_type = lowered(sofa.room_type);
    if (room_type != "free field" && room_type != "free-field" && room_type != "anechoic" &&
        room_type != "hemi-anechoic") {
        return field_fail(JOC_ERR_HRTF_FORMAT,
                          "RoomType must explicitly be free-field, got " + sofa.room_type);
    }
    const std::size_t measurements = sofa.ir_count;
    const std::size_t taps = sofa.ir_length;
    if (measurements == 0u || taps == 0u || sofa.ir.size() != measurements * 2u * taps) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "Data.IR must have shape [M,R=2,N]");
    }
    for (const double value : sofa.ir) {
        if (!std::isfinite(value)) {
            return field_fail(JOC_ERR_HRTF_FORMAT, "Data.IR contains non-finite values");
        }
    }
    if (!std::isfinite(sofa.sample_rate) || sofa.sample_rate <= 0.0) {
        return field_fail(JOC_ERR_HRTF_FORMAT,
                          "Data.SamplingRate must contain one positive finite value");
    }
    const std::string rate_units = lowered(sofa.sampling_rate_units);
    if (rate_units != "hertz" && rate_units != "hz") {
        return field_fail(JOC_ERR_HRTF_FORMAT,
                          "Data.SamplingRate Units must be hertz, got " + sofa.sampling_rate_units);
    }
    std::vector<double> delay(2u, 0.0);
    for (std::size_t ear = 0u; ear < 2u; ++ear) {
        const double value = sofa.delay[ear];
        if (!std::isfinite(value)) {
            return field_fail(JOC_ERR_HRTF_FORMAT, "Data.Delay must be finite");
        }
        if (value < -1.0e-9) {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              "negative Data.Delay is outside the supported causal contract");
        }
        delay[ear] = std::max(0.0, value);
    }

    const std::vector<double> emitter_values(sofa.emitter_position,
                                             sofa.emitter_position + 3);
    std::vector<double> emitter;
    Status status = convert_coordinates(emitter_values, 1u, sofa.emitter_position_coordinates.type,
                                        sofa.emitter_position_coordinates.units, "EmitterPosition",
                                        &emitter);
    if (!status.ok()) {
        return status;
    }
    for (const double value : emitter) {
        if (std::abs(value) > 1.0e-9) {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              "non-zero EmitterPosition needs a separate source-pose adapter");
        }
    }

    const std::vector<double> position_values(sofa.listener_position,
                                              sofa.listener_position + 3);
    const std::vector<double> view_values(sofa.listener_view, sofa.listener_view + 3);
    const std::vector<double> up_values(sofa.listener_up, sofa.listener_up + 3);
    std::vector<double> listener_position;
    std::vector<double> listener_view;
    std::vector<double> listener_up_raw;
    status = convert_coordinates(position_values, 1u, sofa.listener_position_coordinates.type,
                                 sofa.listener_position_coordinates.units, "ListenerPosition",
                                 &listener_position);
    if (!status.ok()) {
        return status;
    }
    status = convert_coordinates(view_values, 1u, sofa.listener_view_coordinates.type,
                                 sofa.listener_view_coordinates.units, "ListenerView", &listener_view);
    if (!status.ok()) {
        return status;
    }
    // ListenerUp inherits ListenerView's Type/Units when it declares none.
    const std::string up_type = sofa.listener_up_coordinates.type.empty() ? sofa.listener_view_coordinates.type
                                                                    : sofa.listener_up_coordinates.type;
    const std::string up_units = sofa.listener_up_coordinates.units.empty()
                                     ? sofa.listener_view_coordinates.units
                                     : sofa.listener_up_coordinates.units;
    status = convert_coordinates(up_values, 1u, up_type, up_units, "ListenerUp", &listener_up_raw);
    if (!status.ok()) {
        return status;
    }
    double forward[3] = {listener_view[0], listener_view[1], listener_view[2]};
    if (!normalize3(forward)) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "ListenerView must be non-zero");
    }
    double left[3];
    cross3(listener_up_raw.data(), forward, left);
    if (!normalize3(left)) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "ListenerUp must not be parallel to ListenerView");
    }
    double up[3];
    cross3(forward, left, up);

    std::vector<double> source_world;
    status = convert_coordinates(sofa.source_position, measurements, sofa.source_position_coordinates.type,
                                 sofa.source_position_coordinates.units, "SourcePosition",
                                 &source_world);
    if (!status.ok()) {
        return status;
    }
    if (source_world.size() != measurements * 3u) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "SourcePosition must have shape [M,C=3]");
    }
    std::vector<double> source_local(measurements * 3u, 0.0);
    std::vector<double> radii(measurements, 0.0);
    for (std::size_t row = 0u; row < measurements; ++row) {
        const double relative[3] = {source_world[row * 3u] - listener_position[0],
                                    source_world[row * 3u + 1u] - listener_position[1],
                                    source_world[row * 3u + 2u] - listener_position[2]};
        source_local[row * 3u] = dot3(relative, forward);
        source_local[row * 3u + 1u] = dot3(relative, left);
        source_local[row * 3u + 2u] = dot3(relative, up);
        radii[row] = std::sqrt(source_local[row * 3u] * source_local[row * 3u] +
                               source_local[row * 3u + 1u] * source_local[row * 3u + 1u] +
                               source_local[row * 3u + 2u] * source_local[row * 3u + 2u]);
        if (!std::isfinite(radii[row]) || radii[row] <= 1.0e-8) {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              "every source measurement must have a positive radius");
        }
    }

    // ReceiverPosition is [R=2,C=3,I]; the two receivers are converted separately
    // so their units are validated the same way as every other coordinate row.
    std::vector<double> receiver_cartesian(6u, 0.0);
    for (std::size_t receiver = 0u; receiver < 2u; ++receiver) {
        const std::vector<double> row(sofa.receiver_position + receiver * 3u,
                                      sofa.receiver_position + receiver * 3u + 3u);
        std::vector<double> converted;
        status = convert_coordinates(row, 1u, sofa.receiver_position_coordinates.type,
                                     sofa.receiver_position_coordinates.units, "ReceiverPosition",
                                     &converted);
        if (!status.ok()) {
            return status;
        }
        for (int axis = 0; axis < 3; ++axis) {
            receiver_cartesian[receiver * 3u + static_cast<std::size_t>(axis)] =
                converted[static_cast<std::size_t>(axis)];
        }
    }
    int left_index = 0;
    int right_index = 1;
    const double lateral = receiver_cartesian[1] - receiver_cartesian[4];
    if (lateral <= 1.0e-5) {
        if (lateral >= -1.0e-5) {
            return field_fail(JOC_ERR_HRTF_FORMAT,
                              "ReceiverPosition does not identify one consistently-left and one "
                              "consistently-right receiver");
        }
        left_index = 1;
        right_index = 0;
    }

    CanonicalHrtf canonical;
    canonical.source_path = sofa.source_path;
    canonical.source_sha256 = to_upper(sofa.source_sha256);
    canonical.convention = sofa.sofa_conventions;
    canonical.convention_version = sofa.convention_version;
    canonical.processing_label = processing_label(sofa);
    canonical.sample_rate_hz = sofa.sample_rate;
    canonical.measurements = static_cast<std::uint32_t>(measurements);
    canonical.taps = static_cast<std::uint32_t>(taps);
    canonical.left_receiver_index = left_index;
    canonical.right_receiver_index = right_index;
    canonical.source_position_cartesian_m = source_local;
    canonical.unit_directions.resize(measurements * 3u, 0.0);
    for (std::size_t row = 0u; row < measurements; ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            canonical.unit_directions[row * 3u + static_cast<std::size_t>(axis)] =
                source_local[row * 3u + static_cast<std::size_t>(axis)] / radii[row];
        }
    }
    canonical.measurement_radius_m = radii;
    canonical.hrir.assign(measurements * 2u * taps, 0.0);
    canonical.delay_samples.assign(measurements * 2u, 0.0);
    for (std::size_t row = 0u; row < measurements; ++row) {
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            const std::size_t source_ear = ear == 0u ? static_cast<std::size_t>(left_index)
                                                     : static_cast<std::size_t>(right_index);
            std::copy(sofa.ir.begin() +
                          static_cast<std::ptrdiff_t>((row * 2u + source_ear) * taps),
                      sofa.ir.begin() +
                          static_cast<std::ptrdiff_t>((row * 2u + source_ear + 1u) * taps),
                      canonical.hrir.begin() + static_cast<std::ptrdiff_t>((row * 2u + ear) * taps));
            canonical.delay_samples[row * 2u + ear] = delay[source_ear];
        }
    }
    *out = std::move(canonical);
    return Status::success();
}

Status compile_canonical_field(const CanonicalHrtf& canonical, const CompileOptions& options,
                               Field* out) {
    if (out == nullptr) {
        return field_fail(JOC_ERR_INVALID_ARGUMENT, "null field destination");
    }
    if (options.order != kFieldOrder) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "the runtime HRTF field is fixed at fifth order");
    }
    if (!std::isfinite(canonical.sample_rate_hz) ||
        std::abs(canonical.sample_rate_hz - kFieldSampleRateHz) > 1.0e-9) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "SOFA HRTF fields must be compiled at 48 kHz");
    }
    if (!std::isfinite(options.shell_radius_m) || options.shell_radius_m <= 0.0) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "shell_radius_m must be positive and finite");
    }
    if (!std::isfinite(options.projection_ridge) || options.projection_ridge < 0.0 ||
        !std::isfinite(options.sh_ridge) || options.sh_ridge < 0.0) {
        return field_fail(JOC_ERR_HRTF_FORMAT,
                          "compiler ridge values must be finite and non-negative");
    }
    if (canonical.measurements == 0u || canonical.taps == 0u) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "canonical HRTF is empty");
    }
    const std::size_t terms = static_cast<std::size_t>(kFieldTerms);
    const std::size_t gains_per_direction = static_cast<std::size_t>(kParameterCount);
    const std::size_t columns = static_cast<std::size_t>(kParameterCount);

    const AlignedHrtf aligned = time_align_hrtf(canonical);
    double shell_radius = options.shell_radius_m;
    const std::vector<std::size_t> indices =
        canonical_shell_indices(canonical, options.shell_radius_m, &shell_radius);
    if (indices.size() < terms) {
        return field_fail(JOC_ERR_HRTF_FORMAT,
                          "fifth-order SH needs at least 36 measurements on one shell, got " +
                              std::to_string(indices.size()));
    }
    const std::size_t selected = indices.size();

    // Project the selected HRIRs onto the public analysis/gain/synthesis dictionary.
    const Kernels& kernels = builtin_kernels();
    const std::vector<double> dictionary = hybrid_gain_synthesis_dictionary(canonical.taps, kernels);
    std::vector<double> system(columns * columns, 0.0);
    for (std::size_t left = 0u; left < columns; ++left) {
        for (std::size_t right = 0u; right < columns; ++right) {
            double sum = 0.0;
            for (std::size_t tap = 0u; tap < canonical.taps; ++tap) {
                sum += dictionary[tap * columns + left] * dictionary[tap * columns + right];
            }
            system[left * columns + right] = sum;
        }
    }
    double trace = 0.0;
    for (std::size_t parameter = 0u; parameter < columns; ++parameter) {
        trace += system[parameter * columns + parameter];
    }
    const double scale = trace / static_cast<double>(columns);
    for (std::size_t parameter = 0u; parameter < columns; ++parameter) {
        system[parameter * columns + parameter] += options.projection_ridge * scale;
    }

    // target[tap][measurement*2 + ear], the reference's reshape(-1, N).T.
    const std::size_t target_columns = selected * 2u;
    std::vector<double> target(canonical.taps * target_columns, 0.0);
    for (std::size_t row = 0u; row < selected; ++row) {
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            const std::size_t column = row * 2u + ear;
            for (std::size_t tap = 0u; tap < canonical.taps; ++tap) {
                target[tap * target_columns + column] =
                    canonical.hrir[(indices[row] * 2u + ear) * canonical.taps + tap];
            }
        }
    }
    std::vector<double> parameters(columns * target_columns, 0.0);
    for (std::size_t tap = 0u; tap < canonical.taps; ++tap) {
        for (std::size_t parameter = 0u; parameter < columns; ++parameter) {
            const double value = dictionary[tap * columns + parameter];
            if (value == 0.0) {
                continue;
            }
            for (std::size_t column = 0u; column < target_columns; ++column) {
                parameters[parameter * target_columns + column] +=
                    value * target[tap * target_columns + column];
            }
        }
    }
    if (!lu_solve_real(&system, static_cast<int>(columns), &parameters,
                       static_cast<int>(target_columns))) {
        return field_fail(JOC_ERR_HRTF_FORMAT, "HRTF projection system is singular");
    }

    // Hybrid gains [measurement][ear*77 + band] with the known embedded delay
    // removed exactly once (parameters[parameter][measurement*2 + ear]).
    std::vector<double> selected_removed(selected * 2u, 0.0);
    std::vector<Complex> gains_by_direction(selected * gains_per_direction, Complex(0.0, 0.0));
    for (std::size_t row = 0u; row < selected; ++row) {
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            const std::size_t column = row * 2u + ear;
            const double removed = aligned.embedded_removed[indices[row] * 2u + ear];
            selected_removed[column] = removed;
            for (int band = 0; band < kHybridBands; ++band) {
                const double real = parameters[static_cast<std::size_t>(2 * band) * target_columns + column];
                const double imaginary =
                    parameters[static_cast<std::size_t>(2 * band + 1) * target_columns + column];
                const double removal =
                    2.0 * kPi * removed * kBandCenters[band] / canonical.sample_rate_hz;
                gains_by_direction[row * gains_per_direction + ear * kHybridBands +
                                   static_cast<std::size_t>(band)] =
                    Complex(real, imaginary) * std::polar(1.0, removal);
            }
        }
    }
    std::vector<double> signal_to_noise(target_columns, 0.0);
    double maximum_gain = 0.0;
    for (const Complex value : gains_by_direction) {
        maximum_gain = std::max(maximum_gain, std::abs(value));
    }
    for (std::size_t column = 0u; column < target_columns; ++column) {
        double reference_energy = 0.0;
        double error_energy = 0.0;
        for (std::size_t tap = 0u; tap < canonical.taps; ++tap) {
            double reconstructed = 0.0;
            for (std::size_t parameter = 0u; parameter < columns; ++parameter) {
                reconstructed += dictionary[tap * columns + parameter] *
                                 parameters[parameter * target_columns + column];
            }
            const double value = target[tap * target_columns + column];
            reference_energy += value * value;
            const double error = value - reconstructed;
            error_energy += error * error;
        }
        signal_to_noise[column] =
            10.0 * std::log10(std::max(reference_energy, 1.0e-300) /
                              std::max(error_energy, 1.0e-300));
    }

    std::vector<double> directions;
    std::vector<Complex> gains;
    std::vector<double> runtime_delay;
    std::vector<double> selected_directions(selected * 3u, 0.0);
    std::vector<double> selected_delays(selected * 2u, 0.0);
    for (std::size_t row = 0u; row < selected; ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            selected_directions[row * 3u + static_cast<std::size_t>(axis)] =
                canonical.unit_directions[indices[row] * 3u + static_cast<std::size_t>(axis)];
        }
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            selected_delays[row * 2u + ear] = aligned.runtime_delay[indices[row] * 2u + ear];
        }
    }
    group_coincident(selected_directions, selected, gains_by_direction, gains_per_direction,
                     selected_delays, &directions, &gains, &runtime_delay);
    const std::size_t unique = directions.size() / 3u;
    if (unique < terms) {
        return field_fail(JOC_ERR_HRTF_FORMAT,
                          "coincident-direction merging left fewer than 36 directions");
    }

    std::vector<double> basis;
    real_spherical_harmonics(directions, unique, options.order, &basis);
    const std::vector<double> weights = spherical_voronoi_weights(directions, unique);
    const std::vector<Complex> coefficients = fit_complex(
        basis, unique, kFieldTerms, weights, options.sh_ridge, gains, gains_per_direction);
    const std::vector<double> delay_coefficients =
        fit_real(basis, unique, kFieldTerms, weights, options.sh_ridge, runtime_delay, 2u);

    // Reconstruction error of the reported fit.
    std::vector<double> relative_error;
    std::vector<double> magnitude_error_db;
    std::vector<double> delay_error;
    relative_error.reserve(unique * kHybridBands);
    magnitude_error_db.reserve(unique * kHybridBands);
    delay_error.reserve(unique * 2u);
    for (std::size_t row = 0u; row < unique; ++row) {
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            double reconstructed_delay = 0.0;
            for (std::size_t term = 0u; term < terms; ++term) {
                reconstructed_delay += basis[row * terms + term] * delay_coefficients[term * 2u + ear];
            }
            const double reference_delay = runtime_delay[row * 2u + ear];
            delay_error.push_back(reconstructed_delay - reference_delay);
            for (int band = 0; band < kHybridBands; ++band) {
                Complex reconstructed_aligned(0.0, 0.0);
                for (std::size_t term = 0u; term < terms; ++term) {
                    reconstructed_aligned +=
                        basis[row * terms + term] *
                        coefficients[term * gains_per_direction + ear * kHybridBands +
                                     static_cast<std::size_t>(band)];
                }
                const Complex reference =
                    gains[row * gains_per_direction + ear * kHybridBands +
                          static_cast<std::size_t>(band)] *
                    std::polar(1.0, -2.0 * kPi * reference_delay * kBandCenters[band] /
                                        canonical.sample_rate_hz);
                const Complex reconstructed =
                    reconstructed_aligned *
                    std::polar(1.0, -2.0 * kPi * reconstructed_delay * kBandCenters[band] /
                                        canonical.sample_rate_hz);
                const double magnitude_reference = std::max(std::abs(reference), 1.0e-12);
                relative_error.push_back(std::abs(reconstructed - reference) / magnitude_reference);
                magnitude_error_db.push_back(
                    std::abs(20.0 * std::log10(std::max(std::abs(reconstructed), 1.0e-12)) -
                             20.0 * std::log10(magnitude_reference)));
            }
        }
    }
    double delay_square_sum = 0.0;
    double delay_maximum = 0.0;
    for (const double value : delay_error) {
        delay_square_sum += value * value;
        delay_maximum = std::max(delay_maximum, std::abs(value));
    }

    // Delay bounds per ear, and the four arrays the cache stores.
    std::vector<double> delay_bounds(4u, 0.0);
    for (std::size_t ear = 0u; ear < 2u; ++ear) {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t row = 0u; row < unique; ++row) {
            minimum = std::min(minimum, runtime_delay[row * 2u + ear]);
            maximum = std::max(maximum, runtime_delay[row * 2u + ear]);
        }
        delay_bounds[ear * 2u] = minimum;
        delay_bounds[ear * 2u + 1u] = maximum;
    }
    std::vector<double> flat_coefficients(terms * 2u * kHybridBands * 2u, 0.0);
    for (std::size_t term = 0u; term < terms; ++term) {
        for (std::size_t ear = 0u; ear < 2u; ++ear) {
            for (int band = 0; band < kHybridBands; ++band) {
                const Complex value =
                    coefficients[term * gains_per_direction + ear * kHybridBands +
                                 static_cast<std::size_t>(band)];
                const std::size_t offset =
                    ((term * 2u + ear) * kHybridBands + static_cast<std::size_t>(band)) * 2u;
                flat_coefficients[offset] = value.real();
                flat_coefficients[offset + 1u] = value.imag();
            }
        }
    }

    Field field;
    field.source_sha256 = to_upper(canonical.source_sha256);
    field.source_display_name = basename_of(canonical.source_path);
    field.measurement_radius_m = shell_radius;
    field.order = options.order;
    field.projection_ridge = options.projection_ridge;
    field.spherical_harmonic_ridge = options.sh_ridge;
    field.compiler_version = kCompilerVersion;
    field.phase_policy_version = kPhasePolicyVersion;
    field.sh_convention = kShConvention;
    field.filterbank_json = filterbank_fingerprint_json();
    field.delay_source = aligned.delay_source;
    field.band_centers_hz.assign(kBandCenters, kBandCenters + kHybridBands);
    field.coefficients = std::move(flat_coefficients);
    field.delay_coefficients = std::move(delay_coefficients);
    field.delay_bounds = std::move(delay_bounds);
    field.cache_key = compiled_hrtf_cache_key(field.source_sha256, canonical.sample_rate_hz,
                                              shell_radius, options.order,
                                              options.projection_ridge, options.sh_ridge);
    field.fit_report_json = json_object({
        {"complex_relative_error_median", json_number(percentile(relative_error, 50.0))},
        {"complex_relative_error_p95", json_number(percentile(relative_error, 95.0))},
        {"delay_error_samples_max", json_number(delay_maximum)},
        {"delay_error_samples_rms",
         json_number(std::sqrt(delay_square_sum / static_cast<double>(delay_error.size())))},
        {"delay_source", json_escape(aligned.delay_source)},
        {"format", json_escape("SOFA FIR -> public 64-QMF/77-hybrid -> ACN/N3D real SH")},
        {"input_measurements", std::to_string(selected)},
        {"magnitude_error_db_median", json_number(percentile(magnitude_error_db, 50.0))},
        {"magnitude_error_db_p95", json_number(percentile(magnitude_error_db, 95.0))},
        {"order", std::to_string(options.order)},
        {"phase_policy_version", json_escape(kPhasePolicyVersion)},
        {"precision", json_escape("float64/complex128")},
        {"projection",
         json_object({
             {"dictionary_shape",
              json_array({std::to_string(canonical.taps), std::to_string(kParameterCount)})},
             {"embedded_delay_samples_max", json_number(maximum_of(selected_removed))},
             {"embedded_delay_samples_min", json_number(minimum_of(selected_removed))},
             {"fir_reconstruction_snr_db_median", json_number(percentile(signal_to_noise, 50.0))},
             {"fir_reconstruction_snr_db_min", json_number(minimum_of(signal_to_noise))},
             {"fir_reconstruction_snr_db_p05", json_number(percentile(signal_to_noise, 5.0))},
             {"maximum_absolute_hybrid_gain", json_number(maximum_gain)},
             {"method", json_escape("regularized public analysis/gain/synthesis dictionary")},
             {"precision", json_escape("float64/complex128")},
             {"real_parameters", std::to_string(kParameterCount)},
             {"ridge", json_number(options.projection_ridge)},
         })},
        {"shell_radius_m", json_number(shell_radius)},
        {"spherical_harmonic_ridge", json_number(options.sh_ridge)},
        {"terms", std::to_string(kFieldTerms)},
        {"unique_directions", std::to_string(unique)},
    });
    field.payload_sha256 = field_payload_sha256(field);
    field.metadata_json = json_object({
        {"cache_key", json_escape(field.cache_key)},
        {"cache_key_version", "1"},
        {"cache_schema", json_escape(std::string(kCacheSchema))},
        {"compiler_version", json_escape(field.compiler_version)},
        {"delay_source", json_escape(field.delay_source)},
        {"filterbank", field.filterbank_json},
        {"fit_report", field.fit_report_json},
        {"format_version", std::to_string(kFormatVersion)},
        {"magic", json_escape(std::string(kMagic))},
        {"measurement_radius_m", json_number(field.measurement_radius_m)},
        {"order", std::to_string(field.order)},
        {"payload_sha256", json_escape(field.payload_sha256)},
        {"phase_policy_version", json_escape(field.phase_policy_version)},
        {"projection_ridge", json_number(field.projection_ridge)},
        {"sample_rate_hz", json_number(canonical.sample_rate_hz)},
        {"sh_convention", json_escape(field.sh_convention)},
        {"source_display_name", json_escape(field.source_display_name)},
        {"source_sha256", json_escape(field.source_sha256)},
        {"spherical_harmonic_ridge", json_number(field.spherical_harmonic_ridge)},
    });
    *out = std::move(field);
    return Status::success();
}

void shell_directions_and_weights_for_check(const SofaHrir& sofa, double radius_m,
                                            std::vector<double>* directions,
                                            std::vector<double>* weights) {
    CanonicalHrtf canonical;
    if (!canonicalize_sofa(sofa, &canonical).ok()) {
        return;
    }
    double actual = radius_m;
    const std::vector<std::size_t> indices = canonical_shell_indices(canonical, radius_m, &actual);
    directions->assign(indices.size() * 3u, 0.0);
    for (std::size_t row = 0u; row < indices.size(); ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            (*directions)[row * 3u + static_cast<std::size_t>(axis)] =
                canonical.unit_directions[indices[row] * 3u + static_cast<std::size_t>(axis)];
        }
    }
    *weights = spherical_voronoi_weights(*directions, indices.size());
}
Status compile_sofa_field(const SofaHrir& sofa, const CompileOptions& options, Field* out) {
    CanonicalHrtf canonical;
    const Status status = canonicalize_sofa(sofa, &canonical);
    if (!status.ok()) {
        return status;
    }
    return compile_canonical_field(canonical, options, out);
}

}  // namespace joc::hrtf
