#include "hrtf/rosella_renderer.h"

#include "simd/simd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "foundation/fft.h"
#include "hrtf/public_filterbank.h"

namespace joc::hrtf {

namespace {

constexpr int kSourceChannels = 16;
constexpr int kOutputChannels = 2;
constexpr int kDirectionTerms = 36;
constexpr int kBlockSamples = 512;
constexpr int kFrameSamples = 1536;
constexpr int kRoomBands = 64;
constexpr double kPi = 3.14159265358979323846;
constexpr double kRoomInputGain = 0.70710677;
constexpr double kCentreEqual = 0.9998489618301392;
constexpr double kCentreAlternate = 0.7070000171661377;

// The fixed sixteen-band low-pass a special (LFE) source is rendered through; the
// constants are the float32 bit patterns of rosella_direct._SPECIAL_LFE_LOW_16.
constexpr std::uint32_t kSpecialLfeLowBits[16] = {
    0x402695EAu, 0x3FE75979u, 0x3F28CAAAu, 0xBCE1FB2Eu, 0xBDD8AF65u, 0xBD8F426Eu,
    0x3D996821u, 0xBC16B3A0u, 0x3B64BAF1u, 0xBC81ECFDu, 0xBA3D892Fu, 0x3AF6A9F0u,
    0xB9DD1C5Fu, 0x380A193Fu, 0x38052059u, 0x351BCB34u};

Status render_fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kRender, message);
}

std::vector<double> special_lfe_low() {
    std::vector<double> values(kHybridBands, 0.0);
    for (int index = 0; index < 16; ++index) {
        float narrowed = 0.0f;
        const std::uint32_t bits = kSpecialLfeLowBits[index];
        std::memcpy(&narrowed, &bits, sizeof(float));
        values[static_cast<std::size_t>(index)] = static_cast<double>(narrowed);
    }
    return values;
}

// rosella_model.direction_basis: the observed 36-term basis.  The operation order is
// part of the model, so the intermediates are named after the reference's.
void direction_basis(double x, double y, double z, double* out) {
    const double yz = y * z;
    const double x2 = x * x;
    const double y2 = y * y;
    const double x2m02 = x2 - 0.2;
    const double xy = x * y;
    out[0] = 1.0;
    out[1] = x;
    out[2] = y;
    out[3] = z;
    out[4] = x2 - (1.0 / 3.0);
    out[5] = xy;
    out[6] = x * z;
    out[7] = y2 - (1.0 / 3.0);
    out[8] = yz;
    out[9] = (x2 - 0.6) * x;
    out[10] = x2m02 * y;
    out[11] = x2m02 * z;
    out[12] = (y2 - 0.2) * x;
    out[13] = yz * x;
    out[14] = (y2 - 0.6) * y;
    out[15] = (y2 - 0.2) * z;
    out[16] = x2 * x2 - 0.2;
    out[17] = xy * x2;
    out[18] = (x * z) * x2;
    out[19] = y2 * x2 - (1.0 / 15.0);
    out[20] = yz * x2;
    out[21] = x * y2 * y;
    out[22] = x * y2 * z;
    out[23] = y2 * y2 - 0.2;
    const double x4 = x2 * x2;
    const double x2y2 = y2 * x2;
    const double y4 = y2 * y2;
    out[24] = yz * y2;
    out[25] = (x4 - (3.0 / 7.0)) * x;
    out[26] = (x4 - (3.0 / 35.0)) * y;
    out[27] = (x4 - (3.0 / 35.0)) * z;
    out[28] = (x2y2 - (3.0 / 35.0)) * x;
    out[29] = (x2 * z) * xy;
    out[30] = (x2y2 - (3.0 / 35.0)) * y;
    out[31] = (x2y2 - (1.0 / 35.0)) * z;
    out[32] = (y4 - (3.0 / 35.0)) * x;
    out[33] = (y2 * z) * xy;
    out[34] = (y4 - (3.0 / 7.0)) * y;
    out[35] = (y4 - (3.0 / 35.0)) * z;
}

// rosella_direct._logical_field: the padded lane grid as [77][36][2] float64.
void logical_field(const std::vector<float>& padded, std::vector<double>* out) {
    out->assign(static_cast<std::size_t>(kHybridBands) * kDirectionTerms * 2u, 0.0);
    for (int band = 0; band < kHybridBands; ++band) {
        const int block = band / 4;
        const int lane = band % 4;
        for (int term = 0; term < kDirectionTerms; ++term) {
            for (int component = 0; component < 2; ++component) {
                const std::size_t source =
                    static_cast<std::size_t>(lane + 4 * (term * 2 + component + 72 * block));
                (*out)[(static_cast<std::size_t>(band) * kDirectionTerms +
                        static_cast<std::size_t>(term)) * 2u +
                       static_cast<std::size_t>(component)] =
                    source < padded.size() ? static_cast<double>(padded[source]) : 0.0;
            }
        }
    }
}

// rosella_direct._round_away_from_zero.
double round_away_from_zero(double value) {
    return value >= 0.0 ? std::floor(value + 0.5) : std::ceil(value - 0.5);
}

// rosella_direct._q15_position: ADM coordinates on the Rosella metadata grid.
void q15_position(const double position[3], int encoded[3]) {
    const double raw[3] = {
        std::min(std::max((position[0] + 1.0) * 0.5, 0.0), 1.0),
        std::min(std::max((1.0 - position[1]) * 0.5, 0.0), 1.0),
        std::min(std::max(position[2], -1.0), 1.0),
    };
    for (int axis = 0; axis < 3; ++axis) {
        encoded[axis] = static_cast<int>(
            std::min(round_away_from_zero(raw[axis] * 32768.0), 32767.0));
    }
}

struct ProfileGeometry {
    double direction[3] = {1.0, 0.0, 0.0};
    double radius = 0.0;
    double clamped = 0.0;
    double alpha = 0.0;
};

// rosella_direct._profile_geometry.
ProfileGeometry profile_geometry(const RosellaModel& model, const double position[3],
                                 int profile_index) {
    const RosellaDistanceProfile& profile = model.profiles[static_cast<std::size_t>(profile_index)];
    int encoded[3] = {0, 0, 0};
    q15_position(position, encoded);
    const double q_front = 1.0 - 2.0 * static_cast<double>(encoded[1]) / 32768.0;
    const double q_x = 2.0 * static_cast<double>(encoded[0]) / 32768.0 - 1.0;
    const double q_vertical = static_cast<double>(encoded[2]) / 32768.0;
    double mapped_front = q_front;
    double mapped_lateral = -q_x;
    double mapped_vertical = q_vertical;
    if (model.header_integer_fields[0] != 0) {
        if (q_x == 0.0 && q_front == 0.0) {
            mapped_front = 0.0;
            mapped_lateral = 0.0;
            mapped_vertical = q_vertical;
        } else {
            const double horizontal_max = std::max(std::abs(q_x), std::abs(q_front));
            const double horizontal_norm = (q_x / horizontal_max) * (q_x / horizontal_max) +
                                           (q_front / horizontal_max) * (q_front / horizontal_max);
            double vertical_norm = 1.0;
            if (q_vertical != 0.0) {
                const double smaller = std::min(std::abs(q_vertical), horizontal_max);
                const double larger = std::max(std::abs(q_vertical), horizontal_max);
                vertical_norm = 1.0 + (smaller / larger) * (smaller / larger);
            }
            const double horizontal_factor = 1.0 / std::sqrt(horizontal_norm * vertical_norm);
            const double vertical_factor = 1.0 / std::sqrt(vertical_norm);
            mapped_front = q_front * horizontal_factor;
            mapped_lateral = -q_x * horizontal_factor;
            mapped_vertical = q_vertical * vertical_factor;
        }
    }
    double scaled[3] = {mapped_front * static_cast<double>(profile.axis_scales_internal[2]),
                        mapped_lateral * static_cast<double>(profile.axis_scales_internal[0]),
                        mapped_vertical * static_cast<double>(profile.axis_scales_internal[1])};
    double ray = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double value = scaled[axis];
        const double lower = static_cast<double>(profile.bounds[static_cast<std::size_t>(axis) * 2u]);
        const double upper =
            static_cast<double>(profile.bounds[static_cast<std::size_t>(axis) * 2u + 1u]);
        if (value < lower) {
            ray = std::min(ray, lower / value);
        } else if (value > upper) {
            ray = std::min(ray, upper / value);
        }
    }
    if (ray < 1.0) {
        scaled[0] *= ray;
        scaled[1] *= ray;
        scaled[2] *= ray;
    }
    ProfileGeometry result;
    result.radius = std::sqrt(scaled[0] * scaled[0] + scaled[1] * scaled[1] +
                              scaled[2] * scaled[2]);
    result.clamped = std::max(result.radius, static_cast<double>(profile.minimum_normalized_radius));
    result.alpha = result.radius / result.clamped;
    if (result.radius > 1.0e-30) {
        for (int axis = 0; axis < 3; ++axis) {
            result.direction[axis] = scaled[axis] / result.radius;
        }
    }
    return result;
}

struct EarGeometry {
    double basis_minus[kDirectionTerms] = {};
    double basis_plus[kDirectionTerms] = {};
    double path_minus = 0.0;
    double path_plus = 0.0;
};

// rosella_direct._ear_geometry.
EarGeometry ear_geometry(const RosellaModel& model, const RosellaDistanceProfile& profile,
                         const double direction[3], double clamped, double offset,
                         double correction) {
    const double x = direction[0];
    const double y = direction[1];
    const double z = direction[2];
    const double inverse_distance = static_cast<double>(profile.inverse_distance_per_m);
    const double ear = offset * inverse_distance / clamped;
    const double y_minus = y - ear;
    const double y_plus = y + ear;
    const double common = x * x + z * z;
    const double length_minus = std::sqrt(y_minus * y_minus + common);
    const double length_plus = std::sqrt(y_plus * y_plus + common);
    EarGeometry result;
    direction_basis(x / length_minus, y_minus / length_minus, z / length_minus,
                    result.basis_minus);
    direction_basis(x / length_plus, y_plus / length_plus, z / length_plus, result.basis_plus);
    result.path_minus = length_minus * clamped;
    result.path_plus = length_plus * clamped;
    if (correction != 0.0) {
        const double multiplier = 2.0 * correction * inverse_distance;
        double left = 0.0;
        double right = 0.0;
        for (int term = 0; term < kDirectionTerms; ++term) {
            left += static_cast<double>(model.vector_left[static_cast<std::size_t>(term)]) *
                    result.basis_minus[term];
            right += static_cast<double>(model.vector_right[static_cast<std::size_t>(term)]) *
                     result.basis_plus[term];
        }
        result.path_minus += std::max(left, 0.0) * multiplier;
        result.path_plus += std::max(right, 0.0) * multiplier;
    }
    return result;
}

// rosella_direct._phase_groups.
void phase_groups(const RosellaModel& model, double delay_samples, Complex* out) {
    // The reference starts from np.ones(77) and only rewrites the flagged bands.
    std::fill(out, out + kHybridBands, Complex(1.0, 0.0));
    Complex current(1.0, 0.0);
    Complex step(1.0, 0.0);
    std::size_t value_index = 0u;
    for (std::size_t band = 0u; band < model.hybrid_flags.size() && band < kHybridBands; ++band) {
        const int flag = model.hybrid_flags[band];
        if (flag != 2) {
            if (flag == 1) {
                const double angle =
                    static_cast<double>(model.hybrid_values[value_index]) * delay_samples;
                ++value_index;
                step = Complex(std::cos(angle), std::sin(angle));
            }
            current *= step;
        }
        out[band] = current;
    }
}

struct DirectResult {
    Complex gains[2][kHybridBands] = {};
    double room_send = 0.0;
    double physical_radius_m = 0.0;
    double normalized_radius = 0.0;
    double clamped_radius = 0.0;
    double delay_samples = 0.0;
    int delayed_ear = -1;
};

// rosella_direct.special_lfe_direct.
const DirectResult& special_lfe_direct() {
    static const DirectResult result = [] {
        DirectResult direct;
        const std::vector<double> low = special_lfe_low();
        for (int ear = 0; ear < 2; ++ear) {
            for (int band = 0; band < kHybridBands; ++band) {
                direct.gains[ear][band] = Complex(low[static_cast<std::size_t>(band)], 0.0);
            }
        }
        return direct;
    }();
    return result;
}

// rosella_direct.direct_and_room_send.
DirectResult direct_and_room_send(const RosellaModel& model,
                                  const std::vector<double>& field_left,
                                  const std::vector<double>& field_right, const double position[3],
                                  int profile_index) {
    const RosellaDistanceProfile& profile =
        model.profiles[static_cast<std::size_t>(profile_index)];
    const ProfileGeometry geometry = profile_geometry(model, position, profile_index);

    const EarGeometry delay_ears =
        ear_geometry(model, profile, geometry.direction, geometry.clamped,
                     static_cast<double>(model.model_scalars[1]),
                     static_cast<double>(model.model_scalars[2]));
    const double delay = std::abs(delay_ears.path_plus - delay_ears.path_minus) *
                         static_cast<double>(profile.distance_scale_m) *
                         (static_cast<double>(model.sample_rate) / 343.3) * geometry.alpha;
    int delayed_ear = -1;
    if (delay_ears.path_minus > delay_ears.path_plus) {
        delayed_ear = 0;
    } else if (delay_ears.path_plus > delay_ears.path_minus) {
        delayed_ear = 1;
    }

    const EarGeometry weight_ears =
        ear_geometry(model, profile, geometry.direction, geometry.clamped,
                     static_cast<double>(model.model_scalars[3]),
                     static_cast<double>(model.model_scalars[4]));
    const double weight_norm =
        std::sqrt(weight_ears.path_minus * weight_ears.path_minus +
                  weight_ears.path_plus * weight_ears.path_plus);
    const double weight_left = weight_ears.path_plus / weight_norm;
    const double weight_right = weight_ears.path_minus / weight_norm;

    const double final_offset = static_cast<double>(model.model_scalars[0]);
    double basis_minus[kDirectionTerms] = {};
    double basis_plus[kDirectionTerms] = {};
    if (final_offset == 0.0) {
        direction_basis(geometry.direction[0], geometry.direction[1], geometry.direction[2],
                        basis_minus);
        std::copy(basis_minus, basis_minus + kDirectionTerms, basis_plus);
    } else {
        const double x = geometry.direction[0];
        const double y = geometry.direction[1];
        const double z = geometry.direction[2];
        const double ear =
            final_offset * static_cast<double>(profile.inverse_distance_per_m) / geometry.clamped;
        const double y_minus = y - ear;
        const double y_plus = y + ear;
        const double common_length = x * x + z * z;
        const double length_minus = std::sqrt(y_minus * y_minus + common_length);
        const double length_plus = std::sqrt(y_plus * y_plus + common_length);
        direction_basis(x / length_minus, y_minus / length_minus, z / length_minus, basis_minus);
        direction_basis(x / length_plus, y_plus / length_plus, z / length_plus, basis_plus);
    }

    DirectResult result;
    Complex left[kHybridBands] = {};
    Complex right[kHybridBands] = {};
    for (int band = 0; band < kHybridBands; ++band) {
        double left_real = 0.0;
        double left_imag = 0.0;
        double right_real = 0.0;
        double right_imag = 0.0;
        for (int term = 0; term < kDirectionTerms; ++term) {
            const std::size_t index =
                (static_cast<std::size_t>(band) * kDirectionTerms + static_cast<std::size_t>(term)) *
                2u;
            left_real += field_left[index] * basis_minus[term];
            left_imag += field_left[index + 1u] * basis_minus[term];
            right_real += field_right[index] * basis_plus[term];
            right_imag += field_right[index + 1u] * basis_plus[term];
        }
        left[band] = Complex(left_real, left_imag);
        right[band] = Complex(right_real, right_imag);
    }
    if (delayed_ear >= 0) {
        Complex phase[kHybridBands];
        phase_groups(model, delay, phase);
        Complex* target = delayed_ear == 0 ? left : right;
        for (int band = 0; band < kHybridBands; ++band) {
            target[band] *= phase[band];
        }
    }

    const double effective_radius =
        geometry.radius * static_cast<double>(model.header_float_scalars[0]) *
        static_cast<double>(profile.distance_scale_m);
    double common = 1.0;
    double room_send = 0.0;
    if (profile_index == static_cast<int>(RosellaProfile::Far) ||
        profile_index == static_cast<int>(RosellaProfile::Mid)) {
        common = 1.0 / std::sqrt(1.0 + static_cast<double>(model.header_float_scalars[1]) *
                                           effective_radius * effective_radius);
        room_send = effective_radius * common;
    }

    Complex left_term0[kHybridBands];
    Complex right_term0[kHybridBands];
    for (int band = 0; band < kHybridBands; ++band) {
        const std::size_t index = static_cast<std::size_t>(band) * kDirectionTerms * 2u;
        left_term0[band] = Complex(field_left[index], field_left[index + 1u]);
        right_term0[band] = Complex(field_right[index], field_right[index + 1u]);
    }
    const bool weights_are_default_equal =
        static_cast<double>(model.model_scalars[3]) == 0.0 &&
        static_cast<double>(model.model_scalars[4]) == 0.0;
    double centre_left = 0.0;
    double centre_right = 0.0;
    double right_direction_weight = weight_right;
    if (weights_are_default_equal) {
        centre_left = weight_left * (1.0 - geometry.alpha) * kCentreEqual;
        centre_right = centre_left;
        right_direction_weight = weight_left;
    } else {
        centre_left = (1.0 - geometry.alpha) * kCentreAlternate;
        centre_right = centre_left;
    }
    for (int band = 0; band < kHybridBands; ++band) {
        result.gains[0][band] =
            common * (left[band] * (weight_left * geometry.alpha) + left_term0[band] * centre_left);
        result.gains[1][band] = common * (right[band] * (right_direction_weight * geometry.alpha) +
                                          right_term0[band] * centre_right);
    }
    result.room_send = room_send;
    result.physical_radius_m = static_cast<double>(profile.distance_scale_m) * geometry.radius;
    result.normalized_radius = geometry.radius;
    result.clamped_radius = geometry.clamped;
    result.delay_samples = delay;
    result.delayed_ear = delayed_ear;
    return result;
}

// rosella_room._RosellaRoomState: the recursive table-A room model.
class RoomState {
public:
    explicit RoomState(const RosellaModel& model) : model_(model) {
        bands_ = std::min(64, model.table_a_dimension);
        delays_ = model.table_a_four_integers;
        capacity_ = 0;
        for (const int delay : delays_) {
            capacity_ = std::max(capacity_, delay);
        }
        // The vectors are stored column-major (numpy's order="F").
        matrix_.assign(16u, 0.0);
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                matrix_[static_cast<std::size_t>(row) + 4u * static_cast<std::size_t>(column)] =
                    static_cast<double>(model.table_a_vector16[static_cast<std::size_t>(row) +
                                                               4u * static_cast<std::size_t>(column)]);
            }
        }
        const std::vector<float>& f8 = model.table_a_filter_8x64_padded;
        const std::vector<float>& f4 = model.table_a_filter_4x64_padded;
        const std::vector<float>& f16 = model.table_a_filter_16x64_padded;
        feedback_real_.assign(static_cast<std::size_t>(bands_) * 4u, 0.0);
        feedback_imag_.assign(static_cast<std::size_t>(bands_) * 4u, 0.0);
        output_tap_.assign(static_cast<std::size_t>(bands_) * 4u, 0.0);
        left_real_.assign(static_cast<std::size_t>(bands_) * 4u, 0.0);
        left_imag_.assign(static_cast<std::size_t>(bands_) * 4u, 0.0);
        right_real_.assign(static_cast<std::size_t>(bands_) * 4u, 0.0);
        right_imag_.assign(static_cast<std::size_t>(bands_) * 4u, 0.0);
        for (int band = 0; band < bands_; ++band) {
            const int group = band / 4;
            const int lane = band % 4;
            for (int tap = 0; tap < 4; ++tap) {
                const std::size_t target = static_cast<std::size_t>(band) * 4u +
                                           static_cast<std::size_t>(tap);
                const std::size_t g8 = ((static_cast<std::size_t>(group) * 4u +
                                         static_cast<std::size_t>(tap)) * 2u) *
                                           4u +
                                       static_cast<std::size_t>(lane);
                feedback_real_[target] = static_cast<double>(f8[g8]);
                feedback_imag_[target] = static_cast<double>(f8[g8 + 4u]);
                output_tap_[target] = static_cast<double>(
                    f4[(static_cast<std::size_t>(group) * 4u + static_cast<std::size_t>(tap)) * 4u +
                       static_cast<std::size_t>(lane)]);
                const std::size_t g16 = ((static_cast<std::size_t>(group) * 4u +
                                          static_cast<std::size_t>(tap)) * 4u) *
                                        4u;
                left_real_[target] = static_cast<double>(f16[g16 + static_cast<std::size_t>(lane)]);
                left_imag_[target] =
                    static_cast<double>(f16[g16 + 4u + static_cast<std::size_t>(lane)]);
                right_real_[target] =
                    static_cast<double>(f16[g16 + 8u + static_cast<std::size_t>(lane)]);
                right_imag_[target] =
                    static_cast<double>(f16[g16 + 12u + static_cast<std::size_t>(lane)]);
            }
        }
        allpass_gain_.resize(model.table_a_option_values.size());
        for (std::size_t index = 0u; index < model.table_a_option_values.size(); ++index) {
            allpass_gain_[index] = static_cast<double>(model.table_a_option_values[index]);
        }
        allpass_delay_ = model.table_a_option_ids;
        for (const int delay : allpass_delay_) {
            allpass_real_.emplace_back(static_cast<std::size_t>(delay) *
                                           static_cast<std::size_t>(bands_),
                                       0.0);
            allpass_imag_.emplace_back(static_cast<std::size_t>(delay) *
                                           static_cast<std::size_t>(bands_),
                                       0.0);
        }
        allpass_position_.assign(allpass_real_.size(), 0);
        memory_real_.assign(static_cast<std::size_t>(capacity_) * static_cast<std::size_t>(bands_) * 4u,
                            0.0);
        memory_imag_.assign(memory_real_.size(), 0.0);
        extra_matrices_.resize(model.table_a_extra_indices.size());
        for (std::size_t index = 0u; index < extra_matrices_.size(); ++index) {
            extra_matrices_[index].assign(16u, 0.0);
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    extra_matrices_[index][static_cast<std::size_t>(row) +
                                           4u * static_cast<std::size_t>(column)] =
                        static_cast<double>(
                            model.table_a_extra_vectors[index * 16u + static_cast<std::size_t>(row) +
                                                        4u * static_cast<std::size_t>(column)]);
                }
            }
        }
    }

    void reset() {
        for (std::vector<double>& value : allpass_real_) {
            std::fill(value.begin(), value.end(), 0.0);
        }
        for (std::vector<double>& value : allpass_imag_) {
            std::fill(value.begin(), value.end(), 0.0);
        }
        std::fill(allpass_position_.begin(), allpass_position_.end(), 0);
        std::fill(memory_real_.begin(), memory_real_.end(), 0.0);
        std::fill(memory_imag_.begin(), memory_imag_.end(), 0.0);
        position_ = 0;
    }

    Status process_slot(const Complex* room_send, Complex* output) {
        if (static_cast<double>(model_.table_a_scalar) >= 0.5) {
            return render_fail(JOC_ERR_NOT_SUPPORTED, "alternate Rosella table-A room mode");
        }
        const std::size_t bands = static_cast<std::size_t>(bands_);
        std::vector<double> input_real(bands, 0.0);
        std::vector<double> input_imag(bands, 0.0);
        for (std::size_t band = 0u; band < bands; ++band) {
            input_real[band] = room_send[band].real() * kRoomInputGain;
            input_imag[band] = room_send[band].imag() * kRoomInputGain;
        }
        for (std::size_t index = 0u; index < allpass_gain_.size(); ++index) {
            const std::size_t delay = static_cast<std::size_t>(allpass_delay_[index]);
            const std::size_t position = static_cast<std::size_t>(allpass_position_[index]);
            const double gain = allpass_gain_[index];
            for (std::size_t band = 0u; band < bands; ++band) {
                const std::size_t cell = position * bands + band;
                const double previous_real = allpass_real_[index][cell];
                const double previous_imag = allpass_imag_[index][cell];
                const double residual_real = input_real[band] - previous_real * gain;
                const double residual_imag = input_imag[band] - previous_imag * gain;
                input_real[band] = residual_real * gain + previous_real;
                input_imag[band] = residual_imag * gain + previous_imag;
                allpass_real_[index][cell] = residual_real;
                allpass_imag_[index][cell] = residual_imag;
            }
            allpass_position_[index] = static_cast<int>((position + 1u) % delay);
        }

        std::vector<double> branch_real(bands * 4u, 0.0);
        std::vector<double> branch_imag(bands * 4u, 0.0);
        std::vector<double> delayed_real(bands * 4u, 0.0);
        std::vector<double> delayed_imag(bands * 4u, 0.0);
        for (int branch = 0; branch < 4; ++branch) {
            const std::size_t slot =
                static_cast<std::size_t>((position_ - delays_[static_cast<std::size_t>(branch)] +
                                          capacity_) %
                                         capacity_);
            for (std::size_t band = 0u; band < bands; ++band) {
                const std::size_t source = (slot * bands + band) * 4u + static_cast<std::size_t>(branch);
                delayed_real[band * 4u + static_cast<std::size_t>(branch)] = memory_real_[source];
                delayed_imag[band * 4u + static_cast<std::size_t>(branch)] = memory_imag_[source];
            }
        }
        for (std::size_t band = 0u; band < bands; ++band) {
            for (int row = 0; row < 4; ++row) {
                double real = input_real[band];
                double imag = input_imag[band];
                for (int column = 0; column < 4; ++column) {
                    const double coefficient =
                        matrix_[static_cast<std::size_t>(row) + 4u * static_cast<std::size_t>(column)];
                    real += delayed_real[band * 4u + static_cast<std::size_t>(column)] * coefficient;
                    imag += delayed_imag[band * 4u + static_cast<std::size_t>(column)] * coefficient;
                }
                branch_real[band * 4u + static_cast<std::size_t>(row)] = real;
                branch_imag[band * 4u + static_cast<std::size_t>(row)] = imag;
            }
        }

        const std::size_t tap_slot = static_cast<std::size_t>(
            (position_ - model_.table_a_integer + capacity_) % capacity_);
        std::vector<double> tap_real(bands * 4u, 0.0);
        std::vector<double> tap_imag(bands * 4u, 0.0);
        for (std::size_t band = 0u; band < bands; ++band) {
            for (int lane = 0; lane < 4; ++lane) {
                const std::size_t source = (tap_slot * bands + band) * 4u + static_cast<std::size_t>(lane);
                tap_real[band * 4u + static_cast<std::size_t>(lane)] = memory_real_[source];
                tap_imag[band * 4u + static_cast<std::size_t>(lane)] = memory_imag_[source];
            }
        }
        for (std::size_t cell = 0u; cell < branch_real.size(); ++cell) {
            const double real = branch_real[cell];
            const double imag = branch_imag[cell];
            memory_real_[(static_cast<std::size_t>(position_) * bands) * 4u + cell] =
                real * feedback_real_[cell] - imag * feedback_imag_[cell];
            memory_imag_[(static_cast<std::size_t>(position_) * bands) * 4u + cell] =
                imag * feedback_real_[cell] + real * feedback_imag_[cell];
        }
        position_ = (position_ + 1) % capacity_;

        std::vector<double> extra_real(bands * 4u, 0.0);
        std::vector<double> extra_imag(bands * 4u, 0.0);
        for (std::size_t index = 0u; index < model_.table_a_extra_indices.size(); ++index) {
            const std::size_t delay =
                static_cast<std::size_t>(model_.table_a_extra_indices[index]);
            const std::size_t slot = static_cast<std::size_t>(
                (position_ - static_cast<int>(delay) - 1 + capacity_ * 2) % capacity_);
            for (std::size_t band = 0u; band < bands; ++band) {
                for (int row = 0; row < 4; ++row) {
                    double mixed_real = 0.0;
                    double mixed_imag = 0.0;
                    for (int column = 0; column < 4; ++column) {
                        const std::size_t source =
                            (slot * bands + band) * 4u + static_cast<std::size_t>(column);
                        const double coefficient =
                            extra_matrices_[index][static_cast<std::size_t>(row) +
                                                   4u * static_cast<std::size_t>(column)];
                        mixed_real += memory_real_[source] * coefficient;
                        mixed_imag += memory_imag_[source] * coefficient;
                    }
                    const int group = static_cast<int>(band) / 4;
                    const int lane = static_cast<int>(band) % 4;
                    const std::size_t field =
                        (index * 20u + static_cast<std::size_t>(group)) * 2u * 4u;
                    const double coefficient_real = static_cast<double>(
                        model_.table_a_extra_fields_padded[field + static_cast<std::size_t>(lane)]);
                    const double coefficient_imag = static_cast<double>(
                        model_.table_a_extra_fields_padded[field + 4u + static_cast<std::size_t>(lane)]);
                    const std::size_t target = band * 4u + static_cast<std::size_t>(row);
                    extra_real[target] +=
                        mixed_real * coefficient_real - mixed_imag * coefficient_imag;
                    extra_imag[target] +=
                        mixed_imag * coefficient_real + mixed_real * coefficient_imag;
                }
            }
        }

        for (std::size_t band = 0u; band < bands; ++band) {
            double left_real = 0.0;
            double left_imag = 0.0;
            double right_real = 0.0;
            double right_imag = 0.0;
            for (int lane = 0; lane < 4; ++lane) {
                const std::size_t cell = band * 4u + static_cast<std::size_t>(lane);
                const double real = tap_real[cell] * output_tap_[cell] + extra_real[cell];
                const double imag = tap_imag[cell] * output_tap_[cell] + extra_imag[cell];
                left_real += left_real_[cell] * real - left_imag_[cell] * imag;
                left_imag += left_imag_[cell] * real + left_real_[cell] * imag;
                right_real += right_real_[cell] * real - right_imag_[cell] * imag;
                right_imag += right_imag_[cell] * real + right_real_[cell] * imag;
            }
            output[band] = Complex(left_real, left_imag);
            output[kHybridBands + band] = Complex(right_real, right_imag);
        }
        return Status::success();
    }

private:
    const RosellaModel& model_;
    int bands_ = 0;
    std::array<int, 4> delays_{};
    int capacity_ = 0;
    std::vector<double> matrix_;
    std::vector<double> feedback_real_;
    std::vector<double> feedback_imag_;
    std::vector<double> output_tap_;
    std::vector<double> left_real_;
    std::vector<double> left_imag_;
    std::vector<double> right_real_;
    std::vector<double> right_imag_;
    std::vector<double> allpass_gain_;
    std::vector<int> allpass_delay_;
    std::vector<std::vector<double>> allpass_real_;
    std::vector<std::vector<double>> allpass_imag_;
    std::vector<int> allpass_position_;
    std::vector<double> memory_real_;
    std::vector<double> memory_imag_;
    int position_ = 0;
    std::vector<std::vector<double>> extra_matrices_;
};

// rosella_room.RosellaRoomFir: the room FIR plus its overlap-add convolution.
class RoomFir {
public:
    RoomFir(const RosellaModel& model, int impulse_slots) : length_(impulse_slots) {
        RoomState reference(model);
        kernel_.assign(static_cast<std::size_t>(length_) * 2u * kRoomBands, Complex(0.0, 0.0));
        Complex slot_output[2 * kHybridBands];
        for (int slot = 0; slot < length_; ++slot) {
            Complex impulse[kHybridBands];
            for (int band = 0; band < kHybridBands; ++band) {
                impulse[band] = Complex(slot == 0 && band < kRoomBands ? 1.0 : 0.0, 0.0);
            }
            const Status status = reference.process_slot(impulse, slot_output);
            if (!status.ok()) {
                return;
            }
            for (int ear = 0; ear < 2; ++ear) {
                for (int band = 0; band < kRoomBands; ++band) {
                    kernel_[(static_cast<std::size_t>(slot) * 2u + static_cast<std::size_t>(ear)) *
                                kRoomBands +
                            static_cast<std::size_t>(band)] =
                        slot_output[ear * kHybridBands + band];
                }
            }
        }
        tail_.assign(static_cast<std::size_t>(length_ - 1) * 2u * kRoomBands, Complex(0.0, 0.0));
        valid_ = true;
    }

    bool valid() const { return valid_; }

    void reset() {
        std::fill(tail_.begin(), tail_.end(), Complex(0.0, 0.0));
        fft_cache_size_ = 0u;
        forward_plan_.reset();
        inverse_plan_.reset();
    }

    // rosella_room.RosellaRoomFir.process_chunk.
    Status process_chunk(const std::vector<Complex>& values, std::size_t slots,
                         std::vector<Complex>* output) {
        output->assign(slots * 2u * kHybridBands, Complex(0.0, 0.0));
        if (slots == 0u) {
            return Status::success();
        }
        const std::size_t needed = slots + static_cast<std::size_t>(length_) - 1u;
        std::size_t fft_size = 1u;
        while (fft_size < needed) {
            fft_size <<= 1u;
        }
        if (fft_size != fft_cache_size_) {
            kernel_fft_.assign(fft_size * 2u * kRoomBands, Complex(0.0, 0.0));
            // The room transforms a fixed length for the whole chunk, so the radix-2
            // twiddle recurrences are materialised once here rather than re-derived
            // inside every butterfly of every one of the 3 * kRoomBands transforms.
            forward_plan_ = std::make_unique<dsp::FftPlan>(fft_size, false);
            inverse_plan_ = std::make_unique<dsp::FftPlan>(fft_size, true);
            line_.assign(fft_size, Complex(0.0, 0.0));
            for (int ear = 0; ear < 2; ++ear) {
                for (int band = 0; band < kRoomBands; ++band) {
                    for (int slot = 0; slot < length_; ++slot) {
                        line_[static_cast<std::size_t>(slot)] =
                            kernel_[(static_cast<std::size_t>(slot) * 2u +
                                     static_cast<std::size_t>(ear)) *
                                        kRoomBands +
                                    static_cast<std::size_t>(band)];
                    }
                    std::fill(line_.begin() + length_, line_.end(), Complex(0.0, 0.0));
                    dsp::fft_radix2(&line_, *forward_plan_);
                    for (std::size_t index = 0u; index < fft_size; ++index) {
                        // Band-major, so that the spectrum multiply below reads both
                        // of its operands as runs instead of striding through them
                        // one complex per cache line.  Same values, same places.
                        kernel_fft_[(static_cast<std::size_t>(ear) * kRoomBands +
                                     static_cast<std::size_t>(band)) *
                                        fft_size +
                                    index] = line_[index];
                    }
                }
            }
            fft_cache_size_ = fft_size;
        }
        // input_fft_ and block_ are indexed linearly and every entry of each is written
        // before it is read (the per-band forward transform fills input_fft_ column by
        // column, and the inverse transform fills block_ entry by entry), so their zero
        // fills were dead.  The line buffer, by contrast, is handed to fft_radix2,
        // which transforms exactly data->size() entries -- it therefore has to keep the
        // exact length and the exact initialisation it had as a local.
        if (line_.size() != fft_size) {
            line_.assign(fft_size, Complex(0.0, 0.0));
        }
        const std::size_t input_fft_size = fft_size * kRoomBands;
        if (input_fft_.size() < input_fft_size) {
            input_fft_.resize(input_fft_size);
        }
        for (int band = 0; band < kRoomBands; ++band) {
            for (std::size_t slot = 0u; slot < slots; ++slot) {
                line_[slot] = values[slot * kHybridBands + static_cast<std::size_t>(band)];
            }
            std::fill(line_.begin() + static_cast<std::ptrdiff_t>(slots), line_.end(),
                      Complex(0.0, 0.0));
            dsp::fft_radix2(&line_, *forward_plan_);
            for (std::size_t index = 0u; index < fft_size; ++index) {
                // Band-major, matching kernel_fft_, so that the spectrum multiply
                // reads both operands as runs.
                input_fft_[static_cast<std::size_t>(band) * fft_size + index] = line_[index];
            }
        }
        const std::size_t block_size = needed * 2u * kRoomBands;
        if (block_.size() < block_size) {
            block_.resize(block_size);
        }
        for (int ear = 0; ear < 2; ++ear) {
            for (int band = 0; band < kRoomBands; ++band) {
                // Both planes are band-major now, so this is one contiguous complex
                // product per (ear, band) and the dispatched kernel puts neighbouring
                // complexes in its lanes; each product keeps the caller's four
                // multiplies and two roundings.
                simd::complex_multiply(
                    reinterpret_cast<double*>(line_.data()),
                    reinterpret_cast<const double*>(
                        input_fft_.data() + static_cast<std::size_t>(band) * fft_size),
                    reinterpret_cast<const double*>(
                        kernel_fft_.data() +
                        (static_cast<std::size_t>(ear) * kRoomBands +
                         static_cast<std::size_t>(band)) *
                            fft_size),
                    fft_size);
                dsp::fft_radix2(&line_, *inverse_plan_);
                for (std::size_t slot = 0u; slot < needed; ++slot) {
                    block_[(slot * 2u + static_cast<std::size_t>(ear)) * kRoomBands +
                           static_cast<std::size_t>(band)] = line_[slot];
                }
            }
        }
        const std::size_t tail_length = static_cast<std::size_t>(length_ - 1);
        for (std::size_t index = 0u; index < tail_length * 2u * kRoomBands; ++index) {
            block_[index] += tail_[index];
        }
        for (std::size_t slot = 0u; slot < slots; ++slot) {
            for (int ear = 0; ear < 2; ++ear) {
                for (int band = 0; band < kRoomBands; ++band) {
                    (*output)[slot * 2u * kHybridBands + static_cast<std::size_t>(ear) * kHybridBands +
                              static_cast<std::size_t>(band)] =
                        block_[(slot * 2u + static_cast<std::size_t>(ear)) * kRoomBands +
                               static_cast<std::size_t>(band)];
                }
            }
        }
        for (std::size_t index = 0u; index < tail_length * 2u * kRoomBands; ++index) {
            tail_[index] = block_[slots * 2u * kRoomBands + index];
        }
        return Status::success();
    }

private:
    int length_ = 0;
    bool valid_ = false;
    std::vector<Complex> kernel_;
    std::vector<Complex> tail_;
    std::vector<Complex> kernel_fft_;
    std::size_t fft_cache_size_ = 0u;
    std::vector<Complex> line_;       // scratch, exactly fft_size entries
    std::vector<Complex> input_fft_;  // scratch, [fft_size][kRoomBands], fully written
    std::vector<Complex> block_;      // scratch, [needed][2][kRoomBands], fully written
    std::unique_ptr<dsp::FftPlan> forward_plan_;
    std::unique_ptr<dsp::FftPlan> inverse_plan_;
};

// rosella_core.RosellaRenderer: per-source direct gains plus the room send.
class Core {
public:
    Status configure(const RosellaModel& model, const std::vector<double>& field_left,
                     const std::vector<double>& field_right, int room_impulse_slots) {
        model_ = &model;
        field_left_ = &field_left;
        field_right_ = &field_right;
        room_ = std::make_unique<RoomFir>(model, room_impulse_slots);
        if (!room_->valid()) {
            return render_fail(JOC_ERR_NOT_SUPPORTED, "alternate Rosella table-A room mode");
        }
        gains_.assign(static_cast<std::size_t>(kSourceChannels) * 2u * kHybridBands,
                      Complex(0.0, 0.0));
        room_sends_.assign(kSourceChannels, 0.0);
        keys_.assign(kSourceChannels, Key{});
        for (int source = 0; source < kSourceChannels; ++source) {
            const double position[3] = {0.0, 1.0, 0.0};
            Status status = set_source(source, position,
                                       static_cast<int>(RosellaProfile::Mid), false);
            if (!status.ok()) {
                return status;
            }
        }
        return Status::success();
    }

    void reset() { room_->reset(); }

    Status set_source(int source, const double position[3], int profile, bool special_lfe) {
        const int effective_profile = special_lfe ? 0 : profile;
        Key key;
        key.special_lfe = special_lfe;
        key.profile = effective_profile;
        key.position[0] = position[0];
        key.position[1] = position[1];
        key.position[2] = position[2];
        if (keys_[static_cast<std::size_t>(source)].equals(key)) {
            return Status::success();
        }
        DirectResult result;
        if (special_lfe) {
            result = special_lfe_direct();
        } else {
            result = direct_and_room_send(*model_, *field_left_, *field_right_, position,
                                          effective_profile);
        }
        for (int ear = 0; ear < 2; ++ear) {
            for (int band = 0; band < kHybridBands; ++band) {
                gains_[(static_cast<std::size_t>(source) * 2u + static_cast<std::size_t>(ear)) *
                           kHybridBands +
                       static_cast<std::size_t>(band)] = result.gains[ear][band];
            }
        }
        room_sends_[static_cast<std::size_t>(source)] = result.room_send;
        keys_[static_cast<std::size_t>(source)] = key;
        return Status::success();
    }

    // rosella_core.RosellaRenderer.direct_and_send_static: the sources are summed in
    // reverse index order, which is part of the reference's arithmetic.
    void direct_and_send_static(const std::vector<Complex>& hybrid, std::size_t slots,
                                std::vector<Complex>* direct, std::vector<Complex>* room_send) {
        direct->assign(slots * 2u * kHybridBands, Complex(0.0, 0.0));
        room_send->assign(slots * kHybridBands, Complex(0.0, 0.0));
        for (int source = kSourceChannels - 1; source >= 0; --source) {
            const std::size_t base = static_cast<std::size_t>(source) * kHybridBands;
            for (std::size_t slot = 0u; slot < slots; ++slot) {
                const std::size_t input = slot * kSourceChannels * kHybridBands + base;
                const std::size_t room_target = slot * kHybridBands;
                for (int band = 0; band < kHybridBands; ++band) {
                    const Complex value = hybrid[input + static_cast<std::size_t>(band)];
                    for (int ear = 0; ear < 2; ++ear) {
                        (*direct)[slot * 2u * kHybridBands +
                                  static_cast<std::size_t>(ear) * kHybridBands +
                                  static_cast<std::size_t>(band)] +=
                            value * gains_[(static_cast<std::size_t>(source) * 2u +
                                            static_cast<std::size_t>(ear)) *
                                               kHybridBands +
                                           static_cast<std::size_t>(band)];
                    }
                    (*room_send)[room_target + static_cast<std::size_t>(band)] +=
                        value * room_sends_[static_cast<std::size_t>(source)];
                }
            }
        }
    }

    Status process_chunk(const std::vector<Complex>& hybrid, std::size_t slots,
                         std::vector<Complex>* output) {
        std::vector<Complex> direct;
        std::vector<Complex> room_send;
        direct_and_send_static(hybrid, slots, &direct, &room_send);
        std::vector<Complex> room;
        const Status status = room_->process_chunk(room_send, slots, &room);
        if (!status.ok()) {
            return status;
        }
        for (std::size_t index = 0u; index < direct.size(); ++index) {
            direct[index] += room[index];
        }
        *output = std::move(direct);
        return Status::success();
    }

    Status room_process(const std::vector<Complex>& room_send, std::size_t slots,
                        std::vector<Complex>* output) {
        return room_->process_chunk(room_send, slots, output);
    }

private:
    struct Key {
        bool special_lfe = false;
        int profile = 0;
        double position[3] = {0.0, 1.0, 0.0};
        bool equals(const Key& other) const {
            return special_lfe == other.special_lfe && profile == other.profile &&
                   position[0] == other.position[0] && position[1] == other.position[1] &&
                   position[2] == other.position[2];
        }
    };

    const RosellaModel* model_ = nullptr;
    const std::vector<double>* field_left_ = nullptr;
    const std::vector<double>* field_right_ = nullptr;
    std::unique_ptr<RoomFir> room_;
    std::vector<Complex> gains_;
    std::vector<double> room_sends_;
    std::vector<Key> keys_;
};

}  // namespace

struct RosellaRuntime::Impl {
    RosellaModel model;
    RosellaRenderOptions options;
    std::vector<double> field_left;
    std::vector<double> field_right;
    Core core;
    std::unique_ptr<PublicFilterbank> analysis_bank;
    std::unique_ptr<PublicFilterbank> synthesis_bank;
    timeline::OamdPositionTimeline timeline{timeline::kTimelineObjects};

    std::vector<double> buffer;  // [chunk_samples][16]
    std::size_t buffer_used = 0;
    std::vector<double> output;
    std::uint64_t input_samples = 0;
    std::uint64_t processed_input_samples = 0;
    std::uint64_t raw_output_samples = 0;
    std::uint64_t metadata_block_updates = 0;
    bool open = false;
    bool finished = false;

    std::size_t chunk_samples() const {
        return static_cast<std::size_t>(options.chunk_frames) * kFrameSamples;
    }

    // rosella_binaural_renderer._process_samples.
    Status process_samples(const double* values, std::size_t count) {
        const std::size_t blocks = count / kBlockSamples;
        const std::size_t slots = blocks * (kBlockSamples / kQmfHop);
        // Every plane below is written in full before it is read, so they live as
        // reusable buffers rather than as per-chunk allocations: the chunk geometry is
        // constant, which turns each one into a no-op size check after the first chunk.
        const std::size_t hops_size = slots * kSourceChannels * kQmfHop;
        if (hops_.size() < hops_size) {
            hops_.resize(hops_size);
        }
        for (std::size_t block = 0u; block < blocks; ++block) {
            for (std::size_t hop = 0u; hop < kBlockSamples / kQmfHop; ++hop) {
                for (std::size_t channel = 0u; channel < kSourceChannels; ++channel) {
                    for (int index = 0; index < kQmfHop; ++index) {
                        hops_[((block * (kBlockSamples / kQmfHop) + hop) * kSourceChannels +
                               channel) *
                                  kQmfHop +
                              static_cast<std::size_t>(index)] =
                            values[(block * kBlockSamples + hop * kQmfHop +
                                    static_cast<std::size_t>(index)) *
                                       kSourceChannels +
                                   channel];
                    }
                }
            }
        }
        analysis_bank->analyze_qmf(hops_, slots, &qmf_);
        analysis_bank->analyze_hybrid(qmf_, slots, &hybrid_);

        if (rendered_.size() < slots * 2u * kHybridBands) {
            rendered_.resize(slots * 2u * kHybridBands);
        }
        if (room_sends_buffer_.size() != slots * kHybridBands) {
            room_sends_buffer_.resize(slots * kHybridBands);
        }
        for (std::size_t block = 0u; block < blocks; ++block) {
            const std::size_t first_slot = block * (kBlockSamples / kQmfHop);
            const std::size_t block_slots = kBlockSamples / kQmfHop;
            const std::int64_t sample =
                static_cast<std::int64_t>(processed_input_samples) +
                static_cast<std::int64_t>(block * kBlockSamples);
            Status status = set_block_parameters(sample);
            if (!status.ok()) {
                return status;
            }
            std::vector<Complex> block_hybrid(
                hybrid_.begin() + static_cast<std::ptrdiff_t>(first_slot * kSourceChannels *
                                                              kHybridBands),
                hybrid_.begin() + static_cast<std::ptrdiff_t>((first_slot + block_slots) *
                                                              kSourceChannels * kHybridBands));
            std::vector<Complex> direct;
            std::vector<Complex> room_send;
            core.direct_and_send_static(block_hybrid, block_slots, &direct, &room_send);
            std::copy(direct.begin(), direct.end(),
                      rendered_.begin() + static_cast<std::ptrdiff_t>(first_slot * 2u * kHybridBands));
            std::copy(room_send.begin(), room_send.end(),
                      room_sends_buffer_.begin() +
                          static_cast<std::ptrdiff_t>(first_slot * kHybridBands));
        }
        // The room runs once for the whole chunk, as the reference does.  Its output
        // keeps a zero fill of its own: only the first kRoomBands of each ear are
        // written, and the caller adds the whole 77-band slot back into `rendered`.
        Status status = core.room_process(room_sends_buffer_, slots, &room_out_);
        if (!status.ok()) {
            return status;
        }
        for (std::size_t index = 0u; index < rendered_.size(); ++index) {
            rendered_[index] += room_out_[index];
        }
        synthesis_bank->synthesize_hybrid(rendered_, slots, &qmf_out_);
        // The synthesis bank returns [slots, ears, 64]; the reference transposes it to
        // sample-major stereo before the latency skip.
        synthesis_bank->synthesize_qmf(qmf_out_, slots, &bands_);
        const std::size_t produced = blocks * kBlockSamples;
        if (time_.size() != produced * kOutputChannels) {
            time_.resize(produced * kOutputChannels);
        }
        for (std::size_t slot = 0u; slot < slots; ++slot) {
            for (std::size_t channel = 0u; channel < kOutputChannels; ++channel) {
                for (int band = 0; band < kQmfHop; ++band) {
                    time_[(slot * kQmfHop + static_cast<std::size_t>(band)) * kOutputChannels +
                          channel] =
                        bands_[(slot * kOutputChannels + channel) * kQmfHop +
                               static_cast<std::size_t>(band)];
                }
            }
        }
        for (std::size_t sample = 0u; sample < produced; ++sample) {
            for (std::size_t channel = 0u; channel < kOutputChannels; ++channel) {
                time_[sample * kOutputChannels + channel] *= options.output_gain;
            }
        }
        append_output(time_.data(), produced);
        processed_input_samples += count;
        return Status::success();
    }

    // rosella_binaural_renderer._set_block_parameters.
    Status set_block_parameters(std::int64_t sample) {
        double positions[timeline::kTimelineObjects][3] = {};
        Status status = timeline.positions_at(sample, positions);
        if (!status.ok()) {
            return status;
        }
        const double lfe[3] = {0.0, 1.0, 0.0};
        status = core.set_source(0, lfe, static_cast<int>(RosellaProfile::Mid), true);
        if (!status.ok()) {
            return status;
        }
        for (int object = 0; object < timeline::kTimelineObjects; ++object) {
            status = core.set_source(object + 1, positions[object],
                                     static_cast<int>(options.profile), false);
            if (!status.ok()) {
                return status;
            }
        }
        return Status::success();
    }

    // The reference drops the first 961 samples of the raw stream (the filterbank's
    // latency) and returns everything after that.
    void append_output(const double* stereo, std::size_t count) {
        const std::int64_t remaining = static_cast<std::int64_t>(kLatencySamples) -
                                       static_cast<std::int64_t>(raw_output_samples);
        const std::size_t skip = static_cast<std::size_t>(
            std::max<std::int64_t>(0, std::min<std::int64_t>(static_cast<std::int64_t>(count),
                                                             remaining)));
        raw_output_samples += count;
        for (std::size_t sample = skip; sample < count; ++sample) {
            output.push_back(stereo[sample * kOutputChannels]);
            output.push_back(stereo[sample * kOutputChannels + 1u]);
        }
    }

    // Per-chunk scratch.  Every one of these planes is written in full before it is
    // read, so they are sized once and reused instead of being allocated per chunk.
    std::vector<double> hops_;          // [slots][kSourceChannels][64]
    std::vector<Complex> qmf_;          // [slots][kSourceChannels][64]
    std::vector<Complex> hybrid_;       // [slots][kSourceChannels][77]
    std::vector<Complex> rendered_;     // [slots][2][77]
    std::vector<Complex> room_out_;     // [slots][2][77], partially written (see above)
    std::vector<Complex> qmf_out_;      // [slots][2][64]
    std::vector<double> bands_;         // [slots][2][64]
    std::vector<double> time_;          // [produced][2]
    std::vector<Complex> room_sends_buffer_;
};

RosellaRuntime::RosellaRuntime() : impl_(std::make_unique<Impl>()) {}
RosellaRuntime::~RosellaRuntime() = default;

Status RosellaRuntime::open(const RosellaModel& model, const RosellaRenderOptions& options) {
    if (impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is already open");
    }
    if (options.chunk_frames <= 0) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "chunk_frames must be positive");
    }
    if (options.room_impulse_slots <= 0) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "room_impulse_slots must be positive");
    }
    if (!(options.tail_seconds >= 0.0)) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "tail_seconds must be non-negative");
    }
    if (!std::isfinite(options.output_gain)) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "output_gain must be finite");
    }
    if (options.object_delay_samples < 0) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "object_delay_samples must be non-negative");
    }
    impl_->model = model;
    impl_->options = options;
    logical_field(impl_->model.field_left_padded, &impl_->field_left);
    logical_field(impl_->model.field_right_padded, &impl_->field_right);
    // The core and the room keep references, so they must point at this runtime's
    // own copy of the model rather than at the caller's.
    Status status = impl_->core.configure(impl_->model, impl_->field_left, impl_->field_right,
                                          options.room_impulse_slots);
    if (!status.ok()) {
        return status;
    }
    // The analysis bank runs sixteen sources, the synthesis bank two output ears.
    impl_->analysis_bank = std::make_unique<PublicFilterbank>(builtin_kernels(), kSourceChannels);
    impl_->synthesis_bank = std::make_unique<PublicFilterbank>(builtin_kernels(), kOutputChannels);
    impl_->buffer.assign(impl_->chunk_samples() * kSourceChannels, 0.0);
    impl_->room_sends_buffer_.assign(
        (impl_->chunk_samples() / kQmfHop) * kHybridBands, Complex(0.0, 0.0));
    impl_->buffer_used = 0;
    impl_->output.clear();
    impl_->input_samples = 0u;
    impl_->processed_input_samples = 0u;
    impl_->raw_output_samples = 0u;
    impl_->metadata_block_updates = 0u;
    impl_->finished = false;
    impl_->open = true;
    return Status::success();
}

Status RosellaRuntime::submit_frame(const float* objects16_planar, const oamd::OamdUpdate* update,
                                    std::int64_t frame_index, std::int64_t outer_sample_offset,
                                    std::int64_t object_delay_samples) {
    // The timeline is addressed by the number of samples submitted so far, exactly as
    // the reference addresses it, rather than by the caller's frame index.
    (void)frame_index;
    if (!impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is not open");
    }
    if (impl_->finished) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is already finished");
    }
    if (objects16_planar == nullptr) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "null frame");
    }
    if (update != nullptr) {
        const Status submitted = impl_->timeline.submit_update(
            *update, static_cast<std::int64_t>(impl_->input_samples), outer_sample_offset,
            object_delay_samples, static_cast<std::int64_t>(impl_->processed_input_samples));
        if (!submitted.ok()) {
            return submitted;
        }
        ++impl_->metadata_block_updates;
    }
    const std::size_t chunk = impl_->chunk_samples();
    std::size_t offset = 0u;
    while (offset < kFrameSamples) {
        const std::size_t room = chunk - impl_->buffer_used;
        const std::size_t count = std::min<std::size_t>(room, kFrameSamples - offset);
        for (std::size_t sample = 0u; sample < count; ++sample) {
            double* row = impl_->buffer.data() + (impl_->buffer_used + sample) * kSourceChannels;
            for (std::size_t channel = 0u; channel < kSourceChannels; ++channel) {
                row[channel] = static_cast<double>(
                    objects16_planar[channel * kFrameSamples + offset + sample]);
            }
        }
        impl_->buffer_used += count;
        offset += count;
        if (impl_->buffer_used == chunk) {
            const Status status = impl_->process_samples(impl_->buffer.data(), chunk);
            if (!status.ok()) {
                return status;
            }
            impl_->buffer_used = 0u;
        }
    }
    impl_->input_samples += kFrameSamples;
    return Status::success();
}

Status RosellaRuntime::finish(std::uint32_t flush_samples, std::vector<double>* out) {
    if (!impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is not open");
    }
    if (out != nullptr) {
        out->clear();
    }
    if (impl_->finished) {
        return Status::success();
    }
    // Only the flush tail is returned: the caller has already taken the program.
    const std::size_t program_end = impl_->output.size();
    if (impl_->buffer_used != 0u) {
        const Status status = impl_->process_samples(impl_->buffer.data(), impl_->buffer_used);
        if (!status.ok()) {
            return status;
        }
        impl_->buffer_used = 0u;
    }
    const std::size_t chunk = impl_->chunk_samples();
    std::vector<double> silence;
    std::uint32_t remaining = flush_samples;
    while (remaining > 0u) {
        const std::size_t count = std::min<std::size_t>(remaining, chunk);
        silence.assign(count * kSourceChannels, 0.0);
        const Status status = impl_->process_samples(silence.data(), count);
        if (!status.ok()) {
            return status;
        }
        remaining -= static_cast<std::uint32_t>(count);
    }
    impl_->finished = true;
    if (out != nullptr) {
        out->assign(impl_->output.begin() + static_cast<std::ptrdiff_t>(program_end),
                    impl_->output.end());
    }
    return Status::success();
}

std::uint32_t RosellaRuntime::finish_capacity(double tail_seconds) const {
    const double samples = tail_seconds * 48000.0 + static_cast<double>(kLatencySamples) +
                           static_cast<double>(kBlockSamples);
    const double blocks = std::ceil(samples / static_cast<double>(kBlockSamples));
    return static_cast<std::uint32_t>(blocks) * kBlockSamples;
}

Status RosellaRuntime::reset() {
    if (!impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is not open");
    }
    impl_->core.reset();
    impl_->analysis_bank->reset();
    impl_->synthesis_bank->reset();
    impl_->buffer_used = 0u;
    impl_->output.clear();
    impl_->input_samples = 0u;
    impl_->processed_input_samples = 0u;
    impl_->raw_output_samples = 0u;
    impl_->metadata_block_updates = 0u;
    impl_->finished = false;
    impl_->timeline = timeline::OamdPositionTimeline(timeline::kTimelineObjects);
    return Status::success();
}

const std::vector<double>& RosellaRuntime::output() const { return impl_->output; }

void RosellaRuntime::take_output(std::vector<double>* out) {
    if (out == nullptr) {
        return;
    }
    // Replace semantics, exactly like SofaBinauralRuntime::take_output: the caller's
    // previous contents are discarded, and the internal buffer is emptied afterwards
    // because the swap leaves it holding whatever the caller had.
    out->swap(impl_->output);
    impl_->output.clear();
}

std::uint64_t RosellaRuntime::input_samples() const { return impl_->input_samples; }
std::uint64_t RosellaRuntime::processed_input_samples() const {
    return impl_->processed_input_samples;
}
std::uint64_t RosellaRuntime::metadata_block_updates() const {
    return impl_->metadata_block_updates;
}
const timeline::OamdPositionTimeline& RosellaRuntime::timeline() const { return impl_->timeline; }

}  // namespace joc::hrtf
