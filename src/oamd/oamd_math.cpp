#include "oamd/oamd_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "joc_bitstream/bed_layout_map.h"

namespace joc::oamd {

namespace {

// 2**-126, the reference's TINY.
constexpr float kTiny = 1.17549435e-38f;
// The float32 constants of the reference equations, written as the reference
// writes them so that the rounding matches.
constexpr float kQuarter = 0.25f;
constexpr float kHalf = 0.5f;
constexpr float kPanAngle = 0.78539819f;
constexpr float kLayerAngle = 0.39269909f;
constexpr float kBoundaryAngle = 0.34147748f;
constexpr float kBoundaryPower = 26.575424f;
constexpr float kBoundaryShape = 0.6328125f;
constexpr float kSizeStep = 0.036842104f;
constexpr float kSizeCurveScale = 0.71428573f;
constexpr float kQ15Scale = 32768.0f;
constexpr float kLogScale = 4.0793457f;
constexpr float kLogOffset = 2.6931152f;
constexpr float kLogSquare = 1.3862305f;
constexpr float kExpFactor = -0.1660964f;
constexpr std::size_t kMaxPoints = ::ejoc::speaker_tables::kMaxPoints;
constexpr std::size_t kMaxGroups = ::ejoc::speaker_tables::kMaxGroups;

inline float max_float(const float left, const float right) {
    return left > right ? left : right;
}

inline float min_float(const float left, const float right) {
    return left < right ? left : right;
}

}  // namespace

float exp2_approx(const float value) {
    const float x = value;
    const int n = static_cast<int>(std::floor(static_cast<double>(x)));
    const float f = x - static_cast<float>(n);
    float poly = f * 0.6862793f;
    poly = poly + 1.0f;
    poly = poly + (f * f) * 0.25482178f;
    poly = poly + (f * (f * f)) * 0.058898926f;
    return static_cast<float>(std::ldexp(static_cast<double>(poly), n));
}

float pow_approx(const float value, const float power) {
    const float x = value;
    if (std::fabs(x) < kTiny) {
        return 0.0f;
    }
    int exponent = 0;
    const double fraction = std::frexp(static_cast<double>(x), &exponent);
    const float fraction32 = static_cast<float>(fraction);
    float logarithm = fraction32 * kLogScale;
    logarithm = logarithm - kLogOffset;
    logarithm = logarithm - (fraction32 * fraction32) * kLogSquare;
    logarithm = logarithm + static_cast<float>(exponent);
    return exp2_approx(logarithm * power);
}

std::vector<float> norm(const std::vector<float>& vector) {
    float total = 0.0f;
    for (const float value : vector) {
        total = total + value * value;
    }
    if (std::fabs(total) < kTiny) {
        total = 0.0f;
    }
    const float root = static_cast<float>(std::sqrt(static_cast<double>(total)));
    const float denominator = max_float(root, 1e-5f);
    std::vector<float> result(vector.size());
    for (std::size_t index = 0; index < vector.size(); ++index) {
        float value = vector[index] / denominator;
        if (std::fabs(value) < kTiny) {
            value = 0.0f;
        }
        result[index] = value;
    }
    return result;
}

float boundary(const int axis, const int edge, const float position, const float t) {
    const float edge32 = static_cast<float>(edge);
    const float position32 = position;
    const float t32 = t;
    const float delta = (edge32 - position32) * kQuarter;
    if (t32 < 0.000025f || std::fabs(delta) > t32) {
        return 0.0f;
    }
    const float ratio = static_cast<float>(static_cast<double>(delta) /
                                           static_cast<double>(t32));
    const float ratio2 = ratio * ratio;
    float weight = exp2_approx((-(ratio2 * ratio2) * kBoundaryShape) * kBoundaryPower);
    int exponent = 0;
    std::frexp(static_cast<double>(weight), &exponent);
    if (exponent < -31 || exponent > 1) {
        weight = 0.0f;
    }
    if (axis == 2) {
        const float angle = (edge32 * kBoundaryAngle) * 4.0f;
        weight = std::cos(angle) * weight;
    }
    return weight;
}

namespace {

struct PanVector {
    std::vector<float> values;
};

struct RegionCoordinates {
    std::array<std::array<float, 3>, kMaxPoints> values{};
    int count = 0;
};

RegionCoordinates region_coordinates(const ::ejoc::speaker_tables::RegionGeometry& region) {
    RegionCoordinates coordinates;
    coordinates.count = region.point_count;
    for (int index = 0; index < region.point_count; ++index) {
        for (int component = 0; component < 3; ++component) {
            coordinates.values[static_cast<std::size_t>(index)][static_cast<std::size_t>(
                component)] = static_cast<float>(region.points[static_cast<std::size_t>(index)]
                                                     .coordinate_q15[static_cast<std::size_t>(
                                                         component)]) /
                               kQ15Scale;
        }
    }
    return coordinates;
}

using GroupList = std::vector<std::vector<int>>;

GroupList collect_groups(const std::array<::ejoc::speaker_tables::AxisGroup, kMaxGroups>& groups,
                         const int count) {
    GroupList result;
    result.reserve(static_cast<std::size_t>(count));
    for (int row = 0; row < count; ++row) {
        const auto& group = groups[static_cast<std::size_t>(row)];
        std::vector<int> indices;
        indices.reserve(group.size);
        for (std::size_t index = 0; index < group.size; ++index) {
            indices.push_back(static_cast<int>(group.indices[index]));
        }
        result.push_back(std::move(indices));
    }
    return result;
}

int group_total(const GroupList& groups) {
    int total = 0;
    for (const auto& group : groups) {
        total += static_cast<int>(group.size());
    }
    return total;
}

void group_pan_x(const RegionCoordinates& coordinates, const GroupList& groups,
                 const float position, std::vector<float>& output) {
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        const std::size_t first = static_cast<std::size_t>(group.front());
        const std::size_t last = static_cast<std::size_t>(group.back());
        const float x_first = coordinates.values[first][0];
        const float x_last = coordinates.values[last][0];
        if (position < x_first) {
            output[first] = 1.0f;
            continue;
        }
        if (position > x_last) {
            output[last] = 1.0f;
            continue;
        }
        for (std::size_t index = 0; index + 1 < group.size(); ++index) {
            const std::size_t left = static_cast<std::size_t>(group[index]);
            const std::size_t right = static_cast<std::size_t>(group[index + 1]);
            const float x_left = coordinates.values[left][0];
            const float x_right = coordinates.values[right][0];
            if (!(x_left < x_right && x_left <= position && position <= x_right)) {
                continue;
            }
            const float fraction = (position - x_left) / (x_right - x_left);
            if (fraction <= kHalf) {
                const float angle = ((fraction * kPanAngle) * kHalf) * 4.0f;
                output[left] = std::cos(angle);
                output[right] = std::sin(angle);
            } else {
                const float angle = (((1.0f - fraction) * kPanAngle) * kHalf) * 4.0f;
                output[left] = std::sin(angle);
                output[right] = std::cos(angle);
            }
            break;
        }
    }
}

void group_pan_y(const RegionCoordinates& coordinates, const GroupList& groups,
                 const float position, std::vector<float>& output) {
    if (groups.empty()) {
        return;
    }
    std::vector<float> rows;
    rows.reserve(groups.size());
    for (const auto& group : groups) {
        rows.push_back(coordinates.values[static_cast<std::size_t>(group.front())][1]);
    }
    if (position <= rows.front()) {
        for (const int index : groups.front()) {
            output[static_cast<std::size_t>(index)] = 1.0f;
        }
        return;
    }
    if (position > rows.back()) {
        for (const int index : groups.back()) {
            output[static_cast<std::size_t>(index)] = 1.0f;
        }
        return;
    }
    for (std::size_t row = 0; row + 1 < groups.size(); ++row) {
        const float y_low = rows[row];
        const float y_high = rows[row + 1];
        if (!(y_low <= position && position <= y_high && y_low < y_high)) {
            continue;
        }
        const float fraction = (position - y_low) / (y_high - y_low);
        float low_gain = 0.0f;
        float high_gain = 0.0f;
        if (fraction <= kHalf) {
            const float angle = ((fraction * kPanAngle) * kHalf) * 4.0f;
            low_gain = std::cos(angle);
            high_gain = std::sin(angle);
        } else {
            const float angle = (((1.0f - fraction) * kPanAngle) * kHalf) * 4.0f;
            low_gain = std::sin(angle);
            high_gain = std::cos(angle);
        }
        for (const int index : groups[row]) {
            output[static_cast<std::size_t>(index)] = low_gain;
        }
        for (const int index : groups[row + 1]) {
            output[static_cast<std::size_t>(index)] = high_gain;
        }
        return;
    }
}

std::vector<float> layer_pan_z(const RegionCoordinates& coordinates, const float position) {
    float floor_gain = 0.0f;
    float height_gain = 0.0f;
    if (position <= 0.0f) {
        floor_gain = 1.0f;
        height_gain = 0.0f;
    } else if (position < 0.99996948f) {
        const float angle = (position * kLayerAngle) * 4.0f;
        floor_gain = std::cos(angle);
        height_gain = std::sin(angle);
    } else {
        floor_gain = 0.0f;
        height_gain = 1.0f;
    }
    std::vector<float> output(static_cast<std::size_t>(coordinates.count), 0.0f);
    for (int index = 0; index < coordinates.count; ++index) {
        output[static_cast<std::size_t>(index)] =
            coordinates.values[static_cast<std::size_t>(index)][2] >= kHalf ? height_gain
                                                                           : floor_gain;
    }
    return output;
}

std::vector<float> pan_x_all(const RegionCoordinates& coordinates, const GroupList& floor_groups,
                             const GroupList& height_groups, const float position,
                             const int mode) {
    std::vector<float> output(static_cast<std::size_t>(coordinates.count), 0.0f);
    group_pan_x(coordinates, floor_groups, position, output);
    if (mode == 3 && !height_groups.empty()) {
        std::vector<float> extra(static_cast<std::size_t>(coordinates.count), 0.0f);
        group_pan_x(coordinates, height_groups, position, extra);
        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] = output[index] + extra[index];
        }
    }
    return output;
}

std::vector<float> pan_y_all(const RegionCoordinates& coordinates, const GroupList& floor_groups,
                             const GroupList& height_groups, const float position,
                             const int mode) {
    std::vector<float> output(static_cast<std::size_t>(coordinates.count), 0.0f);
    group_pan_y(coordinates, floor_groups, position, output);
    if (mode == 3 && !height_groups.empty()) {
        std::vector<float> extra(static_cast<std::size_t>(coordinates.count), 0.0f);
        group_pan_y(coordinates, height_groups, position, extra);
        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] = output[index] + extra[index];
        }
    }
    return output;
}

using BoundaryRows = std::array<std::vector<float>, 5>;

BoundaryRows boundary_rows(const RegionCoordinates& coordinates, const GroupList& floor_groups,
                           const GroupList& height_groups, const int mode) {
    BoundaryRows rows;
    rows[0] = pan_x_all(coordinates, floor_groups, height_groups, 0.0f, mode);
    rows[1] = pan_x_all(coordinates, floor_groups, height_groups, 1.0f, mode);
    rows[2] = pan_y_all(coordinates, floor_groups, height_groups, 0.0f, mode);
    rows[3] = pan_y_all(coordinates, floor_groups, height_groups, 1.0f, mode);
    if (mode == 3) {
        rows[4] = layer_pan_z(coordinates, 1.0f);
    } else {
        rows[4] = std::vector<float>(static_cast<std::size_t>(coordinates.count), 0.0f);
    }
    return rows;
}

// Per-virtual-point S448 weight used by the generated size tables.
float size_weight(const float position, const float u, const float t, const int axis) {
    const float offset = ((u - position) * kQuarter);
    if (t < 0.000025f || std::fabs(offset) > t) {
        return 0.0f;
    }
    const float ratio = offset / t;
    const float squared = ratio * ratio;
    const float power = (-(squared * squared)) * kBoundaryShape;
    float weight = exp2_approx(power * kBoundaryPower);
    if (axis == 2) {
        const float angle = (u * kBoundaryAngle) * 4.0f;
        weight = std::cos(angle) * weight;
    }
    return weight;
}

float size_m(const float t) {
    return t <= 0.125f ? 6.0f : (6.869565f - t * 6.956522f);
}

struct AxisTable {
    int position_nodes = 0;
    int size_nodes = 0;
    int floor_stride = 0;
    int height_stride = 0;
    std::vector<float> floor_values;
    std::vector<float> height_values;
};

AxisTable axis_table(const std::vector<PanVector>& pan_vectors, const int position_nodes,
                     const int speaker_count, const int offset_start, const int axis) {
    AxisTable table;
    table.position_nodes = position_nodes;
    table.size_nodes = 20;
    table.floor_stride = 0;
    table.height_stride = 0;
    table.floor_values.assign(
        static_cast<std::size_t>(position_nodes) * 20u *
            static_cast<std::size_t>(speaker_count),
        0.0f);
    const float denominator = static_cast<float>(pan_vectors.size() - 1u);
    for (int position_index = 0; position_index < position_nodes; ++position_index) {
        const float position = static_cast<float>(position_index) /
                               static_cast<float>(position_nodes - 1);
        for (int size_index = 0; size_index < 20; ++size_index) {
            const float t = static_cast<float>(size_index) * kSizeStep;
            const float m = size_m(t);
            for (std::size_t u = 0; u < pan_vectors.size(); ++u) {
                const float weight =
                    size_weight(position, static_cast<float>(u) / denominator, t, axis);
                if (weight == 0.0f) {
                    continue;
                }
                for (int speaker = 0; speaker < speaker_count; ++speaker) {
                    const float value =
                        pan_vectors[u].values[static_cast<std::size_t>(offset_start + speaker)] *
                        weight;
                    const std::size_t index =
                        (static_cast<std::size_t>(position_index) * 20u +
                         static_cast<std::size_t>(size_index)) *
                            static_cast<std::size_t>(speaker_count) +
                        static_cast<std::size_t>(speaker);
                    table.floor_values[index] =
                        table.floor_values[index] + pow_approx(value, m);
                }
            }
        }
    }
    return table;
}

AxisTable axis2_table(const std::vector<PanVector>& pan_vectors, const int speaker_index) {
    AxisTable table;
    table.position_nodes = 4;
    table.size_nodes = 20;
    table.floor_stride = 1;
    table.height_stride = 1;
    // The z axis tables are stored scalar per (position node, size node); the
    // reference repeats the scalar over the group members when it consumes it.
    table.floor_values.assign(4u * 20u, 0.0f);
    table.height_values.assign(4u * 20u, 0.0f);
    if (pan_vectors.empty() ||
        static_cast<std::size_t>(speaker_index) >= pan_vectors.front().values.size()) {
        return table;
    }
    std::vector<float>& target = table.floor_values;
    for (int position_index = 0; position_index < 4; ++position_index) {
        const float position = static_cast<float>(position_index) / 3.0f;
        for (int size_index = 0; size_index < 20; ++size_index) {
            const float t = static_cast<float>(size_index) * kSizeStep;
            const float m = size_m(t);
            float total = 0.0f;
            for (std::size_t u = 0; u < pan_vectors.size(); ++u) {
                const float weight = size_weight(position, static_cast<float>(u) / 7.0f, t, 2);
                if (weight == 0.0f) {
                    continue;
                }
                const float value =
                    pan_vectors[u].values[static_cast<std::size_t>(speaker_index)] * weight;
                total = total + pow_approx(value, m);
            }
            target[static_cast<std::size_t>(position_index) * 20u +
                   static_cast<std::size_t>(size_index)] = total;
        }
    }
    return table;
}

struct RegionSizeTables {
    int floor_count = 0;
    int height_count = 0;
    int mode = 0;
    int floor_axis_group_count = 0;
    int height_axis_group_count = 0;
    BoundaryRows boundary_3d;
    BoundaryRows boundary_2d;
    AxisTable axes[3];
};

RegionSizeTables build_region_size_tables(const ::ejoc::speaker_tables::RegionGeometry& region) {
    RegionSizeTables entry;
    const RegionCoordinates coordinates = region_coordinates(region);
    const GroupList floor_groups = collect_groups(region.axis0_groups,
                                                  region.axis0_group_count);
    const GroupList height_groups =
        region.mode == 3 ? collect_groups(region.axis1_groups, region.axis1_group_count)
                         : GroupList{};
    entry.floor_count = group_total(floor_groups);
    entry.height_count = group_total(height_groups);
    entry.mode = region.mode;
    entry.floor_axis_group_count = static_cast<int>(floor_groups.size());
    entry.height_axis_group_count = static_cast<int>(height_groups.size());

    std::vector<PanVector> x_vectors;
    std::vector<PanVector> y_vectors;
    std::vector<PanVector> z_vectors;
    x_vectors.reserve(20);
    y_vectors.reserve(20);
    z_vectors.reserve(8);
    for (int index = 0; index < 20; ++index) {
        const float position = static_cast<float>(index) / 19.0f;
        x_vectors.push_back(PanVector{pan_x_all(coordinates, floor_groups, height_groups,
                                                position, region.mode)});
        y_vectors.push_back(PanVector{pan_y_all(coordinates, floor_groups, height_groups,
                                                position, region.mode)});
    }
    if (region.mode == 3) {
        for (int index = 0; index < 8; ++index) {
            z_vectors.push_back(
                PanVector{layer_pan_z(coordinates, static_cast<float>(index) / 7.0f)});
        }
    } else {
        for (int index = 0; index < 8; ++index) {
            z_vectors.push_back(PanVector{
                std::vector<float>(static_cast<std::size_t>(coordinates.count), 0.0f)});
        }
    }

    entry.axes[0] = axis_table(x_vectors, 35, entry.floor_count, 0, 0);
    entry.axes[0].height_values =
        axis_table(x_vectors, 35, entry.height_count, entry.floor_count, 0).floor_values;
    entry.axes[0].height_stride = entry.height_count;
    entry.axes[0].floor_stride = entry.floor_count;

    entry.axes[1] = axis_table(y_vectors, 35, entry.floor_count, 0, 1);
    entry.axes[1].height_values =
        axis_table(y_vectors, 35, entry.height_count, entry.floor_count, 1).floor_values;
    entry.axes[1].height_stride = entry.height_count;
    entry.axes[1].floor_stride = entry.floor_count;

    if (region.mode == 3) {
        entry.axes[2] = axis2_table(z_vectors, 0);
        entry.axes[2].height_values = axis2_table(z_vectors, entry.floor_count).floor_values;
    } else {
        entry.axes[2] = AxisTable{};
        entry.axes[2].position_nodes = 4;
        entry.axes[2].size_nodes = 20;
        entry.axes[2].floor_stride = 1;
        entry.axes[2].height_stride = 1;
    }

    entry.boundary_3d = boundary_rows(coordinates, floor_groups, height_groups, region.mode);
    entry.boundary_2d = boundary_rows(coordinates, floor_groups, GroupList{}, 2);
    return entry;
}

const RegionSizeTables& region_size_tables(const ::ejoc::speaker_tables::RegionGeometry& region) {
    static std::unordered_map<const void*, RegionSizeTables> cache;
    const auto found = cache.find(&region);
    if (found != cache.end()) {
        return found->second;
    }
    return cache.emplace(&region, build_region_size_tables(region)).first->second;
}

void bracket(const float position, const int axis, int* low, int* high, float* alpha) {
    const int last = axis == 2 ? 3 : 34;
    const float step = 1.0f / static_cast<float>(last);
    int low_value = static_cast<int>(
        std::floor(static_cast<double>(static_cast<float>(last) * position)));
    int high_value = std::min(low_value + 1, last);
    if (low_value > last) {
        low_value = last;
        high_value = last;
    }
    *low = low_value;
    *high = high_value;
    if (low_value == high_value) {
        *alpha = 0.0f;
    } else {
        const float position32 = position;
        const float remainder = position32 -
                                (static_cast<float>(low_value) * step);
        *alpha = max_float(0.0f, remainder / step);
    }
}

void size_bracket(const float t, int* low, int* high, float* beta) {
    const float step = kSizeStep;
    int low_value = static_cast<int>(std::floor(static_cast<double>(
        ((t * kSizeCurveScale) * 2.0f) * 19.0f)));
    if (low_value > 19) {
        *low = 19;
        *high = 19;
        *beta = 0.0f;
        return;
    }
    const int high_value = std::min(low_value + 1, 19);
    *low = low_value;
    *high = high_value;
    if (low_value == high_value) {
        *beta = 0.0f;
    } else {
        const float remainder = t - (static_cast<float>(low_value) * step);
        *beta = max_float(
            0.0f, static_cast<float>(static_cast<double>(remainder) /
                                     static_cast<double>(step)));
    }
}

std::vector<float> axis_weights(const RegionSizeTables& entry, const int axis,
                                const float position, const float t, const bool height) {
    int low = 0;
    int high = 0;
    float alpha = 0.0f;
    bracket(position, axis, &low, &high, &alpha);
    int low_t = 0;
    int high_t = 0;
    float beta = 0.0f;
    size_bracket(t, &low_t, &high_t, &beta);
    const float weight0 = 1.0f - max_float(beta, alpha);
    const float weight1 = 1.0f - max_float(beta, 1.0f - alpha);
    const float weight2 = 1.0f - max_float(1.0f - beta, alpha);
    const float weight3 = 1.0f - max_float(1.0f - beta, 1.0f - alpha);
    const float total = ((weight1 + weight0) + weight2) + weight3;
    const float scales[4] = {weight0 / total, weight1 / total, weight2 / total,
                             weight3 / total};

    const AxisTable& table = entry.axes[axis];
    std::vector<float> result;
    result.reserve(static_cast<std::size_t>(entry.floor_count +
                                            (height ? entry.height_count : 0)));
    const auto append_group = [&](const std::vector<float>& values, const int count,
                                  const int stride) {
        if (count == 0) {
            return;
        }
        const auto index = [&](const int position_index, const int size_index,
                               const int speaker) {
            const int column = stride == 1 ? 0 : speaker;
            return (static_cast<std::size_t>(position_index) * 20u +
                    static_cast<std::size_t>(size_index)) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(column);
        };
        for (int speaker = 0; speaker < count; ++speaker) {
            const float value =
                ((scales[0] * values[index(low, low_t, speaker)] +
                  scales[1] * values[index(high, low_t, speaker)]) +
                 scales[2] * values[index(low, high_t, speaker)]) +
                scales[3] * values[index(high, high_t, speaker)];
            result.push_back(value);
        }
    };
    append_group(table.floor_values, entry.floor_count, table.floor_stride);
    if (height && entry.height_count > 0) {
        append_group(table.height_values, entry.height_count, table.height_stride);
    }
    return result;
}

float size_curve(const double size_value) {
    static constexpr std::array<std::array<float, 2>, 5> kNodes{{
        {0.0f, 0.0f}, {0.2f, 0.075f}, {0.5f, 0.25f}, {0.75f, 0.45f}, {1.0f, 0.7f}}};
    const float size = static_cast<float>(size_value);
    if (size <= kNodes.front()[0]) {
        return 0.0f;
    }
    if (size >= kNodes.back()[0]) {
        return 1.0f;
    }
    for (std::size_t index = 0; index + 1 < kNodes.size(); ++index) {
        const float size_low = kNodes[index][0];
        const float size_high = kNodes[index + 1][0];
        if (size <= size_high) {
            const float span = size_high - size_low;
            const float fraction = span != 0.0f ? (size - size_low) / span : 0.0f;
            const float t_low = kNodes[index][1];
            const float t_high = kNodes[index + 1][1];
            return t_low + fraction * (t_high - t_low);
        }
    }
    return 1.0f;
}

}  // namespace

float size_curve_value(const double size) {
    return size_curve(size);
}

std::vector<float> size_kernel(const ::ejoc::speaker_tables::RegionGeometry& region,
                               const float position[3], const float t, const bool height) {
    const RegionSizeTables& entry = region_size_tables(region);
    const int floor_count = entry.floor_count;
    const int height_count = height ? entry.height_count : 0;
    const int total = floor_count + height_count;
    if (total == 0) {
        return std::vector<float>();
    }
    const int dim = height ? entry.mode
                           : (entry.floor_axis_group_count == 1 ? 1 : 2);
    const float m = size_m(t);
    const float inverse_m = 1.0f / m;
    std::array<float, 3> position32{position[0], position[1], position[2]};
    std::vector<float> axes[3];
    for (int axis = 0; axis < dim; ++axis) {
        axes[axis] = axis_weights(entry, axis, position32[static_cast<std::size_t>(axis)], t,
                                  height);
    }

    const BoundaryRows& source = dim == 3 ? entry.boundary_3d : entry.boundary_2d;
    // The reference reorders the five boundary rows to z1, x0, x1, y0, y1 and
    // keeps the first ``total`` columns.
    std::array<std::vector<float>, 5> boundary_table;
    constexpr std::array<int, 5> kOrder{4, 0, 1, 2, 3};
    for (std::size_t row = 0; row < 5; ++row) {
        const std::vector<float>& line = source[static_cast<std::size_t>(kOrder[row])];
        boundary_table[row].assign(line.begin(),
                                   line.begin() + std::min<std::size_t>(
                                                      line.size(),
                                                      static_cast<std::size_t>(total)));
        boundary_table[row].resize(static_cast<std::size_t>(total), 0.0f);
    }
    constexpr std::array<std::array<int, 2>, 5> kRowAxes{{
        {2, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}}};
    for (std::size_t row = 0; row < 5; ++row) {
        const int axis = kRowAxes[row][0];
        const int edge = kRowAxes[row][1];
        const float weight = boundary(axis, edge, position32[static_cast<std::size_t>(axis)], t);
        for (float& value : boundary_table[row]) {
            value = value * weight;
        }
    }

    std::vector<float> inside = axes[0];
    if (dim > 1) {
        for (std::size_t index = 0; index < inside.size(); ++index) {
            inside[index] = inside[index] * axes[1][index];
        }
    }
    if (dim > 2) {
        for (std::size_t index = 0; index < inside.size(); ++index) {
            inside[index] = inside[index] * axes[2][index];
        }
    }
    inside = norm(inside);
    if (dim == 1) {
        std::vector<float> result(inside.size());
        for (std::size_t index = 0; index < inside.size(); ++index) {
            result[index] = pow_approx(inside[index], inverse_m);
        }
        return result;
    }

    std::array<std::vector<float>, 5> powered;
    for (std::size_t row = 0; row < 5; ++row) {
        powered[row].resize(boundary_table[row].size());
        for (std::size_t index = 0; index < boundary_table[row].size(); ++index) {
            powered[row][index] = pow_approx(boundary_table[row][index], m);
        }
    }
    std::vector<float> edge(static_cast<std::size_t>(total), 0.0f);
    if (dim == 2) {
        const std::vector<float>& x = axes[0];
        const std::vector<float>& y = axes[1];
        for (std::size_t index = 0; index < edge.size(); ++index) {
            edge[index] = ((powered[1][index] * y[index] + powered[2][index] * y[index]) +
                           powered[3][index] * x[index]) +
                          powered[4][index] * x[index];
        }
    } else {
        const std::vector<float>& x = axes[0];
        const std::vector<float>& y = axes[1];
        const std::vector<float>& z = axes[2];
        for (std::size_t index = 0; index < edge.size(); ++index) {
            const float xy = x[index] * y[index];
            const float yz = y[index] * z[index];
            const float xz = x[index] * z[index];
            edge[index] = ((((xy * powered[0][index] + yz * powered[1][index]) +
                             yz * powered[2][index]) +
                            xz * powered[3][index]) +
                           xz * powered[4][index]);
        }
    }
    edge = norm(edge);

    float distance = std::min(position32[0], 1.0f - position32[0]);
    distance = std::min(distance, position32[1]);
    distance = std::min(distance, 1.0f - position32[1]);
    if (dim == 3) {
        distance = std::min(distance, 1.0f - position32[2]);
    }
    const float quarter_distance = kQuarter * distance;
    float mix = 0.0f;
    if (t < 0.000025f || (quarter_distance > t && quarter_distance > 0.05f)) {
        mix = 1.0f;
    } else {
        const float four_t = t * 4.0f;
        const float first = static_cast<float>(static_cast<double>(distance) /
                                               static_cast<double>(four_t));
        const float scaled = distance * 5.0f;
        mix = first * (scaled * scaled);
    }
    std::vector<float> result(static_cast<std::size_t>(total));
    for (std::size_t index = 0; index < result.size(); ++index) {
        const float joined = ((mix * inside[index] + edge[index]) * kHalf) * 0.0625f;
        result[index] = pow_approx(joined, inverse_m);
    }
    return result;
}

std::vector<float> size_targets(const MetadataParams& meta,
                                const ::ejoc::speaker_tables::LayoutGeometry& layout,
                                const int region_index, const double position[3],
                                const double size, const bool height) {
    std::vector<float> result(static_cast<std::size_t>(layout.channel_count), 0.0f);
    if (region_index < 0 || region_index >= static_cast<int>(layout.regions.size())) {
        return result;
    }
    const ::ejoc::speaker_tables::RegionGeometry& region =
        layout.regions[static_cast<std::size_t>(region_index)];
    if (region.point_count == 0) {
        return result;
    }
    (void)meta;
    const float t = size_curve_value(size);
    const float position32[3] = {static_cast<float>(position[0]),
                                 static_cast<float>(position[1]),
                                 static_cast<float>(position[2])};
    const std::vector<float> kernel = size_kernel(region, position32, t, height);
    const std::size_t count = std::min<std::size_t>(
        kernel.size(), static_cast<std::size_t>(region.point_count));
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t speaker = region.points[index].speaker_id;
        if (speaker < result.size()) {
            result[speaker] = kernel[index];
        }
    }
    return result;
}

IsfBasis assemble_isf(const MetadataParams& meta, const char* layout_key,
                      const int isf_idx) {
    IsfBasis basis;
    if (isf_idx < 0 || static_cast<std::size_t>(isf_idx) >= meta.configs.size()) {
        return basis;
    }
    const IsfConfig& config = meta.configs[static_cast<std::size_t>(isf_idx)];
    const IsfLayout* geometry = meta.layout(layout_key);
    if (geometry == nullptr || geometry->output_count <= 0) {
        return basis;
    }
    basis.rows = config.row_count();
    basis.columns = geometry->output_count;
    basis.values.assign(static_cast<std::size_t>(basis.rows) *
                            static_cast<std::size_t>(basis.columns),
                        0.0f);
    const std::size_t output_ids = geometry->output_ids.size();
    std::size_t cursor = 0;
    const auto add_group = [&](const char* name, const int selector) {
        const std::vector<IsfGroupRecord>* records = meta.layout_groups.find(name);
        if (records == nullptr) {
            return;
        }
        const IsfGroupRecord* record = nullptr;
        for (const IsfGroupRecord& candidate : *records) {
            if (candidate.selector == selector) {
                record = &candidate;
                break;
            }
        }
        if (record == nullptr) {
            return;
        }
        for (int column = record->begin; column < record->end; ++column) {
            const bool paired = column >= 0 && static_cast<std::size_t>(column) <
                                                  meta.paired_columns.size() &&
                                meta.paired_columns[static_cast<std::size_t>(column)];
            const std::vector<int>* mappings[2] = {&config.primary_row_map,
                                                   &config.secondary_row_map};
            const int mapping_count = paired ? 2 : 1;
            for (int which = 0; which < mapping_count; ++which) {
                std::size_t speaker = 0;
                if (cursor < output_ids) {
                    speaker = geometry->output_ids[cursor];
                } else if (cursor >= output_ids && output_ids > 0) {
                    // The reference indexes the padded 37-entry table; a cursor
                    // past the table is impossible for the shipped data.
                    speaker = 0;
                }
                const std::vector<int>& mapping = *mappings[which];
                for (int row = 0; row < basis.rows; ++row) {
                    const int source_row = mapping[static_cast<std::size_t>(row)];
                    const std::size_t matrix_index =
                        static_cast<std::size_t>(source_row) * 73u +
                        static_cast<std::size_t>(column);
                    if (matrix_index < config.matrix.size() &&
                        speaker < static_cast<std::size_t>(basis.columns)) {
                        float& target =
                            basis.values[static_cast<std::size_t>(row) *
                                             static_cast<std::size_t>(basis.columns) +
                                         speaker];
                        target = target + config.matrix[matrix_index];
                    }
                }
                ++cursor;
            }
        }
    };
    const auto& selectors = geometry->selectors;
    add_group("floor_a", selectors[0]);
    cursor = cursor >= 2 ? cursor - 2 : 0;
    add_group("floor_b", selectors[1]);
    cursor = cursor >= 2 ? cursor - 2 : 0;
    add_group("height", selectors[2]);
    const int counts[5] = {config.counts[0], config.counts[1], config.counts[2],
                           config.counts[3], config.counts[4]};
    if (selectors[3] != 0) {
        const int mid = counts[0];
        const int upper = counts[1];
        const int lower = counts[2];
        const int zenith = counts[3];
        const auto clear_rows = [&](int begin, int end) {
            begin = std::max(begin, 0);
            end = std::min(end, basis.rows);
            for (int row = begin; row < end; ++row) {
                for (int column = 0; column < basis.columns; ++column) {
                    basis.values[static_cast<std::size_t>(row) *
                                     static_cast<std::size_t>(basis.columns) +
                                 static_cast<std::size_t>(column)] = 0.0f;
                }
            }
        };
        clear_rows(mid, mid + upper);
        clear_rows(mid + upper + lower, mid + upper + lower + zenith);
        add_group("additional", selectors[3]);
    }
    return basis;
}

float isf_compensation(const MetadataParams& meta, const char* layout_key,
                       const int isf_idx, const int row, const bool disabled) {
    if (disabled) {
        return 1.0f;
    }
    if (isf_idx < 0 || static_cast<std::size_t>(isf_idx) >= meta.configs.size()) {
        return 1.0f;
    }
    const IsfConfig& config = meta.configs[static_cast<std::size_t>(isf_idx)];
    const IsfLayout* geometry = meta.layout(layout_key);
    if (geometry == nullptr) {
        return 1.0f;
    }
    const int first = config.counts[0];
    float value = static_cast<float>(geometry->compensation_value);
    if (row < first) {
        const float distance = static_cast<float>(std::min(row, first - row));
        const float lower = max_float(static_cast<float>(first) * 0.16666667f, 1.0f);
        const float width = static_cast<float>(first) * 0.11111111f;
        float fraction = 0.0f;
        if (distance <= lower) {
            fraction = 0.0f;
        } else if (distance > lower + width) {
            fraction = 1.0f;
        } else {
            fraction = (distance - lower) / width;
        }
        value = fraction * value;
    }
    return exp2_approx(value * 0.1660964f);
}

namespace {

constexpr int kDirectBedLabels[17] = {0, 1, 2, 3, 4, 5, 6, 7, 13, 14,
                                      17, 18, 21, 22, 11, 12, 3};
constexpr float kAWeight = 0.5011872053146362f;
constexpr float kBWeight = 0.7079457640647888f;
constexpr float kCWeight = 0.3548133969306946f;
constexpr float kRow537 = 0.5370317697525024f;
constexpr float kRow288 = 0.2884031534194946f;
constexpr float kRow813 = 0.8128305077552795f;
constexpr float kRow661 = 0.6606934666633606f;
constexpr float kRow437 = 0.43651583790779114f;

int direct_bed_label(const int label) {
    if (label >= 1 && label <= 17) {
        return kDirectBedLabels[label - 1];
    }
    return 3;
}

struct RowEntry {
    int target = 0;
    float weight = 0.0f;
};

const std::vector<RowEntry>* bed_rows(const int label, const std::uint32_t mask) {
    static const std::vector<RowEntry> row9{{13, 1.0f}, {15, kBWeight}, {17, kAWeight}};
    static const std::vector<RowEntry> row10{{14, 1.0f}, {16, kBWeight}, {18, kAWeight}};
    static const std::vector<RowEntry> row11{{13, kAWeight}, {15, kBWeight}, {17, 1.0f},
                                             {19, kBWeight}, {21, kAWeight}};
    static const std::vector<RowEntry> row12{{14, kAWeight}, {16, kBWeight}, {18, 1.0f},
                                             {20, kBWeight}, {22, kAWeight}};
    static const std::vector<RowEntry> row13{{17, kAWeight}, {19, kBWeight}, {21, 1.0f}};
    static const std::vector<RowEntry> row14{{18, kAWeight}, {20, kBWeight}, {22, 1.0f}};
    static const std::vector<RowEntry> maskC05{{4, 1.0f}, {25, kBWeight}, {27, kBWeight}};
    static const std::vector<RowEntry> maskC06{{5, 1.0f}, {26, kBWeight}, {28, kBWeight}};
    static const std::vector<RowEntry> maskC07{{6, 1.0f}, {8, kCWeight}, {29, kBWeight},
                                              {31, kBWeight}, {33, kAWeight}};
    static const std::vector<RowEntry> maskC08{{7, 1.0f}, {8, kCWeight}, {30, kBWeight},
                                              {32, kBWeight}, {34, kAWeight}};
    static const std::vector<RowEntry> mask305{{4, 1.0f}, {6, kRow537}, {8, kRow288},
                                              {25, kBWeight}, {27, kRow813}, {29, kRow661},
                                              {31, kRow437}, {33, kCWeight}};
    static const std::vector<RowEntry> mask306{{5, 1.0f}, {7, kRow537}, {8, kRow288},
                                              {26, kBWeight}, {28, kRow813}, {30, kRow661},
                                              {32, kRow437}, {34, kCWeight}};
    if (label >= 9 && label <= 14) {
        switch (label) {
            case 9: return &row9;
            case 10: return &row10;
            case 11: return &row11;
            case 12: return &row12;
            case 13: return &row13;
            default: return &row14;
        }
    }
    if ((mask & 0xC0u) != 0u) {
        switch (label) {
            case 5: return &maskC05;
            case 6: return &maskC06;
            case 7: return &maskC07;
            case 8: return &maskC08;
            default: return nullptr;
        }
    }
    if ((mask & 0x30u) != 0u) {
        switch (label) {
            case 5: return &mask305;
            case 6: return &mask306;
            default: return nullptr;
        }
    }
    return nullptr;
}

}  // namespace

float bed_weight(const int label, const std::uint32_t mask, const int target_label) {
    const float value =
        target_label == direct_bed_label(label) ? 1.0f : 0.0f;
    if (label >= 9 && label <= 14) {
        const std::vector<RowEntry>* rows = bed_rows(label, mask);
        for (const RowEntry& entry : *rows) {
            if (entry.target == target_label) {
                return entry.weight;
            }
        }
        return value;
    }
    if ((mask & 0xC0u) == 0u && (mask & 0x30u) == 0u) {
        return value;
    }
    if (label == 5 && (mask & 0x4000u) == 0u && target_label == 11) {
        return kAWeight;
    }
    if (label == 6 && (mask & 0x8000u) == 0u && target_label == 12) {
        return kAWeight;
    }
    const std::vector<RowEntry>* rows = bed_rows(label, mask);
    if (rows == nullptr) {
        return value;
    }
    for (const RowEntry& entry : *rows) {
        if (entry.target == target_label) {
            return entry.weight;
        }
    }
    return value;
}

std::array<int, 5> bed_candidates(const int label, const std::uint32_t mask,
                                  const std::vector<int>& output_labels) {
    const auto present = [&](const int value) {
        return std::find(output_labels.begin(), output_labels.end(), value) !=
                       output_labels.end()
                   ? 1
                   : 0;
    };
    std::array<int, 5> available{present(13), present(15), present(17), present(19),
                                 present(21)};
    std::array<int, 5> remaining = available;
    const bool front_bits = label == 9 || label == 11 || label == 13;
    const int front_bit = front_bits ? 8 : 9;
    const int middle_bit = front_bits ? 10 : 11;
    const int rear_bit = front_bits ? 12 : 13;
    const int front = static_cast<int>((mask >> static_cast<unsigned>(front_bit)) & 1u);
    const int middle = static_cast<int>((mask >> static_cast<unsigned>(middle_bit)) & 1u);
    const int rear = static_cast<int>((mask >> static_cast<unsigned>(rear_bit)) & 1u);
    if (front != 0) {
        if (available[0] != 0) {
            remaining[0] = 0;
        } else if (available[1] != 0) {
            remaining[1] = 0;
        }
    }
    if (middle != 0 && available[2] != 0) {
        remaining[2] = 0;
    }
    if (rear != 0) {
        if (available[4] != 0) {
            remaining[4] = 0;
        } else if (available[3] != 0) {
            remaining[3] = 0;
        }
    }
    const int available_sum =
        available[0] + available[1] + available[2] + available[3] + available[4];
    const bool special = available[2] == 0 && available_sum == 2 && front == 0 &&
                         rear == 0 && middle != 0 &&
                         (available[0] != 0 || available[1] != 0) &&
                         (available[3] != 0 || available[4] != 0);
    if (label == 9 || label == 10) {
        if (available[0] != 0 || available[1] != 0 || remaining[2] != 0) {
            return {available[0], available[1], remaining[2], 0, 0};
        }
        return {0, 0, 0, 0, 0};
    }
    if (label == 11 || label == 12) {
        if (special) {
            return {0, 0, 0, 0, 0};
        }
        return {remaining[0], remaining[1], available[2], remaining[3], remaining[4]};
    }
    if (label == 13 || label == 14) {
        if (available[3] != 0 || available[4] != 0 || remaining[2] != 0) {
            return {0, 0, remaining[2], available[3], available[4]};
        }
        return {0, 0, 0, 0, 0};
    }
    return {0, 0, 0, 0, 0};
}

BedDistribution bed_distribution(const ::joc::speaker::LayoutInfo& layout, const int label,
                                 const std::uint32_t mask, const bool distribute) {
    const ::joc::joc::BedOutputDescription description =
        ::joc::joc::bed_output_description(layout);
    const std::vector<int>& labels = description.output_labels;
    BedDistribution result;
    result.weights.assign(labels.size(), 0.0f);
    const int bed_index = label - 1;
    if (bed_index < 0 || bed_index >= kBedLabelCount) {
        return result;
    }
    const int slot = description.label_to_slot[bed_index];
    const int fallback = description.direct_output_labels[bed_index];
    const bool height = label >= 9 && label <= 14;
    if (slot == -2 && (!height || !distribute)) {
        result.route = fallback;
        return result;
    }
    if (slot == -1) {
        result.route = -1;
        return result;
    }
    if (slot >= 0 && static_cast<std::size_t>(slot) < result.weights.size()) {
        result.weights[static_cast<std::size_t>(slot)] = 1.0f;
    }
    if (!distribute) {
        result.route = -1;
        return result;
    }
    std::vector<float> weights;
    if (height) {
        const std::array<int, 5> candidates = bed_candidates(label, mask, labels);
        if (std::none_of(candidates.begin(), candidates.end(),
                         [](const int value) { return value != 0; })) {
            result.route = fallback;
            return result;
        }
        static constexpr std::array<int, 5> kSideOdd{13, 15, 17, 19, 21};
        static constexpr std::array<int, 5> kSideEven{14, 16, 18, 20, 22};
        const std::array<int, 5>& side_labels = (label % 2) != 0 ? kSideOdd : kSideEven;
        weights.assign(labels.size(), 0.0f);
        for (std::size_t index = 0; index < labels.size(); ++index) {
            bool allowed = false;
            for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
                if (candidates[candidate] != 0 && side_labels[candidate] == labels[index]) {
                    allowed = true;
                    break;
                }
            }
            weights[index] = allowed ? bed_weight(label, mask, labels[index]) : 0.0f;
        }
    } else {
        weights.resize(labels.size());
        bool binary = true;
        for (std::size_t index = 0; index < labels.size(); ++index) {
            weights[index] = bed_weight(label, mask, labels[index]);
            if (weights[index] != 0.0f && weights[index] != 1.0f) {
                binary = false;
            }
        }
        if (binary) {
            result.route = -1;
            return result;
        }
    }
    std::size_t count = 0;
    for (const float value : weights) {
        if (value != 0.0f) {
            ++count;
        }
    }
    if (count == 1) {
        for (float& value : weights) {
            if (value != 0.0f) {
                value = 1.0f;
            }
        }
    } else if (count > 1) {
        weights = norm(weights);
    } else {
        result.route = fallback;
        return result;
    }
    result.route = -1;
    result.weights = std::move(weights);
    return result;
}

void bed_reference_position(const int index, int q15[3]) {
    q15[0] = 0;
    q15[1] = 0;
    q15[2] = 0;
    if (index < 0 || index >= ::joc::joc::kBedReferencePositionCount) {
        return;
    }
    const std::int32_t(*positions)[3] = ::joc::joc::bed_reference_positions_q15();
    if (positions == nullptr) {
        return;
    }
    q15[0] = static_cast<int>(positions[index][0]);
    q15[1] = static_cast<int>(positions[index][1]);
    q15[2] = static_cast<int>(positions[index][2]);
}

float bed_attenuation_db(const ::joc::speaker::LayoutInfo& layout) {
    return static_cast<float>(::joc::joc::bed_output_description(layout).automatic_attenuation_db);
}

float bed_compensation(const ::joc::speaker::LayoutInfo& layout, const int label,
                       const int mode, const bool bypass, const double levels[3]) {
    if (label == 4 || label == 17 || bypass || mode == 1) {
        return 1.0f;
    }
    if (label < 1 || label > kBedLabelCount) {
        return 1.0f;
    }
    const ::joc::joc::BedOutputDescription description =
        ::joc::joc::bed_output_description(layout);
    const int source = description.direct_output_labels[label - 1];
    int position[3] = {0, 0, 0};
    bed_reference_position(source, position);
    const float x = static_cast<float>(static_cast<double>(position[0]) / 32768.0);
    const float y = static_cast<float>(static_cast<double>(position[1]) / 32768.0);
    const float z = static_cast<float>(static_cast<double>(position[2]) / 32768.0);
    float value_db = 0.0f;
    if (mode == 0) {
        const float y_term = y >= 0.6f ? 1.0f
                                       : (y <= 0.0f
                                              ? 0.0f
                                              : static_cast<float>(static_cast<double>(y) / 0.6));
        const float z_term =
            z >= 1.0f ? 1.0f
                      : (z <= 0.2f
                             ? 0.0f
                             : static_cast<float>(static_cast<double>(z - 0.2f) / 0.8));
        const float amount = max_float(0.0f, min_float(y_term + z_term, 1.0f));
        value_db = amount * static_cast<float>(description.automatic_attenuation_db);
    } else {
        const float surround = static_cast<float>(levels[0]);
        const float height = static_cast<float>(levels[1]);
        const float centre = static_cast<float>(levels[2]);
        const float surround_factor =
            y >= 0.5f ? surround
                      : static_cast<float>(static_cast<double>(y * surround) * 2.0);
        float centre_position = max_float(std::fabs(z) - 0.05f, y - 0.05f);
        centre_position =
            max_float(centre_position, std::fabs(x - 0.5f) - 0.05f);
        const float centre_factor =
            1.0f - min_float(max_float(10.0f * centre_position, 0.0f), 1.0f);
        value_db = ((surround_factor * (1.0f - z)) + (z * height)) +
                   (centre_factor * centre);
    }
    return exp2_approx(value_db * 0.1660964f);
}

}  // namespace joc::oamd
