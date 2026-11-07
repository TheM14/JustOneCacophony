"""Float32 equations for the metadata application path.

The routines implement the fixed float32 equations for the size tables, ISF
basis assembly, bed channel routing and the position-dependent compensation.
The size tables are generated from the region geometry with grouped equal-power
panning, an S448 weight and a power sum; every operation is evaluated in
float32 in a fixed order so the coefficients are reproducible.

The caller passes the parameter mappings from :mod:`oamd_params`; this module
does not read files itself.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from bed_layout import BED_REFERENCE_POSITIONS_Q15, bed_output_description
from speaker_layouts import get_speaker_layout

F = np.float32
TINY = F(2 ** -126)


def exp2_approx(value):
    """Specified ``2**x`` approximation used by the compensation kernels."""
    x = F(value)
    n = math.floor(float(x))
    f = F(x - F(n))
    poly = F(F(F(F(f * F(0.6862793)) + F(1)) + F(F(f * f) * F(0.25482178)))
             + F(F(f * F(f * f)) * F(0.058898926)))
    return F(math.ldexp(float(poly), n))


def pow_approx(x, power):
    """Specified ``x**power`` approximation; ``power`` is a float32."""
    x = F(x)
    if abs(x) < TINY:
        return F(0)
    fraction, exponent = math.frexp(float(x))
    fraction = F(fraction)
    logarithm = F(F(F(F(fraction * F(4.0793457)) - F(2.6931152))
                   - F(F(fraction * fraction) * F(1.3862305))) + F(exponent))
    return exp2_approx(F(logarithm * F(power)))


def norm(vector):
    """Stable float32 normalization with the tiny-norm floor."""
    vector = np.asarray(vector, dtype=np.float32)
    total = F(0)
    for value in vector:
        total = F(total + F(value * value))
    if abs(total) < TINY:
        total = F(0)
    denominator = max(F(math.sqrt(float(total))), F(1e-5))
    result = (vector / denominator).astype(np.float32)
    result[np.abs(result) < TINY] = 0
    return result


def boundary(axis, edge, position, t):
    """One boundary weight of the size kernel."""
    edge, position, t = F(edge), F(position), F(t)
    delta = F(F(edge - position) * F(0.25))
    if t < F(0.000025) or abs(delta) > t:
        return F(0)
    ratio = F(float(delta) / float(t))
    ratio2 = F(ratio * ratio)
    weight = exp2_approx(F(F(-F(ratio2 * ratio2) * F(0.6328125))
                           * F(26.575424)))
    _, exponent = math.frexp(float(weight))
    if exponent < -31 or exponent > 1:
        weight = F(0)
    if axis == 2:
        weight = F(F(math.cos(float(F(F(edge * F(0.34147748)) * F(4)))))
                   * weight)
    return weight


def _region_coordinates(region):
    """Normalize the region's Q15 coordinates to the 0..1 room grid."""
    return (np.asarray(region.coordinates_q15, dtype=np.float32)
            / F(32768))


def _group_pan_x(coordinates, groups, position, output):
    """Region X panning: equal-power between x-adjacent speakers."""
    position = F(position)
    for group in groups:
        first = group[0]
        last = group[-1]
        x_first = coordinates[first][0]
        x_last = coordinates[last][0]
        if position < x_first:
            output[first] = F(1)
            continue
        if position > x_last:
            output[last] = F(1)
            continue
        for left, right in zip(group, group[1:]):
            x_left = coordinates[left][0]
            x_right = coordinates[right][0]
            if not (x_left < x_right and x_left <= position <= x_right):
                continue
            fraction = F(F(position - x_left) / F(x_right - x_left))
            if fraction <= F(0.5):
                angle = F(F(F(fraction * F(0.78539819)) * F(0.5)) * F(4))
                output[left] = F(np.cos(np.float32(angle)))
                output[right] = F(np.sin(np.float32(angle)))
            else:
                angle = F(F(F(F(1) - fraction) * F(0.78539819))
                          * F(0.5) * F(4))
                output[left] = F(np.sin(np.float32(angle)))
                output[right] = F(np.cos(np.float32(angle)))
            break


def _group_pan_y(coordinates, groups, position, output):
    """Region Y panning: equal-power between y-adjacent speaker rows."""
    if not groups:
        return
    position = F(position)
    rows = [coordinates[group[0]][1] for group in groups]
    if position <= rows[0]:
        for index in groups[0]:
            output[index] = F(1)
        return
    if position > rows[-1]:
        for index in groups[-1]:
            output[index] = F(1)
        return
    for row in range(len(groups) - 1):
        y_low = rows[row]
        y_high = rows[row + 1]
        if not (y_low <= position <= y_high and y_low < y_high):
            continue
        fraction = F(F(position - y_low) / F(y_high - y_low))
        if fraction <= F(0.5):
            angle = F(F(F(fraction * F(0.78539819)) * F(0.5)) * F(4))
            low_gain = F(np.cos(np.float32(angle)))
            high_gain = F(np.sin(np.float32(angle)))
        else:
            angle = F(F(F(F(1) - fraction) * F(0.78539819))
                      * F(0.5) * F(4))
            low_gain = F(np.sin(np.float32(angle)))
            high_gain = F(np.cos(np.float32(angle)))
        for index in groups[row]:
            output[index] = low_gain
        for index in groups[row + 1]:
            output[index] = high_gain
        return


def _layer_pan_z(coordinates, position):
    """Region Z panning: floor/height equal-power split."""
    position = F(position)
    if position <= F(0):
        floor_gain = F(1)
        height_gain = F(0)
    elif position < F(0.99996948):
        angle = F(F(position * F(0.39269909)) * F(4))
        floor_gain = F(np.cos(np.float32(angle)))
        height_gain = F(np.sin(np.float32(angle)))
    else:
        floor_gain = F(0)
        height_gain = F(1)
    output = np.zeros(len(coordinates), dtype=np.float32)
    for index, coordinate in enumerate(coordinates):
        output[index] = (height_gain if coordinate[2] >= F(0.5)
                         else floor_gain)
    return output


def _pan_x_all(coordinates, floor_groups, height_groups, position, mode):
    output = np.zeros(len(coordinates), dtype=np.float32)
    _group_pan_x(coordinates, floor_groups, position, output)
    if mode == 3 and height_groups:
        extra = np.zeros(len(coordinates), dtype=np.float32)
        _group_pan_x(coordinates, height_groups, position, extra)
        output = F(output + extra)
    return output


def _pan_y_all(coordinates, floor_groups, height_groups, position, mode):
    output = np.zeros(len(coordinates), dtype=np.float32)
    _group_pan_y(coordinates, floor_groups, position, output)
    if mode == 3 and height_groups:
        extra = np.zeros(len(coordinates), dtype=np.float32)
        _group_pan_y(coordinates, height_groups, position, extra)
        output = F(output + extra)
    return output


def _boundary_rows(coordinates, floor_groups, height_groups, mode):
    x0 = _pan_x_all(coordinates, floor_groups, height_groups, F(0), mode)
    x1 = _pan_x_all(coordinates, floor_groups, height_groups, F(1), mode)
    y0 = _pan_y_all(coordinates, floor_groups, height_groups, F(0), mode)
    y1 = _pan_y_all(coordinates, floor_groups, height_groups, F(1), mode)
    if mode == 3:
        z1 = _layer_pan_z(coordinates, F(1))
    else:
        z1 = np.zeros(len(coordinates), dtype=np.float32)
    return [x0, x1, y0, y1, z1]


def _size_weight(position, u, t, axis):
    """Per-virtual-point S448 weight used by the generated size tables."""
    t = F(t)
    offset = F(F(F(u - position)) * F(0.25))
    if t < F(0.000025) or abs(float(offset)) > float(t):
        return F(0)
    ratio = F(offset / t)
    squared = F(ratio * ratio)
    power = F(F(-F(squared * squared)) * F(0.6328125))
    weight = exp2_approx(F(power * F(26.575424)))
    if axis == 2:
        angle = F(F(F(u * F(0.34147748))) * F(4))
        weight = F(F(np.cos(np.float32(angle))) * weight)
    return weight


def _axis_table(pan_vectors, position_nodes, speaker_count, offset_start,
                axis):
    table = np.zeros((position_nodes, 20, speaker_count), dtype=np.float32)
    for position_index in range(position_nodes):
        position = F(F(position_index) / F(position_nodes - 1))
        for size_index in range(20):
            t = F(F(size_index) * F(0.036842104))
            m = (F(6) if t <= F(0.125)
                 else F(F(6.869565) - F(t * F(6.956522))))
            for u, pan in enumerate(pan_vectors):
                weight = _size_weight(position, F(F(u) / F(len(pan_vectors) - 1)),
                                      t, axis)
                if weight == F(0):
                    continue
                for speaker in range(speaker_count):
                    value = F(pan[offset_start + speaker] * weight)
                    table[position_index, size_index, speaker] = F(
                        table[position_index, size_index, speaker]
                        + pow_approx(value, m))
    return table


def _axis2_table(pan_vectors, speaker_index):
    table = np.zeros((4, 20), dtype=np.float32)
    if speaker_index >= len(pan_vectors[0]):
        return table
    for position_index in range(4):
        position = F(F(position_index) / F(3))
        for size_index in range(20):
            t = F(F(size_index) * F(0.036842104))
            m = (F(6) if t <= F(0.125)
                 else F(F(6.869565) - F(t * F(6.956522))))
            total = F(0)
            for u, pan in enumerate(pan_vectors):
                weight = _size_weight(position, F(F(u) / F(7)), t, 2)
                if weight == F(0):
                    continue
                total = F(total + pow_approx(
                    F(pan[speaker_index] * weight), m))
            table[position_index, size_index] = total
    return table


@lru_cache(maxsize=256)
def _region_size_tables(region):
    """Generate the region size tables from the region geometry."""
    coordinates = _region_coordinates(region)
    floor_groups = tuple(region.axis0_groups)
    height_groups = tuple(region.axis1_groups) if region.mode == 3 else ()
    floor_count = sum(len(group) for group in floor_groups)
    height_count = sum(len(group) for group in height_groups)
    mode = region.mode

    x_vectors = [_pan_x_all(coordinates, floor_groups, height_groups,
                            F(F(index) / F(19)), mode) for index in range(20)]
    y_vectors = [_pan_y_all(coordinates, floor_groups, height_groups,
                            F(F(index) / F(19)), mode) for index in range(20)]
    z_vectors = ([_layer_pan_z(coordinates, F(F(index) / F(7)))
                  for index in range(8)] if mode == 3
                 else [np.zeros(len(coordinates), dtype=np.float32)
                       for _ in range(8)])

    x_floor = _axis_table(x_vectors, 35, floor_count, 0, 0)
    x_height = _axis_table(x_vectors, 35, height_count, floor_count, 0)
    y_floor = _axis_table(y_vectors, 35, floor_count, 0, 1)
    y_height = _axis_table(y_vectors, 35, height_count, floor_count, 1)
    if mode == 3:
        z_floor = _axis2_table(z_vectors, 0)
        z_height = _axis2_table(z_vectors, floor_count)
    else:
        z_floor = np.zeros(0, dtype=np.float32)
        z_height = np.zeros(0, dtype=np.float32)

    entry = {
        "floor_count": floor_count,
        "height_count": height_count,
        "mode": mode,
        "floor_axis_group_count": len(floor_groups),
        "height_axis_group_count": len(height_groups),
        "boundary_3d": _boundary_rows(coordinates, floor_groups,
                                      height_groups, mode),
        "boundary_2d": _boundary_rows(coordinates, floor_groups, (), 2),
        "axis_tables": {
            "0": {"position_nodes": 35, "size_nodes": 20,
                  "floor_stride": floor_count,
                  "height_stride": height_count,
                  "floor": x_floor.reshape(-1).copy(),
                  "height": x_height.reshape(-1).copy()},
            "1": {"position_nodes": 35, "size_nodes": 20,
                  "floor_stride": floor_count,
                  "height_stride": height_count,
                  "floor": y_floor.reshape(-1).copy(),
                  "height": y_height.reshape(-1).copy()},
            "2": {"position_nodes": 4, "size_nodes": 20,
                  "floor_stride": 1, "height_stride": 1,
                  "floor": z_floor.reshape(-1).copy(),
                  "height": z_height.reshape(-1).copy()},
        },
    }
    return entry


def _bracket(position, axis):
    position = F(position)
    step = F(F(1) / F(3 if axis == 2 else 34))
    last = 3 if axis == 2 else 34
    low = math.floor(float(F(F(last) * position)))
    high = min(low + 1, last)
    if low > last:
        low = high = last
    alpha = (F(0) if low == high
             else max(F(0), F(F(position - F(F(low) * step)) / step)))
    return low, high, alpha


def _size_bracket(t):
    t = F(t)
    step = F(0.036842104)
    low = math.floor(float(F(F(F(t * F(0.71428573)) * F(2)) * F(19))))
    if low > 19:
        return 19, 19, F(0)
    high = min(low + 1, 19)
    beta = (F(0) if low == high
            else max(F(0), F(float(F(t - F(F(low) * step))) / float(step))))
    return low, high, beta


def _axis_weights(entry, axis, position, t, height):
    low, high, alpha = _bracket(position, axis)
    low_t, high_t, beta = _size_bracket(t)
    weights = np.asarray([
        F(1) - max(beta, alpha),
        F(1) - max(beta, F(1) - alpha),
        F(1) - max(F(1) - beta, alpha),
        F(1) - max(F(1) - beta, F(1) - alpha)], dtype=np.float32)
    total = F(F(F(weights[1] + weights[0]) + weights[2]) + weights[3])
    weights = np.asarray([F(float(value) / float(total)) for value in weights],
                         dtype=np.float32)
    table = entry["axis_tables"][str(axis)]
    result = []
    for group, count in (("floor", entry["floor_count"]),
                         ("height", entry["height_count"] if height else 0)):
        if not count:
            continue
        stride = table[group + "_stride"]
        values = np.asarray(table[group], dtype=np.float32).reshape(
            table["position_nodes"], 20, stride)
        value = F(F(F(weights[0] * values[low, low_t]
                      + weights[1] * values[high, low_t])
                    + weights[2] * values[low, high_t])
                  + weights[3] * values[high, high_t])
        result.extend([value[0]] * count if axis == 2 else value[:count])
    return np.asarray(result, dtype=np.float32)


def size_kernel(region, position, t, height=True):
    """Size kernel response in region-speaker order."""
    entry = _region_size_tables(region)
    position = np.asarray(position, dtype=np.float32)
    t = F(t)
    floor_count = entry["floor_count"]
    height_count = entry["height_count"] if height else 0
    total = floor_count + height_count
    if not total:
        return np.zeros(0, dtype=np.float32)
    dim = (entry["mode"] if height
           else (1 if entry["floor_axis_group_count"] == 1 else 2))
    m = (F(6) if t <= F(0.125)
         else F(F(6.869565) - F(t * F(6.956522))))
    inverse_m = F(F(1) / m)
    axes = [_axis_weights(entry, axis, position[axis], t, height)
            for axis in range(dim)]
    key = "boundary_3d" if dim == 3 else "boundary_2d"
    boundary_table = np.asarray(entry[key], dtype=np.float32)[
        [4, 0, 1, 2, 3], :total].copy()
    for row, (axis, edge) in enumerate([(2, 1), (0, 0), (0, 1),
                                        (1, 0), (1, 1)]):
        boundary_table[row] *= boundary(axis, edge, position[axis], t)
    inside = axes[0].copy()
    if dim > 1:
        inside *= axes[1]
    if dim > 2:
        inside *= axes[2]
    inside = norm(inside)
    if dim == 1:
        return np.asarray([pow_approx(value, inverse_m) for value in inside],
                          dtype=np.float32)
    powered = np.asarray([[pow_approx(value, m) for value in row]
                          for row in boundary_table], dtype=np.float32)
    x, y = axes[0], axes[1]
    if dim == 2:
        edge = F(F(F(powered[1] * y + powered[2] * y)
                   + powered[3] * x) + powered[4] * x)
    else:
        z = axes[2]
        edge = F(F(F(F(F(x * y) * powered[0]
                        + F(y * z) * powered[1])
                    + F(y * z) * powered[2])
                  + F(x * z) * powered[3])
              + F(x * z) * powered[4])
    edge = norm(edge)
    distances = [position[0], F(1) - position[0],
                 position[1], F(1) - position[1]]
    if dim == 3:
        distances.append(F(1) - position[2])
    distance = min(distances)
    if t < F(0.000025) or (F(F(0.25) * distance) > t
                           and F(F(0.25) * distance) > F(0.05)):
        mix = F(1)
    else:
        mix = F(F(float(distance) / float(F(t * F(4))))
                * F(F(distance * F(5)) * F(distance * F(5))))
    joined = F(F(F(mix * inside) + edge) * F(0.5)) * F(0.0625)
    return np.asarray([pow_approx(value, inverse_m) for value in joined],
                      dtype=np.float32)


_SIZE_CURVE_NODES = ((0.0, 0.0), (0.2, 0.075), (0.5, 0.25),
                     (0.75, 0.45), (1.0, 0.7))


def size_curve_value(size):
    """Map a scalar size in ``[0, 1]`` to ``t`` on the size curve."""
    points = [(F(point[0]), F(point[1])) for point in _SIZE_CURVE_NODES]
    size = F(size)
    if size <= points[0][0]:
        return F(0)
    if size >= points[-1][0]:
        return F(1)
    for (size_low, t_low), (size_high, t_high) in zip(points, points[1:]):
        if size <= size_high:
            span = F(size_high - size_low)
            fraction = F(F(size - size_low) / span) if span else F(0)
            return F(t_low + F(fraction * F(t_high - t_low)))
    return F(1)


def size_targets(meta, layout, region_index, position, size, height=True):
    """Return a size-kernel target vector in ``layout`` channel order."""
    region = layout.regions[int(region_index)]
    result = np.zeros(layout.channel_count, dtype=np.float32)
    if not len(region.speaker_ids):
        return result
    t = size_curve_value(size)
    kernel = size_kernel(region, position, t, height=height)
    for value, channel in zip(kernel, region.speaker_ids):
        result[channel] = value
    return result


def assemble_isf(meta, layout, isf_idx):
    """Assemble one ISF basis matrix from the compressed base tables."""
    config = meta["isf"]["configs"][isf_idx]
    geometry = meta["isf"]["layouts"][layout]
    row_count = config["counts"][4]
    matrix = np.asarray(config["matrix_values"], dtype=np.float32).reshape(
        -1, 73)
    result = np.zeros((row_count, geometry["output_count"]), dtype=np.float32)
    cursor = 0

    def add_group(name, selector):
        nonlocal cursor
        record = next((item for item in meta["isf"]["layout_groups"][name]
                       if item[0] == selector), None)
        if record is None:
            return
        for column in range(record[1], record[2]):
            mappings = [config["primary_row_map"]]
            if meta["isf"]["paired_columns"][column]:
                mappings.append(config["secondary_row_map"])
            for mapping in mappings:
                speaker = geometry["output_ids"][cursor]
                result[:, speaker] += matrix[np.asarray(mapping), column]
                cursor += 1

    floor_a, floor_b, height, additional = geometry["selectors"]
    add_group("floor_a", floor_a)
    cursor -= 2
    add_group("floor_b", floor_b)
    cursor -= 2
    add_group("height", height)
    if additional:
        mid, upper, lower, zenith, _ = config["counts"]
        result[mid:mid + upper] = 0
        result[mid + upper + lower:mid + upper + lower + zenith] = 0
        add_group("additional", additional)
    return result


def isf_matrix(meta, layout, isf_idx):
    """Return the assembled ISF basis matrix ``(Nisf, output_count)``."""
    return assemble_isf(meta, layout, isf_idx)


def isf_compensation(meta, layout, isf_idx, row, disabled=False):
    """Return the per-basis ISF compensation gain."""
    if disabled:
        return F(1)
    config = meta["isf"]["configs"][isf_idx]
    geom = meta["isf"]["layouts"][layout]
    first = config["counts"][0]
    value = F(geom["compensation_value"])
    if row < first:
        distance = F(min(row, first - row))
        lower = max(F(F(first) * F(0.16666667)), F(1))
        width = F(F(first) * F(0.11111111))
        if distance <= lower:
            fraction = F(0)
        elif distance > F(lower + width):
            fraction = F(1)
        else:
            fraction = F(F(distance - lower) / width)
        value = F(fraction * value)
    return exp2_approx(F(value * F(0.1660964)))


DIRECT_BED_LABEL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
                    9: 13, 10: 14, 11: 17, 12: 18, 13: 21, 14: 22,
                    15: 11, 16: 12, 17: 3}
A_WEIGHT = F(0.5011872053146362)
B_WEIGHT = F(0.7079457640647888)
C_WEIGHT = F(0.3548133969306946)


def bed_weight(label, mask, target_label):
    """Return the per-label bed redistribution weight."""
    value = F(1 if target_label == DIRECT_BED_LABEL.get(label, 3) else 0)
    if 9 <= label <= 14:
        rows = {9: {13: 1, 15: B_WEIGHT, 17: A_WEIGHT},
                10: {14: 1, 16: B_WEIGHT, 18: A_WEIGHT},
                11: {13: A_WEIGHT, 15: B_WEIGHT, 17: 1,
                     19: B_WEIGHT, 21: A_WEIGHT},
                12: {14: A_WEIGHT, 16: B_WEIGHT, 18: 1,
                     20: B_WEIGHT, 22: A_WEIGHT},
                13: {17: A_WEIGHT, 19: B_WEIGHT, 21: 1},
                14: {18: A_WEIGHT, 20: B_WEIGHT, 22: 1}}
        return F(rows[label].get(target_label, value))
    if mask & 0xC0:
        rows = {5: {4: 1, 25: B_WEIGHT, 27: B_WEIGHT},
                6: {5: 1, 26: B_WEIGHT, 28: B_WEIGHT},
                7: {6: 1, 8: C_WEIGHT, 29: B_WEIGHT, 31: B_WEIGHT,
                    33: A_WEIGHT},
                8: {7: 1, 8: C_WEIGHT, 30: B_WEIGHT, 32: B_WEIGHT,
                    34: A_WEIGHT}}
    elif mask & 0x30:
        rows = {5: {4: 1, 6: F(0.5370317697525024),
                    8: F(0.2884031534194946), 25: B_WEIGHT,
                    27: F(0.8128305077552795),
                    29: F(0.6606934666633606),
                    31: F(0.43651583790779114), 33: C_WEIGHT},
                6: {5: 1, 7: F(0.5370317697525024),
                    8: F(0.2884031534194946), 26: B_WEIGHT,
                    28: F(0.8128305077552795),
                    30: F(0.6606934666633606),
                    32: F(0.43651583790779114), 34: C_WEIGHT}}
    else:
        return value
    if label == 5 and not mask & 0x4000 and target_label == 11:
        return A_WEIGHT
    if label == 6 and not mask & 0x8000 and target_label == 12:
        return A_WEIGHT
    return F(rows.get(label, {}).get(target_label, value))


def bed_candidates(label, mask, output_labels):
    """Return the five height-candidate flags for one input bed label."""
    available = [int(value in output_labels)
                 for value in (13, 15, 17, 19, 21)]
    a, b, m, d, e = available
    remaining = available.copy()
    bits = (8, 10, 12) if label in (9, 11, 13) else (9, 11, 13)
    front, middle, rear = [(mask >> bit) & 1 for bit in bits]
    if front:
        if a:
            remaining[0] = 0
        elif b:
            remaining[1] = 0
    if middle and m:
        remaining[2] = 0
    if rear:
        if e:
            remaining[4] = 0
        elif d:
            remaining[3] = 0
    special = (not m and sum(available) == 2 and not front and not rear
               and middle and (a or b) and (d or e))
    if label in (9, 10):
        return [a, b, remaining[2], 0, 0] if (a or b or remaining[2]) \
            else [0] * 5
    if label in (11, 12):
        return [0] * 5 if special else [remaining[0], remaining[1], m,
                                        remaining[3], remaining[4]]
    if label in (13, 14):
        return [0, 0, remaining[2], d, e] if (d or e or remaining[2]) \
            else [0] * 5
    return [0] * 5


def bed_distribution(layout_key, label, mask, distribute):
    """Return ``(route, weights)`` for one input bed label.

    ``route == -1`` means the returned weights are the target; otherwise
    ``route`` is the reference position used to synthesize a point source.
    The slot map is derived from the speaker layout, not stored separately.
    """
    layout = bed_output_description(get_speaker_layout(layout_key))
    labels = layout["output_labels"]
    base = np.zeros(len(labels), dtype=np.float32)
    slot = layout["label_to_slot"][label - 1]
    fallback = layout["direct_output_labels"][label - 1]
    height = 9 <= label <= 14
    if slot == -2 and (not height or not distribute):
        return fallback, base
    if slot == -1:
        return -1, base
    if slot >= 0:
        base[slot] = F(1)
    if not distribute:
        return -1, base
    if height:
        candidate = bed_candidates(label, mask, labels)
        if not any(candidate):
            return fallback, base
        side_labels = (13, 15, 17, 19, 21) if label % 2 \
            else (14, 16, 18, 20, 22)
        allowed = {value for value, active in zip(side_labels, candidate)
                   if active}
        weights = np.array([bed_weight(label, mask, value)
                            if value in allowed else F(0)
                            for value in labels], dtype=np.float32)
    else:
        weights = np.array([bed_weight(label, mask, value)
                            for value in labels], dtype=np.float32)
        if all(value == F(0) or value == F(1) for value in weights):
            return -1, base
    count = np.count_nonzero(weights)
    if count == 1:
        weights[weights != 0] = F(1)
    elif count > 1:
        weights = norm(weights)
    else:
        return fallback, base
    return -1, weights


def bed_reference_position(layout_key, index):
    """Return the Q15 reference position used by the bed fallback path."""
    return BED_REFERENCE_POSITIONS_Q15[int(index)]


def bed_compensation(layout_key, label, mode=0, bypass=False,
                     levels=(0, 0, 0)):
    """Return the bed path compensation gain for one input bed label."""
    if label in (4, 17) or bypass or mode == 1:
        return F(1)
    layout = bed_output_description(get_speaker_layout(layout_key))
    source = layout["direct_output_labels"][label - 1]
    position = BED_REFERENCE_POSITIONS_Q15[source]
    x, y, z = [F(value / 32768) for value in position]
    if mode == 0:
        y_term = F(1) if y >= F(0.6) else (
            F(0) if y <= 0 else F(float(y) / 0.6))
        z_term = F(1) if z >= 1 else (
            F(0) if z <= F(0.2) else F(float(F(z - F(0.2))) / 0.8))
        amount = max(F(0), min(F(y_term + z_term), F(1)))
        value_db = F(amount * F(layout["automatic_attenuation_db"]))
    else:
        surround, height, centre = map(F, levels)
        surround_factor = F(surround if y >= F(0.5)
                            else F(F(y * surround) * F(2)))
        centre_position = max(F(abs(z) - F(0.05)), F(y - F(0.05)),
                              F(abs(F(x - F(0.5))) - F(0.05)))
        centre_factor = F(F(1) - min(max(F(10) * centre_position, F(0)),
                                     F(1)))
        value_db = F(F(F(surround_factor * F(F(1) - z))
                       + F(z * height)) + F(centre_factor * centre))
    return exp2_approx(F(value_db * F(0.1660964)))
