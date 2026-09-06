"""Float64 Rosella direction, distance, HRTF, and room-send calculations."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rosella_model import RosellaModel, direction_basis

PROFILE_NEAR = 1
PROFILE_FAR = 2
PROFILE_MID = 3
BINAURAL_PROFILE_NAMES = {
    "near": PROFILE_NEAR,
    "far": PROFILE_FAR,
    "mid": PROFILE_MID,
}

_SPECIAL_LFE_LOW_16 = np.asarray([
    0x402695EA, 0x3FE75979, 0x3F28CAAA, 0xBCE1FB2E,
    0xBDD8AF65, 0xBD8F426E, 0x3D996821, 0xBC16B3A0,
    0x3B64BAF1, 0xBC81ECFD, 0xBA3D892F, 0x3AF6A9F0,
    0xB9DD1C5F, 0x380A193F, 0x38052059, 0x351BCB34,
], dtype=np.uint32).view(np.float32).astype(np.float64)
_CENTRE_EQUAL = 0.9998489618301392
_CENTRE_ALTERNATE = 0.7070000171661377

_FIELD_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class DirectResult:
    gains: np.ndarray                 # complex128 [ear=2, hybrid_band=77]
    room_send: np.float64
    physical_radius_m: np.float64
    normalized_radius: np.float64
    clamped_radius: np.float64
    delay_samples: np.float64
    delayed_ear: int | None


def special_lfe_direct() -> DirectResult:
    """Return the fixed 16-band low-pass used by a special/LFE source."""
    mono = np.zeros(77, dtype=np.complex128)
    mono[:16] = _SPECIAL_LFE_LOW_16
    return DirectResult(
        gains=np.repeat(mono[None, :], 2, axis=0),
        room_send=np.float64(0.0),
        physical_radius_m=np.float64(0.0),
        normalized_radius=np.float64(0.0),
        clamped_radius=np.float64(0.0),
        delay_samples=np.float64(0.0),
        delayed_ear=None,
    )


def _round_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def _q15_position(position) -> np.ndarray:
    """Quantize ADM Cartesian coordinates to the Rosella metadata grid.

    Quantization is metadata decoding.  The returned integer lanes are promoted
    to float64 before any geometry is evaluated.
    """
    x, y, z = (float(value) for value in position)
    encoded = (
        min(max((x + 1.0) * 0.5, 0.0), 1.0),
        min(max((1.0 - y) * 0.5, 0.0), 1.0),
        min(max(z, -1.0), 1.0),
    )
    return np.asarray([
        min(_round_away_from_zero(value * 32768.0), 32767)
        for value in encoded
    ], dtype=np.int32)


def _profile_geometry(model: RosellaModel, position, profile_index: int):
    if profile_index not in (PROFILE_NEAR, PROFILE_FAR, PROFILE_MID):
        raise ValueError("binaural object profile must be near, mid, or far")
    profile = model.profiles[profile_index]
    encoded = _q15_position(position)
    q_front = 1.0 - 2.0 * float(encoded[1]) / 32768.0
    q_x = 2.0 * float(encoded[0]) / 32768.0 - 1.0
    q_vertical = float(encoded[2]) / 32768.0

    if int(model.header_integer_fields[0]) != 0:
        if q_x == 0.0 and q_front == 0.0:
            mapped_front = 0.0
            mapped_lateral = 0.0
            mapped_vertical = q_vertical
        else:
            horizontal_max = max(abs(q_x), abs(q_front))
            horizontal_norm = ((q_x / horizontal_max) ** 2
                               + (q_front / horizontal_max) ** 2)
            if q_vertical == 0.0:
                vertical_norm = 1.0
            else:
                smaller = min(abs(q_vertical), horizontal_max)
                larger = max(abs(q_vertical), horizontal_max)
                vertical_norm = 1.0 + (smaller / larger) ** 2
            horizontal_factor = 1.0 / math.sqrt(horizontal_norm * vertical_norm)
            vertical_factor = 1.0 / math.sqrt(vertical_norm)
            mapped_front = q_front * horizontal_factor
            mapped_lateral = -q_x * horizontal_factor
            mapped_vertical = q_vertical * vertical_factor
    else:
        mapped_front = q_front
        mapped_lateral = -q_x
        mapped_vertical = q_vertical

    scales = np.asarray(profile.axis_scales_internal, dtype=np.float64)
    scaled = np.asarray([
        mapped_front * scales[2],
        mapped_lateral * scales[0],
        mapped_vertical * scales[1],
    ], dtype=np.float64)
    bounds = np.asarray(profile.bounds, dtype=np.float64)
    ray = 1.0
    for axis in range(3):
        value = scaled[axis]
        lower, upper = bounds[axis * 2:axis * 2 + 2]
        if value < lower:
            ray = min(ray, lower / value)
        elif value > upper:
            ray = min(ray, upper / value)
    if ray < 1.0:
        scaled *= ray

    radius = float(np.linalg.norm(scaled))
    clamped = max(radius, float(profile.minimum_normalized_radius))
    alpha = radius / clamped
    direction = (scaled / radius if radius > 1.0e-30
                 else np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    return profile, direction, radius, clamped, alpha


def _logical_field(padded: np.ndarray) -> np.ndarray:
    result = np.empty((77, 36, 2), dtype=np.float64)
    source = np.asarray(padded, dtype=np.float64)
    for band in range(77):
        block, lane = divmod(band, 4)
        for term in range(36):
            for component in range(2):
                result[band, term, component] = source[
                    lane + 4 * (term * 2 + component + 72 * block)]
    return result


def _model_fields(model: RosellaModel) -> tuple[np.ndarray, np.ndarray]:
    key = id(model)
    fields = _FIELD_CACHE.get(key)
    if fields is None:
        fields = (_logical_field(model.field_left_padded),
                  _logical_field(model.field_right_padded))
        _FIELD_CACHE[key] = fields
    return fields


def _ear_geometry(model: RosellaModel, profile, direction, clamped: float,
                  offset: float, correction: float):
    x, y, z = (float(value) for value in direction)
    inverse_distance = float(profile.inverse_distance_per_m)
    ear = float(offset) * inverse_distance / clamped
    y_minus = y - ear
    y_plus = y + ear
    common = x * x + z * z
    length_minus = math.sqrt(y_minus * y_minus + common)
    length_plus = math.sqrt(y_plus * y_plus + common)
    basis_minus = direction_basis(
        x / length_minus, y_minus / length_minus, z / length_minus,
        dtype=np.float64)
    basis_plus = direction_basis(
        x / length_plus, y_plus / length_plus, z / length_plus,
        dtype=np.float64)
    path_minus = length_minus * clamped
    path_plus = length_plus * clamped
    if correction != 0.0:
        multiplier = 2.0 * float(correction) * inverse_distance
        path_minus += max(float(np.dot(
            np.asarray(model.vector_left, dtype=np.float64), basis_minus)), 0.0) * multiplier
        path_plus += max(float(np.dot(
            np.asarray(model.vector_right, dtype=np.float64), basis_plus)), 0.0) * multiplier
    return basis_minus, basis_plus, path_minus, path_plus


def _phase_groups(model: RosellaModel, delay_samples: float) -> np.ndarray:
    result = np.ones(77, dtype=np.complex128)
    current = 1.0 + 0.0j
    step = 1.0 + 0.0j
    value_index = 0
    for band, flag in enumerate(model.hybrid_flags):
        if flag != 2:
            if flag == 1:
                angle = float(model.hybrid_values[value_index]) * delay_samples
                value_index += 1
                step = complex(math.cos(angle), math.sin(angle))
            current *= step
        result[band] = current
    return result


def direct_and_room_send(model: RosellaModel, position,
                         profile_index: int) -> DirectResult:
    """Evaluate one ordinary source using float64/complex128 throughout."""
    profile, direction, radius, clamped, alpha = _profile_geometry(
        model, position, profile_index)

    _, _, path_minus, path_plus = _ear_geometry(
        model, profile, direction, clamped,
        float(model.model_scalars[1]), float(model.model_scalars[2]))
    delay = (abs(path_plus - path_minus)
             * float(profile.distance_scale_m)
             * (float(model.sample_rate) / 343.3) * alpha)
    delayed_ear = 0 if path_minus > path_plus else (
        1 if path_plus > path_minus else None)

    _, _, weight_minus_path, weight_plus_path = _ear_geometry(
        model, profile, direction, clamped,
        float(model.model_scalars[3]), float(model.model_scalars[4]))
    weight_norm = math.sqrt(
        weight_minus_path * weight_minus_path
        + weight_plus_path * weight_plus_path)
    weight_left = weight_plus_path / weight_norm
    weight_right = weight_minus_path / weight_norm

    final_offset = float(model.model_scalars[0])
    if final_offset == 0.0:
        basis_minus = direction_basis(*direction, dtype=np.float64)
        basis_plus = basis_minus.copy()
    else:
        x, y, z = (float(value) for value in direction)
        ear = final_offset * float(profile.inverse_distance_per_m) / clamped
        y_minus = y - ear
        y_plus = y + ear
        common_length = x * x + z * z
        length_minus = math.sqrt(y_minus * y_minus + common_length)
        length_plus = math.sqrt(y_plus * y_plus + common_length)
        basis_minus = direction_basis(
            x / length_minus, y_minus / length_minus, z / length_minus,
            dtype=np.float64)
        basis_plus = direction_basis(
            x / length_plus, y_plus / length_plus, z / length_plus,
            dtype=np.float64)

    field_left, field_right = _model_fields(model)
    left_components = np.einsum(
        "bjc,j->bc", field_left, basis_minus,
        dtype=np.float64, optimize=False)
    right_components = np.einsum(
        "bjc,j->bc", field_right, basis_plus,
        dtype=np.float64, optimize=False)
    left = left_components[:, 0] + 1j * left_components[:, 1]
    right = right_components[:, 0] + 1j * right_components[:, 1]
    if delayed_ear is not None:
        phase = _phase_groups(model, delay)
        if delayed_ear == 0:
            left *= phase
        else:
            right *= phase

    effective_radius = (radius * float(model.header_float_scalars[0])
                        * float(profile.distance_scale_m))
    if profile_index in (PROFILE_FAR, PROFILE_MID):
        common = 1.0 / math.sqrt(
            1.0 + float(model.header_float_scalars[1])
            * effective_radius * effective_radius)
        room_send = effective_radius * common
    else:
        common = 1.0
        room_send = 0.0

    left_term0 = field_left[:, 0, 0] + 1j * field_left[:, 0, 1]
    right_term0 = field_right[:, 0, 0] + 1j * field_right[:, 0, 1]
    weights_are_default_equal = (
        float(model.model_scalars[3]) == 0.0
        and float(model.model_scalars[4]) == 0.0)
    if weights_are_default_equal:
        centre_left = weight_left * (1.0 - alpha) * _CENTRE_EQUAL
        centre_right = centre_left
        right_direction_weight = weight_left
    else:
        centre_left = (1.0 - alpha) * _CENTRE_ALTERNATE
        centre_right = centre_left
        right_direction_weight = weight_right

    gains = np.empty((2, 77), dtype=np.complex128)
    gains[0] = common * (
        left * (weight_left * alpha) + left_term0 * centre_left)
    gains[1] = common * (
        right * (right_direction_weight * alpha) + right_term0 * centre_right)
    return DirectResult(
        gains=gains,
        room_send=np.float64(room_send),
        physical_radius_m=np.float64(float(profile.distance_scale_m) * radius),
        normalized_radius=np.float64(radius),
        clamped_radius=np.float64(clamped),
        delay_samples=np.float64(delay),
        delayed_ear=delayed_ear,
    )
