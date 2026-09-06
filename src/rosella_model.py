"""Parser for ``.personalized_headphone`` and raw ``rp`` models."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


Q15 = np.float32(1.0 / 32768.0)


def _f32(value) -> np.float32:
    return np.float32(value)


def _q15(value: int) -> np.float32:
    return _f32(_f32(value) * Q15)


def _q15_exp(value: int, exponent: int) -> np.float32:
    return _f32(_q15(value) * _f32(np.ldexp(1.0, exponent)))


@dataclass(frozen=True)
class DistanceProfile:
    bounds: np.ndarray
    distance_scale_m: np.float32
    inverse_distance_per_m: np.float32
    axis_scales_internal: np.ndarray
    minimum_normalized_radius: np.float32

    @property
    def floats(self) -> np.ndarray:
        return np.concatenate((
            self.bounds,
            np.asarray([self.distance_scale_m,
                        self.inverse_distance_per_m], dtype=np.float32),
            self.axis_scales_internal,
            np.asarray([self.minimum_normalized_radius], dtype=np.float32),
        ))


@dataclass(frozen=True)
class RosellaModel:
    source_path: str
    coefficients: np.ndarray
    coefficient_sha256: str
    coefficient_version: str | None
    room_model: str | None
    table_a_dimension: int
    table_a_option: int
    table_a_extra: int
    table_a_header_field: int
    table_a_header_25: int
    table_a_control: int
    table_a_option_ids: np.ndarray
    table_a_option_values: np.ndarray
    table_a_scalar: np.float32
    table_a_filter_16x64_padded: np.ndarray
    table_a_four_integers: np.ndarray
    table_a_integer: int
    table_a_filter_8x64_padded: np.ndarray
    table_a_vector16: np.ndarray
    table_a_filter_4x64_padded: np.ndarray
    table_a_extra_indices: np.ndarray
    table_a_extra_fields_padded: np.ndarray
    table_a_extra_vectors: np.ndarray
    sample_rate: int
    matrix_exponent: int
    field_exponent: int
    matrix_left: np.ndarray
    matrix_right: np.ndarray
    vector_left: np.ndarray
    vector_right: np.ndarray
    field_left_padded: np.ndarray
    field_right_padded: np.ndarray
    field_left_odd_serialized_zero: bool
    hybrid_flags: np.ndarray
    hybrid_values: np.ndarray
    model_scalars: np.ndarray
    header_float_scalars: np.ndarray
    header_integer_fields: np.ndarray
    profiles: tuple[DistanceProfile, ...]
    profile_tail: np.ndarray
    post_fields: np.ndarray
    table_a_main_serialized: np.ndarray


def _lane(data: bytes, index: int) -> int:
    if (index + 1) * 4 > len(data):
        raise ValueError(f"Rosella rp truncated before int32 lane {index}")
    return struct.unpack_from("<I", data, index * 4)[0]


def inspect_rp(data: bytes) -> dict:
    """Return the active-lane layout and checksum status for one raw rp image."""
    if len(data) < 20 or len(data) % 4:
        raise ValueError("Rosella rp must contain whole little-endian int32 lanes")
    if _lane(data, 0) != 0x7072:
        raise ValueError(f"bad Rosella rp magic: 0x{_lane(data, 0):08X}")

    low16 = lambda value: value & 0xFFFF
    checksum = low16(_lane(data, 1))
    table_a_present = low16(_lane(data, 2))
    table_b_present = low16(_lane(data, 3))
    table_c_present = low16(_lane(data, 4))
    index = 5
    if table_a_present:
        table_a_dimension = low16(_lane(data, index))
        table_a_option = low16(_lane(data, index + 1))
        table_a_extra = low16(_lane(data, index + 2))
        index += 5
    else:
        table_a_dimension, table_a_option, table_a_extra = 77, 0, 0
    if table_b_present:
        if not table_a_present:
            raise ValueError("Rosella rp table B cannot be present without table A")
        table_b_dimension = low16(_lane(data, index))
        table_b_extra = low16(_lane(data, index + 1))
        table_b_groups = low16(_lane(data, index + 2))
        index += 3
    else:
        table_b_dimension = table_b_extra = table_b_groups = 0
    if table_c_present:
        table_c_dimension = low16(_lane(data, index))
        index += 1
    else:
        table_c_dimension = 0

    payload_words = (
        (index - 2)
        + table_b_present * (
            table_b_dimension + 380 * table_b_groups + table_b_extra + 79)
        + table_a_present * (
            171 * table_a_extra + 79 + 2 * (table_a_option + 14 * table_a_dimension))
        + 11
        + table_c_present * (314 * table_c_dimension + 1)
    )
    total_lanes = 2 + payload_words
    if len(data) < total_lanes * 4:
        raise ValueError(
            f"Rosella rp truncated: need {total_lanes * 4} bytes, have {len(data)}")
    computed = 0xA569
    for lane_index in range(2, total_lanes):
        computed ^= low16(_lane(data, lane_index))
    computed &= 0xFFFF
    return {
        "stored_checksum": checksum,
        "computed_checksum": computed,
        "checksum_valid": computed == checksum,
        "table_a_present": table_a_present,
        "table_b_present": table_b_present,
        "table_c_present": table_c_present,
        "table_a_dimension": table_a_dimension,
        "table_a_option": table_a_option,
        "table_a_extra": table_a_extra,
        "table_b_dimension": table_b_dimension,
        "table_b_extra": table_b_extra,
        "table_b_groups": table_b_groups,
        "table_c_dimension": table_c_dimension,
        "active_int32_lanes": total_lanes,
    }


def _json_int32(values) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError("rosella_coefficients must be a JSON array")
    result = np.empty(len(values), dtype=np.int32)
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"rosella_coefficients[{index}] is not a number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != math.trunc(numeric):
            raise ValueError(
                f"rosella_coefficients[{index}] is not an exact integer: {value!r}")
        integer = int(value)
        if integer < -(1 << 31) or integer > (1 << 31) - 1:
            raise ValueError(
                f"rosella_coefficients[{index}] is outside signed int32: {integer}")
        result[index] = integer
    return result


def _load_coefficients(path: Path) -> tuple[np.ndarray, str | None, str | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_bytes()
    stripped = source.lstrip()
    if stripped.startswith(b"{"):
        try:
            document = json.loads(source.decode("utf-8"))
            virtualizer = document["personalized_hrtf"]["virtualizer_parameters"]
            coefficients = _json_int32(virtualizer["rosella_coefficients"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid personalized_headphone JSON: {exc}") from exc
        version = virtualizer.get("rosella_coefficients_version")
        room = virtualizer.get("room_model")
    else:
        if len(source) % 4:
            raise ValueError("raw rp payload must contain complete int32 lanes")
        coefficients = np.frombuffer(source, dtype="<i4").copy()
        version = room = None
    return coefficients, version, room


def _unpack_field(serialized: np.ndarray, directions: int,
                  exponent: int) -> np.ndarray:
    expected = 154 * directions
    if serialized.size != expected:
        raise ValueError(f"expected {expected} field lanes, got {serialized.size}")
    padded = np.zeros(160 * directions, dtype=np.float32)
    stride8 = 8 * directions
    stride2 = 2 * directions
    for source_index, value in enumerate(serialized):
        group4 = (source_index % stride8) // stride2
        destination = ((group4 & 3) + 4 * (
            source_index % stride2 +
            2 * directions * (source_index // stride8 + (group4 >> 2))))
        padded[destination] = _q15_exp(int(value), exponent)
    return padded


def _unpack_table_a_grid(serialized: np.ndarray, dimension: int,
                         serialized_rows: int, padded_rows: int,
                         lane_group: int) -> np.ndarray:
    if serialized.size != serialized_rows * dimension:
        raise ValueError("unexpected table-A grid size")
    padded = np.zeros(padded_rows * dimension, dtype=np.float32)
    group_width = lane_group * 4
    for source_index, value in enumerate(serialized):
        remainder = source_index % group_width
        destination = ((remainder // lane_group) + 4 * (
            remainder % lane_group +
            group_width // 4 * (source_index // group_width)))
        padded[destination] = _q15(int(value))
    return padded


def _unpack_table_a_extra(serialized: np.ndarray) -> np.ndarray:
    if serialized.size != 154:
        raise ValueError("table-A extra field must contain 154 serialized values")
    padded = np.zeros(160, dtype=np.float32)
    for source_index, value in enumerate(serialized):
        remainder = source_index & 7
        destination = ((remainder >> 1) + 4 * (
            (source_index & 1) + 2 * (source_index >> 3)))
        padded[destination] = _q15(int(value))
    return padded


def _parse_profile(values: np.ndarray, position: int) -> tuple[DistanceProfile, int]:
    bounds = np.asarray([_q15(int(value)) for value in values[position:position + 6]],
                        dtype=np.float32)
    position += 6
    distance = _q15_exp(int(values[position]), int(values[position + 1]))
    position += 2
    remaining = np.asarray(
        [_q15(int(value)) for value in values[position:position + 5]],
        dtype=np.float32)
    position += 5
    return DistanceProfile(
        bounds=bounds,
        distance_scale_m=distance,
        inverse_distance_per_m=remaining[0],
        axis_scales_internal=remaining[1:4],
        minimum_normalized_radius=remaining[4],
    ), position


def load_personalized_headphone(path: str | Path) -> RosellaModel:
    path = Path(path).resolve()
    coefficients, version, room = _load_coefficients(path)
    raw = coefficients.astype("<i4", copy=False).tobytes()
    header = inspect_rp(raw)
    if not header["checksum_valid"] or header["active_int32_lanes"] != coefficients.size:
        raise ValueError("invalid or non-active Rosella rp coefficient sequence")
    if not header["table_a_present"] or not header["table_b_present"] or header["table_c_present"]:
        raise NotImplementedError("current local renderer requires table A+B and no table C")
    if header["table_a_dimension"] != 64 or header["table_a_option"] != 3:
        raise NotImplementedError("current local renderer requires the observed 64-channel HQMF layout")
    if header["table_b_dimension"] != 20 or header["table_b_groups"] != 36:
        raise NotImplementedError("current local renderer requires 20 hybrid groups and 36 direction terms")

    values = coefficients
    extra = header["table_a_extra"]
    table_a_main_start = 13
    position = table_a_main_start
    table_a_control = int(values[position]) & 0xFFFF
    field_exponent = int(values[position])
    position += 1
    option_count = header["table_a_option"]
    option_ids = (values[position:position + option_count].astype(np.int64) &
                  0xFFFF).astype(np.int32)
    position += option_count
    option_values = np.asarray(
        [_q15(int(value)) for value in values[position:position + option_count]],
        dtype=np.float32)
    position += option_count
    table_a_scalar = _q15(int(values[position]))
    position += 1
    dimension = header["table_a_dimension"]
    table_a_filter_16x64 = _unpack_table_a_grid(
        values[position:position + 16 * dimension], dimension, 16, 20, 16)
    position += 16 * dimension
    table_a_four_integers = (values[position:position + 4].astype(np.int64) &
                             0xFFFF).astype(np.int32)
    position += 4
    table_a_integer = int(values[position]) & 0xFFFF
    position += 1
    table_a_filter_8x64 = _unpack_table_a_grid(
        values[position:position + 8 * dimension], dimension, 8, 10, 8)
    position += 8 * dimension
    table_a_vector16 = np.asarray(
        [_q15(int(value)) for value in values[position:position + 16]],
        dtype=np.float32)
    position += 16
    table_a_filter_4x64 = _unpack_table_a_grid(
        values[position:position + 4 * dimension], dimension, 4, 5, 4)
    position += 4 * dimension
    extra_indices = (values[position:position + extra].astype(np.int64) &
                     0xFFFF).astype(np.int32)
    position += extra
    extra_fields = np.empty((extra, 160), dtype=np.float32)
    for index in range(extra):
        extra_fields[index] = _unpack_table_a_extra(values[position:position + 154])
        position += 154
    extra_vectors = np.empty((extra, 16), dtype=np.float32)
    for index in range(extra):
        extra_vectors[index] = np.asarray(
            [_q15(int(value)) for value in values[position:position + 16]],
            dtype=np.float32)
        position += 16
    table_b_start = position
    expected_table_b_start = table_a_main_start + 1821 + 171 * extra
    if table_b_start != expected_table_b_start:
        raise AssertionError(
            f"table-A parser ended at {table_b_start}, expected {expected_table_b_start}")
    table_a_main = values[table_a_main_start:table_b_start].copy()

    sample_rate = 2 * (int(values[position]) & 0xFFFF)
    position += 1
    matrix_exponent = int(values[position])
    position += 1
    matrix_count = 36 * 36
    scale_matrix = lambda block: np.asarray(
        [_q15_exp(int(value), matrix_exponent) for value in block],
        dtype=np.float32).reshape(36, 36)
    matrix_left = scale_matrix(values[position:position + matrix_count])
    position += matrix_count
    matrix_right = scale_matrix(values[position:position + matrix_count])
    position += matrix_count
    vector_left = np.asarray(
        [_q15_exp(int(value), matrix_exponent)
         for value in values[position:position + 36]], dtype=np.float32)
    position += 36
    vector_right = np.asarray(
        [_q15_exp(int(value), matrix_exponent)
         for value in values[position:position + 36]], dtype=np.float32)
    position += 36

    serialized_count = 154 * 36
    field_left_serialized = values[position:position + serialized_count]
    field_left = _unpack_field(field_left_serialized, 36, field_exponent)
    field_left_odd_zero = not np.any(
        np.abs(np.asarray([_q15_exp(int(value), field_exponent)
                           for value in field_left_serialized[1::2]],
                          dtype=np.float32)) > np.float32(1e-6))
    position += serialized_count
    field_right = _unpack_field(
        values[position:position + serialized_count], 36, field_exponent)
    position += serialized_count

    hybrid_flags = (values[position:position + 20].astype(np.int64) & 0xFFFF).astype(np.int32)
    position += 20
    active_hybrid_values = int(np.count_nonzero(hybrid_flags == 1))
    if active_hybrid_values != header["table_b_extra"]:
        raise ValueError(
            f"hybrid value count {active_hybrid_values} != header {header['table_b_extra']}")
    hybrid_values = np.asarray(
        [_q15(int(value)) for value in values[position:position + active_hybrid_values]],
        dtype=np.float32)
    position += active_hybrid_values
    model_scalars = np.asarray(
        [_q15(int(value)) for value in values[position:position + 5]],
        dtype=np.float32)
    position += 5

    expected_table_a_tail = table_b_start + (
        header["table_b_dimension"] +
        380 * header["table_b_groups"] +
        header["table_b_extra"] + 79)
    if position != expected_table_a_tail:
        raise AssertionError(f"table-B parser ended at {position}, expected {expected_table_a_tail}")

    header_float_scalars = np.asarray([
        _q15(int(values[position])),
        _f32(_q15(int(values[position + 1])) * _f32(16.0)),
    ], dtype=np.float32)
    header_integer_fields = np.asarray([
        int(values[position + 2]),
        int(values[position + 3]) & 0xFFFF,
    ], dtype=np.int32)
    position += 4
    profiles = []
    for _ in range(4):
        profile, position = _parse_profile(values, position)
        profiles.append(profile)
    profile_tail = np.asarray(
        [_q15(int(value)) for value in values[position:position + 8]],
        dtype=np.float32)
    position += 8
    post_fields = values[position:position + 3].astype(np.int32, copy=True)
    position += 3
    if position != values.size:
        raise AssertionError(f"unparsed coefficient lanes: {values.size - position}")

    return RosellaModel(
        source_path=str(path),
        coefficients=coefficients,
        coefficient_sha256=hashlib.sha256(raw).hexdigest(),
        coefficient_version=version,
        room_model=room,
        table_a_dimension=header["table_a_dimension"],
        table_a_option=header["table_a_option"],
        table_a_extra=extra,
        table_a_header_field=int(values[8]) & 0xFFFF,
        table_a_header_25=int(values[9]) & 0xFFFF,
        table_a_control=table_a_control,
        table_a_option_ids=option_ids,
        table_a_option_values=option_values,
        table_a_scalar=table_a_scalar,
        table_a_filter_16x64_padded=table_a_filter_16x64,
        table_a_four_integers=table_a_four_integers,
        table_a_integer=table_a_integer,
        table_a_filter_8x64_padded=table_a_filter_8x64,
        table_a_vector16=table_a_vector16,
        table_a_filter_4x64_padded=table_a_filter_4x64,
        table_a_extra_indices=extra_indices,
        table_a_extra_fields_padded=extra_fields,
        table_a_extra_vectors=extra_vectors,
        sample_rate=sample_rate,
        matrix_exponent=matrix_exponent,
        field_exponent=field_exponent,
        matrix_left=matrix_left,
        matrix_right=matrix_right,
        vector_left=vector_left,
        vector_right=vector_right,
        field_left_padded=field_left,
        field_right_padded=field_right,
        field_left_odd_serialized_zero=field_left_odd_zero,
        hybrid_flags=hybrid_flags,
        hybrid_values=hybrid_values,
        model_scalars=model_scalars,
        header_float_scalars=header_float_scalars,
        header_integer_fields=header_integer_fields,
        profiles=tuple(profiles),
        profile_tail=profile_tail,
        post_fields=post_fields,
        table_a_main_serialized=table_a_main,
    )


def direction_basis(x: float, y: float, z: float,
                    dtype=np.float64) -> np.ndarray:
    """Return the observed 36-term Rosella direction basis."""
    f = dtype
    x, y, z = f(x), f(y), f(z)
    out = np.empty(36, dtype=dtype)
    yz = f(y * z)
    x2 = f(x * x)
    y2 = f(y * y)
    x2m02 = f(x2 - f(0.2))
    xy = f(x * y)
    out[0:4] = (f(1.0), x, y, z)
    out[4] = f(x2 - f(1.0 / 3.0))
    out[5] = xy
    out[6] = f(x * z)
    out[7] = f(y2 - f(1.0 / 3.0))
    out[8] = yz
    out[9] = f(f(x2 - f(0.6)) * x)
    out[10] = f(x2m02 * y)
    out[11] = f(x2m02 * z)
    out[12] = f(f(y2 - f(0.2)) * x)
    out[13] = f(yz * x)
    out[14] = f(f(y2 - f(0.6)) * y)
    out[15] = f(f(y2 - f(0.2)) * z)
    out[16] = f(f(x2 * x2) - f(0.2))
    out[17] = f(xy * x2)
    out[18] = f(f(x * z) * x2)
    out[19] = f(f(y2 * x2) - f(1.0 / 15.0))
    out[20] = f(yz * x2)
    out[21] = f(x * y2 * y)
    out[22] = f(x * y2 * z)
    out[23] = f(f(y2 * y2) - f(0.2))
    x4 = f(x2 * x2)
    x2y2 = f(y2 * x2)
    y4 = f(y2 * y2)
    out[24] = f(yz * y2)
    out[25] = f(f(x4 - f(3.0 / 7.0)) * x)
    out[26] = f(f(x4 - f(3.0 / 35.0)) * y)
    out[27] = f(f(x4 - f(3.0 / 35.0)) * z)
    out[28] = f(f(x2y2 - f(3.0 / 35.0)) * x)
    out[29] = f(f(x2 * z) * xy)
    out[30] = f(f(x2y2 - f(3.0 / 35.0)) * y)
    out[31] = f(f(x2y2 - f(1.0 / 35.0)) * z)
    out[32] = f(f(y4 - f(3.0 / 35.0)) * x)
    out[33] = f(f(y2 * z) * xy)
    out[34] = f(f(y4 - f(3.0 / 7.0)) * y)
    out[35] = f(f(y4 - f(3.0 / 35.0)) * z)
    return out
