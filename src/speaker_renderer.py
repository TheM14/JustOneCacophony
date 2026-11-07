"""High-precision object-to-speaker renderer.

All spatial calculations, gain ramps, and object accumulation use float64. Input
PCM may be float32, but precision is reduced only when the caller explicitly
writes a lower-precision output format.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
from typing import Iterable

import numpy as np

import oamd_math
from oamd_bits import JocFieldState, frame_update, parse_trim
from oamd_params import metadata_params
from variant_error import UnsupportedVariantError
from speaker_layouts import RegionGeometry, SpeakerLayout, get_speaker_layout
from trim_balance import (SpeakerRenderParameters, apply_balance,
                          explicit_position_gain, parameters_from_oamd)

Q15_SCALE = 32768.0
Q15_MAX = 32767.0 / Q15_SCALE
GAIN_SNAP_THRESHOLD = 1.0e-4
ZONE_TO_REGION = (0, 1, 2, 3, 4, 5, 6, 0)
LFE_BED_LABELS = ("RC_LFE", "RC_LFE2")
BED_LABEL_INDEX = {label: index + 1 for index, label in
                   enumerate(("RC_L", "RC_R", "RC_C", "RC_LFE",
                              "RC_LS", "RC_RS", "RC_LB", "RC_RB",
                              "RC_TFL", "RC_TFR", "RC_TSL", "RC_TSR",
                              "RC_TBL", "RC_TBR", "RC_LW", "RC_RW",
                              "RC_LFE2"))}


def _stable_norm(vector):
    """Reference vector normalization: tiny-norm floor and unit clamp."""
    values = np.asarray(vector, dtype=np.float64)
    total = float(np.dot(values, values))
    if total == 0.0:
        return np.zeros_like(values)
    result = values / max(math.sqrt(total), 1.0e-5)
    return np.minimum(result, 1.0)


def _region_vector(layout, region_index, u, floor_v, top_v, w,
                   enable_height):
    """Uncompensated point-source vector in ``layout`` internal order."""
    region = layout.regions[int(region_index)]
    coordinates = _coordinates(region)
    floor = _plane_gains(region, coordinates, region.axis0_groups, u,
                         floor_v, region.mode)
    gains = floor
    if region.mode == 3:
        top = _plane_gains(region, coordinates, region.axis1_groups, u,
                           top_v, 3)
        height = min(max(float(w) if enable_height else 0.0, 0.0),
                     Q15_MAX)
        if height >= Q15_MAX:
            gains = top
        elif height > 0.0:
            floor_weight, height_weight = equal_power_pair(height)
            gains = floor * floor_weight + top * height_weight
    vector = np.zeros(layout.channel_count, dtype=np.float64)
    for value, speaker in zip(gains, region.speaker_ids):
        vector[speaker] = value
    return vector


def _fold_vector(calc, final, warp_mode, vector):
    """Apply the existing warp 2/3 surround fold to a calculation vector."""
    if calc is final:
        return vector
    return (_surround_fold_matrix(calc.speaker_bitfield,
                                 final.speaker_bitfield, warp_mode)
            @ vector)


def bed_object_targets(bed_channels) -> dict[int, tuple[float, float, float]]:
    """JOC 对象槽 → 固定方位 ``(u, v, w)``；只包含 bed 声道。

    bed 声道的方位由声道标签定义（ETSI TS 103 420 Table B.10），与目标扬声器布局
    无关：同一个声道无论输出哪种布局都落在同一个源位置，再交给与动态对象相同的
    点声源渲染路径。LFE 不占对象槽，单独处理。
    """
    from bed_layout import adm_xyz_to_render_coordinates, bed_object_positions

    return {slot: adm_xyz_to_render_coordinates(*xyz)
            for slot, xyz in bed_object_positions(bed_channels or ()).items()}


def _all_object_targets(layout: SpeakerLayout, state: JocFieldState,
                        bed_targets=None, parameters=None) -> np.ndarray:
    bed_targets = bed_targets or {}
    parameters = parameters or parameters_from_oamd(None, layout)
    result = np.zeros((15, layout.channel_count), dtype=np.float64)
    for object_index in range(1, 16):
        fixed = bed_targets.get(object_index)
        if fixed is not None:
            u, v, w = fixed
        else:
            u = state.q[(object_index, "q1")] / Q15_SCALE
            v = state.q[(object_index, "q2")] / Q15_SCALE
            w = state.q[(object_index, "q3")] / Q15_SCALE
        result[object_index - 1] = render_point_gains(
            layout, u, v, w, standard_order=False, parameters=parameters,
            bypass_trim=parameters.bypasses(object_index))
    return result


def expand_speaker_bitfield(compact_mask: int, center_height: bool = False) -> int:
    """Expand the compact target-layout bitfield into individual speakers."""
    expansions = (
        0x00000003, 0x00000004, 0x00000008, 0x00000030,
        0x000000C0, 0x00000100, 0x00000600, 0x00001800,
        0x00006000, 0x00018000, 0x00060000, 0x00180000,
        0x00600000, 0x01800000, 0x06000000, 0x18000000,
        0x60000000, 0x080000000, 0x600000000, 0x800000000,
        0x1000000000, 0x2000000000,
    )
    expanded = 0
    for bit, value in enumerate(expansions):
        if int(compact_mask) & (1 << bit):
            expanded |= value
    if center_height:
        expanded |= 0x4000000000
    return expanded


def layout_attenuation_db(compact_mask: int) -> float:
    """Compute the layout-dependent maximum positional compensation in dB."""
    expanded = expand_speaker_bitfield(compact_mask)
    height_channels = 2 * sum((expanded >> bit) & 1 for bit in (13, 15, 17, 19, 21))
    floor_channels = ((expanded >> 8) & 1) + 2 * sum(
        (expanded >> bit) & 1 for bit in (31, 4, 6, 11, 25, 27, 29, 33)
    )
    height_factor = min(height_channels / 4.0, 1.0)
    floor_factor = min(floor_channels / 4.0, 1.0)
    return -max(4.5 - 1.5 * height_factor - 3.0 * floor_factor, 0.0)


def floor_y_exponent(compact_mask: int, metadata_scaling_enabled: bool = True) -> int:
    if not metadata_scaling_enabled:
        return 0
    low_mask = expand_speaker_bitfield(compact_mask) & 0xFFFFFFFF
    return int(bool(low_mask & 0x130) and not bool(low_mask & 0x18C0))


def position_gain(compact_mask: int, v: float, w: float) -> float:
    """Return the high-precision position-dependent layout compensation."""
    y_term = min(max(float(v) / 0.6, 0.0), 1.0)
    z_term = min(max((float(w) - 0.2) / 0.8, 0.0), 1.0)
    amount = min(max(y_term + z_term, 0.0), 1.0)
    return math.pow(10.0, layout_attenuation_db(compact_mask) * amount / 20.0)


def equal_power_pair(position: float) -> tuple[float, float]:
    angle = math.pi * 0.5 * float(position)
    return math.cos(angle), math.sin(angle)


def _coordinates(region: RegionGeometry) -> np.ndarray:
    return np.asarray(region.coordinates_q15, dtype=np.float64) / Q15_SCALE


def _normalized(value: float, lower: float, upper: float) -> float:
    return (float(value) - float(lower)) / (float(upper) - float(lower))


def _axis0_gains(region: RegionGeometry, coordinates: np.ndarray, value: float) -> np.ndarray:
    result = np.zeros(len(region.speaker_ids), dtype=np.float64)
    position = float(value)
    for group in region.axis0_groups:
        if not group:
            continue
        first, last = group[0], group[-1]
        if position <= coordinates[first, 0]:
            result[first] = 1.0
            continue
        if position >= coordinates[last, 0]:
            result[last] = 1.0
            continue
        for lower_index, upper_index in zip(group, group[1:]):
            lower = coordinates[lower_index, 0]
            upper = coordinates[upper_index, 0]
            if position > lower and position <= upper:
                result[lower_index], result[upper_index] = equal_power_pair(
                    _normalized(position, lower, upper)
                )
                break
    return result


def _axis1_gains(region: RegionGeometry, coordinates: np.ndarray, groups,
                 value: float) -> np.ndarray:
    result = np.zeros(len(region.speaker_ids), dtype=np.float64)
    if not groups:
        return result
    position = float(value)
    first_value = coordinates[groups[0][0], 1]
    last_value = coordinates[groups[-1][0], 1]
    if position <= first_value:
        result[list(groups[0])] = 1.0
        return result
    if position > last_value:
        result[list(groups[-1])] = 1.0
        return result
    for lower_group, upper_group in zip(groups, groups[1:]):
        lower = coordinates[lower_group[0], 1]
        upper = coordinates[upper_group[0], 1]
        if position >= lower and position <= upper:
            lower_gain, upper_gain = equal_power_pair(_normalized(position, lower, upper))
            result[list(lower_group)] = lower_gain
            result[list(upper_group)] = upper_gain
            break
    return result


def _plane_gains(region: RegionGeometry, coordinates: np.ndarray, groups,
                  u: float, v: float, mode: int) -> np.ndarray:
    gains = _axis0_gains(
        RegionGeometry(region.coordinates_q15, region.speaker_ids, tuple(groups), (), region.mode),
        coordinates,
        u,
    )
    if mode >= 2:
        gains *= _axis1_gains(region, coordinates, groups, v)
    return gains


def render_point_gains(layout: str | SpeakerLayout, u: float, v: float, w: float,
                       *, region_index: int = 0, enable_height: bool = True,
                       object_gain: float = 1.0, standard_order: bool = True,
                       parameters: SpeakerRenderParameters | None = None,
                       bypass_trim: bool = False) -> np.ndarray:
    """Render one point object to a target speaker layout using float64."""
    target = get_speaker_layout(layout) if isinstance(layout, str) else layout
    controls = parameters or parameters_from_oamd(None, target)
    source = _surround_fold_layout(target) if controls.warp_mode in (2, 3) else target
    # Geometry may expand to 7.1.x; calibration always uses the final layout.
    internal = _render_point_gains(source, u, v, w, region_index=region_index,
                                   enable_height=enable_height, object_gain=object_gain,
                                   parameters=controls, bypass_trim=bypass_trim,
                                   compensation_bitfield=target.speaker_bitfield)
    if source is not target:
        internal = _surround_fold_matrix(source.speaker_bitfield, target.speaker_bitfield,
                                         controls.warp_mode) @ internal
    if standard_order:
        return internal[list(target.internal_to_standard)]
    return internal


@lru_cache(maxsize=16)
def _surround_fold_layout(target: SpeakerLayout) -> SpeakerLayout:
    """Add a rear pair for one-pair surround layouts, keeping the height layer."""
    if not floor_y_exponent(target.speaker_bitfield):
        return target
    expanded = {"51": "71", "512": "712", "514": "714"}
    if target.name in expanded:
        return get_speaker_layout(expanded[target.name])
    if target.name == "60":
        return target
    if target.name != "40":
        raise ValueError(f"unsupported surround fold layout: {target.name}")
    # 4.0 expands to six full-band speakers without adding centre or LFE.
    base = get_speaker_layout("71")
    regions = []
    speaker_ids = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    for region in base.regions:
        kept = [i for i, speaker in enumerate(region.speaker_ids) if speaker in speaker_ids]
        indices = {old: new for new, old in enumerate(kept)}
        def groups(items):
            return tuple(tuple(indices[i] for i in group if i in indices)
                         for group in items if any(i in indices for i in group))
        regions.append(RegionGeometry(
            tuple(region.coordinates_q15[i] for i in kept),
            tuple(speaker_ids[region.speaker_ids[i]] for i in kept),
            groups(region.axis0_groups), groups(region.axis1_groups), region.mode))
    return SpeakerLayout("60", 255, target.speaker_bitfield | 0x10,
                         ("L", "R", "Ls", "Rs", "Lrs", "Rrs"),
                         tuple(range(6)), tuple(regions))


@lru_cache(maxsize=32)
def _surround_fold_matrix(source_bitfield: int, target_bitfield: int,
                          warp_mode: int) -> np.ndarray:
    """Copy shared speakers and fold the rear pair into the target surround pair."""
    def speakers(bitfield):
        mask = expand_speaker_bitfield(bitfield)
        return tuple(i for i in range(mask.bit_length()) if mask & (1 << i))
    source, target = speakers(source_bitfield), speakers(target_bitfield)
    matrix = np.zeros((len(target), len(source)), dtype=np.float64)
    same_side, cross_side = (math.sqrt(3.0) / 2.0, 0.5) if warp_mode == 2 else (1.0, 0.0)
    # Speaker IDs 4/5 are side surrounds; 6/7 are the rear pair.
    for column, speaker in enumerate(source):
        if speaker in target:
            matrix[target.index(speaker), column] = 1.0
        elif speaker in (6, 7):
            matrix[target.index(speaker - 2), column] = same_side
            matrix[target.index(5 if speaker == 6 else 4), column] = cross_side
    matrix.flags.writeable = False
    return matrix


def _render_point_gains(target: SpeakerLayout, u: float, v: float, w: float,
                        *, region_index: int, enable_height: bool, object_gain: float,
                        parameters: SpeakerRenderParameters, bypass_trim: bool,
                        compensation_bitfield: int) -> np.ndarray:
    """Evaluate point gains before any surround fold or output-channel reorder."""
    controls = parameters
    region = target.regions[int(region_index)]
    coordinates = _coordinates(region)
    v, w = float(v), max(float(w), 0.0)
    floor_v = min(max(math.ldexp(v, floor_y_exponent(
        target.speaker_bitfield, controls.warp_mode != 0)), 0.0), 1.0)
    top_v = min(max(v, 0.0), 1.0)
    trim_v = v
    if controls.trim_mode == "explicit" and not bypass_trim:
        # Apply warp -> balance for floor, balance(original Y) for
        # top and trim. These coordinates must not be collapsed into one Y.
        listener, overhead = controls.listener_balance, controls.overhead_balance
        floor_v = apply_balance(floor_v, w, listener, overhead)
        top_v = apply_balance(top_v, w, listener, overhead)
        trim_v = apply_balance(v, w, listener, overhead)
    floor = _plane_gains(region, coordinates, region.axis0_groups, u, floor_v, region.mode)
    point_gains = floor
    if region.mode == 3:
        top = _plane_gains(region, coordinates, region.axis1_groups, u, top_v, 3)
        height = min(max(float(w) if enable_height else 0.0, 0.0), Q15_MAX)
        if height >= Q15_MAX:
            point_gains = top
        elif height > 0.0:
            floor_weight, height_weight = equal_power_pair(height)
            point_gains = floor * floor_weight + top * height_weight
    internal = np.zeros(target.channel_count, dtype=np.float64)
    gain = float(object_gain)
    if not bypass_trim:
        if controls.trim_mode == "automatic":
            gain *= position_gain(compensation_bitfield, trim_v, w)
        elif controls.trim_mode == "explicit":
            gain *= explicit_position_gain(controls, u, trim_v, w)
    for point_gain, speaker_id in zip(point_gains, region.speaker_ids):
        internal[speaker_id] = point_gain * gain
    return internal


def align_metadata_sample(sample: int, block_size: int = 32) -> int:
    return block_size * ((int(sample) + block_size // 2 - 1) // block_size)


def ramp_block_count(duration_samples: int, block_size: int = 32,
                     rate_scale: int = 1) -> int:
    return (int(duration_samples) * int(rate_scale) + block_size // 2 - 1) // block_size


class PythonSpeakerRenderer:
    """Stateful float64 speaker renderer.

    Two input shapes are accepted. Payloads that expand to block-level object
    state (``frame_update`` with ``object_blocks``) use the metadata path:
    object gain/inactive, region/elevation, size/snap/divergence/refinement,
    priority budget, ISF and bed distribution. Older payload dictionaries keep
    the previous point-position path so existing callers and tests still work.
    """

    def __init__(self, layout: str | SpeakerLayout, block_size: int = 32,
                 bed_channels=None, trim_meta=None, *, parameters=None,
                 size_budget: int = 0):
        self.layout = get_speaker_layout(layout) if isinstance(layout, str) else layout
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.bed_targets = bed_object_targets(bed_channels)
        if parameters is not None and trim_meta is not None:
            raise ValueError("pass normalized parameters or OAMD trim_meta, not both")
        self._parameters_override = parameters
        self._initial_parameters = parameters or parameters_from_oamd(trim_meta, self.layout)
        self._size_budget = int(size_budget)
        if self._size_budget < 0:
            raise ValueError("size_budget must be non-negative")
        self._raw_trim_mode = (trim_meta or {}).get("global_trim_mode", 0)
        self._meta_params = metadata_params()
        self.reset()

    def reset(self):
        self.parameters = self._initial_parameters
        channels = self.layout.channel_count
        self._state = JocFieldState()
        self._current = np.zeros((15, channels), dtype=np.float64)
        self._target = np.zeros_like(self._current)
        self._step = np.zeros_like(self._current)
        self._remaining = np.zeros((15, channels), dtype=np.int32)
        self._window = np.arange(self.block_size, dtype=np.float64) / float(self.block_size)
        self._meta_mode = False
        self._meta_ready = False
        self._n_meta = 0
        self._meta_slots: list[int | None] = []
        self._meta_labels: list[str | None] = []
        self._mc = None
        self._mt = None
        self._ms = None
        self._mr = None

    # -------------------------------------------------------------- metadata state
    def _update_trim(self, payload, update=None):
        trim = update.get("trim") if update is not None else None
        if trim is None and update is None:
            trim = parse_trim(payload)
        if trim is not None and self._parameters_override is None:
            self.parameters = parameters_from_oamd(trim, self.layout)
            if isinstance(trim, dict) and "global_trim_mode" in trim:
                self._raw_trim_mode = trim["global_trim_mode"]

    def _init_meta(self, update):
        self._isf_cache = {}
        program = update["program"]
        kinds = update["kinds"]
        count = int(update["object_count"])
        labels = [label for group in program["bed_channels"] for label in group]
        num_bed = int(program["num_bed_objects"])
        if len(labels) != num_bed:
            if program["dynamic_object_only"]:
                labels = ["RC_LFE"] if program["lfe_present"] else []
            if len(labels) != num_bed:
                raise UnsupportedVariantError(
                    "speaker", "bed_declaration_mismatch",
                    "OAMD bed 声明数量与节目分区不一致",
                    details={"declared": len(labels), "num_bed_objects": num_bed})
        slots = []
        slot = 1
        for object_index, kind in enumerate(kinds):
            label = labels[object_index] if object_index < len(labels) else None
            if kind == "bed" and label in LFE_BED_LABELS:
                slots.append(0)
            else:
                if slot > 15:
                    raise UnsupportedVariantError(
                        "speaker", "object_slot_overflow",
                        "元数据对象数量超出 15 个 JOC 对象槽",
                        details={"object_count": count})
                slots.append(slot)
                slot += 1
        if not program["dynamic_object_only"]:
            declared = program["num_dynamic_objects"]
            if (declared is not None
                    and num_bed + int(program["num_isf_objects"]) + int(declared)
                    != count):
                raise UnsupportedVariantError(
                    "speaker", "program_object_count_mismatch",
                    "OAMD 节目声明的对象分区与对象数不一致",
                    details={"object_count": count,
                             "num_bed_objects": num_bed,
                             "num_isf_objects": program["num_isf_objects"],
                             "num_dynamic_objects": declared})
        if program["num_isf_objects"]:
            isf_idx = program["isf_idx"]
            config = (self._meta_params["isf"]["configs"][isf_idx]
                      if isf_idx is not None and isf_idx < 6 else None)
            if config is None or config["counts"][4] != program["num_isf_objects"]:
                raise UnsupportedVariantError(
                    "speaker", "isf_configuration_mismatch",
                    "ISF 配置数量与节目声明不一致",
                    details={"isf_idx": isf_idx,
                             "num_isf_objects": program["num_isf_objects"]})
        if count != self._n_meta or slots != self._meta_slots:
            channels = self.layout.channel_count
            self._n_meta = count
            self._meta_slots = slots
            self._mc = np.zeros((count, channels), dtype=np.float64)
            self._mt = np.zeros((count, channels), dtype=np.float64)
            self._ms = np.zeros((count, channels), dtype=np.float64)
            self._mr = np.zeros((count, channels), dtype=np.int32)
        self._meta_labels = [labels[object_index]
                             if object_index < len(labels) else None
                             for object_index in range(count)]
        self._meta_mode = True

    def _block_targets(self, update, block_index):
        count = int(update["object_count"])
        channels = self.layout.channel_count
        targets = np.zeros((count, channels), dtype=np.float64)
        blocks = update["object_blocks"]
        kinds = update["kinds"]
        zero_size = set()
        size_objects = []
        for object_index in range(count):
            state = blocks[object_index][block_index]
            if (kinds[object_index] == "dynamic" and not state["not_active"]
                    and state["size_mode"] in (1, 2)
                    and not state["snap"] and state["divergence_q"] == 0):
                size_objects.append(object_index)
        if 0 < self._size_budget < len(size_objects):
            ranked = sorted(
                size_objects,
                key=lambda index: (-((int(blocks[index][block_index]["priority_q"])
                                      + 512) >> 10), index))
            zero_size = set(ranked[self._size_budget:])
        for object_index in range(count):
            state = blocks[object_index][block_index]
            if state["not_active"]:
                continue
            spatial = self._object_spatial(update, object_index, block_index,
                                           object_index in zero_size)
            gain = float(state["gain_q"]) / 4096.0
            targets[object_index] = spatial * gain
        return targets

    def _object_spatial(self, update, object_index, block_index, force_zero_size):
        kind = update["kinds"][object_index]
        if kind == "dynamic":
            return self._dynamic_spatial(update, object_index, block_index,
                                         force_zero_size)
        if kind == "bed":
            return self._bed_spatial(update, object_index)
        if kind == "isf":
            return self._isf_spatial(update, object_index)
        return np.zeros(self.layout.channel_count, dtype=np.float64)

    # -------------------------------------------------------------- dynamic path
    def _dynamic_spatial(self, update, object_index, block_index,
                         force_zero_size):
        state = update["object_blocks"][object_index][block_index]
        final = self.layout
        warp = self.parameters.warp_mode
        calc = _surround_fold_layout(final) if warp in (2, 3) else final
        region_index = ZONE_TO_REGION[int(state["zone_idx"]) & 7]
        u = float(state["qx"]) / Q15_SCALE
        v = float(state["qy"]) / Q15_SCALE
        w = max(float(state["qz"]) / Q15_SCALE, 0.0)
        enable_height = bool(state["enable_elevation"])
        if not enable_height:
            w = 0.0
        bypass = self.parameters.bypasses(object_index)
        floor_v = min(max(math.ldexp(v, floor_y_exponent(
            calc.speaker_bitfield, warp != 0)), 0.0), 1.0)
        top_v = min(max(v, 0.0), 1.0)
        trim_v = v
        if self.parameters.trim_mode == "explicit" and not bypass:
            listener = self.parameters.listener_balance
            overhead = self.parameters.overhead_balance
            floor_v = apply_balance(floor_v, w, listener, overhead)
            top_v = apply_balance(top_v, w, listener, overhead)
            trim_v = apply_balance(v, w, listener, overhead)
        region = calc.regions[region_index]
        coordinates = _coordinates(region)
        if state["snap"]:
            point = _region_vector(calc, region_index, u, floor_v, top_v, w,
                                   enable_height)
            best = -1.0
            chosen = 0
            for index, speaker in enumerate(region.speaker_ids):
                value = float(point[speaker])
                if value > best:
                    best = value
                    chosen = index
            speaker_u, speaker_v, speaker_w = coordinates[chosen]
            scale = 1.0
            distance = ((u - speaker_u) ** 2
                        + (min(v, scale) - scale * speaker_v) ** 2
                        + (w - speaker_w) ** 2)
            if distance < 0.04:
                vector = np.zeros(calc.channel_count, dtype=np.float64)
                vector[region.speaker_ids[chosen]] = 1.0
                comp = (float(speaker_u), float(speaker_v), float(speaker_w))
            else:
                vector = point
                comp = (u, trim_v, w)
        elif state["divergence_q"] > 0:
            spread = min(float(state["divergence_q"]) / Q15_SCALE, u, 1.0 - u)
            left = _region_vector(calc, region_index, u - spread, floor_v,
                                  top_v, w, enable_height)
            right = _region_vector(calc, region_index, u + spread, floor_v,
                                   top_v, w, enable_height)
            vector = _stable_norm(left + right)
            comp = (u, trim_v, w)
        elif state["size_mode"] in (1, 2) and not force_zero_size:
            size_q = state["size_q"]
            if size_q[0] == size_q[1] == size_q[2]:
                q_size = int(size_q[0])
            else:
                q_size = (int(size_q[0]) + int(size_q[1]) + int(size_q[2])) // 3
            size = q_size / Q15_SCALE
            t = oamd_math.size_curve_value(size)
            point = _region_vector(calc, region_index, u, floor_v, top_v, w,
                                   enable_height)
            if t < 0.000025:
                vector = point
            else:
                query = (u, (1.0 - w) * floor_v + w * top_v, w)
                spread = np.asarray(oamd_math.size_targets(
                    self._meta_params, calc, region_index, query, size,
                    height=enable_height), dtype=np.float64)
                if t > 0.05:
                    vector = spread
                else:
                    angle = 20.0 * t
                    if angle >= 0.99996948:
                        cosine, sine = 0.0, 1.0
                    else:
                        cosine = math.cos(angle * math.pi / 2.0)
                        sine = math.sin(angle * math.pi / 2.0)
                    vector = _stable_norm(0.5 * cosine * point
                                          + 0.5 * sine * spread)
            comp = (u, trim_v, w)
        else:
            vector = _region_vector(calc, region_index, u, floor_v, top_v, w,
                                    enable_height)
            comp = (u, trim_v, w)
        compensation = 1.0
        if not bypass:
            if self.parameters.trim_mode == "automatic":
                compensation = position_gain(final.speaker_bitfield,
                                             comp[1], comp[2])
            elif self.parameters.trim_mode == "explicit":
                compensation = explicit_position_gain(self.parameters,
                                                      comp[0], comp[1], comp[2])
        return _fold_vector(calc, final, warp, vector) * compensation

    # -------------------------------------------------------------- bed path
    def _bed_spatial(self, update, object_index):
        final = self.layout
        warp = self.parameters.warp_mode
        calc = _surround_fold_layout(final) if warp in (2, 3) else final
        label = self._meta_labels[object_index]
        bed_index = BED_LABEL_INDEX.get(label)
        if bed_index is None:
            return np.zeros(final.channel_count, dtype=np.float64)
        program = update["program"]
        route, weights = oamd_math.bed_distribution(
            calc.name, bed_index, int(program["bed_mask"]),
            bool(program["b_bed_chan_distribute"]))
        bypass = self.parameters.bypasses(object_index)
        mode = {"automatic": 0, "disabled": 1, "explicit": 2}[
            self.parameters.trim_mode]
        if route != -1:
            position = oamd_math.bed_reference_position(calc.name, route)
            qx, qy, qz = (int(value) for value in position)
            if qz > 0:
                qx = 32767 if qx > 16384 else 0
            u, v = qx / Q15_SCALE, qy / Q15_SCALE
            w = max(qz, 0) / Q15_SCALE
            floor_v = min(max(math.ldexp(v, floor_y_exponent(
                calc.speaker_bitfield, warp != 0)), 0.0), 1.0)
            top_v = min(max(v, 0.0), 1.0)
            vector = _region_vector(calc, 0, u, floor_v, top_v, w, True)
            compensation = 1.0
            if not bypass and self.parameters.trim_mode != "disabled":
                if self.parameters.trim_mode == "automatic":
                    compensation = position_gain(final.speaker_bitfield, v, w)
                else:
                    compensation = explicit_position_gain(
                        self.parameters, u, v, w)
        else:
            vector = np.asarray(weights, dtype=np.float64).copy()
            levels = (self.parameters.surround_db,
                      self.parameters.height_db,
                      self.parameters.centre_db)
            compensation = float(oamd_math.bed_compensation(
                final.name, bed_index, mode, bypass, levels))
        return _fold_vector(calc, final, warp, vector) * compensation

    # -------------------------------------------------------------- ISF path
    def _isf_spatial(self, update, object_index):
        program = update["program"]
        isf_idx = int(program["isf_idx"])
        row = object_index - int(program["num_bed_objects"])
        final = self.layout
        warp = self.parameters.warp_mode
        calc = _surround_fold_layout(final) if warp in (2, 3) else final
        key = (calc.name, isf_idx)
        matrix = self._isf_cache.get(key)
        if matrix is None:
            matrix = oamd_math.isf_matrix(self._meta_params, calc.name, isf_idx)
            self._isf_cache[key] = matrix
        vector = np.asarray(matrix[row], dtype=np.float64)
        disabled = (int(self._raw_trim_mode) == 1
                    or self.parameters.bypasses(object_index))
        compensation = float(oamd_math.isf_compensation(
            self._meta_params, final.name, isf_idx, row, disabled))
        return _fold_vector(calc, final, warp, vector) * compensation

    # -------------------------------------------------------------- scheduling
    def _apply_meta_targets(self, targets, ramp_samples):
        channels = self.layout.channel_count
        if self._mc is None or self._mc.shape != (len(targets), channels):
            self._mc = np.zeros((len(targets), channels), dtype=np.float64)
            self._mt = np.zeros_like(self._mc)
            self._ms = np.zeros_like(self._mc)
            self._mr = np.zeros((len(targets), channels), dtype=np.int32)
        difference = np.asarray(targets, dtype=np.float64) - self._mc
        significant = np.abs(difference) >= GAIN_SNAP_THRESHOLD
        self._mt[...] = targets
        self._mc[~significant] = self._mt[~significant]
        self._ms[~significant] = 0.0
        self._mr[~significant] = 0
        blocks = ramp_block_count(int(ramp_samples), self.block_size)
        if blocks:
            self._ms[significant] = difference[significant] / float(blocks)
            self._mr[significant] = blocks
        else:
            self._mc[significant] = self._mt[significant]
            self._ms[significant] = 0.0
            self._mr[significant] = 0
        self._meta_ready = True

    def _schedule_payload(self, update, event_sample):
        if "object_blocks" not in update or not update["object_count"]:
            return False
        if update.get("trim") is not None:
            self._update_trim(None, update)
        self._init_meta(update)
        for block_index, block in enumerate(update["blocks"]):
            sample = (int(event_sample) + int(update["sample_offset"])
                      + int(block["block_offset_samples"]))
            aligned = align_metadata_sample(sample, self.block_size)
            targets = self._block_targets(update, block_index)
            self._pending.setdefault(aligned // self.block_size, []).append(
                (targets, block["ramp_duration_samples"]))
        return True

    # -------------------------------------------------------------- legacy path
    def _apply_legacy(self, update):
        trim = parse_trim(self._current_payload)
        if trim is not None and self._parameters_override is None:
            self.parameters = parameters_from_oamd(trim, self.layout)
        self._state.apply(update["values"])
        new_target = _all_object_targets(self.layout, self._state,
                                         self.bed_targets, self.parameters)
        blocks = ramp_block_count(update["ramp_duration_samples"], self.block_size)
        difference = new_target - self._current
        significant = np.abs(difference) >= GAIN_SNAP_THRESHOLD
        self._target[...] = new_target
        self._current[~significant] = new_target[~significant]
        self._step[~significant] = 0.0
        self._remaining[~significant] = 0
        if blocks:
            self._step[significant] = difference[significant] / float(blocks)
            self._remaining[significant] = blocks
        else:
            self._current[significant] = new_target[significant]
            self._step[significant] = 0.0
            self._remaining[significant] = 0

    def _apply_payload(self, payload):
        self._current_payload = payload
        update = frame_update(payload)
        if "object_blocks" in update and update["object_count"]:
            self._update_trim(payload, update)
            self._init_meta(update)
            targets = self._block_targets(update, 0)
            self._apply_meta_targets(targets,
                                     update["blocks"][0]["ramp_duration_samples"])
        else:
            self._apply_legacy(update)

    # -------------------------------------------------------------- process
    def _render_meta_block(self, source, start, stop, out_block):
        for object_index in range(self._n_meta):
            slot = self._meta_slots[object_index]
            if slot is None:
                continue
            active = self._mr[object_index] > 0
            gains = np.broadcast_to(self._mt[object_index],
                                    (self.block_size, self.layout.channel_count)).copy()
            if np.any(active):
                gains[:, active] = (
                    self._mc[object_index, active][None, :]
                    + self._window[:, None] * self._ms[object_index, active][None, :]
                )
            out_block += source[start:stop, slot, None] * gains
            if np.any(active):
                self._mc[object_index, active] += self._ms[object_index, active]
                self._mr[object_index, active] -= 1
            finished = self._mr[object_index] == 0
            self._mc[object_index, finished] = self._mt[object_index, finished]

    def _render_legacy_block(self, source, start, stop, out_block):
        channels = self.layout.channel_count
        if "LFE" in self.layout.channels:
            out_block[:, 3] = source[start:stop, 0].astype(np.float64, copy=False)
        for object_index in range(15):
            active = self._remaining[object_index] > 0
            gains = np.broadcast_to(self._target[object_index],
                                    (self.block_size, channels)).copy()
            if np.any(active):
                gains[:, active] = (
                    self._current[object_index, active][None, :]
                    + self._window[:, None] * self._step[object_index, active][None, :]
                )
            out_block += (
                source[start:stop, object_index + 1, None].astype(np.float64, copy=False)
                * gains
            )
            if np.any(active):
                self._current[object_index, active] += self._step[object_index, active]
                self._remaining[object_index, active] -= 1
            finished = self._remaining[object_index] == 0
            self._current[object_index, finished] = self._target[object_index, finished]

    def process(self, objects16: np.ndarray, events=(), *, standard_order: bool = True,
                output: np.ndarray | None = None) -> np.ndarray:
        source = np.asarray(objects16)
        if source.ndim != 2 or source.shape[1] != 16:
            raise ValueError(f"objects16 must have shape [samples,16], got {source.shape}")
        if len(source) % self.block_size:
            raise ValueError(f"sample count must be divisible by {self.block_size}")
        total_samples = len(source)
        channels = self.layout.channel_count
        if output is None:
            output = np.zeros((total_samples, channels), dtype=np.float64)
        elif output.shape != (total_samples, channels) or output.dtype != np.float64:
            raise ValueError((output.shape, output.dtype))
        else:
            output[...] = 0.0

        events_by_block: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
        for sample_offset, payload in events:
            if not 0 <= int(sample_offset) <= total_samples:
                raise ValueError(f"metadata offset outside process call: {sample_offset}")
            block = align_metadata_sample(sample_offset, self.block_size) // self.block_size
            events_by_block[block].append((align_metadata_sample(
                sample_offset, self.block_size), bytes(payload)))

        self._pending: dict[int, list] = {}
        total_blocks = total_samples // self.block_size
        for block in range(total_blocks):
            for event_sample, payload in events_by_block.get(block, ()):
                update = frame_update(payload)
                if "object_blocks" in update and update["object_count"]:
                    self._schedule_payload(update, event_sample)
                else:
                    self._current_payload = payload
                    self._apply_legacy(update)
            for targets, ramp_samples in self._pending.pop(block, ()):
                self._apply_meta_targets(targets, ramp_samples)
            start = block * self.block_size
            stop = start + self.block_size
            out_block = output[start:stop]
            if self._meta_mode and self._meta_ready:
                self._render_meta_block(source, start, stop, out_block)
            else:
                self._render_legacy_block(source, start, stop, out_block)

        for event_sample, payload in events_by_block.get(total_blocks, ()):
            update = frame_update(payload)
            if "object_blocks" in update and update["object_count"]:
                self._schedule_payload(update, event_sample)
            else:
                self._current_payload = payload
                self._apply_legacy(update)
        for targets, ramp_samples in self._pending.pop(total_blocks, ()):
            self._apply_meta_targets(targets, ramp_samples)
        for block_index in sorted(self._pending):
            for targets, ramp_samples in self._pending[block_index]:
                self._apply_meta_targets(targets, ramp_samples)
        self._pending.clear()
        if standard_order:
            return output[:, list(self.layout.internal_to_standard)]
        return output

    def render_frame(self, objects16, payload=None, metadata_offset=1473):
        source = np.asarray(objects16)
        if source.shape != (1536, 16):
            raise ValueError(f"frame must have shape (1536,16), got {source.shape}")
        events = () if payload is None else ((int(metadata_offset), bytes(payload)),)
        return self.process(source, events)


def render_objects16(objects16: np.ndarray, payload_events: Iterable[tuple[int, bytes]],
                     layout: str | SpeakerLayout, *, block_size: int = 32,
                     standard_order: bool = True, output: np.ndarray | None = None,
                     trim_meta=None, parameters=None, size_budget: int = 0) -> np.ndarray:
    """Render a complete interleaved LFE + 15 object stream."""
    renderer = PythonSpeakerRenderer(layout, block_size=block_size,
                                     trim_meta=trim_meta, parameters=parameters,
                                     size_budget=size_budget)
    return renderer.process(objects16, payload_events, standard_order=standard_order,
                            output=output)


def payload_events_from_index(index, frame_count: int, *, frame_samples: int = 1536,
                              payload_id: int = 11, outer_offset_samples: int = 1473):
    for frame, row in enumerate(index.rows[:frame_count]):
        subpayloads = index.subpayloads(row)
        if payload_id in subpayloads:
            timing = frame_update(subpayloads[payload_id])
            yield (
                frame * frame_samples + outer_offset_samples
                + int(timing["sample_offset"]),
                subpayloads[payload_id],
            )
