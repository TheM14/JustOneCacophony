"""High-precision object-to-speaker renderer.

All spatial calculations, gain ramps, and object accumulation use float64. Input
PCM may be float32, but precision is reduced only when the caller explicitly
writes a lower-precision output format.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

import numpy as np

from oamd_bits import JocFieldState, frame_update
from speaker_layouts import RegionGeometry, SpeakerLayout, get_speaker_layout

Q15_SCALE = 32768.0
Q15_MAX = 32767.0 / Q15_SCALE
GAIN_SNAP_THRESHOLD = 1.0e-4


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
                       object_gain: float = 1.0, standard_order: bool = True) -> np.ndarray:
    """Render one point object to a target speaker layout using float64."""
    target = get_speaker_layout(layout) if isinstance(layout, str) else layout
    region = target.regions[int(region_index)]
    coordinates = _coordinates(region)
    floor_v = min(max(math.ldexp(float(v), floor_y_exponent(target.speaker_bitfield)), 0.0), 1.0)
    floor = _plane_gains(region, coordinates, region.axis0_groups, u, floor_v, region.mode)
    point_gains = floor
    if region.mode == 3:
        top = _plane_gains(region, coordinates, region.axis1_groups, u, float(v), 3)
        height = min(max(float(w) if enable_height else 0.0, 0.0), Q15_MAX)
        if height >= Q15_MAX:
            point_gains = top
        elif height > 0.0:
            floor_weight, height_weight = equal_power_pair(height)
            point_gains = floor * floor_weight + top * height_weight
    internal = np.zeros(target.channel_count, dtype=np.float64)
    gain = position_gain(target.speaker_bitfield, v, w) * float(object_gain)
    for point_gain, speaker_id in zip(point_gains, region.speaker_ids):
        internal[speaker_id] = point_gain * gain
    if standard_order:
        return internal[list(target.internal_to_standard)]
    return internal


def align_metadata_sample(sample: int, block_size: int = 32) -> int:
    return block_size * ((int(sample) + block_size // 2 - 1) // block_size)


def ramp_block_count(duration_samples: int, block_size: int = 32,
                     rate_scale: int = 1) -> int:
    return (int(duration_samples) * int(rate_scale) + block_size // 2 - 1) // block_size


def _all_object_targets(layout: SpeakerLayout, state: JocFieldState) -> np.ndarray:
    result = np.zeros((15, layout.channel_count), dtype=np.float64)
    for object_index in range(1, 16):
        result[object_index - 1] = render_point_gains(
            layout,
            state.q[(object_index, "q1")] / Q15_SCALE,
            state.q[(object_index, "q2")] / Q15_SCALE,
            state.q[(object_index, "q3")] / Q15_SCALE,
            standard_order=False,
        )
    return result


class PythonSpeakerRenderer:
    """Stateful float64 speaker renderer with the same frame API as native."""

    def __init__(self, layout: str | SpeakerLayout, block_size: int = 32):
        self.layout = get_speaker_layout(layout) if isinstance(layout, str) else layout
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.reset()

    def reset(self):
        channels = self.layout.channel_count
        self._state = JocFieldState()
        self._current = np.zeros((15, channels), dtype=np.float64)
        self._target = np.zeros_like(self._current)
        self._step = np.zeros_like(self._current)
        self._remaining = np.zeros((15, channels), dtype=np.int32)
        self._window = np.arange(self.block_size, dtype=np.float64) / float(self.block_size)

    def _apply_payload(self, payload):
        update = frame_update(payload)
        self._state.apply(update["values"])
        new_target = _all_object_targets(self.layout, self._state)
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
        if "LFE" in self.layout.channels:
            output[:, 3] = source[:, 0].astype(np.float64, copy=False)

        events_by_block: dict[int, list[bytes]] = defaultdict(list)
        for sample_offset, payload in events:
            if not 0 <= int(sample_offset) <= total_samples:
                raise ValueError(f"metadata offset outside process call: {sample_offset}")
            block = align_metadata_sample(sample_offset, self.block_size) // self.block_size
            events_by_block[block].append(bytes(payload))

        total_blocks = total_samples // self.block_size
        for block in range(total_blocks):
            for payload in events_by_block.get(block, ()):
                self._apply_payload(payload)
            start = block * self.block_size
            stop = start + self.block_size
            out_block = output[start:stop]
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

        for payload in events_by_block.get(total_blocks, ()):
            self._apply_payload(payload)
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
                     standard_order: bool = True, output: np.ndarray | None = None) -> np.ndarray:
    """Render a complete interleaved LFE + 15 object stream."""
    renderer = PythonSpeakerRenderer(layout, block_size=block_size)
    return renderer.process(objects16, payload_events, standard_order=standard_order, output=output)

def payload_events_from_index(index, frame_count: int, *, frame_samples: int = 1536,
                              payload_id: int = 11, outer_offset_samples: int = 1473):
    for frame, row in enumerate(index.rows[:frame_count]):
        subpayloads = index.subpayloads(row)
        if payload_id in subpayloads:
            timing = frame_update(subpayloads[payload_id])
            yield (
                frame * frame_samples + outer_offset_samples + timing["block_offset_samples"],
                subpayloads[payload_id],
            )
