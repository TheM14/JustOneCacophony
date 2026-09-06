"""Direct ID11/OAMD position scheduling for the binaural render path."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adm_atmos import q_to_adm_xyz
from oamd_bits import JocFieldState, frame_update
from variant_error import UnsupportedVariantError

OAMD_UPDATE_QUANTUM_SAMPLES = 64


@dataclass(frozen=True)
class PositionTransition:
    start_sample: int
    duration_samples: int
    origin: np.ndarray
    target: np.ndarray

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.duration_samples


class _ObjectPositionTrack:
    def __init__(self):
        self.initial = np.zeros(3, dtype=np.float64)
        self.last_target = self.initial.copy()
        self.transitions: list[PositionTransition] = []
        self.cursor = 0
        self.last_query_sample = -1

    def set_initial(self, position):
        target = np.asarray(position, dtype=np.float64)
        self.initial = target.copy()
        self.last_target = target.copy()

    def append(self, start_sample: int, duration_samples: int, target,
               object_index: int):
        start = int(start_sample)
        duration = int(duration_samples)
        if start < 0 or duration < 0:
            raise ValueError("position transition timing must be non-negative")
        target = np.asarray(target, dtype=np.float64)
        if self.transitions:
            previous = self.transitions[-1]
            if start < previous.end_sample:
                raise UnsupportedVariantError(
                    "oamd", "overlapping_binaural_position_ramps",
                    "同一对象的新位置更新在上一双耳 ramp 完成前到达",
                    details={
                        "object": object_index,
                        "ramp_start_sample": previous.start_sample,
                        "ramp_end_sample": previous.end_sample,
                        "next_update_sample": start,
                    })
            if start == previous.start_sample and previous.duration_samples == 0:
                self.transitions[-1] = PositionTransition(
                    start, duration, previous.origin.copy(), target.copy())
                self.last_target = target.copy()
                return
        self.transitions.append(PositionTransition(
            start, duration, self.last_target.copy(), target.copy()))
        self.last_target = target.copy()

    def position_at(self, sample: int) -> np.ndarray:
        sample = int(sample)
        if sample < self.last_query_sample:
            raise ValueError("binaural metadata positions must be queried monotonically")
        self.last_query_sample = sample
        while self.cursor < len(self.transitions):
            transition = self.transitions[self.cursor]
            if sample < transition.end_sample:
                break
            self.initial = transition.target.copy()
            self.cursor += 1
        if self.cursor >= len(self.transitions):
            return self.initial
        transition = self.transitions[self.cursor]
        if sample < transition.start_sample:
            return self.initial
        if transition.duration_samples == 0:
            return transition.target
        amount = (sample - transition.start_sample) / float(transition.duration_samples)
        return transition.origin + (transition.target - transition.origin) * amount


class OamdPositionTimeline:
    """Convert OAMD state updates into a sample-timed Cartesian trajectory."""

    def __init__(self, object_count: int = 15):
        if object_count != 15:
            raise ValueError("JOC OAMD currently requires 15 object slots")
        self.object_count = int(object_count)
        self.state = JocFieldState()
        self.tracks = [_ObjectPositionTrack() for _ in range(self.object_count)]
        self.initialized = False
        self.previous_targets: list[tuple[float, float, float] | None] = [
            None] * self.object_count
        self.payload_count = 0
        self.transition_count = 0
        self.last_coded_event_sample = -1

    def _targets(self) -> list[tuple[float, float, float]]:
        q = self.state.q
        return [
            q_to_adm_xyz(
                q[(object_index, "q1")],
                q[(object_index, "q2")],
                q[(object_index, "q3")],
            )
            for object_index in range(1, self.object_count + 1)
        ]

    def submit_update(self, update: dict, *, frame_start_sample: int,
                      outer_sample_offset: int = 0,
                      object_delay_samples: int = 1473,
                      processed_sample: int = 0):
        """Schedule one already-parsed :func:`oamd_bits.frame_update` result."""
        frame_start = int(frame_start_sample)
        outer_offset = int(outer_sample_offset)
        object_delay = int(object_delay_samples)
        if min(frame_start, outer_offset, object_delay) < 0:
            raise ValueError("OAMD frame, outer offset, and object delay must be non-negative")
        self.state.apply(update["values"])
        targets = self._targets()
        coded_event = (
            frame_start + outer_offset + int(update["block_offset_samples"]))
        if coded_event < self.last_coded_event_sample:
            raise UnsupportedVariantError(
                "oamd", "non_monotonic_binaural_updates",
                "双耳 OAMD 更新时间倒退",
                details={
                    "event_sample": coded_event,
                    "previous_event_sample": self.last_coded_event_sample,
                })
        self.last_coded_event_sample = coded_event

        if not self.initialized:
            if int(processed_sample) > 0:
                raise UnsupportedVariantError(
                    "oamd", "late_initial_binaural_state",
                    "首个 OAMD 状态在双耳 PCM 已处理后才出现，无法回填 sample 0",
                    details={
                        "processed_sample": int(processed_sample),
                        "first_event_sample": coded_event,
                    })
            for index, target in enumerate(targets):
                self.tracks[index].set_initial(target)
                self.previous_targets[index] = target
            self.initialized = True
            self.payload_count += 1
            return

        ramp_duration = int(update["ramp_duration_samples"])
        effective_ramp = max(0, ramp_duration - OAMD_UPDATE_QUANTUM_SAMPLES)
        transition_start = coded_event + object_delay
        if effective_ramp:
            transition_start += OAMD_UPDATE_QUANTUM_SAMPLES
        for index, target in enumerate(targets):
            if self.previous_targets[index] == target:
                continue
            self.tracks[index].append(
                transition_start, effective_ramp, target, index + 1)
            self.previous_targets[index] = target
            self.transition_count += 1
        self.payload_count += 1

    def submit_payload(self, payload, *, frame_start_sample: int,
                       outer_sample_offset: int = 0,
                       object_delay_samples: int = 1473,
                       processed_sample: int = 0):
        update = frame_update(payload)
        self.submit_update(
            update,
            frame_start_sample=frame_start_sample,
            outer_sample_offset=outer_sample_offset,
            object_delay_samples=object_delay_samples,
            processed_sample=processed_sample,
        )
        return update

    def positions_at(self, sample: int) -> np.ndarray:
        return np.stack(
            [track.position_at(sample) for track in self.tracks], axis=0
        ).astype(np.float64, copy=False)
