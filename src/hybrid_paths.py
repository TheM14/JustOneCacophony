"""Stateful direct and reflected paths for compact HRTF fields."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from public_filterbank import HYBRID_BANDS


@dataclass(frozen=True)
class HybridPath:
    label: str
    delay_slots: np.ndarray  # whole-QMF delay per ear, [2]
    transfer: np.ndarray  # [ear,77], includes gain and residual delay

    def __post_init__(self):
        slots = np.asarray(self.delay_slots)
        if slots.shape == ():
            slots = np.repeat(slots, 2)
        if slots.shape != (2,) or slots.dtype.kind not in "iu":
            raise ValueError("hybrid path delay_slots must contain two integers")
        slots = np.asarray(slots, dtype=np.int64)
        transfer = np.asarray(self.transfer, dtype=np.complex128)
        if transfer.shape != (2, HYBRID_BANDS) or not np.isfinite(transfer).all():
            raise ValueError("hybrid path transfer must have finite shape [2,77]")
        if np.any(slots < 0):
            raise ValueError("hybrid path delay_slots must be non-negative")
        slots.setflags(write=False)
        transfer.setflags(write=False)
        object.__setattr__(self, "delay_slots", slots)
        object.__setattr__(self, "transfer", transfer)


class HybridObjectPathRenderer:
    """Per-object hybrid histories and path transitions."""

    def __init__(self, source_count: int, *, history_slots: int = 256,
                 transition_slots: int = 8):
        self.source_count = int(source_count)
        self.history_slots = int(history_slots)
        self.transition_slots = int(transition_slots)
        if min(self.source_count, self.history_slots) <= 0 or self.transition_slots < 0:
            raise ValueError("invalid hybrid path renderer dimensions")
        self.history = np.zeros(
            (self.source_count, self.history_slots, HYBRID_BANDS), dtype=np.complex128)
        self.position = 0
        self.current: list[tuple[HybridPath, ...]] = [tuple() for _ in range(self.source_count)]
        self.target: list[tuple[HybridPath, ...] | None] = [None] * self.source_count
        self.fade_position = np.zeros(self.source_count, dtype=np.int32)
        self.fade_total = np.zeros(self.source_count, dtype=np.int32)
        self.processed_slots = 0

    def reset(self) -> None:
        self.history.fill(0.0)
        self.position = 0
        self.target = [None] * self.source_count
        self.fade_position.fill(0)
        self.fade_total.fill(0)
        self.processed_slots = 0

    def set_paths(self, source: int, paths, *, fade_slots: int | None = None) -> None:
        source = int(source)
        if not 0 <= source < self.source_count:
            raise IndexError(source)
        values = tuple(paths)
        for path in values:
            if np.any(path.delay_slots >= self.history_slots):
                raise ValueError(
                    f"path {path.label!r} needs {path.delay_slots.tolist()} slots, "
                    f"history capacity is {self.history_slots}")
        fade = self.transition_slots if fade_slots is None else int(fade_slots)
        if fade < 0:
            raise ValueError("path fade must be non-negative")
        if self.target[source] is not None:
            self.current[source] = self.target[source]
            self.target[source] = None
        if not self.processed_slots or fade == 0:
            self.current[source] = values
            self.target[source] = None
            self.fade_position[source] = 0
            self.fade_total[source] = 0
        else:
            self.target[source] = values
            self.fade_position[source] = 0
            self.fade_total[source] = fade

    def _render_paths(self, source: int, paths: tuple[HybridPath, ...]) -> np.ndarray:
        result = np.zeros((2, HYBRID_BANDS), dtype=np.complex128)
        for path in paths:
            indices = (self.position - path.delay_slots) % self.history_slots
            delayed = self.history[source, indices, :]
            result += delayed * path.transfer
        return result

    def process(self, hybrid) -> np.ndarray:
        values = np.asarray(hybrid, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.source_count, HYBRID_BANDS):
            raise ValueError(
                f"hybrid input must have shape [slots,{self.source_count},77]")
        if not np.isfinite(values).all():
            raise ValueError("hybrid input contains non-finite values")
        output = np.zeros((len(values), 2, HYBRID_BANDS), dtype=np.complex128)
        for slot in range(len(values)):
            self.history[:, self.position, :] = values[slot]
            for source in range(self.source_count - 1, -1, -1):
                current = self._render_paths(source, self.current[source])
                target_paths = self.target[source]
                if target_paths is None:
                    output[slot] += current
                    continue
                target = self._render_paths(source, target_paths)
                self.fade_position[source] += 1
                amount = min(
                    1.0, self.fade_position[source] / float(self.fade_total[source]))
                output[slot] += current * (1.0 - amount) + target * amount
                if self.fade_position[source] >= self.fade_total[source]:
                    self.current[source] = target_paths
                    self.target[source] = None
                    self.fade_position[source] = 0
                    self.fade_total[source] = 0
            self.position = (self.position + 1) % self.history_slots
            self.processed_slots += 1
        return output
