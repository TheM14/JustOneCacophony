"""Stateful float64/complex128 Rosella hybrid-band renderer."""
from __future__ import annotations

import numpy as np

from rosella_direct import (
    PROFILE_MID,
    direct_and_room_send,
    special_lfe_direct,
)
from rosella_model import RosellaModel
from rosella_room import RosellaRoomFir


class RosellaRenderer:
    """Hold per-source direct parameters and the cross-block room state."""

    def __init__(self, model: RosellaModel, source_count: int,
                 room_impulse_slots: int = 4096, *, create_room: bool = True):
        if source_count <= 0:
            raise ValueError("source_count must be positive")
        self.model = model
        self.source_count = int(source_count)
        self.room = (RosellaRoomFir(model, impulse_slots=room_impulse_slots)
                     if create_room else None)
        self.positions = np.zeros((self.source_count, 3), dtype=np.float64)
        self.positions[:, 1] = 1.0
        self.profiles = np.full(self.source_count, PROFILE_MID, dtype=np.int32)
        self.special_lfe = np.zeros(self.source_count, dtype=bool)
        self.gains = np.empty(
            (self.source_count, 2, 77), dtype=np.complex128)
        self.room_sends = np.empty(self.source_count, dtype=np.float64)
        self._parameter_keys = [None] * self.source_count
        for source in range(self.source_count):
            self.set_source(source, self.positions[source], PROFILE_MID)

    def reset(self):
        if self.room is not None:
            self.room.reset()

    def set_source(self, source: int, position, profile: int = PROFILE_MID,
                   *, special_lfe: bool = False):
        source = int(source)
        if not 0 <= source < self.source_count:
            raise IndexError(source)
        coordinates = np.asarray(position, dtype=np.float64)
        if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
            raise ValueError(f"source position must be three finite values, got {position!r}")
        effective_profile = 0 if special_lfe else int(profile)
        key = ((bool(special_lfe), effective_profile)
               + tuple(float(value) for value in coordinates))
        if self._parameter_keys[source] == key:
            return
        self.positions[source] = coordinates
        self.profiles[source] = effective_profile
        self.special_lfe[source] = bool(special_lfe)
        parameters = (special_lfe_direct() if special_lfe else
                      direct_and_room_send(self.model, coordinates, effective_profile))
        self.gains[source] = parameters.gains
        self.room_sends[source] = parameters.room_send
        self._parameter_keys[source] = key

    def direct_and_send_static(self, sources):
        """Mix one static-parameter slot chunk without advancing room state."""
        values = np.asarray(sources, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.source_count, 77):
            raise ValueError(
                f"expected [slots,{self.source_count},77], got {values.shape}")
        direct = np.zeros((values.shape[0], 2, 77), dtype=np.complex128)
        room_send = np.zeros((values.shape[0], 77), dtype=np.complex128)
        for source in range(self.source_count - 1, -1, -1):
            direct += values[:, source, None, :] * self.gains[source][None, :, :]
            room_send += values[:, source, :] * self.room_sends[source]
        return direct, room_send

    def process_static_chunk(self, sources) -> np.ndarray:
        direct, room_send = self.direct_and_send_static(sources)
        if self.room is None:
            raise RuntimeError("room renderer is not configured")
        direct += self.room.process_chunk(room_send)
        return direct
