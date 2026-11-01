"""Full-resolution coupling to the existing project-owned early/late room."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import fft

from generate_sofa_room import SCHEMA, ROOM_PATHS
from partitioned_convolution import UniformPartitionedConvolver
from public_room import ShoeboxRoomConfig, first_order_image_sources
from reference_distance import DistanceState, ReferenceDistanceProfileV1


class SofaRoom:
    batch_samples = 1536

    def __init__(self, source_count, reference, *, implementation_latency, recording_origin,
                 path=None, preset=None):
        self.reference = reference
        self.preset = ("legacy" if reference.mode == "far" else "compact") if preset is None else str(preset)
        if self.preset not in ROOM_PATHS:
            raise ValueError("room preset must be compact or legacy")
        self.path = Path(ROOM_PATHS[self.preset] if path is None else path)
        with np.load(self.path, allow_pickle=False) as data:
            if str(data["schema"]) != SCHEMA or int(data["sample_rate_hz"]) != 48000:
                raise ValueError("unsupported SOFA room asset schema or sample rate")
            kernel = np.asarray(data["late_fir"], dtype="<f8")
            self.metadata = json.loads(str(data["metadata_json"]))
        if (kernel.ndim != 2 or kernel.shape[0] != 2 or kernel.shape[1] < 1
                or not np.isfinite(kernel).all()
                or hashlib.sha256(kernel.tobytes()).hexdigest() != self.metadata["kernel_sha256"]):
            raise ValueError("invalid SOFA room FIR or checksum")
        self.config = ShoeboxRoomConfig(**self.metadata["early_config"])
        self.config.validate()
        self.alignment_samples = int(implementation_latency) + max(0, int(recording_origin))
        kernel = np.pad(kernel, ((0, 0), (self.alignment_samples, 0)))
        self.convolver = UniformPartitionedConvolver(kernel, block_size=self.batch_samples)
        self.current = np.zeros(int(source_count))
        self.target = self.current.copy()
        self.pending_mono, self.pending_direct = [], []
        self.maximum_early_delay_samples = 0.

    def distance_state(self, state):
        return DistanceState(
            self.reference.mode, state.normalized_radius, self.reference.reference_distance_m,
            max(ReferenceDistanceProfileV1.MINIMUM_DISTANCE_M, state.distance_m),
            state.direction[[1, 0, 2]] * [-1, 1, 1])

    def paths(self, state):
        distance = self.distance_state(state)
        try:
            paths = first_order_image_sources(distance.direction_adm, distance.physical_distance_m,
                                              48000, self.config)
        except ValueError as error:
            raise ValueError("--sofa-room预设房间无法表示当前声源位置；关闭该开关可保留直接声") from error
        self.maximum_early_delay_samples = max(
            self.maximum_early_delay_samples, max(p.extra_delay_samples for p in paths))
        return paths

    def direct_gain(self, state):
        return ReferenceDistanceProfileV1.direct_level_gain(self.distance_state(state))

    def early_gain(self, path, measurement_radius_m):
        return (ReferenceDistanceProfileV1.PUBLIC_ROOM_CALIBRATION_GAIN * path.reflection_gain
                * measurement_radius_m / path.path_distance_m)

    def set_source(self, source, state, gain, *, first=False, special_lfe=False):
        value = 0. if special_lfe else (float(gain) * ReferenceDistanceProfileV1.PUBLIC_ROOM_CALIBRATION_GAIN
                                       * ReferenceDistanceProfileV1.late_send(self.distance_state(state)))
        self.target[source] = value
        if first:
            self.current[source] = value

    def filter_budget(self, field):
        needed = field.fft_size + int(math.ceil(self.early_delay_bound_samples)) + 64
        return 1 << (needed - 1).bit_length()

    @property
    def early_delay_bound_samples(self):
        dimensions = np.asarray(self.config.dimensions_m)
        listener = np.asarray(self.config.listener_position_m)
        return 2 * float(np.max(np.maximum(listener, dimensions - listener))) * 48000 / self.config.speed_of_sound_m_s

    def full_hrir(self, field, state):
        length = self.filter_budget(field)
        omega = 2 * math.pi * fft.rfftfreq(length)
        direct = np.roll(np.fft.irfft(field.spectrum(state.direction), field.fft_size),
                         field.common_latency_samples, axis=-1)
        spectrum = fft.rfft(direct, length) * self.direct_gain(state)
        for path in self.paths(state):
            q = path.direction_adm[[1, 0, 2]] * [1, -1, 1]
            hrir = np.roll(np.fft.irfft(field.spectrum(q), field.fft_size),
                           field.common_latency_samples, axis=-1)
            spectrum += (fft.rfft(hrir, length) * np.exp(-1j * omega * path.extra_delay_samples)
                         * self.early_gain(path, field.measurement_radius_m))
        return fft.irfft(spectrum, length, axis=-1)

    def process(self, source_block, direct_block):
        amount = np.arange(1, 129, dtype=np.float64)[:, None] / 128
        sends = self.current + (self.target - self.current) * amount
        self.pending_mono.append(np.einsum("ts,ts->t", source_block, sends))
        self.pending_direct.append(direct_block)
        self.current[:] = self.target
        if len(self.pending_mono) * 128 < self.batch_samples:
            return np.empty((0, 2))
        result = np.concatenate(self.pending_direct) + self.convolver.process_block(np.concatenate(self.pending_mono))
        self.pending_mono.clear()
        self.pending_direct.clear()
        return result

    def info(self):
        return dict(enabled=True, schema=SCHEMA, preset=self.preset, asset=str(self.path.resolve()),
                    late_fir=self.metadata, early_config=asdict(self.config),
                    maximum_early_delay_samples=self.maximum_early_delay_samples,
                    late_alignment_samples=self.alignment_samples,
                    distance_coupling=ReferenceDistanceProfileV1.info(),
                    high_precision_adaptation="full-HRIR early paths with fractional delay; PCM float64 late FIR",
                    normalization="original project room coupling; not a measured BRIR")
