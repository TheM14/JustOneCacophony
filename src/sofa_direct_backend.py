"""Direct rendering of full SOFA HRIRs and explicitly selected compact fields."""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import warnings
import numpy as np
from scipy import signal

from partitioned_convolution import MultiSourcePartitionedConvolver
from point_source import PointSourceReference
from sofa_hrir import FullHrirField


class _CompiledField:
    def __init__(self, path):
        from sofa_hrtf_field import SofaHrtfField
        self.field = SofaHrtfField.load(path)
        self.fft_size = 2048
        self.measurement_radius_m = self.field.measurement_radius_m
        i = np.arange(128)
        z = 1 - 2 * (i + .5) / 128
        phi = i * np.pi * (3 - math.sqrt(5))
        directions = np.column_stack((np.sqrt(1 - z * z) * np.cos(phi),
                                      np.sqrt(1 - z * z) * np.sin(phi), z))
        self.recording_origin_samples = int(round(float(np.median([
            self.field.evaluate_sofa(q).delay_samples.mean() for q in directions]))))

    def info(self):
        return dict(source_sha256=self.field.source_sha256, method="compiled-band-field",
                    measurement_radius_m=self.measurement_radius_m,
                    cache_key=self.field.cache_key, format_version=self.field.format_version,
                    recording_origin_samples=self.recording_origin_samples,
                    radial_adaptation="none")


class DirectSofaBackend:
    fixed_reference = True
    default_profile = "mid"
    dsp_backend = "python-full-hrir"
    block_samples = 128

    def __init__(self, source, *, compiled=False, source_count=16, output_gain=1.,
                 method="lf_logmag_phase", shell_radius_m=None, mode=None,
                 reference_distance_m=None, sofa_room=False, room_preset=None):
        if source_count <= 0 or not math.isfinite(output_gain):
            raise ValueError("invalid source count or output gain")
        self.source_count, self.output_gain = int(source_count), float(output_gain)
        self.reference = PointSourceReference(reference_distance_m, mode=mode)
        self.default_profile = self.reference.mode
        self.compiled = bool(compiled)
        self.field = (_CompiledField(source) if compiled else
                      FullHrirField(source, method=method, shell_radius_m=shell_radius_m))
        self.hrtf_input_kind = "compiled_cache" if compiled else "sofa"
        self.response_kind = "hrir" if compiled else getattr(self.field, "response_kind", "hrir")
        if self.response_kind == "brir":
            message = "输入为包含房间响应的BRIR，将直接渲染文件响应。"
            if sofa_room:
                message += "已忽略--sofa-room。"
            warnings.warn(message, UserWarning)
            sofa_room = False
        self.hrtf_input_path = str(Path(source).expanduser().resolve())
        self._last = [None] * self.source_count
        self.distance_states = [None] * self.source_count
        self.processed_input_samples = self.output_samples = self.parameter_updates = 0
        self.finished = False
        self._filter_updates = deque()
        self.origin = self.field.recording_origin_samples
        if compiled:
            from hybrid_paths import HybridObjectPathRenderer
            from rosella_bank import RosellaBank
            self.dsp_backend = "python-compiled-hrtf"
            self.bank = RosellaBank(1, source_channels=self.source_count, output_channels=2)
            maximum = int(math.ceil(float(np.max(self.field.field.delay_bounds)) / 64))
            self.paths = HybridObjectPathRenderer(self.source_count, history_slots=maximum + 8,
                                                  transition_slots=2)
            self.latency_samples = self.bank.latency_samples
            from rosella_direct import special_lfe_direct
            self.lfe_transfer = special_lfe_direct().gains
        else:
            self.latency_samples = self.field.common_latency_samples
            self.convolver = MultiSourcePartitionedConvolver(
                self.source_count, block_size=128, max_filter_taps=self.field.fft_size)
            self.lfe_filter = np.zeros((2, self.field.fft_size), dtype=np.float64)
            start = self.latency_samples + max(self.origin, 0) - 64
            if start < 0 or start + 129 > self.field.fft_size:
                raise ValueError("LFE latency exceeds filter budget")
            self.lfe_filter[:, start:start + 129] = signal.firwin(129, 150., fs=48000) / math.sqrt(2)
        self._latency_remaining = self.latency_samples
        self.room = None
        self.filter_taps = self.field.fft_size
        if sofa_room:
            from sofa_room import SofaRoom
            self.room = SofaRoom(self.source_count, self.reference,
                                 implementation_latency=self.latency_samples, recording_origin=self.origin,
                                 preset=room_preset)
            if self.compiled:
                maximum += int(math.ceil(self.room.early_delay_bound_samples / 64))
                self.paths = HybridObjectPathRenderer(self.source_count, history_slots=maximum + 8,
                                                      transition_slots=2)
            else:
                self.filter_taps = self.room.filter_budget(self.field)
                self.convolver = MultiSourcePartitionedConvolver(
                    self.source_count, block_size=128, max_filter_taps=self.filter_taps)

    def set_source(self, source, position_adm, *, profile=None, gain=1., enabled=True,
                   special_lfe=False, fade=True):
        if self.finished:
            raise RuntimeError("renderer is finished")
        source = int(source)
        if not 0 <= source < self.source_count or not math.isfinite(gain):
            raise ValueError("invalid source or gain")
        if profile not in (None, self.default_profile):
            raise ValueError("source profile must match the selected reference mode")
        position = np.asarray(position_adm, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("source position must be finite")
        gain = float(gain) if enabled else 0.
        key = (tuple(position), gain, bool(special_lfe))
        if key == self._last[source]:
            return
        first = self.processed_input_samples == 0
        if special_lfe:
            if self.room is not None:
                self.room.set_source(source, None, gain, first=first, special_lfe=True)
            if self.compiled:
                from hybrid_paths import HybridPath
                f = self.field.field.band_center_frequencies_hz
                transfer = self.lfe_transfer * gain * np.exp(-2j * np.pi * (self.origin % 64) * f / 48000)
                path = HybridPath("lfe", np.full(2, self.origin // 64, dtype=np.int64), transfer)
                self.paths.set_paths(source, (path,), fade_slots=0 if first else 2)
            else:
                self._set_filter(source, self.lfe_filter * gain, first=first, fade=fade)
        else:
            previous = self.distance_states[source]
            state = self.reference.map_position(
                position, previous_direction=None if previous is None else previous.direction)
            if self.compiled:
                from hybrid_paths import HybridPath
                evaluation = self.field.field.evaluate_sofa(state.direction)
                slots = np.floor(evaluation.delay_samples / 64).astype(np.int64)
                residual = evaluation.delay_samples - slots * 64
                f = self.field.field.band_center_frequencies_hz
                transfer = evaluation.aligned_gains * np.exp(-2j * np.pi * residual[:, None] * f / 48000)
                direct_gain = 1. if self.room is None else self.room.direct_gain(state)
                paths = [HybridPath("direct", slots, transfer * gain * direct_gain)]
                if self.room is not None:
                    for reflection in self.room.paths(state):
                        q = reflection.direction_adm[[1, 0, 2]] * [1, -1, 1]
                        e = self.field.field.evaluate_sofa(q)
                        total = e.delay_samples + reflection.extra_delay_samples
                        early_slots = np.floor(total / 64).astype(np.int64)
                        residual = total - early_slots * 64
                        h = e.aligned_gains * np.exp(-2j * np.pi * residual[:, None] * f / 48000)
                        h *= gain * self.room.early_gain(reflection, self.field.measurement_radius_m)
                        paths.append(HybridPath("early:" + reflection.wall, early_slots, h))
                self.paths.set_paths(source, paths, fade_slots=0 if first or not fade else 2)
            else:
                if self.room is None:
                    spectrum = self.field.spectrum(state.direction)
                    hrir = np.roll(np.fft.irfft(spectrum, self.field.fft_size),
                                   self.field.common_latency_samples, axis=-1)
                else:
                    hrir = self.room.full_hrir(self.field, state)
                self._set_filter(source, hrir * gain, first=first, fade=fade)
            self.distance_states[source] = state
            if self.room is not None:
                self.room.set_source(source, state, gain, first=first)
        self._last[source] = key
        self.parameter_updates += 1

    def _set_filter(self, source, hrir, *, first, fade):
        if first:
            self.convolver.set_filter(source, hrir, fade_blocks=0)
        else:
            # Keep parameter changes on the source timeline after removing the common FIR shift.
            due = self.processed_input_samples + self.field.common_latency_samples
            update = (due, source, hrir, int(bool(fade)))
            for index in range(len(self._filter_updates) - 1, -1, -1):
                item = self._filter_updates[index]
                if item[0] != due:
                    break
                if item[1] == source:
                    self._filter_updates[index] = update
                    return
            self._filter_updates.append(update)

    def process(self, source):
        if self.finished:
            raise RuntimeError("renderer is finished")
        values = np.asarray(source, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.source_count or len(values) % 128
                or not np.isfinite(values).all()):
            raise ValueError("input must contain finite complete 128-sample source blocks")
        outputs = []
        for start in range(0, len(values), 128):
            block = values[start:start + 128]
            if self.compiled:
                hops = block.reshape(2, 64, self.source_count).transpose(0, 2, 1)
                hybrid = self.bank.hybrid_analysis.process_chunk(self.bank.qmf_analysis.process_chunk(hops))
                direct = self.paths.process(hybrid)
                time = self.bank.qmf_synthesis.process_chunk(self.bank.hybrid_synthesis.process_chunk(direct))
                output = time.transpose(0, 2, 1).reshape(-1, 2)
            else:
                while self._filter_updates and self._filter_updates[0][0] <= self.processed_input_samples:
                    _, source, hrir, fade_blocks = self._filter_updates.popleft()
                    self.convolver.set_filter(source, hrir, fade_blocks=fade_blocks)
                output = self.convolver.process(block)
            self.processed_input_samples += 128
            if self.room is not None:
                output = self.room.process(block, output)
                if len(output) == 0:
                    continue
            skip = min(len(output), self._latency_remaining)
            self._latency_remaining -= skip
            outputs.append(output[skip:] * self.output_gain)
        result = np.concatenate(outputs) if outputs else np.empty((0, 2))
        self.output_samples += len(result)
        return result

    def finish_output_capacity(self, tail_seconds):
        tail = float(tail_seconds)
        if not math.isfinite(tail) or tail < 0:
            raise ValueError("tail must be finite and nonnegative")
        return int(math.ceil((tail * 48000 + self.latency_samples + self.filter_taps) / 128) * 128)

    def finish(self, *, tail_seconds=5.):
        if self.finished:
            return np.empty((0, 2))
        count = self.finish_output_capacity(tail_seconds)
        expected = max(0, self.processed_input_samples + count - self.latency_samples - self.output_samples)
        if self.room is not None:
            pending = len(self.room.pending_direct) * 128
            count = int(math.ceil((count + pending) / self.room.batch_samples) * self.room.batch_samples) - pending
        output = self.process(np.zeros((count, self.source_count)))
        self.output_samples -= max(0, len(output) - expected)
        self.finished = True
        return output[:expected]

    def info(self):
        return dict(hrtf_input_kind=self.hrtf_input_kind, hrtf_input_path=self.hrtf_input_path,
                    cache_policy=None, field=self.field.info(), reference=self.reference.info(),
                    level_calibration=1., latency_compensated_samples=self.latency_samples,
                    processing_batch_samples=128 if self.room is None else self.room.batch_samples,
                    recording_origin_compensated_samples=0,
                    recording_origin_preserved_samples=self.origin,
                    parameter_alignment_delay_samples=0 if self.compiled else self.field.common_latency_samples,
                    current_reference_distances_m=[s.distance_m for s in self.distance_states if s is not None],
                    parameter_updates=self.parameter_updates,
                    direct_representation="77-band" if self.compiled else "full-HRIR",
                    radial_model="none", room_processing=self.room is not None,
                    room={"enabled": False} if self.room is None else self.room.info(),
                    response_kind=self.response_kind, precision="float64/complex128")
