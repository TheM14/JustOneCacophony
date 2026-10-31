"""Direct HRIR rendering with a configured reference environment."""
from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import signal

from binaural_reference import ReferenceEnvironment
from partitioned_convolution import MultiSourcePartitionedConvolver
from sofa_hrir import FullHrirField, radial_ratio
from sofa_hrtf_field import SofaHrtfField
from sofa_binaural_backend import HybridPath, HybridObjectPathRenderer, _StereoDelay


class _CompiledField:
    def __init__(self, path, receivers):
        if receivers is None:
            raise ValueError("compiled input requires configured receiver positions")
        self.field = SofaHrtfField.load(path)
        self.fft_size = 2048
        self.frequencies = np.fft.rfftfreq(self.fft_size, 1/48000)
        self.measurement_radius_m = self.field.measurement_radius_m
        receivers = np.asarray(receivers, dtype=np.float64)
        if receivers.shape != (2, 3) or not np.isfinite(receivers).all():
            raise ValueError("invalid compiled receiver positions")
        self.head_radius_m = float(np.linalg.norm(receivers[0]))
        if not .04 < self.head_radius_m < .15 or not np.allclose(receivers[0], -receivers[1], atol=1e-6):
            raise ValueError("unsupported compiled receiver geometry")
        self.ear_directions = receivers/self.head_radius_m
        self.band_geometry = SimpleNamespace(frequencies=self.field.band_center_frequencies_hz,
            measurement_radius_m=self.measurement_radius_m, head_radius_m=self.head_radius_m,
            ear_directions=self.ear_directions)
        i = np.arange(128)
        z = 1-2*(i+.5)/128
        phi = i*np.pi*(3-math.sqrt(5))
        directions = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
        evaluations = [self.field.evaluate_sofa(q) for q in directions]
        self.recording_origin_samples = int(round(float(np.median([e.delay_samples.mean() for e in evaluations]))))
        self.centre_band = np.sqrt(np.mean([np.abs(e.transfer_gains)**2 for e in evaluations], axis=(0, 1)))
        from public_filterbank import hybrid_gain_synthesis_dictionary
        self.dictionary = hybrid_gain_synthesis_dictionary(self.fft_size)

    def bands(self, state):
        e = self.field.evaluate_sofa(state.direction)
        ratio = radial_ratio(self.band_geometry, state.direction, state.ear_distance_m)
        return e, ratio

    def spectrum_at(self, state):
        e, ratio = self.bands(state)
        f = self.field.band_center_frequencies_hz
        centre = self.centre_band*np.exp(-2j*np.pi*f*self.recording_origin_samples/48000)
        gains = state.centre_weight*e.transfer_gains*ratio+(1-state.centre_weight)*centre
        values = np.empty((2, 154))
        values[:, 0::2], values[:, 1::2] = gains.real, gains.imag
        return np.fft.rfft(values @ self.dictionary.T, self.fft_size)

    def info(self):
        return dict(source_sha256=self.field.source_sha256, method="compiled-band-field",
                    measurement_radius_m=self.measurement_radius_m,
                    receiver_geometry_source="reference-configuration",
                    cache_key=self.field.cache_key, format_version=self.field.format_version,
                    recording_origin_samples=self.recording_origin_samples,
                    radial_model_validation="analytic-identity-and-series-checks; approximate-human-head")


class ReferenceSofaBackend:
    fixed_reference = True
    default_profile = "mid"
    dsp_backend = "python-full-hrir"
    block_samples = 128
    environment_batch_samples = 1536

    def __init__(self, source, *, reference_config=None, compiled=False, source_count=16,
                 output_gain=1., method="lf_logmag_phase", shell_radius_m=None, mode=None):
        if source_count <= 0 or not math.isfinite(output_gain):
            raise ValueError("invalid source count or output gain")
        self.source_count, self.output_gain = int(source_count), float(output_gain)
        self.environment = ReferenceEnvironment(reference_config, mode=mode)
        self.default_profile = self.environment.mode
        self.compiled = bool(compiled)
        self.field = (_CompiledField(source, self.environment.compiled_receivers) if compiled else
                      FullHrirField(source, method=method, shell_radius_m=shell_radius_m))
        self.hrtf_input_kind = "compiled_cache" if compiled else "sofa"
        self.hrtf_input_path = str(Path(source).expanduser().resolve())
        self.cache_policy = None
        if self.environment.minimum_distance_m <= self.field.head_radius_m:
            raise ValueError("reference near-centre regularization lies inside the head")
        self._spectra = OrderedDict()
        self.level_calibration = self.environment.calibrate(self.field, self.spectrum_at)
        self.bank = self.environment.create_bank(self.source_count)
        self.room = self.environment.create_room()
        self._last = [None]*self.source_count
        self.distance_states = [None]*self.source_count
        self._sends = np.zeros(self.source_count, dtype=np.float64)
        self._send_targets = self._sends.copy()
        self.processed_input_samples = self.output_samples = self.parameter_updates = 0
        self.finished = False
        self._pending_room = []
        self._pending_direct = []
        self.origin = self.field.recording_origin_samples
        if self.origin < 0:
            raise ValueError("negative recording origin requires an explicit time-base adapter")
        self.implementation_latency = (self.bank.latency_samples if compiled else self.field.common_latency_samples)
        self.latency_samples = max(self.implementation_latency, self.bank.latency_samples)+max(self.origin, 0)
        self._latency_remaining = self.latency_samples
        self.room_delay = _StereoDelay(self.latency_samples-self.bank.latency_samples)
        if compiled:
            self.dsp_backend = "python-compiled-hrtf"
            maximum = int(math.ceil(float(np.max(self.field.field.delay_bounds))/64))
            self.paths = HybridObjectPathRenderer(self.source_count, history_slots=maximum+8, transition_slots=2)
            self.direct_delay = _StereoDelay(max(0, self.latency_samples-self.bank.latency_samples-self.origin))
            self.lfe_transfer = self._compiled_lfe()
            self.direct_bank = self.environment.create_bank(self.source_count)
        else:
            self.convolver = MultiSourcePartitionedConvolver(self.source_count, block_size=self.block_samples,
                max_filter_taps=self.field.fft_size)
            self.direct_delay = _StereoDelay(max(0, self.latency_samples-self.implementation_latency-max(self.origin, 0)))
            lowpass = signal.firwin(129, 150., fs=48000)
            h = np.zeros((2, self.field.fft_size), dtype=np.float64)
            start = self.latency_samples-64
            if start < 0 or start+len(lowpass) > self.field.fft_size:
                raise ValueError("LFE latency exceeds filter budget")
            h[:, start:start+len(lowpass)] = lowpass/math.sqrt(2)
            self.lfe_filter = h

    def spectrum_at(self, state):
        if self.compiled:
            return self.field.spectrum_at(state)
        key = (tuple(state.direction), state.ear_distance_m, state.centre_weight)
        if key in self._spectra:
            self._spectra.move_to_end(key)
            return self._spectra[key].copy()
        value = self.field.spectrum(state.direction)*radial_ratio(self.field, state.direction, state.ear_distance_m)
        value = value*state.centre_weight+(1-state.centre_weight)*self.field.centre_spectrum
        self._spectra[key] = value.copy()
        if len(self._spectra) > 128:
            self._spectra.popitem(last=False)
        return value

    def _compiled_lfe(self):
        from rosella_direct import special_lfe_direct
        return special_lfe_direct().gains

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
        self._last[source] = key
        first = self.processed_input_samples == 0
        if special_lfe:
            self._send_targets[source] = 0.
            if self.compiled:
                f = self.field.field.band_center_frequencies_hz
                transfer = self.lfe_transfer*gain*np.exp(-2j*np.pi*(self.origin % 64)*f/48000)
                path = HybridPath("lfe", np.full(2, self.origin//64, dtype=np.int64), transfer)
                self.paths.set_paths(source, (path,), fade_slots=0 if first else 2)
            else:
                self.convolver.set_filter(source, self.lfe_filter*gain, fade_blocks=0 if first else 1)
        else:
            state = self.environment.map_position(position)
            self.distance_states[source] = state
            self._send_targets[source] = state.room_send*gain
            if self.compiled:
                evaluation, ratio = self.field.bands(state)
                slots = np.floor(evaluation.delay_samples/64).astype(np.int64)
                residual = evaluation.delay_samples-slots*64
                f = self.field.field.band_center_frequencies_hz
                transfer = evaluation.aligned_gains*np.exp(-2j*np.pi*residual[:, None]*f/48000)*ratio
                transfer *= state.centre_weight*state.direct_gain*self.level_calibration*gain
                paths = [HybridPath("direct", slots, transfer)]
                if state.centre_weight < 1:
                    delay = max(0, self.origin)
                    centre = self.field.centre_band*np.exp(-2j*np.pi*(delay % 64)*f/48000)
                    centre = np.tile(centre, (2, 1))*(1-state.centre_weight)*state.direct_gain*self.level_calibration*gain
                    paths.append(HybridPath("centre", np.full(2, delay//64, dtype=np.int64), centre))
                self.paths.set_paths(source, paths, fade_slots=0 if first or not fade else 2)
            else:
                spectrum = self.spectrum_at(state)*state.direct_gain*self.level_calibration*gain
                hrir = np.roll(np.fft.irfft(spectrum, self.field.fft_size), self.field.common_latency_samples, axis=-1)
                self.convolver.set_filter(source, hrir, fade_blocks=0 if first or not fade else 1)
        if first:
            self._sends[source] = self._send_targets[source]
        self.parameter_updates += 1

    def process(self, source):
        if self.finished:
            raise RuntimeError("renderer is finished")
        values = np.asarray(source, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.source_count or len(values) % self.block_samples or not np.isfinite(values).all():
            raise ValueError("input must contain finite complete 128-sample source blocks")
        outputs = []
        for start in range(0, len(values), self.block_samples):
            block = values[start:start+self.block_samples]
            hops = block.reshape(2, 64, self.source_count).transpose(0, 2, 1)
            hybrid = self.bank.hybrid_analysis.process_chunk(self.bank.qmf_analysis.process_chunk(hops))
            amount = np.array([.5, 1.])[:, None]
            sends = self._sends+(self._send_targets-self._sends)*amount
            room_input = np.einsum("tsf,ts->tf", hybrid, sends)
            if self.compiled:
                self._pending_direct.append(self.paths.process(hybrid))
            else:
                self._pending_direct.append(self.convolver.process(block))
            self._pending_room.append(room_input)
            self._sends[:] = self._send_targets
            self.processed_input_samples += self.block_samples
            if len(self._pending_room)*self.block_samples < self.environment_batch_samples:
                continue
            room = self.room.process_chunk(np.concatenate(self._pending_room))
            time = self.bank.qmf_synthesis.process_chunk(self.bank.hybrid_synthesis.process_chunk(room))
            wet = time.transpose(0, 2, 1).reshape(-1, 2)
            direct = np.concatenate(self._pending_direct)
            if self.compiled:
                direct = self.direct_bank.qmf_synthesis.process_chunk(
                    self.direct_bank.hybrid_synthesis.process_chunk(direct))
                direct = direct.transpose(0, 2, 1).reshape(-1, 2)
            output = self.direct_delay.process(direct)+self.room_delay.process(wet)
            self._pending_direct.clear()
            self._pending_room.clear()
            skip = min(len(output), self._latency_remaining)
            self._latency_remaining -= skip
            outputs.append(output[skip:]*self.output_gain)
        output = np.concatenate(outputs) if outputs else np.empty((0, 2))
        self.output_samples += len(output)
        return output

    def finish_output_capacity(self, tail_seconds):
        tail = float(tail_seconds)
        if not math.isfinite(tail) or tail < 0:
            raise ValueError("tail must be finite and nonnegative")
        return int(math.ceil((tail*48000+self.latency_samples+self.field.fft_size)/128)*128)

    def finish(self, *, tail_seconds=5.):
        if self.finished:
            return np.empty((0, 2))
        count = self.finish_output_capacity(tail_seconds)
        pending = len(self._pending_room)*self.block_samples
        padded = int(math.ceil((count+pending)/self.environment_batch_samples)*self.environment_batch_samples)-pending
        expected = max(0, self.processed_input_samples+count-self.latency_samples-self.output_samples)
        output = self.process(np.zeros((padded, self.source_count)))
        self.output_samples -= max(0, len(output)-expected)
        output = output[:expected]
        self.finished = True
        return output

    def info(self):
        distances = [s.distance_m for s in self.distance_states if s is not None]
        return dict(hrtf_input_kind=self.hrtf_input_kind, hrtf_input_path=self.hrtf_input_path,
                    cache_policy=None, field=self.field.info(), reference=self.environment.info(),
                    level_calibration=self.level_calibration,
                    latency_compensated_samples=self.latency_samples,
                    processing_batch_samples=self.environment_batch_samples,
                    recording_origin_compensated_samples=self.origin,
                    current_reference_distances_m=distances,
                    parameter_updates=self.parameter_updates, direct_representation=("77-band" if self.compiled else "full-HRIR"),
                    radial_model="approximate-rigid-sphere-ratio", precision="float64/complex128")
