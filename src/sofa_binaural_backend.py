"""Stateful public SOFA binaural renderer.

The runtime topology mirrors the existing multi-object binaural path:
64-QMF -> 77 hybrid -> per-object directional transfer -> stereo synthesis.
The HRTF parameter source is a SOFA-derived fifth-order field.  Early
reflections and the late room use project-owned behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from public_filterbank import (
    ANALYSIS_SYNTHESIS_LATENCY_SAMPLES,
    HYBRID_BANDS,
    QMF_HOP,
    PublicAnalysis77,
    PublicSynthesis77,
    table_info as filterbank_table_info,
)
from public_room import (
    LateFdnConfig,
    SharedUnitaryFdn,
    ShoeboxRoomConfig,
    first_order_image_sources,
)
from reference_distance import DistanceState, ReferenceDistanceProfileV1
from sofa_canonical import CanonicalHrtf
from sofa_hrtf_field import (
    DEFAULT_ORDER,
    DEFAULT_PROJECTION_RIDGE,
    DEFAULT_SH_RIDGE,
    SofaHrtfField,
    compile_sofa_hrtf,
)


@dataclass(frozen=True)
class HybridPath:
    label: str
    delay_slots: np.ndarray  # whole-QMF delay per ear, [2]
    transfer: np.ndarray  # [ear,77], includes residual delay and HRTF delay

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
    """Per-object hybrid histories for direct and image-source paths."""

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
            # Normal 512-sample updates complete an 8-slot transition exactly.
            # If a caller updates faster, use the previous target as the new
            # stable side rather than resetting signal history.
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


class _StereoDelay:
    def __init__(self, delay_samples: int):
        self.delay_samples = int(delay_samples)
        if self.delay_samples < 0:
            raise ValueError("delay must be non-negative")
        self.state = np.zeros((self.delay_samples, 2), dtype=np.float64)

    def reset(self) -> None:
        self.state.fill(0.0)

    def process(self, values) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("stereo delay input must have shape [samples,2]")
        if self.delay_samples == 0:
            return source.copy()
        joined = np.concatenate((self.state, source), axis=0)
        output = joined[:len(source)].copy()
        self.state = joined[len(source):len(source) + self.delay_samples].copy()
        return output


class SofaBinauralBackend:
    """SOFA-derived public 77-band/SH renderer with public room processing."""

    def __init__(
            self,
            field: SofaHrtfField,
            *,
            source_count: int = 16,
            sample_rate_hz: float = 48000.0,
            default_profile: str = "mid",
            enable_early_reflections: bool = True,
            enable_late_room: bool = True,
            room_config: ShoeboxRoomConfig = ShoeboxRoomConfig(),
            fdn_config: LateFdnConfig | None = None,
            transition_slots: int = 8,
            history_slots: int = 256,
            output_gain: float = 1.0):
        self.source_count = int(source_count)
        self.sample_rate_hz = float(sample_rate_hz)
        self.default_profile = ReferenceDistanceProfileV1.validate_profile(default_profile)
        self.enable_early_reflections = bool(enable_early_reflections)
        self.enable_late_room = bool(enable_late_room)
        self.room_config = room_config
        self.room_config.validate()
        self.output_gain = float(output_gain)
        if (self.source_count <= 0 or not math.isfinite(self.sample_rate_hz)
                or self.sample_rate_hz <= 0.0):
            raise ValueError("source_count and sample rate must be positive")
        if not math.isfinite(self.output_gain):
            raise ValueError("output gain must be finite")
        if abs(self.sample_rate_hz - 48000.0) > 1.0e-9:
            raise ValueError("the public binaural runtime requires 48 kHz")

        if not isinstance(field, SofaHrtfField):
            raise TypeError(
                "field must be SofaHrtfField; use from_sofa() or "
                "from_compiled_cache() for file inputs")
        self.field = field
        self.hrtf_input_kind = "field"
        self.hrtf_input_path: str | None = None
        self.cache_policy: str | None = None
        if abs(self.field.sample_rate_hz - self.sample_rate_hz) > 1.0e-9:
            raise ValueError("HRTF field sample rate does not match the renderer")

        self.early_history_slots = int(history_slots)
        if self.early_history_slots <= 0:
            raise ValueError("history_slots must be positive")
        maximum_hrtf_delay = float(np.max(self.field.delay_bounds[:, 1], initial=0.0))
        self.maximum_hrtf_delay_samples = maximum_hrtf_delay
        self.hrtf_history_slots = int(math.ceil(maximum_hrtf_delay / QMF_HOP))
        self.analysis = PublicAnalysis77(self.source_count)
        self.paths = HybridObjectPathRenderer(
            self.source_count,
            history_slots=self.early_history_slots + self.hrtf_history_slots,
            transition_slots=transition_slots)
        self.synthesis = PublicSynthesis77(2)
        actual_fdn_config = fdn_config or LateFdnConfig(sample_rate_hz=self.sample_rate_hz)
        if abs(actual_fdn_config.sample_rate_hz - self.sample_rate_hz) > 1.0e-9:
            raise ValueError("FDN sample rate does not match the renderer")
        self.fdn = SharedUnitaryFdn(actual_fdn_config)
        self.late_delay = _StereoDelay(ANALYSIS_SYNTHESIS_LATENCY_SAMPLES)

        self.positions = np.zeros((self.source_count, 3), dtype=np.float64)
        self.positions[:, 1] = 1.0
        self.profiles = [self.default_profile] * self.source_count
        self.user_gain = np.ones(self.source_count, dtype=np.float64)
        self.special_lfe = np.zeros(self.source_count, dtype=bool)
        self.distance_state: list[DistanceState | None] = [None] * self.source_count
        self.late_current = np.zeros(self.source_count, dtype=np.float64)
        self.late_start = np.zeros(self.source_count, dtype=np.float64)
        self.late_target = np.zeros(self.source_count, dtype=np.float64)
        self.late_fade_position = np.zeros(self.source_count, dtype=np.int64)
        self.late_fade_total = np.zeros(self.source_count, dtype=np.int64)
        self.maximum_early_delay_samples = 0.0
        self.latency_to_discard = ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
        self.processed_input_samples = 0
        self.output_samples = 0
        self.parameter_updates = 0
        self.finished = False
        for source in range(self.source_count):
            self.set_source(
                source, self.positions[source], profile=self.default_profile,
                fade=False)

    @classmethod
    def from_sofa(
            cls, sofa: str | Path | CanonicalHrtf, *,
            cache_policy: str = "memory",
            cache_dir: str | Path | None = None,
            shell_radius_m: float = 1.0,
            order: int = DEFAULT_ORDER,
            projection_ridge: float = DEFAULT_PROJECTION_RIDGE,
            sh_ridge: float = DEFAULT_SH_RIDGE,
            **renderer_options) -> "SofaBinauralBackend":
        """Compile a SOFA source once and construct the runtime renderer."""
        field = compile_sofa_hrtf(
            sofa,
            target_sample_rate_hz=float(
                renderer_options.get("sample_rate_hz", 48000.0)),
            shell_radius_m=shell_radius_m,
            order=order,
            projection_ridge=projection_ridge,
            sh_ridge=sh_ridge,
            cache_policy=cache_policy,
            cache_dir=cache_dir)
        result = cls(field, **renderer_options)
        result.hrtf_input_kind = "sofa"
        result.hrtf_input_path = (
            str(Path(sofa).expanduser().resolve())
            if not isinstance(sofa, CanonicalHrtf) else sofa.source_path)
        result.cache_policy = str(cache_policy).lower()
        return result

    @classmethod
    def from_compiled_cache(
            cls, cache: str | Path, **renderer_options
            ) -> "SofaBinauralBackend":
        """Load an explicitly selected validated JOC compiled HRTF cache."""
        path = Path(cache).expanduser().resolve()
        field = SofaHrtfField.load(path)
        result = cls(field, **renderer_options)
        result.hrtf_input_kind = "compiled_cache"
        result.hrtf_input_path = str(path)
        result.cache_policy = None
        return result

    def _make_path(self, label: str, direction_adm, path_distance_m: float,
                   extra_delay_samples: float, amplitude: float) -> HybridPath:
        del path_distance_m
        evaluation = self.field.evaluate_adm(direction_adm)
        extra_delay = float(extra_delay_samples)
        if not math.isfinite(extra_delay) or extra_delay < 0.0:
            raise ValueError("path delay must be finite and non-negative")
        early_delay_slots = int(math.floor(extra_delay / QMF_HOP))
        if early_delay_slots >= self.early_history_slots:
            raise ValueError(
                f"path {label!r} needs {early_delay_slots} early-delay slots, "
                f"early history capacity is {self.early_history_slots}")
        total_delay = np.asarray(evaluation.delay_samples, dtype=np.float64) + extra_delay
        delay_slots = np.floor(total_delay / QMF_HOP).astype(np.int64)
        residual = total_delay - delay_slots * QMF_HOP
        propagation_phase = np.exp(
            -2j * np.pi * self.field.band_center_frequencies_hz[None, :]
            * residual[:, None]
            / self.sample_rate_hz)
        transfer = np.asarray(
            evaluation.aligned_gains * propagation_phase * float(amplitude),
            dtype=np.complex128)
        return HybridPath(label, delay_slots, transfer)

    def _ordinary_paths(self, state: DistanceState, gain: float) -> tuple[HybridPath, ...]:
        # Object PCM is programme-normalized.  Physical distance controls room
        # geometry, while the project profile supplies the direct presentation
        # coefficient instead of applying a second free-field 1/r attenuation.
        direct_amplitude = gain * ReferenceDistanceProfileV1.direct_level_gain(state)
        room_gain = ReferenceDistanceProfileV1.room_calibration_gain(state)
        paths = [self._make_path(
            "direct", state.direction_adm, state.physical_distance_m, 0.0,
            direct_amplitude)]
        if self.enable_early_reflections:
            reflections = first_order_image_sources(
                state.direction_adm, state.physical_distance_m,
                self.sample_rate_hz, self.room_config)
            for reflection in reflections:
                amplitude = (
                    gain * room_gain * reflection.reflection_gain
                    * ReferenceDistanceProfileV1.inverse_distance_gain(
                        self.field.measurement_radius_m,
                        reflection.path_distance_m))
                paths.append(self._make_path(
                    f"early:{reflection.wall}", reflection.direction_adm,
                    reflection.path_distance_m, reflection.extra_delay_samples,
                    amplitude))
                self.maximum_early_delay_samples = max(
                    self.maximum_early_delay_samples,
                    reflection.extra_delay_samples)
        return tuple(paths)

    def _lfe_paths(self, gain: float) -> tuple[HybridPath, ...]:
        frequency = self.field.band_center_frequencies_hz
        lowpass = np.ones(HYBRID_BANDS, dtype=np.float64)
        lowpass[frequency >= 180.0] = 0.0
        transition = (frequency > 120.0) & (frequency < 180.0)
        amount = (frequency[transition] - 120.0) / 60.0
        lowpass[transition] = np.cos(0.5 * np.pi * amount) ** 2
        transfer = np.repeat(
            (gain * lowpass / math.sqrt(2.0))[None, :], 2, axis=0
        ).astype(np.complex128)
        return (HybridPath(
            "public_lfe_lowpass", np.zeros(2, dtype=np.int64), transfer),)

    def _set_late_target(self, source: int, value: float, fade: bool) -> None:
        value = float(value)
        fade_samples = (self.paths.transition_slots * QMF_HOP
                        if fade and self.processed_input_samples else 0)
        if fade_samples == 0:
            self.late_current[source] = value
            self.late_start[source] = value
            self.late_target[source] = value
            self.late_fade_position[source] = 0
            self.late_fade_total[source] = 0
        else:
            self.late_start[source] = self.late_current[source]
            self.late_target[source] = value
            self.late_fade_position[source] = 0
            self.late_fade_total[source] = fade_samples

    def set_source(self, source: int, position_adm, *, profile: str | None = None,
                   gain: float = 1.0, enabled: bool = True,
                   special_lfe: bool = False, fade: bool = True) -> None:
        if self.finished:
            raise RuntimeError("SOFA renderer is finished")
        source = int(source)
        if not 0 <= source < self.source_count:
            raise IndexError(source)
        gain = float(gain)
        if not math.isfinite(gain):
            raise ValueError("source gain must be finite")
        effective_gain = gain if enabled else 0.0
        name = self.default_profile if profile is None else profile
        state = ReferenceDistanceProfileV1.map_adm_position(position_adm, name)
        path_set = (self._lfe_paths(effective_gain) if special_lfe
                    else self._ordinary_paths(state, effective_gain))
        self.paths.set_paths(
            source, path_set,
            fade_slots=(self.paths.transition_slots if fade else 0))
        late_send = (0.0 if special_lfe or not self.enable_late_room or not enabled
                     else effective_gain
                     * ReferenceDistanceProfileV1.room_calibration_gain(state)
                     * ReferenceDistanceProfileV1.late_send(state))
        self._set_late_target(source, late_send, fade)
        self.positions[source] = np.asarray(position_adm, dtype=np.float64)
        self.profiles[source] = state.profile
        self.user_gain[source] = gain
        self.special_lfe[source] = bool(special_lfe)
        self.distance_state[source] = state
        self.parameter_updates += 1

    def _late_send_envelope(self, sample_count: int) -> np.ndarray:
        envelope = np.empty((sample_count, self.source_count), dtype=np.float64)
        for source in range(self.source_count):
            total = int(self.late_fade_total[source])
            if total == 0:
                envelope[:, source] = self.late_current[source]
                continue
            start_position = int(self.late_fade_position[source])
            position = start_position + np.arange(1, sample_count + 1)
            amount = np.clip(position / float(total), 0.0, 1.0)
            envelope[:, source] = (
                self.late_start[source] * (1.0 - amount)
                + self.late_target[source] * amount)
            new_position = start_position + sample_count
            if new_position >= total:
                self.late_current[source] = self.late_target[source]
                self.late_start[source] = self.late_target[source]
                self.late_fade_position[source] = 0
                self.late_fade_total[source] = 0
            else:
                self.late_current[source] = float(envelope[-1, source])
                self.late_fade_position[source] = new_position
        return envelope

    def _process(self, sources) -> np.ndarray:
        values = np.asarray(sources, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.source_count:
            raise ValueError(f"sources must have shape [samples,{self.source_count}]")
        if len(values) % QMF_HOP:
            raise ValueError("SOFA backend input must be divisible by 64 samples")
        if not np.isfinite(values).all():
            raise ValueError("SOFA backend input contains non-finite values")
        hybrid = self.analysis.process(values)
        direct_and_early = self.paths.process(hybrid)
        direct_pcm = self.synthesis.process(direct_and_early)
        if self.enable_late_room:
            sends = self._late_send_envelope(len(values))
            mono = np.sum(values * sends, axis=1, dtype=np.float64)
            late_pcm = self.late_delay.process(self.fdn.process(mono))
        else:
            # Still advance any pending send fade deterministically.
            self._late_send_envelope(len(values))
            late_pcm = np.zeros_like(direct_pcm)
        mixed = np.asarray((direct_pcm + late_pcm) * self.output_gain, dtype=np.float64)
        skip = min(self.latency_to_discard, len(mixed))
        self.latency_to_discard -= skip
        self.processed_input_samples += len(values)
        output = mixed[skip:]
        self.output_samples += len(output)
        return output

    def process(self, sources) -> np.ndarray:
        if self.finished:
            raise RuntimeError("SOFA renderer is finished")
        return self._process(sources)

    def finish(self, *, tail_seconds: float | None = None) -> np.ndarray:
        if self.finished:
            return np.zeros((0, 2), dtype=np.float64)
        if tail_seconds is not None and (
                not math.isfinite(float(tail_seconds)) or float(tail_seconds) < 0.0):
            raise ValueError("tail_seconds must be finite and non-negative")
        requested = (self.fdn.tail_samples if tail_seconds is None
                     else int(math.ceil(float(tail_seconds) * self.sample_rate_hz)))
        drain = max(
            requested if self.enable_late_room else 0,
            int(math.ceil(
                self.maximum_hrtf_delay_samples
                + self.maximum_early_delay_samples)) + 2048,
        ) + ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
        drain = int(math.ceil(drain / QMF_HOP) * QMF_HOP)
        output = self._process(np.zeros((drain, self.source_count), dtype=np.float64))
        self.finished = True
        return output

    def finish_output_capacity(self, tail_seconds: float | None = None) -> int:
        """Return a conservative bound for one future :meth:`finish` output."""
        if tail_seconds is not None and (
                not math.isfinite(float(tail_seconds)) or float(tail_seconds) < 0.0):
            raise ValueError("tail_seconds must be finite and non-negative")
        requested = (self.fdn.tail_samples if tail_seconds is None
                     else int(math.ceil(float(tail_seconds) * self.sample_rate_hz)))
        hrtf_bound = self.hrtf_history_slots * QMF_HOP
        early_bound = hrtf_bound + 2048
        if self.enable_early_reflections:
            early_bound += self.early_history_slots * QMF_HOP
        drain = max(requested if self.enable_late_room else 0, early_bound)
        drain += ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
        return int(math.ceil(drain / QMF_HOP) * QMF_HOP)

    def reset(self) -> None:
        self.analysis.reset()
        self.paths.reset()
        self.synthesis.reset()
        self.fdn.reset()
        self.late_delay.reset()
        self.latency_to_discard = ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
        self.processed_input_samples = 0
        self.output_samples = 0
        self.finished = False
        for source in range(self.source_count):
            self.set_source(
                source, self.positions[source], profile=self.profiles[source],
                gain=float(self.user_gain[source]),
                special_lfe=bool(self.special_lfe[source]), fade=False)

    def info(self) -> dict:
        return {
            "name": "SofaBinauralBackend",
            "source_count": self.source_count,
            "sample_rate_hz": self.sample_rate_hz,
            "precision": "float64/complex128",
            "signal_path": (
                "public 64-QMF -> public 77-hybrid -> SOFA order-5 real-SH "
                "direct/early -> public synthesis + shared unitary FDN"),
            "hrtf_input_kind": self.hrtf_input_kind,
            "hrtf_input_path": self.hrtf_input_path,
            "cache_policy": self.cache_policy,
            "latency_compensated_samples": ANALYSIS_SYNTHESIS_LATENCY_SAMPLES,
            "enable_early_reflections": self.enable_early_reflections,
            "enable_late_room": self.enable_late_room,
            "early_history_slots": self.early_history_slots,
            "hrtf_history_slots": self.hrtf_history_slots,
            "maximum_hrtf_delay_samples": self.maximum_hrtf_delay_samples,
            "maximum_early_delay_samples": self.maximum_early_delay_samples,
            "parameter_updates": self.parameter_updates,
            "processed_input_samples_including_flush": self.processed_input_samples,
            "output_samples_before_trim": self.output_samples,
            "distance": ReferenceDistanceProfileV1.info(),
            "filterbank": filterbank_table_info(),
            "field": self.field.info(),
            "late_room": self.fdn.info(),
        }
