"""Native float64 SOFA binaural DSP (ctypes bridge to eac3joc_core).

The C++ side mirrors the Python :class:`sofa_binaural_backend.SofaBinauralBackend`
mathematics: 64-QMF/77-hybrid analysis and synthesis, fifth-order ACN/N3D real
spherical-harmonic field evaluation, whole-QMF-slot per-object delay histories,
six image-source early reflections, the shared unitary FDN late room, the LFE
low-pass and the 961-sample latency policy.  The compiled HRTF field, the
filterbank tables and the room constants are uploaded once; per 512-sample
block the adapter updates every source and streams PCM through the DLL.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

from native_renderer import ABI_VERSION, find_native_library
from public_filterbank import DEFAULT_FILTERBANK_DATA, load_filterbank_tables
from public_room import LateFdnConfig, SharedUnitaryFdn, ShoeboxRoomConfig
from reference_distance import ReferenceDistanceProfileV1
from sofa_binaural_backend import SofaBinauralBackend
from sofa_hrtf_field import (
    DEFAULT_HRTF_CACHE_DIR,
    SofaHrtfField,
    compile_sofa_hrtf,
)

BLOCK_SAMPLES = 512
INPUT_CHANNELS = 16
OUTPUT_CHANNELS = 2
QMF_HOP = 64
LATENCY_SAMPLES = 961

_PROFILE_INDEX = {"near": 0, "mid": 1, "far": 2}


def _room_numbers(fdn_config: LateFdnConfig) -> dict:
    """Derive the FDN delays/feedback with the same arithmetic as the Python room."""
    fdn = SharedUnitaryFdn(fdn_config)
    return {
        "fdn_delays": np.asarray(fdn.delays, dtype=np.uint32),
        "fdn_feedback": np.asarray(fdn.feedback_gain, dtype=np.float64),
        "damping": fdn.damping,
        "output_gain": fdn.output_gain,
        "allpass_delays": np.asarray(
            [diffuser.delay_samples for diffuser in fdn.diffusers], dtype=np.uint32),
        "allpass_gains": np.asarray(fdn_config.allpass_gain, dtype=np.float64),
        "tail_samples": fdn.tail_samples,
    }


class NativeSofaBinauralDsp:
    """Duck-type compatible with SofaBinauralBackend for the JOC adapter."""

    def __init__(
            self,
            field: SofaHrtfField,
            *,
            source_count: int = INPUT_CHANNELS,
            default_profile: str = "mid",
            enable_early_reflections: bool = True,
            enable_late_room: bool = True,
            room_config: ShoeboxRoomConfig = ShoeboxRoomConfig(),
            fdn_config: LateFdnConfig | None = None,
            library_path: str | Path | None = None):
        if not isinstance(field, SofaHrtfField):
            raise TypeError("field must be SofaHrtfField")
        self.source_count = int(source_count)
        self.default_profile = ReferenceDistanceProfileV1.validate_profile(default_profile)
        self.enable_early_reflections = bool(enable_early_reflections)
        self.enable_late_room = bool(enable_late_room)
        self.field = field
        if self.source_count != INPUT_CHANNELS:
            raise ValueError(f"native SOFA backend requires {INPUT_CHANNELS} sources")
        if abs(self.field.sample_rate_hz - 48000.0) > 1.0e-9:
            raise ValueError("the native SOFA binaural runtime requires 48 kHz")
        self.room_config = room_config
        self.room_config.validate()
        self.dsp_backend = "native-sofa"
        self.hrtf_input_kind = "field"
        self.hrtf_input_path = None
        self.cache_policy = None

        self.library_path = find_native_library(library_path)
        self._lib = ctypes.CDLL(str(self.library_path))
        self._bind()
        version = int(self._lib.ejoc_abi_version())
        if version != ABI_VERSION:
            raise RuntimeError(
                f"native ABI mismatch: expected {ABI_VERSION}, got {version}")
        self._handle = self._lib.ejoc_sofa_binaural_create()
        if not self._handle:
            raise RuntimeError("native SOFA binaural renderer creation failed")
        try:
            self._configure_kernels()
            self._configure_field()
            self._configure_room(fdn_config)
        except Exception:
            self.close()
            raise

        self.positions = np.zeros((self.source_count, 3), dtype=np.float64)
        self.positions[:, 1] = 1.0
        self.profiles = [self.default_profile] * self.source_count
        self.user_gain = np.ones(self.source_count, dtype=np.float64)
        self.special_lfe = np.zeros(self.source_count, dtype=bool)
        self.parameter_updates = 0
        self.finished = False
        for source in range(self.source_count):
            self.set_source(
                source, self.positions[source], profile=self.default_profile,
                fade=False)

    def _bind(self):
        void_p = ctypes.c_void_p
        f64_p = ctypes.POINTER(ctypes.c_double)
        i16_p = ctypes.POINTER(ctypes.c_int16)
        u32_p = ctypes.POINTER(ctypes.c_uint32)
        self._lib.ejoc_abi_version.argtypes = []
        self._lib.ejoc_abi_version.restype = ctypes.c_uint32
        self._lib.ejoc_sofa_binaural_create.argtypes = []
        self._lib.ejoc_sofa_binaural_create.restype = void_p
        self._lib.ejoc_sofa_binaural_destroy.argtypes = [void_p]
        self._lib.ejoc_sofa_binaural_destroy.restype = None
        self._lib.ejoc_sofa_binaural_reset.argtypes = [void_p]
        self._lib.ejoc_sofa_binaural_reset.restype = ctypes.c_int
        self._lib.ejoc_sofa_binaural_last_error.argtypes = [void_p]
        self._lib.ejoc_sofa_binaural_last_error.restype = ctypes.c_char_p
        self._lib.ejoc_sofa_binaural_configure_kernels.argtypes = [
            void_p, f64_p, f64_p, i16_p, f64_p, ctypes.c_uint32, f64_p, f64_p]
        self._lib.ejoc_sofa_binaural_configure_kernels.restype = ctypes.c_int
        self._lib.ejoc_sofa_binaural_configure_field.argtypes = [
            void_p, f64_p, f64_p, f64_p, f64_p, ctypes.c_double]
        self._lib.ejoc_sofa_binaural_configure_field.restype = ctypes.c_int
        self._lib.ejoc_sofa_binaural_configure_room.argtypes = [
            void_p, f64_p, f64_p, f64_p, ctypes.c_double, u32_p, f64_p,
            ctypes.c_double, ctypes.c_double, u32_p, f64_p,
            ctypes.c_uint32, ctypes.c_uint32]
        self._lib.ejoc_sofa_binaural_configure_room.restype = ctypes.c_int
        self._lib.ejoc_sofa_binaural_set_source.argtypes = [
            void_p, ctypes.c_uint32, f64_p, ctypes.c_uint32, ctypes.c_double,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.ejoc_sofa_binaural_set_source.restype = ctypes.c_int
        self._lib.ejoc_sofa_binaural_process.argtypes = [
            void_p, f64_p, ctypes.c_uint32, ctypes.c_double, f64_p]
        self._lib.ejoc_sofa_binaural_process.restype = ctypes.c_int
        self._lib.ejoc_sofa_binaural_finish.argtypes = [
            void_p, ctypes.c_uint32, f64_p, ctypes.c_uint32]
        self._lib.ejoc_sofa_binaural_finish.restype = ctypes.c_int

    def _raise(self, operation, status):
        message = self._lib.ejoc_sofa_binaural_last_error(self._handle)
        detail = (message or b"").decode("utf-8", "replace")
        raise RuntimeError(
            f"native SOFA binaural renderer {operation} failed ({status}): {detail}")

    @staticmethod
    def _f64_pointer(values):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def _configure_kernels(self):
        tables = load_filterbank_tables(DEFAULT_FILTERBANK_DATA)
        qmf_analysis = np.ascontiguousarray(
            tables["qmf_analysis_coefficients"], dtype=np.float64)
        hybrid_low = np.ascontiguousarray(
            tables["hybrid_analysis_low_kernel"], dtype=np.float64)
        hybrid_indices = np.ascontiguousarray(
            tables["hybrid_synthesis_indices"], dtype=np.int16)
        hybrid_values = np.ascontiguousarray(
            tables["hybrid_synthesis_values"], dtype=np.float64)
        qmf_basis = np.ascontiguousarray(
            tables["qmf_synthesis_basis"], dtype=np.float64)
        qmf_taps = np.ascontiguousarray(
            tables["qmf_synthesis_taps"], dtype=np.float64)
        status = self._lib.ejoc_sofa_binaural_configure_kernels(
            self._handle,
            self._f64_pointer(qmf_analysis),
            self._f64_pointer(hybrid_low),
            hybrid_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            self._f64_pointer(hybrid_values),
            len(hybrid_indices),
            self._f64_pointer(qmf_basis),
            self._f64_pointer(qmf_taps))
        if status:
            self._raise("configure_kernels", status)
        self._keepalive = (qmf_analysis, hybrid_low, hybrid_indices,
                           hybrid_values, qmf_basis, qmf_taps)

    def _configure_field(self):
        coefficients = np.ascontiguousarray(
            self.field.coefficients, dtype=np.complex128).view(np.float64)
        delay_coefficients = np.ascontiguousarray(
            self.field.delay_coefficients, dtype=np.float64)
        delay_bounds = np.ascontiguousarray(
            self.field.delay_bounds, dtype=np.float64)
        centers = np.ascontiguousarray(
            self.field.band_center_frequencies_hz, dtype=np.float64)
        status = self._lib.ejoc_sofa_binaural_configure_field(
            self._handle,
            self._f64_pointer(coefficients),
            self._f64_pointer(delay_coefficients),
            self._f64_pointer(delay_bounds),
            self._f64_pointer(centers),
            float(self.field.measurement_radius_m))
        if status:
            self._raise("configure_field", status)

    def _configure_room(self, fdn_config: LateFdnConfig | None):
        actual = fdn_config or LateFdnConfig(sample_rate_hz=48000.0)
        numbers = _room_numbers(actual)
        dims = np.asarray(self.room_config.dimensions_m, dtype=np.float64)
        listener = np.asarray(self.room_config.listener_position_m, dtype=np.float64)
        walls = np.asarray(self.room_config.wall_reflection_gain, dtype=np.float64)
        status = self._lib.ejoc_sofa_binaural_configure_room(
            self._handle,
            self._f64_pointer(dims),
            self._f64_pointer(listener),
            self._f64_pointer(walls),
            float(self.room_config.speed_of_sound_m_s),
            numbers["fdn_delays"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self._f64_pointer(numbers["fdn_feedback"]),
            float(numbers["damping"]),
            float(numbers["output_gain"]),
            numbers["allpass_delays"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self._f64_pointer(numbers["allpass_gains"]),
            1 if self.enable_early_reflections else 0,
            1 if self.enable_late_room else 0)
        if status:
            self._raise("configure_room", status)
        self._fdn = SharedUnitaryFdn(actual)
        self._fdn_tail_samples = numbers["tail_samples"]

    def set_source(self, source: int, position_adm, *, profile: str | None = None,
                   gain: float = 1.0, enabled: bool = True,
                   special_lfe: bool = False, fade: bool = True) -> None:
        if self.finished:
            raise RuntimeError("SOFA renderer is finished")
        source = int(source)
        if not 0 <= source < self.source_count:
            raise IndexError(source)
        name = self.default_profile if profile is None else profile
        position = np.asarray(position_adm, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("ADM position must contain three Cartesian values")
        status = self._lib.ejoc_sofa_binaural_set_source(
            self._handle, source,
            self._f64_pointer(np.ascontiguousarray(position)),
            _PROFILE_INDEX[ReferenceDistanceProfileV1.validate_profile(name)],
            float(gain), 1 if enabled else 0, 1 if special_lfe else 0,
            1 if fade else 0)
        if status:
            self._raise("set_source", status)
        self.positions[source] = position
        self.profiles[source] = name
        self.user_gain[source] = float(gain)
        self.special_lfe[source] = bool(special_lfe)
        self.parameter_updates += 1

    def process(self, sources) -> np.ndarray:
        if self.finished:
            raise RuntimeError("SOFA renderer is finished")
        values = np.ascontiguousarray(sources, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.source_count:
            raise ValueError(f"sources must have shape [samples,{self.source_count}]")
        if len(values) % QMF_HOP or len(values) > BLOCK_SAMPLES:
            raise ValueError("native SOFA backend input must be a 64-aligned block")
        output = np.empty((len(values), OUTPUT_CHANNELS), dtype=np.float64)
        count = self._lib.ejoc_sofa_binaural_process(
            self._handle, self._f64_pointer(values), len(values), 1.0,
            self._f64_pointer(output))
        if count < 0:
            self._raise("process", count)
        return output[:count]

    def finish(self, *, tail_seconds: float | None = None) -> np.ndarray:
        if self.finished:
            return np.zeros((0, OUTPUT_CHANNELS), dtype=np.float64)
        if tail_seconds is not None and (
                not math.isfinite(float(tail_seconds)) or float(tail_seconds) < 0.0):
            raise ValueError("tail_seconds must be finite and non-negative")
        flush = self.finish_output_capacity(tail_seconds)
        pieces = []
        remaining = flush
        while remaining > 0:
            chunk = min(BLOCK_SAMPLES, remaining)
            output = np.empty((chunk, OUTPUT_CHANNELS), dtype=np.float64)
            count = self._lib.ejoc_sofa_binaural_finish(
                self._handle, chunk, self._f64_pointer(output), chunk)
            if count < 0:
                self._raise("finish", count)
            pieces.append(output[:count])
            remaining -= chunk
        self.finished = True
        nonempty = [piece for piece in pieces if len(piece)]
        if not nonempty:
            return np.zeros((0, OUTPUT_CHANNELS), dtype=np.float64)
        return np.concatenate(nonempty, axis=0)

    def finish_output_capacity(self, tail_seconds: float | None = None) -> int:
        if tail_seconds is not None and (
                not math.isfinite(float(tail_seconds)) or float(tail_seconds) < 0.0):
            raise ValueError("tail_seconds must be finite and non-negative")
        requested = (self._fdn_tail_samples if tail_seconds is None
                     else int(math.ceil(float(tail_seconds) * 48000.0)))
        maximum_hrtf = float(np.max(self.field.delay_bounds[:, 1], initial=0.0))
        hrtf_slots = int(math.ceil(maximum_hrtf / QMF_HOP))
        hrtf_bound = hrtf_slots * QMF_HOP
        early_bound = hrtf_bound + 2048
        if self.enable_early_reflections:
            early_bound += 256 * QMF_HOP
        drain = max(requested if self.enable_late_room else 0, early_bound)
        drain += LATENCY_SAMPLES
        return int(math.ceil(drain / QMF_HOP) * QMF_HOP)

    def reset(self) -> None:
        if self._lib.ejoc_sofa_binaural_reset(self._handle):
            self._raise("reset", -1)
        self.finished = False
        for source in range(self.source_count):
            self.set_source(
                source, self.positions[source], profile=self.profiles[source],
                gain=float(self.user_gain[source]),
                special_lfe=bool(self.special_lfe[source]), fade=False)

    def info(self) -> dict:
        return {
            "name": "NativeSofaBinauralDsp",
            "source_count": self.source_count,
            "sample_rate_hz": 48000.0,
            "precision": "float64/complex128",
            "signal_path": (
                "native 64-QMF -> native 77-hybrid -> SOFA order-5 real-SH "
                "direct/early -> native synthesis + shared unitary FDN"),
            "hrtf_input_kind": self.hrtf_input_kind,
            "hrtf_input_path": self.hrtf_input_path,
            "cache_policy": self.cache_policy,
            "latency_compensated_samples": LATENCY_SAMPLES,
            "enable_early_reflections": self.enable_early_reflections,
            "enable_late_room": self.enable_late_room,
            "early_history_slots": 256,
            "hrtf_history_slots": int(math.ceil(
                float(np.max(self.field.delay_bounds[:, 1], initial=0.0)) / QMF_HOP)),
            "maximum_hrtf_delay_samples": float(
                np.max(self.field.delay_bounds[:, 1], initial=0.0)),
            "parameter_updates": self.parameter_updates,
            "distance": ReferenceDistanceProfileV1.info(),
            "field": self.field.info(),
            "library_path": str(self.library_path),
            "native_backend": True,
        }

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.ejoc_sofa_binaural_destroy(handle)
            self._handle = None
        self.finished = True


def create_native_sofa_renderer(
        sofa, *, mode="mid", cache_policy="memory", cache_dir=None,
        shell_radius_m=1.0, object_delay_samples=1473, tail_seconds=5.0,
        output_gain=1.0, chunk_frames=64):
    """Compile a SOFA source and build a JOC adapter over the native DSP."""
    from binaural_renderer import SofaBinauralRenderer, resolve_sofa_hrtf

    source = resolve_sofa_hrtf(sofa)
    field = compile_sofa_hrtf(
        source,
        shell_radius_m=shell_radius_m,
        cache_policy=cache_policy,
        cache_dir=cache_dir)
    backend = NativeSofaBinauralDsp(field, default_profile=mode)
    backend.hrtf_input_kind = "sofa"
    backend.hrtf_input_path = str(source)
    backend.cache_policy = str(cache_policy).lower()
    return SofaBinauralRenderer(
        backend, mode=mode, object_delay_samples=object_delay_samples,
        tail_seconds=tail_seconds, chunk_frames=chunk_frames)


def create_native_compiled_cache_renderer(
        cache, *, mode="mid", object_delay_samples=1473, tail_seconds=5.0,
        output_gain=1.0, chunk_frames=64):
    """Load a compiled cache and build a JOC adapter over the native DSP."""
    from binaural_renderer import SofaBinauralRenderer, resolve_compiled_hrtf_cache

    source = resolve_compiled_hrtf_cache(cache)
    field = SofaHrtfField.load(source)
    backend = NativeSofaBinauralDsp(field, default_profile=mode)
    backend.hrtf_input_kind = "compiled_cache"
    backend.hrtf_input_path = str(source)
    backend.cache_policy = None
    return SofaBinauralRenderer(
        backend, mode=mode, object_delay_samples=object_delay_samples,
        tail_seconds=tail_seconds, chunk_frames=chunk_frames)
