"""JOC frame adapter for the public SOFA binaural backend."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from binaural_metadata import OamdPositionTimeline
from public_filterbank import ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
from sofa_binaural_backend import SofaBinauralBackend
from sofa_hrtf_field import (
    DEFAULT_HRTF_CACHE_DIR,
    DEFAULT_PROJECTION_RIDGE,
    DEFAULT_SH_RIDGE,
)


SAMPLE_RATE = 48000
FRAME_SAMPLES = 1536
BINAURAL_BLOCK_SAMPLES = 512
QMF_HOP_SAMPLES = 64
BINAURAL_LATENCY_SAMPLES = ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
SOURCE_CHANNELS = 16
OUTPUT_CHANNELS = 2
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HRTF_DIR = PROJECT_DIR / "HRTF"
DEFAULT_SOFA_HRTF = DEFAULT_HRTF_DIR / "binaural.sofa"


def _resolve_hrtf_file(path: str | Path, suffix: str, label: str) -> Path:
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != suffix:
        raise ValueError(f"{label} must use the {suffix} extension: {target}")
    if not target.is_file():
        raise FileNotFoundError(f"{label} not found: {target}")
    return target


def resolve_sofa_hrtf(path: str | Path) -> Path:
    """Resolve an explicitly selected public SOFA source."""
    return _resolve_hrtf_file(path, ".sofa", "SOFA HRTF")


def resolve_compiled_hrtf_cache(path: str | Path) -> Path:
    """Resolve an explicitly selected JOC compiled HRTF cache."""
    return _resolve_hrtf_file(path, ".jochrtf", "compiled HRTF cache")


class SofaBinauralRenderer:
    """Render interleaved LFE plus fifteen JOC objects to stereo.

    The adapter owns frame buffering and sample-timed OAMD updates.  The
    backend owns the 64-QMF/77-hybrid state, the 961-sample latency policy,
    per-object direct/early state, and the shared late room.
    """

    def __init__(
            self, backend, *,
            mode: str = "mid",
            object_delay_samples: int = 1473,
            tail_seconds: float = 5.0,
            chunk_frames: int = 64):
        required_interface = (
            "source_count", "default_profile", "set_source", "process",
            "finish", "finish_output_capacity", "info")
        missing = [name for name in required_interface if not hasattr(backend, name)]
        if missing:
            raise TypeError(
                f"backend must implement the binaural backend interface; "
                f"missing: {', '.join(missing)}")
        if backend.source_count != SOURCE_CHANNELS:
            raise ValueError(f"JOC binaural backend must have {SOURCE_CHANNELS} sources")
        if backend.default_profile != str(mode).lower():
            raise ValueError("backend default profile does not match renderer mode")
        if int(object_delay_samples) < 0:
            raise ValueError("object_delay_samples must be non-negative")
        if not math.isfinite(float(tail_seconds)) or float(tail_seconds) < 0.0:
            raise ValueError("tail_seconds must be finite and non-negative")
        if int(chunk_frames) <= 0:
            raise ValueError("chunk_frames must be positive")

        self.backend = backend
        self.mode = str(mode).lower()
        self.object_delay_samples = int(object_delay_samples)
        self.tail_seconds = float(tail_seconds)
        self.chunk_frames = int(chunk_frames)
        self.chunk_samples = self.chunk_frames * FRAME_SAMPLES
        self.dsp_backend = getattr(backend, "dsp_backend", "python-sofa")
        self.timeline = OamdPositionTimeline(15)

        self._input_buffer = np.empty(
            (self.chunk_samples, SOURCE_CHANNELS), dtype=np.float64)
        self._buffer_used = 0
        self.input_samples = 0
        self.processed_input_samples = 0
        self.output_samples = 0
        self.finished = False
        self.metadata_block_updates = 0

    @classmethod
    def from_sofa(
            cls, sofa: str | Path, *,
            mode: str = "mid",
            cache_policy: str = "memory",
            cache_dir: str | Path | None = DEFAULT_HRTF_CACHE_DIR,
            shell_radius_m: float = 1.0,
            projection_ridge: float = DEFAULT_PROJECTION_RIDGE,
            sh_ridge: float = DEFAULT_SH_RIDGE,
            object_delay_samples: int = 1473,
            tail_seconds: float = 5.0,
            output_gain: float = 1.0,
            chunk_frames: int = 64) -> "SofaBinauralRenderer":
        source = resolve_sofa_hrtf(sofa)
        backend = SofaBinauralBackend.from_sofa(
            source,
            source_count=SOURCE_CHANNELS,
            default_profile=mode,
            output_gain=output_gain,
            cache_policy=cache_policy,
            cache_dir=cache_dir,
            shell_radius_m=shell_radius_m,
            projection_ridge=projection_ridge,
            sh_ridge=sh_ridge)
        return cls(
            backend,
            mode=mode,
            object_delay_samples=object_delay_samples,
            tail_seconds=tail_seconds,
            chunk_frames=chunk_frames)

    @classmethod
    def from_compiled_cache(
            cls, cache: str | Path, *,
            mode: str = "mid",
            object_delay_samples: int = 1473,
            tail_seconds: float = 5.0,
            output_gain: float = 1.0,
            chunk_frames: int = 64) -> "SofaBinauralRenderer":
        source = resolve_compiled_hrtf_cache(cache)
        backend = SofaBinauralBackend.from_compiled_cache(
            source,
            source_count=SOURCE_CHANNELS,
            default_profile=mode,
            output_gain=output_gain)
        return cls(
            backend,
            mode=mode,
            object_delay_samples=object_delay_samples,
            tail_seconds=tail_seconds,
            chunk_frames=chunk_frames)

    @property
    def finish_capacity_samples(self) -> int:
        return self.backend.finish_output_capacity(self.tail_seconds)

    def _append_input(self, samples: np.ndarray) -> list[np.ndarray]:
        outputs = []
        source = np.asarray(samples, dtype=np.float64)
        position = 0
        while position < len(source):
            count = min(self.chunk_samples - self._buffer_used, len(source) - position)
            self._input_buffer[self._buffer_used:self._buffer_used + count] = (
                source[position:position + count])
            self._buffer_used += count
            position += count
            if self._buffer_used == self.chunk_samples:
                outputs.append(self._process_samples(self._input_buffer))
                self._buffer_used = 0
        return outputs

    def render_frame(self, objects16, payload=None, metadata_offset=None,
                     *, outer_sample_offset=0) -> np.ndarray:
        """Submit one 1536-sample reconstructed frame and its ID11 payload."""
        if self.finished:
            raise RuntimeError("binaural renderer is already finished")
        source = np.asarray(objects16)
        if source.shape != (FRAME_SAMPLES, SOURCE_CHANNELS):
            raise ValueError(
                f"binaural frame must have shape ({FRAME_SAMPLES},{SOURCE_CHANNELS}), "
                f"got {source.shape}")
        frame_start = self.input_samples
        metadata_delay = (self.object_delay_samples if metadata_offset is None
                          else int(metadata_offset))
        if metadata_delay < 0:
            raise ValueError("metadata_offset must be non-negative")
        if payload is not None:
            self.timeline.submit_payload(
                payload,
                frame_start_sample=frame_start,
                outer_sample_offset=int(outer_sample_offset),
                object_delay_samples=metadata_delay,
                processed_sample=self.processed_input_samples,
            )
            self.metadata_block_updates += 1
        self.input_samples += FRAME_SAMPLES
        chunks = self._append_input(source)
        if not chunks:
            return np.empty((0, OUTPUT_CHANNELS), dtype=np.float64)
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def _set_block_parameters(self, sample: int) -> None:
        positions = self.timeline.positions_at(sample)
        self.backend.set_source(
            0, (0.0, 1.0, 0.0), profile=self.mode,
            special_lfe=True)
        for object_index in range(15):
            self.backend.set_source(
                object_index + 1,
                positions[object_index],
                profile=self.mode)

    def _process_samples(self, source: np.ndarray) -> np.ndarray:
        values = np.asarray(source, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != SOURCE_CHANNELS:
            raise ValueError(f"expected [samples,{SOURCE_CHANNELS}], got {values.shape}")
        if len(values) % BINAURAL_BLOCK_SAMPLES:
            raise ValueError("binaural input must be divisible by 512 samples")
        outputs = []
        block_base = self.processed_input_samples
        for start in range(0, len(values), BINAURAL_BLOCK_SAMPLES):
            sample = block_base + start
            self._set_block_parameters(sample)
            outputs.append(self.backend.process(
                values[start:start + BINAURAL_BLOCK_SAMPLES]))
        self.processed_input_samples += len(values)
        nonempty = [value for value in outputs if len(value)]
        if not nonempty:
            return np.empty((0, OUTPUT_CHANNELS), dtype=np.float64)
        output = np.concatenate(nonempty, axis=0)
        self.output_samples += len(output)
        return output

    def finish(self) -> np.ndarray:
        """Process pending source samples and drain early/late room state once."""
        if self.finished:
            return np.empty((0, OUTPUT_CHANNELS), dtype=np.float64)
        outputs: list[np.ndarray] = []
        if self._buffer_used:
            outputs.append(self._process_samples(
                self._input_buffer[:self._buffer_used]))
            self._buffer_used = 0
        outputs.append(self.backend.finish(tail_seconds=self.tail_seconds))
        self.finished = True
        nonempty = [value for value in outputs if len(value)]
        if not nonempty:
            return np.empty((0, OUTPUT_CHANNELS), dtype=np.float64)
        output = np.concatenate(nonempty, axis=0)
        self.output_samples += len(outputs[-1])
        return output

    def close(self) -> None:
        self.finished = True

    @property
    def backend_info(self) -> dict:
        info = self.backend.info()
        info.update({
            "adapter": "JOC 1536-frame / 512-sample metadata",
            "dsp_backend": self.dsp_backend,
            "mode": self.mode,
            "latency_compensated_samples": BINAURAL_LATENCY_SAMPLES,
            "object_delay_samples": self.object_delay_samples,
            "tail_seconds": self.tail_seconds,
            "metadata_payloads": self.timeline.payload_count,
            "metadata_position_transitions": self.timeline.transition_count,
            "input_samples": self.input_samples,
            "source_samples_processed": self.processed_input_samples,
            "output_samples_before_tail_trim": self.output_samples,
            "thread_safe": False,
        })
        return info
