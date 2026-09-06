"""Rosella .personalized_headphone binaural renderer.

Rosella JSON 解析由本项目自行实现（src/rosella_model.py），不调用任何 Dolby
软件；.personalized_headphone 是用户经官方软件个性化扫描得到的模型文件。
该路径与 SOFA 路径各自独立完成 HRTF/room 参数求值，只在最外层的 JOC 调度
（1536-sample 帧缓冲、sample-timed OAMD timeline、512-sample 参数更新、输出
包装）处汇合。
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from binaural_metadata import OamdPositionTimeline
from binaural_native_renderer import NativeBinauralDsp
from rosella_core import RosellaRenderer
from rosella_direct import BINAURAL_PROFILE_NAMES
from rosella_filterbank import (
    DEFAULT_KERNEL_DATA,
    HybridAnalysis,
    HybridSynthesis,
    QmfAnalysis,
    QmfSynthesis,
)
from rosella_model import RosellaModel, load_personalized_headphone

SAMPLE_RATE = 48000
FRAME_SAMPLES = 1536
ROSSELLA_BLOCK_SAMPLES = 512
QMF_HOP_SAMPLES = 64
ROSSELLA_LATENCY_SAMPLES = 961
SOURCE_CHANNELS = 16
OUTPUT_CHANNELS = 2
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PERSONALIZED_HEADPHONE = (
    PROJECT_DIR / "HRTF" / "binaural.personalized_headphone")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_personalized_headphone(path: str | Path | None = None) -> Path:
    target = (DEFAULT_PERSONALIZED_HEADPHONE if path is None
              else Path(path).expanduser().resolve())
    if not target.is_file():
        raise FileNotFoundError(
            f"未找到双耳模型：{target}\n"
            "请将兼容模型保存为 HRTF/binaural.personalized_headphone，"
            "或通过参数指定文件。"
        )
    return target


class RosellaBinauralRenderer:
    """Render interleaved LFE plus fifteen objects to stereo."""

    def __init__(
            self,
            personalized_headphone: str | Path | RosellaModel,
            *,
            mode: str = "mid",
            kernel_data: str | Path = DEFAULT_KERNEL_DATA,
            object_delay_samples: int = 1473,
            tail_seconds: float = 5.0,
            output_gain: float = 1.0,
            chunk_frames: int = 64,
            room_impulse_slots: int = 4096,
            backend: str = "python",
            native_library=None):
        if mode not in BINAURAL_PROFILE_NAMES:
            raise ValueError("binaural mode must be near, mid, or far")
        if int(object_delay_samples) < 0:
            raise ValueError("object_delay_samples must be non-negative")
        if float(tail_seconds) < 0.0:
            raise ValueError("tail_seconds must be non-negative")
        if int(chunk_frames) <= 0:
            raise ValueError("chunk_frames must be positive")
        if not math.isfinite(float(output_gain)):
            raise ValueError("output_gain must be finite")
        if backend not in ("auto", "native", "python"):
            raise ValueError("backend must be auto, native, or python")

        if isinstance(personalized_headphone, RosellaModel):
            self.model = personalized_headphone
            self.model_path = Path(self.model.source_path)
        else:
            self.model_path = resolve_personalized_headphone(personalized_headphone)
            self.model = load_personalized_headphone(self.model_path)
        if self.model.sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Rosella model sample rate must be {SAMPLE_RATE}, got {self.model.sample_rate}")

        self.mode = mode
        self.profile_index = BINAURAL_PROFILE_NAMES[mode]
        self.kernel_data = Path(kernel_data).expanduser().resolve()
        self.kernel_data_sha256 = _sha256_file(self.kernel_data)
        self.object_delay_samples = int(object_delay_samples)
        self.tail_seconds = float(tail_seconds)
        self.output_gain = np.float64(output_gain)
        self.chunk_frames = int(chunk_frames)
        self.chunk_samples = self.chunk_frames * FRAME_SAMPLES

        self.native_dsp = None
        self.backend_fallback = None
        if backend in ("auto", "native"):
            try:
                self.native_dsp = NativeBinauralDsp(
                    self.model, library_path=native_library,
                    kernel_data=self.kernel_data)
            except (AttributeError, OSError, RuntimeError) as exc:
                if backend == "native":
                    raise RuntimeError(f"native binaural backend unavailable: {exc}") from exc
                self.backend_fallback = str(exc)
        if self.native_dsp is not None:
            self.dsp_backend = "native"
            self.qmf_analysis = None
            self.hybrid_analysis = None
            self.hybrid_synthesis = None
            self.qmf_synthesis = None
            self.core = RosellaRenderer(
                self.model, SOURCE_CHANNELS, create_room=False)
        else:
            self.dsp_backend = "python"
            self.qmf_analysis = QmfAnalysis(SOURCE_CHANNELS, self.kernel_data)
            self.hybrid_analysis = HybridAnalysis(SOURCE_CHANNELS, self.kernel_data)
            self.core = RosellaRenderer(
                self.model, SOURCE_CHANNELS,
                room_impulse_slots=room_impulse_slots)
            self.hybrid_synthesis = HybridSynthesis(OUTPUT_CHANNELS, self.kernel_data)
            self.qmf_synthesis = QmfSynthesis(OUTPUT_CHANNELS, self.kernel_data)
        self.timeline = OamdPositionTimeline(15)

        self._input_buffer = np.empty(
            (self.chunk_samples, SOURCE_CHANNELS), dtype=np.float64)
        self._buffer_used = 0
        self.input_samples = 0
        self.processed_input_samples = 0
        self.raw_output_samples = 0
        self.output_samples = 0
        self.finished = False
        self.metadata_block_updates = 0

    def _append_input(self, samples: np.ndarray) -> list[np.ndarray]:
        outputs = []
        source = np.asarray(samples, dtype=np.float64)
        position = 0
        while position < len(source):
            count = min(self.chunk_samples - self._buffer_used,
                        len(source) - position)
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

    def _set_block_parameters(self, sample: int):
        positions = self.timeline.positions_at(sample)
        self.core.set_source(0, (0.0, 1.0, 0.0), special_lfe=True)
        for object_index in range(15):
            self.core.set_source(
                object_index + 1, positions[object_index], self.profile_index)

    def _process_samples(self, source: np.ndarray) -> np.ndarray:
        values = np.asarray(source, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != SOURCE_CHANNELS:
            raise ValueError(f"expected [samples,{SOURCE_CHANNELS}], got {values.shape}")
        if len(values) % ROSSELLA_BLOCK_SAMPLES:
            raise ValueError("binaural input must be divisible by 512 samples")
        blocks = len(values) // ROSSELLA_BLOCK_SAMPLES
        block_base = self.processed_input_samples

        if self.native_dsp is not None:
            stereo = np.empty((len(values), OUTPUT_CHANNELS), dtype=np.float64)
            for block in range(blocks):
                sample = block_base + block * ROSSELLA_BLOCK_SAMPLES
                self._set_block_parameters(sample)
                start = block * ROSSELLA_BLOCK_SAMPLES
                stop = start + ROSSELLA_BLOCK_SAMPLES
                stereo[start:stop] = self.native_dsp.process_block(
                    values[start:stop], self.core.gains, self.core.room_sends,
                    self.output_gain)
        else:
            hops = values.reshape(
                blocks, ROSSELLA_BLOCK_SAMPLES // QMF_HOP_SAMPLES,
                QMF_HOP_SAMPLES, SOURCE_CHANNELS,
            ).transpose(0, 1, 3, 2).reshape(
                blocks * (ROSSELLA_BLOCK_SAMPLES // QMF_HOP_SAMPLES),
                SOURCE_CHANNELS, QMF_HOP_SAMPLES)
            hybrid = self.hybrid_analysis.process_chunk(
                self.qmf_analysis.process_chunk(hops))
            direct = np.empty((blocks * 8, OUTPUT_CHANNELS, 77), dtype=np.complex128)
            room_send = np.empty((blocks * 8, 77), dtype=np.complex128)
            for block in range(blocks):
                sample = block_base + block * ROSSELLA_BLOCK_SAMPLES
                self._set_block_parameters(sample)
                start = block * 8
                stop = start + 8
                direct[start:stop], room_send[start:stop] = (
                    self.core.direct_and_send_static(hybrid[start:stop]))
            rendered = direct + self.core.room.process_chunk(room_send)
            time_bands = self.qmf_synthesis.process_chunk(
                self.hybrid_synthesis.process_chunk(rendered))
            stereo = time_bands.transpose(0, 2, 1).reshape(
                blocks * ROSSELLA_BLOCK_SAMPLES, OUTPUT_CHANNELS)
            stereo *= self.output_gain

        skip = max(0, min(
            len(stereo), ROSSELLA_LATENCY_SAMPLES - self.raw_output_samples))
        self.raw_output_samples += len(stereo)
        self.processed_input_samples += len(values)
        output = stereo[skip:]
        self.output_samples += len(output)
        return output

    def finish(self) -> np.ndarray:
        """Process pending source samples and preserve the configured room tail."""
        if self.finished:
            return np.empty((0, OUTPUT_CHANNELS), dtype=np.float64)
        outputs: list[np.ndarray] = []
        if self._buffer_used:
            outputs.append(self._process_samples(
                self._input_buffer[:self._buffer_used]))
            self._buffer_used = 0
        flush_samples = math.ceil(
            (self.tail_seconds * SAMPLE_RATE
             + ROSSELLA_LATENCY_SAMPLES + ROSSELLA_BLOCK_SAMPLES)
            / ROSSELLA_BLOCK_SAMPLES) * ROSSELLA_BLOCK_SAMPLES
        while flush_samples:
            count = min(flush_samples, self.chunk_samples)
            zero = np.zeros((count, SOURCE_CHANNELS), dtype=np.float64)
            outputs.append(self._process_samples(zero))
            flush_samples -= count
        self.finished = True
        nonempty = [value for value in outputs if len(value)]
        if not nonempty:
            return np.empty((0, OUTPUT_CHANNELS), dtype=np.float64)
        return np.concatenate(nonempty, axis=0)

    def close(self):
        if self.native_dsp is not None:
            self.native_dsp.close()
        self.finished = True

    @property
    def backend_info(self) -> dict:
        return {
            "name": self.dsp_backend,
            "precision": "float64/complex128",
            "fallback_reason": self.backend_fallback,
            "library": (str(self.native_dsp.library_path)
                        if self.native_dsp is not None else None),
            "model": str(self.model_path.resolve()),
            "model_coefficients": int(len(self.model.coefficients)),
            "model_coefficient_sha256": self.model.coefficient_sha256,
            "model_version": self.model.coefficient_version,
            "kernel_data": str(self.kernel_data),
            "kernel_data_sha256": self.kernel_data_sha256,
            "mode": self.mode,
            "latency_compensated_samples": ROSSELLA_LATENCY_SAMPLES,
            "object_delay_samples": self.object_delay_samples,
            "tail_seconds": self.tail_seconds,
            "metadata_payloads": self.timeline.payload_count,
            "metadata_position_transitions": self.timeline.transition_count,
            "input_samples": self.input_samples,
            "processed_samples_including_flush": self.processed_input_samples,
            "output_samples_before_tail_trim": self.output_samples,
        }
