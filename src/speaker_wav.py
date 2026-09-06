"""Shared PCM spool, peak analysis, and WAV writer for direct outputs."""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
_FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")


class PcmSpool:
    """Temporary interleaved PCM store with float64 peak and clipping analysis.

    ``storage_dtype`` controls only the temporary representation.  Speaker
    output keeps its historical float32 spool, while binaural uses float64 so
    precision is reduced only by the selected final WAV format.
    """

    def __init__(self, path, sample_capacity, channel_count, *,
                 expected_samples=None, storage_dtype="<f4",
                 tail_threshold=None):
        self.path = Path(path)
        self.sample_capacity = int(sample_capacity)
        self.sample_count = int(
            self.sample_capacity if expected_samples is None else expected_samples)
        self.expected_samples = (
            None if expected_samples is None else int(expected_samples))
        self.channel_count = int(channel_count)
        self.storage_dtype = np.dtype(storage_dtype)
        self.tail_threshold = (
            None if tail_threshold is None else float(tail_threshold))
        if self.sample_capacity < 0 or self.channel_count <= 0:
            raise ValueError("invalid PCM spool dimensions")
        if self.expected_samples is not None and not (
                0 <= self.expected_samples <= self.sample_capacity):
            raise ValueError("expected_samples exceeds sample_capacity")
        if self.tail_threshold is not None and (
                not math.isfinite(self.tail_threshold) or self.tail_threshold < 0.0):
            raise ValueError("tail_threshold must be finite and non-negative")
        self.position = 0
        self.peak = 0.0
        self.clipped_values = 0
        self.last_above_threshold = -1
        self.kept_samples = None
        self._values = np.memmap(
            self.path, dtype=self.storage_dtype, mode="w+",
            shape=(self.sample_capacity, self.channel_count),
        )

    @property
    def values(self):
        if self._values is None:
            raise RuntimeError("PCM spool is closed")
        length = (self.position if self.kept_samples is None
                  else self.kept_samples)
        return self._values[:length]

    def write_frame(self, pcm):
        values = np.asarray(pcm, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channel_count:
            raise ValueError(
                f"PCM frame must have shape [samples,{self.channel_count}], got {values.shape}")
        if self.position + len(values) > self.sample_capacity:
            raise ValueError("PCM spool received more samples than allocated")
        if not np.all(np.isfinite(values)):
            raise ValueError("renderer produced NaN or infinity")
        absolute = np.abs(values)
        if absolute.size:
            self.peak = max(self.peak, float(np.max(absolute)))
            self.clipped_values += int(np.count_nonzero(absolute > 1.0))
            if self.tail_threshold is not None:
                per_sample = np.max(absolute, axis=1)
                above = np.flatnonzero(per_sample > self.tail_threshold)
                if above.size:
                    self.last_above_threshold = self.position + int(above[-1])
        self._values[self.position:self.position + len(values)] = values.astype(
            self.storage_dtype, copy=False)
        self.position += len(values)

    def finalize(self, *, minimum_samples=0):
        if (self.expected_samples is not None
                and self.position != self.expected_samples):
            raise ValueError(
                f"PCM spool has {self.position} samples, expected {self.expected_samples}")
        keep = self.position
        if self.tail_threshold is not None:
            keep = min(
                self.position,
                max(int(minimum_samples), self.last_above_threshold + 1),
            )
        self.kept_samples = keep
        self.sample_count = keep
        self._values.flush()
        return self

    def close(self):
        values = self._values
        self._values = None
        if values is not None:
            del values


class SpeakerPcmSpool(PcmSpool):
    """Backward-compatible fixed-length float32 speaker spool."""

    def __init__(self, path, sample_count, channel_count):
        super().__init__(
            path, sample_count, channel_count,
            expected_samples=sample_count, storage_dtype="<f4")


class BinauralPcmSpool(PcmSpool):
    """Float64 variable-tail spool for the binaural renderer."""

    def __init__(self, path, sample_capacity, *, tail_threshold=1.0e-8):
        super().__init__(
            path, sample_capacity, 2,
            expected_samples=None, storage_dtype="<f8",
            tail_threshold=tail_threshold)


def _fmt_chunk(channel_count, rate, sample_format):
    if sample_format == "float32":
        bits = 32
        bytes_per_sample = 4
        simple_tag = WAVE_FORMAT_IEEE_FLOAT
        guid = _FLOAT_GUID
    elif sample_format == "int24":
        bits = 24
        bytes_per_sample = 3
        simple_tag = WAVE_FORMAT_PCM
        guid = _PCM_GUID
    else:
        raise ValueError(f"unsupported WAV format: {sample_format}")
    block_align = channel_count * bytes_per_sample
    byte_rate = rate * block_align
    if channel_count <= 2:
        body = struct.pack(
            "<HHIIHH", simple_tag, channel_count, rate,
            byte_rate, block_align, bits)
    else:
        body = (
            struct.pack(
                "<HHIIHHH", WAVE_FORMAT_EXTENSIBLE, channel_count, rate,
                byte_rate, block_align, bits, 22)
            + struct.pack("<HI", bits, 0)
            + guid
        )
    return body, bits, bytes_per_sample, block_align


def _write_header(stream, channel_count, sample_count, rate, sample_format):
    fmt, bits, bytes_per_sample, block_align = _fmt_chunk(
        channel_count, rate, sample_format)
    data_size = sample_count * block_align
    riff_file_size = 12 + 8 + len(fmt) + 8 + data_size
    use_rf64 = riff_file_size - 8 > 0xFFFFFFFF
    if use_rf64:
        file_size = 12 + 36 + 8 + len(fmt) + 8 + data_size
        stream.write(b"RF64")
        stream.write(struct.pack("<I", 0xFFFFFFFF))
        stream.write(b"WAVE")
        stream.write(b"ds64")
        stream.write(struct.pack(
            "<IQQQI", 28, file_size - 8, data_size, sample_count, 0))
    else:
        stream.write(b"RIFF")
        stream.write(struct.pack("<I", riff_file_size - 8))
        stream.write(b"WAVE")
    stream.write(b"fmt ")
    stream.write(struct.pack("<I", len(fmt)))
    stream.write(fmt)
    stream.write(b"data")
    stream.write(struct.pack("<I", 0xFFFFFFFF if use_rf64 else data_size))
    return {
        "format": sample_format,
        "bits_per_sample": bits,
        "bytes_per_sample": bytes_per_sample,
        "block_align": block_align,
        "data_bytes": data_size,
        "rf64": use_rf64,
    }


def _pack_int24(values):
    source = np.asarray(values)
    scaled = (np.clip(source, -1.0, 1.0) * np.float32(8388607.0)).astype(np.int32)
    unsigned = scaled.reshape(-1).view(np.uint32)
    packed = np.empty((unsigned.size, 3), dtype=np.uint8)
    packed[:, 0] = unsigned & 0xFF
    packed[:, 1] = (unsigned >> 8) & 0xFF
    packed[:, 2] = (unsigned >> 16) & 0xFF
    return packed.tobytes()


def write_pcm_wav(path, pcm, sample_format, *, rate=48000,
                  chunk_samples=262144):
    """Write an interleaved array/memmap as float32 or PCM24 WAV."""
    target = Path(path)
    values = np.asarray(pcm)
    if values.ndim != 2:
        raise ValueError(f"PCM must be 2D, got {values.shape}")
    sample_count, channel_count = values.shape
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        info = _write_header(
            stream, channel_count, sample_count, int(rate), sample_format)
        for start in range(0, sample_count, int(chunk_samples)):
            block = np.asarray(values[start:start + chunk_samples])
            if sample_format == "float32":
                stream.write(block.astype("<f4", copy=False).tobytes(order="C"))
            else:
                stream.write(_pack_int24(block))
    info.update({
        "path": str(target.resolve()),
        "sample_rate": int(rate),
        "sample_count": int(sample_count),
        "channel_count": int(channel_count),
        "file_bytes": target.stat().st_size,
    })
    return info


# Existing imports remain valid.
write_speaker_wav = write_pcm_wav
