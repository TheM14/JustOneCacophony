"""Streaming spool and WAV writer for direct speaker-layout output."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
_FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")


class SpeakerPcmSpool:
    """Temporary interleaved float32 store with float64 peak analysis."""

    def __init__(self, path, sample_count, channel_count):
        self.path = Path(path)
        self.sample_count = int(sample_count)
        self.channel_count = int(channel_count)
        self.position = 0
        self.peak = 0.0
        self.clipped_values = 0
        self.values = np.memmap(
            self.path, dtype="<f4", mode="w+",
            shape=(self.sample_count, self.channel_count),
        )

    def write_frame(self, pcm):
        values = np.asarray(pcm, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channel_count:
            raise ValueError(
                f"speaker frame must have shape [samples,{self.channel_count}], got {values.shape}")
        if self.position + len(values) > self.sample_count:
            raise ValueError("speaker spool received more samples than allocated")
        if not np.all(np.isfinite(values)):
            raise ValueError("speaker renderer produced NaN or infinity")
        absolute = np.abs(values)
        if absolute.size:
            self.peak = max(self.peak, float(np.max(absolute)))
            self.clipped_values += int(np.count_nonzero(absolute > 1.0))
        self.values[self.position:self.position + len(values)] = values.astype(np.float32)
        self.position += len(values)

    def finalize(self):
        if self.position != self.sample_count:
            raise ValueError(
                f"speaker spool has {self.position} samples, expected {self.sample_count}")
        self.values.flush()
        return self

    def close(self):
        values = self.values
        self.values = None
        del values


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
        raise ValueError(f"unsupported speaker WAV format: {sample_format}")
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
        # RF64 + ds64 + fmt + data.
        file_size = 12 + 36 + 8 + len(fmt) + 8 + data_size
        stream.write(b"RF64")
        stream.write(struct.pack("<I", 0xFFFFFFFF))
        stream.write(b"WAVE")
        stream.write(b"ds64")
        stream.write(struct.pack("<IQQQI", 28, file_size - 8, data_size, sample_count, 0))
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
    scaled = (np.clip(values, -1.0, 1.0) * np.float32(8388607.0)).astype(np.int32)
    unsigned = scaled.reshape(-1).view(np.uint32)
    packed = np.empty((unsigned.size, 3), dtype=np.uint8)
    packed[:, 0] = unsigned & 0xFF
    packed[:, 1] = (unsigned >> 8) & 0xFF
    packed[:, 2] = (unsigned >> 16) & 0xFF
    return packed.tobytes()


def write_speaker_wav(path, pcm, sample_format, *, rate=48000, chunk_samples=262144):
    """Write an interleaved float32 array/memmap as float32 or PCM24 WAV."""
    target = Path(path)
    values = np.asarray(pcm)
    if values.ndim != 2:
        raise ValueError(f"speaker PCM must be 2D, got {values.shape}")
    sample_count, channel_count = values.shape
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        info = _write_header(
            stream, channel_count, sample_count, int(rate), sample_format)
        for start in range(0, sample_count, int(chunk_samples)):
            block = np.asarray(values[start:start + chunk_samples], dtype="<f4")
            if sample_format == "float32":
                stream.write(block.tobytes(order="C"))
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
