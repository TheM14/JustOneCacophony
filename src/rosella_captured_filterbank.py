"""Captured Rosella 64-QMF / 77-hybrid filterbank.

The Rosella headphone renderer selects its analysis and synthesis callbacks
during preparation, after initialization.  The callbacks it actually runs are
this bank: a 640-coefficient QMF prototype, an 11-tap hybrid analysis with a
3-slot high-band delay, and a 448-sample analysis/synthesis latency.  The shared
public bank lives in ``rosella_filterbank.py`` and differs in every one of those
parameters (13-tap hybrid, 6-slot delay, 961-sample latency).

The tables are the float32 values the C++ core compiles in as
``src/hrtf/rosella_filterbank_tables.h``; ``data/rosella_captured_kernels.npz``
holds the same numbers.  State is float64/complex128, like the rest of the
reference implementation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CAPTURED_KERNEL_DATA = PROJECT_DIR / "data" / "rosella_captured_kernels.npz"

QMF_BANDS = 64
HYBRID_BANDS = 77
QMF_TAPS = 10
HYBRID_TAPS = 11
HYBRID_DELAY = 3
HYBRID_LOW_BANDS = 16
ANALYSIS_SYNTHESIS_LATENCY_SAMPLES = 448

# The captured renderer's hybrid band order: destination[k] = low[HYBRID_ORDER[k]].
HYBRID_ORDER = (0, 7, 1, 6, 2, 9, 3, 4, 10, 5, 11, 15, 8, 12, 13, 14)

# Bands 1 and 3 are conjugated after the permutation.
_CONJUGATED_BANDS = (1, 3)


def _tap_sign(tap: int) -> float:
    return 1.0 if (tap // 2) % 2 == 0 else -1.0


@lru_cache(maxsize=4)
def _load_tables(path_string: str) -> dict[str, np.ndarray]:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Rosella captured kernel data not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        version = archive["format_version"]
        if version.shape != (1,) or int(version[0]) != 1:
            raise ValueError(
                f"unsupported Rosella captured kernel data version in {path}")
        prototype = np.asarray(archive["qmf_prototype"], dtype=np.float64)
        rotation = np.asarray(archive["rotation"], dtype=np.complex128)
        hybrid = np.asarray(archive["hybrid"], dtype=np.complex128)
    if prototype.shape != (QMF_BANDS, QMF_TAPS):
        raise ValueError("invalid qmf_prototype shape")
    if rotation.shape != (QMF_BANDS,):
        raise ValueError("invalid rotation shape")
    if hybrid.shape != (2 * HYBRID_TAPS,):
        raise ValueError("invalid hybrid shape")
    return {"qmf_prototype": prototype, "rotation": rotation, "hybrid": hybrid}


def load_kernel_tables(path: str | Path = DEFAULT_CAPTURED_KERNEL_DATA
                       ) -> dict[str, np.ndarray]:
    """Load and cache the captured Rosella kernel tables."""
    return _load_tables(str(Path(path).expanduser().resolve()))


class QmfAnalysis:
    """64-band analysis: 640-coefficient prototype, 9-slot chronological history."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_CAPTURED_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_kernel_tables(kernel_data)
        self.prototype = tables["qmf_prototype"]
        self.rotation = tables["rotation"]
        self.channels = int(channels)
        self.history = np.zeros((QMF_TAPS - 1, self.channels, QMF_BANDS),
                                dtype=np.float64)

    def reset(self):
        self.history.fill(0.0)

    def process_chunk(self, hops) -> np.ndarray:
        values = np.asarray(hops, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (self.channels, QMF_BANDS):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        count = values.shape[0]
        joined = np.concatenate((self.history, values), axis=0)
        even = np.zeros_like(values)
        odd = np.zeros_like(values)
        for tap in range(QMF_TAPS):
            # The window of output i is joined[i:i+10], oldest tap first.
            source = joined[tap:tap + count]
            target = even if tap % 2 == 0 else odd
            target += source * (
                0.5 * self.prototype[:, tap] * _tap_sign(tap))[None, None, :]
        self.history[:] = joined[-(QMF_TAPS - 1):] if count else self.history

        spectrum = (odd + 1j * even) * self.rotation[None, None, :]
        transformed = np.fft.fft(spectrum, axis=-1)
        output = np.empty(values.shape, dtype=np.complex128)
        output[..., 0::2] = transformed[..., :QMF_BANDS // 2] / QMF_BANDS
        output[..., 1::2] = np.conj(
            transformed[..., QMF_BANDS // 2:][..., ::-1]) / QMF_BANDS
        return output


class HybridAnalysis:
    """Sparse 64-QMF to 77-hybrid analysis: 11 taps, 3-slot high-band delay."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_CAPTURED_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_kernel_tables(kernel_data)
        self.low_kernel = tables["hybrid"]
        self.channels = int(channels)
        self.history = np.zeros((HYBRID_TAPS - 1, self.channels, QMF_BANDS),
                                dtype=np.complex128)

    def reset(self):
        self.history.fill(0.0)

    def process_chunk(self, qmf) -> np.ndarray:
        values = np.asarray(qmf, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, QMF_BANDS):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        count = values.shape[0]
        joined = np.concatenate((self.history, values), axis=0)
        output = np.zeros((count, self.channels, HYBRID_BANDS), dtype=np.complex128)
        low = np.zeros((count, self.channels, HYBRID_LOW_BANDS), dtype=np.complex128)
        for band, size in ((0, 8), (1, 4), (2, 4)):
            offset = 0 if band == 0 else HYBRID_TAPS
            spectrum = np.zeros((count, self.channels, size), dtype=np.complex128)
            for tap in range(HYBRID_TAPS):
                source = joined[tap:tap + count, :, band]
                spectrum[:, :, (tap + 1) % size] += source * self.low_kernel[offset + tap]
            base = 0 if band == 0 else 4 + band * 4
            low[:, :, base:base + size] = np.fft.fft(spectrum, axis=-1)
        output[:, :, :HYBRID_LOW_BANDS] = low[:, :, list(HYBRID_ORDER)]
        for band in _CONJUGATED_BANDS:
            output[:, :, band] = np.conj(output[:, :, band])
        # The high bands pass through unchanged, delayed by HYBRID_DELAY slots.
        delayed = joined[HYBRID_TAPS - 1 - HYBRID_DELAY:
                         HYBRID_TAPS - 1 - HYBRID_DELAY + count, :, 3:]
        output[:, :, HYBRID_LOW_BANDS:] = delayed
        self.history[:] = joined[-(HYBRID_TAPS - 1):] if count else self.history
        return output


class HybridSynthesis:
    """Instantaneous 77-hybrid to 64-QMF recombination."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_CAPTURED_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)

    def reset(self):
        return None

    def process_chunk(self, hybrid) -> np.ndarray:
        values = np.asarray(hybrid, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, HYBRID_BANDS):
            raise ValueError(f"expected [slots,{self.channels},77], got {values.shape}")
        output = np.empty((values.shape[0], self.channels, QMF_BANDS),
                          dtype=np.complex128)
        output[..., 0] = (values[..., 0] + values[..., 2] + values[..., 4] +
                          values[..., 6] + values[..., 7] + values[..., 9] +
                          np.conj(values[..., 3]) + np.conj(values[..., 1]))
        output[..., 1] = (values[..., 12] + values[..., 5] + values[..., 8] +
                          values[..., 10])
        output[..., 2] = (values[..., 13] + values[..., 14] + values[..., 15] +
                          values[..., 11])
        output[..., 3:] = values[..., HYBRID_LOW_BANDS:]
        return output


class QmfSynthesis:
    """64-band synthesis: prototype-matched 10-tap overlap with a 9-slot history."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_CAPTURED_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_kernel_tables(kernel_data)
        self.prototype = tables["qmf_prototype"]
        self.rotation = tables["rotation"]
        self.channels = int(channels)
        self.history = np.zeros((QMF_TAPS - 1, self.channels, QMF_BANDS),
                                dtype=np.complex128)

    def reset(self):
        self.history.fill(0.0)

    def process_chunk(self, qmf) -> np.ndarray:
        values = np.asarray(qmf, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, QMF_BANDS):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        count = values.shape[0]
        spectrum = np.empty((count, self.channels, QMF_BANDS), dtype=np.complex128)
        spectrum[..., :QMF_BANDS // 2] = np.conj(values[..., 0::2])
        spectrum[..., QMF_BANDS // 2:] = values[..., QMF_BANDS - 1::-2]
        current = np.fft.fft(spectrum, axis=-1) * self.rotation[None, None, :]
        joined = np.concatenate((self.history, current), axis=0)
        output = np.zeros((count, self.channels, QMF_BANDS), dtype=np.float64)
        reversed_bands = QMF_BANDS - 1 - np.arange(QMF_BANDS)
        for lag in range(QMF_TAPS):
            source = joined[QMF_TAPS - 1 - lag:QMF_TAPS - 1 - lag + count]
            coefficient = (self.prototype[reversed_bands, QMF_TAPS - 1 - lag] *
                           _tap_sign(lag) * 2.0)
            component = source.real if lag % 2 == 0 else source.imag
            output += component * coefficient[None, None, :]
        self.history[:] = joined[-(QMF_TAPS - 1):] if count else self.history
        return output
