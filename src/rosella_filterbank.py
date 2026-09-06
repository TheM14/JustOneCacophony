"""Float64/complex128 Rosella QMF and hybrid filterbanks."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_KERNEL_DATA = PROJECT_DIR / "data" / "rosella_kernels.npz"


@lru_cache(maxsize=4)
def _load_tables(path_string: str) -> dict[str, np.ndarray]:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Rosella kernel data not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        version = archive["format_version"]
        if version.shape != (1,) or int(version[0]) != 1:
            raise ValueError(f"unsupported Rosella kernel data version in {path}")
        return {name: archive[name].copy() for name in archive.files}


def load_kernel_tables(path: str | Path = DEFAULT_KERNEL_DATA) -> dict[str, np.ndarray]:
    """Load and cache the compact, production Rosella kernel tables."""
    return _load_tables(str(Path(path).expanduser().resolve()))


class QmfAnalysis:
    """Batchable 64-band analysis with float64 state and complex128 FFTs."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_kernel_tables(kernel_data)
        self.coefficients = np.asarray(
            tables["qmf_analysis_coefficients"], dtype=np.float64)
        if self.coefficients.shape != (64, 10):
            raise ValueError("invalid qmf_analysis_coefficients shape")
        self.channels = int(channels)
        self.history = np.zeros((9, self.channels, 64), dtype=np.float64)
        phase = np.arange(64, dtype=np.float64)
        self.premod = np.exp(-1j * np.pi * phase / 128.0).astype(np.complex128)
        self.post = np.exp(
            -1j * 3.0 * (np.arange(64, dtype=np.float64) + 0.5) * np.pi / 128.0
        ).astype(np.complex128)
        self.even_post = (
            1j * ((-1.0) ** np.arange(64, dtype=np.float64))
        ).astype(np.complex128)

    def reset(self):
        self.history.fill(0.0)

    def process_chunk(self, hops) -> np.ndarray:
        values = np.asarray(hops, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 64):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        count = values.shape[0]
        joined = np.concatenate((self.history, values), axis=0)
        even = np.zeros_like(values)
        odd = np.zeros_like(values)
        for lag in range(10):
            source = joined[9 - lag:9 - lag + count]
            target = even if lag % 2 == 0 else odd
            target += source * self.coefficients[:, lag][None, None, :]
        self.history[:] = joined[-9:]

        def transform(block):
            prepared = block.astype(np.complex128, copy=False) * self.premod
            transformed = np.fft.fft(prepared, n=128, axis=-1)[..., :64]
            return transformed * self.post

        return np.asarray(transform(odd) + transform(even) * self.even_post,
                          dtype=np.complex128)


class HybridAnalysis:
    """Sparse 64-QMF to 77-hybrid analysis in float64/complex128."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_kernel_tables(kernel_data)
        self.low_kernel = np.asarray(
            tables["hybrid_analysis_low_kernel"], dtype=np.float64)
        if self.low_kernel.shape != (3, 2, 13, 16, 2):
            raise ValueError("invalid hybrid_analysis_low_kernel shape")
        self.channels = int(channels)
        self.history = np.zeros((12, self.channels, 3, 2), dtype=np.float64)
        self.high_history = np.zeros(
            (6, self.channels, 61), dtype=np.complex128)

    def reset(self):
        self.history.fill(0.0)
        self.high_history.fill(0.0)

    def process_chunk(self, qmf) -> np.ndarray:
        values = np.asarray(qmf, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 64):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        count = values.shape[0]
        low = np.stack((values[:, :, :3].real, values[:, :, :3].imag), axis=-1)
        joined = np.concatenate((self.history, low), axis=0)
        output = np.zeros((count, self.channels, 77, 2), dtype=np.float64)
        for lag in range(13):
            source = joined[12 - lag:12 - lag + count]
            output[:, :, :16] += np.einsum(
                "tcpi,pibo->tcbo", source, self.low_kernel[:, :, lag],
                dtype=np.float64, optimize=False)
        self.history[:] = joined[-12:]

        high_joined = np.concatenate((self.high_history, values[:, :, 3:]), axis=0)
        high = high_joined[:count]
        output[:, :, 16:, 0] = high.real
        output[:, :, 16:, 1] = high.imag
        self.high_history[:] = high_joined[-6:]
        return np.asarray(output[..., 0] + 1j * output[..., 1], dtype=np.complex128)


class HybridSynthesis:
    """Instantaneous sparse 77-hybrid to 64-QMF synthesis map."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_kernel_tables(kernel_data)
        indices = np.asarray(tables["hybrid_synthesis_indices"], dtype=np.int64)
        values = np.asarray(tables["hybrid_synthesis_values"], dtype=np.float64)
        if indices.ndim != 2 or indices.shape[1] != 4 or len(indices) != len(values):
            raise ValueError("invalid hybrid synthesis sparse table")
        self.mapping = [
            (int(index[0]), int(index[1]), int(index[2]), int(index[3]), float(value))
            for index, value in zip(indices, values)
        ]
        self.channels = int(channels)

    def reset(self):
        return None

    def process_chunk(self, hybrid) -> np.ndarray:
        values = np.asarray(hybrid, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 77):
            raise ValueError(f"expected [slots,{self.channels},77], got {values.shape}")
        source = np.stack((values.real, values.imag), axis=-1)
        output = np.zeros((values.shape[0], self.channels, 64, 2), dtype=np.float64)
        for input_band, input_component, output_band, output_component, gain in self.mapping:
            output[:, :, output_band, output_component] += (
                source[:, :, input_band, input_component] * gain)
        return np.asarray(output[..., 0] + 1j * output[..., 1], dtype=np.complex128)


class QmfSynthesis:
    """Rank-4 64-band synthesis with float64 state and accumulation."""

    def __init__(self, channels: int, kernel_data: str | Path = DEFAULT_KERNEL_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_kernel_tables(kernel_data)
        self.basis = np.asarray(tables["qmf_synthesis_basis"], dtype=np.float64)
        self.taps = np.asarray(tables["qmf_synthesis_taps"], dtype=np.float64)
        if self.basis.shape != (64, 4, 128) or self.taps.shape != (64, 10, 4):
            raise ValueError("invalid QMF synthesis factorization")
        self.channels = int(channels)
        self.rank = 4
        self.history = np.zeros(
            (9, self.channels, 64, self.rank), dtype=np.float64)

    def reset(self):
        self.history.fill(0.0)

    def process_chunk(self, qmf) -> np.ndarray:
        values = np.asarray(qmf, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 64):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        count = values.shape[0]
        flat = np.stack((values.real, values.imag), axis=-1).reshape(
            count * self.channels, 128)
        modulation = self.basis.reshape(64 * self.rank, 128)
        features = (flat @ modulation.T).reshape(
            count, self.channels, 64, self.rank)
        joined = np.concatenate((self.history, features), axis=0)
        output = np.zeros((count, self.channels, 64), dtype=np.float64)
        for lag in range(10):
            output += np.sum(
                joined[9 - lag:9 - lag + count]
                * self.taps[:, lag, :][None, None, :, :],
                axis=-1, dtype=np.float64)
        self.history[:] = joined[-9:]
        return output
