"""Uniform partitioned convolution with float64/complex128 state."""
from __future__ import annotations

import math
import numpy as np


class UniformPartitionedConvolver:
    """Overlap-add FIR convolution with shared input history.

    Filter updates crossfade outputs with independent old/new overlap tails.
    """

    def __init__(self, filters, *, block_size: int = 128,
                 max_filter_taps: int | None = None):
        values = np.asarray(filters, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
            raise ValueError("filters must have shape [outputs,taps]")
        if not np.isfinite(values).all():
            raise ValueError("filters must be finite")
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.fft_size = 2 * self.block_size
        self.frequency_bins = self.block_size + 1
        self.outputs = int(values.shape[0])
        self.max_filter_taps = int(max_filter_taps or values.shape[1])
        if self.max_filter_taps < values.shape[1]:
            raise ValueError("max_filter_taps is shorter than the initial filter")
        self.partition_count = int(math.ceil(self.max_filter_taps / self.block_size))
        self.input_history = np.zeros(
            (self.partition_count, self.frequency_bins), dtype=np.complex128)
        self.write_index = -1
        self.processed_blocks = 0
        self.current_spectra = self._partition(values)
        self.current_overlap = np.zeros(
            (self.outputs, self.block_size), dtype=np.float64)
        self.target_spectra: np.ndarray | None = None
        self.target_overlap: np.ndarray | None = None
        self.fade_samples = 0
        self.fade_position = 0

    @property
    def transition_active(self) -> bool:
        return self.target_spectra is not None

    def _partition(self, filters) -> np.ndarray:
        values = np.asarray(filters, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[0] != self.outputs or values.shape[1] > self.max_filter_taps:
            raise ValueError(
                f"filter update must have shape [{self.outputs}, <= {self.max_filter_taps}]")
        if not np.isfinite(values).all():
            raise ValueError("filter update contains non-finite values")
        padded = np.zeros(
            (self.outputs, self.partition_count, self.fft_size), dtype=np.float64)
        for partition in range(self.partition_count):
            start = partition * self.block_size
            stop = min(start + self.block_size, values.shape[1])
            if stop > start:
                padded[:, partition, :stop - start] = values[:, start:stop]
        return np.asarray(np.fft.rfft(padded, axis=-1), dtype=np.complex128)

    def _ordered_input_history(self) -> np.ndarray:
        if self.write_index < 0:
            return np.zeros_like(self.input_history)
        indices = (self.write_index - np.arange(self.partition_count)) % self.partition_count
        return self.input_history[indices]

    def _convolution_block(self, spectra: np.ndarray,
                           overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordered = self._ordered_input_history()
        frequency = np.einsum(
            "pf,opf->of", ordered, spectra, dtype=np.complex128, optimize=False)
        time = np.asarray(np.fft.irfft(frequency, self.fft_size, axis=-1),
                          dtype=np.float64)
        output = time[:, :self.block_size] + overlap
        return output, time[:, self.block_size:].copy()

    def _overlap_for_previous_block(self, spectra: np.ndarray) -> np.ndarray:
        if self.processed_blocks == 0:
            return np.zeros((self.outputs, self.block_size), dtype=np.float64)
        ordered = self._ordered_input_history()
        frequency = np.einsum(
            "pf,opf->of", ordered, spectra, dtype=np.complex128, optimize=False)
        time = np.asarray(np.fft.irfft(frequency, self.fft_size, axis=-1),
                          dtype=np.float64)
        return time[:, self.block_size:].copy()

    def set_filter(self, filters, *, fade_blocks: int = 0) -> None:
        if self.transition_active:
            raise RuntimeError("finish the current filter crossfade before another update")
        target = self._partition(filters)
        fade_blocks = int(fade_blocks)
        if fade_blocks < 0:
            raise ValueError("fade_blocks must be non-negative")
        if fade_blocks == 0:
            self.current_spectra = target
            self.current_overlap = self._overlap_for_previous_block(target)
            return
        self.target_spectra = target
        self.target_overlap = self._overlap_for_previous_block(target)
        self.fade_samples = fade_blocks * self.block_size
        self.fade_position = 0

    def process_block(self, samples) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float64)
        if values.shape != (self.block_size,) or not np.isfinite(values).all():
            raise ValueError(f"samples must have finite shape ({self.block_size},)")
        padded = np.zeros(self.fft_size, dtype=np.float64)
        padded[:self.block_size] = values
        spectrum = np.asarray(np.fft.rfft(padded), dtype=np.complex128)
        self.write_index = (self.write_index + 1) % self.partition_count
        self.input_history[self.write_index] = spectrum

        current, next_current_overlap = self._convolution_block(
            self.current_spectra, self.current_overlap)
        if not self.transition_active:
            self.current_overlap = next_current_overlap
            self.processed_blocks += 1
            return current.T.copy()

        target, next_target_overlap = self._convolution_block(
            self.target_spectra, self.target_overlap)
        sample_indices = self.fade_position + np.arange(
            1, self.block_size + 1, dtype=np.float64)
        amount = np.clip(sample_indices / float(self.fade_samples), 0.0, 1.0)
        output = current * (1.0 - amount[None, :]) + target * amount[None, :]
        self.fade_position += self.block_size
        self.current_overlap = next_current_overlap
        self.target_overlap = next_target_overlap
        if self.fade_position >= self.fade_samples:
            self.current_spectra = self.target_spectra
            self.current_overlap = self.target_overlap
            self.target_spectra = None
            self.target_overlap = None
            self.fade_samples = 0
            self.fade_position = 0
        self.processed_blocks += 1
        return output.T.copy()

    def reset(self) -> None:
        self.input_history.fill(0.0)
        self.current_overlap.fill(0.0)
        if self.target_overlap is not None:
            self.target_overlap.fill(0.0)
        self.write_index = -1
        self.processed_blocks = 0
        self.target_spectra = None
        self.target_overlap = None
        self.fade_samples = 0
        self.fade_position = 0


class MultiSourcePartitionedConvolver:
    """Sum independent per-source convolution outputs."""

    def __init__(self, source_count: int, *, block_size: int,
                 max_filter_taps: int, output_channels: int = 2):
        self.source_count = int(source_count)
        self.block_size = int(block_size)
        self.output_channels = int(output_channels)
        if min(self.source_count, self.block_size, self.output_channels,
               int(max_filter_taps)) <= 0:
            raise ValueError("source_count, block_size, outputs and taps must be positive")
        zero = np.zeros((self.output_channels, int(max_filter_taps)), dtype=np.float64)
        self.convolvers = [
            UniformPartitionedConvolver(
                zero, block_size=self.block_size, max_filter_taps=max_filter_taps)
            for _ in range(self.source_count)
        ]

    def set_filter(self, source: int, filters, *, fade_blocks: int = 0) -> None:
        source = int(source)
        if not 0 <= source < self.source_count:
            raise IndexError(source)
        self.convolvers[source].set_filter(filters, fade_blocks=fade_blocks)

    def process_block(self, sources) -> np.ndarray:
        values = np.asarray(sources, dtype=np.float64)
        if values.shape != (self.block_size, self.source_count):
            raise ValueError(
                f"sources must have shape ({self.block_size},{self.source_count})")
        if not np.isfinite(values).all():
            raise ValueError("sources contain non-finite values")
        output = np.zeros((self.block_size, self.output_channels), dtype=np.float64)
        for source in range(self.source_count):
            output += self.convolvers[source].process_block(values[:, source])
        return output

    def process(self, sources) -> np.ndarray:
        values = np.asarray(sources, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.source_count:
            raise ValueError(f"sources must have shape [samples,{self.source_count}]")
        if len(values) % self.block_size:
            raise ValueError("partitioned input length must be divisible by block_size")
        blocks = [
            self.process_block(values[start:start + self.block_size])
            for start in range(0, len(values), self.block_size)
        ]
        return (np.concatenate(blocks, axis=0) if blocks
                else np.zeros((0, self.output_channels), dtype=np.float64))

    def flush(self, blocks: int | None = None) -> np.ndarray:
        count = (max(conv.partition_count for conv in self.convolvers) + 2
                 if blocks is None else int(blocks))
        if count < 0:
            raise ValueError("flush block count must be non-negative")
        zero = np.zeros((self.block_size, self.source_count), dtype=np.float64)
        return self.process(np.tile(zero, (count, 1)))

    def reset(self) -> None:
        for convolver in self.convolvers:
            convolver.reset()
