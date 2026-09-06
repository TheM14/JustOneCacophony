"""Float64 Rosella table-A room model and overlap-add realization."""
from __future__ import annotations

import numpy as np

from rosella_model import RosellaModel


class _RosellaRoomState:
    """Recursive table-A state used to generate the stable FIR realization."""

    def __init__(self, model: RosellaModel):
        self.model = model
        self.bands = min(64, model.table_a_dimension)
        self.delays = model.table_a_four_integers.astype(np.int32)
        self.capacity = int(np.max(self.delays))
        self.matrix = np.asarray(model.table_a_vector16, dtype=np.float64).reshape(
            4, 4, order="F")
        f8 = np.asarray(model.table_a_filter_8x64_padded, dtype=np.float64).reshape(
            20, 4, 2, 4)
        f4 = np.asarray(model.table_a_filter_4x64_padded, dtype=np.float64).reshape(
            20, 4, 4)
        f16 = np.asarray(model.table_a_filter_16x64_padded, dtype=np.float64).reshape(
            20, 4, 4, 4)
        self.feedback_real = np.empty((self.bands, 4), dtype=np.float64)
        self.feedback_imag = np.empty_like(self.feedback_real)
        self.output_tap = np.empty_like(self.feedback_real)
        self.left_real = np.empty_like(self.feedback_real)
        self.left_imag = np.empty_like(self.feedback_real)
        self.right_real = np.empty_like(self.feedback_real)
        self.right_imag = np.empty_like(self.feedback_real)
        for band in range(self.bands):
            group, lane = divmod(band, 4)
            self.feedback_real[band] = f8[group, :, 0, lane]
            self.feedback_imag[band] = f8[group, :, 1, lane]
            self.output_tap[band] = f4[group, :, lane]
            self.left_real[band] = f16[group, :, 0, lane]
            self.left_imag[band] = f16[group, :, 1, lane]
            self.right_real[band] = f16[group, :, 2, lane]
            self.right_imag[band] = f16[group, :, 3, lane]

        self.allpass_gain = np.asarray(model.table_a_option_values, dtype=np.float64)
        self.allpass_delay = model.table_a_option_ids.astype(np.int32)
        self.allpass_real = [
            np.zeros((int(delay), self.bands), dtype=np.float64)
            for delay in self.allpass_delay
        ]
        self.allpass_imag = [np.zeros_like(value) for value in self.allpass_real]
        self.allpass_position = np.zeros(len(self.allpass_real), dtype=np.int32)
        self.memory_real = np.zeros(
            (self.capacity, self.bands, 4), dtype=np.float64)
        self.memory_imag = np.zeros_like(self.memory_real)
        self.position = 0
        self.extra_fields = np.asarray(
            model.table_a_extra_fields_padded, dtype=np.float64).reshape(-1, 20, 2, 4)
        self.extra_matrices = [
            np.asarray(value, dtype=np.float64).reshape(4, 4, order="F")
            for value in model.table_a_extra_vectors
        ]

    def reset(self):
        for value in self.allpass_real + self.allpass_imag:
            value.fill(0.0)
        self.allpass_position.fill(0)
        self.memory_real.fill(0.0)
        self.memory_imag.fill(0.0)
        self.position = 0

    def process_slot(self, room_send) -> np.ndarray:
        values = np.asarray(room_send, dtype=np.complex128)
        input_real = values[:self.bands].real * 0.70710677
        input_imag = values[:self.bands].imag * 0.70710677
        if float(self.model.table_a_scalar) >= 0.5:
            raise NotImplementedError("alternate Rosella table-A room mode")

        for index, gain in enumerate(self.allpass_gain):
            position = int(self.allpass_position[index])
            previous_real = self.allpass_real[index][position].copy()
            previous_imag = self.allpass_imag[index][position].copy()
            residual_real = input_real - previous_real * gain
            residual_imag = input_imag - previous_imag * gain
            input_real = residual_real * gain + previous_real
            input_imag = residual_imag * gain + previous_imag
            self.allpass_real[index][position] = residual_real
            self.allpass_imag[index][position] = residual_imag
            self.allpass_position[index] = (
                position + 1) % len(self.allpass_real[index])

        branch_real = np.repeat(input_real[:, None], 4, axis=1)
        branch_imag = np.repeat(input_imag[:, None], 4, axis=1)
        delayed_real = np.empty_like(branch_real)
        delayed_imag = np.empty_like(branch_imag)
        for branch, delay in enumerate(self.delays):
            delayed_real[:, branch] = self.memory_real[
                (self.position - int(delay)) % self.capacity, :, branch]
            delayed_imag[:, branch] = self.memory_imag[
                (self.position - int(delay)) % self.capacity, :, branch]
        branch_real += np.einsum(
            "bj,ij->bi", delayed_real, self.matrix,
            dtype=np.float64, optimize=False)
        branch_imag += np.einsum(
            "bj,ij->bi", delayed_imag, self.matrix,
            dtype=np.float64, optimize=False)

        tap_index = (self.position - self.model.table_a_integer) % self.capacity
        tap_real = self.memory_real[tap_index].copy()
        tap_imag = self.memory_imag[tap_index].copy()
        next_real = branch_real * self.feedback_real - branch_imag * self.feedback_imag
        next_imag = branch_imag * self.feedback_real + branch_real * self.feedback_imag
        self.memory_real[self.position] = next_real
        self.memory_imag[self.position] = next_imag
        self.position = (self.position + 1) % self.capacity

        extra_real = np.zeros_like(branch_real)
        extra_imag = np.zeros_like(branch_imag)
        for index, delay in enumerate(self.model.table_a_extra_indices):
            source_real = self.memory_real[
                (self.position - (int(delay) + 1)) % self.capacity]
            source_imag = self.memory_imag[
                (self.position - (int(delay) + 1)) % self.capacity]
            matrix = self.extra_matrices[index]
            mixed_real = np.einsum(
                "bj,ij->bi", source_real, matrix,
                dtype=np.float64, optimize=False)
            mixed_imag = np.einsum(
                "bj,ij->bi", source_imag, matrix,
                dtype=np.float64, optimize=False)
            coefficient_real = np.empty(self.bands, dtype=np.float64)
            coefficient_imag = np.empty(self.bands, dtype=np.float64)
            for band in range(self.bands):
                group, lane = divmod(band, 4)
                coefficient_real[band] = self.extra_fields[index, group, 0, lane]
                coefficient_imag[band] = self.extra_fields[index, group, 1, lane]
            extra_real += (mixed_real * coefficient_real[:, None]
                           - mixed_imag * coefficient_imag[:, None])
            extra_imag += (mixed_imag * coefficient_real[:, None]
                           + mixed_real * coefficient_imag[:, None])

        output_real = tap_real * self.output_tap + extra_real
        output_imag = tap_imag * self.output_tap + extra_imag
        left = np.sum(
            self.left_real * output_real - self.left_imag * output_imag,
            axis=1, dtype=np.float64)
        left_imag = np.sum(
            self.left_imag * output_real + self.left_real * output_imag,
            axis=1, dtype=np.float64)
        right = np.sum(
            self.right_real * output_real - self.right_imag * output_imag,
            axis=1, dtype=np.float64)
        right_imag = np.sum(
            self.right_imag * output_real + self.right_real * output_imag,
            axis=1, dtype=np.float64)
        result = np.zeros((2, 77), dtype=np.complex128)
        result[0, :self.bands] = left + 1j * left_imag
        result[1, :self.bands] = right + 1j * right_imag
        return result


class RosellaRoomFir:
    """Complex128 overlap-add room FIR generated locally from table-A."""

    def __init__(self, model: RosellaModel, impulse_slots: int = 4096):
        if impulse_slots <= 0:
            raise ValueError("impulse_slots must be positive")
        reference = _RosellaRoomState(model)
        self.length = int(impulse_slots)
        self.kernel = np.empty((self.length, 2, 64), dtype=np.complex128)
        for slot in range(self.length):
            impulse = np.zeros(77, dtype=np.complex128)
            if slot == 0:
                impulse[:64] = 1.0
            self.kernel[slot] = reference.process_slot(impulse)[:, :64]
        self.tail = np.zeros((self.length - 1, 2, 64), dtype=np.complex128)
        self._fft_cache: dict[int, np.ndarray] = {}

    def reset(self):
        self.tail.fill(0.0)

    def process_chunk(self, room_send) -> np.ndarray:
        values = np.asarray(room_send, dtype=np.complex128)
        if values.ndim != 2 or values.shape[1] != 77:
            raise ValueError("room_send must have shape [slots,77]")
        count = len(values)
        if count == 0:
            return np.zeros((0, 2, 77), dtype=np.complex128)
        needed = count + self.length - 1
        fft_size = 1 << (needed - 1).bit_length()
        kernel_fft = self._fft_cache.get(fft_size)
        if kernel_fft is None:
            kernel_fft = np.fft.fft(self.kernel, fft_size, axis=0)
            self._fft_cache[fft_size] = kernel_fft
        input_fft = np.fft.fft(values[:, :64], fft_size, axis=0)
        block = np.fft.ifft(input_fft[:, None, :] * kernel_fft, axis=0)[:needed]
        block[:len(self.tail)] += self.tail
        result = np.zeros((count, 2, 77), dtype=np.complex128)
        result[:, :, :64] = block[:count]
        self.tail = block[count:count + self.length - 1].copy()
        return result
