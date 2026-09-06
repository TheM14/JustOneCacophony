"""ctypes bridge for the native float64 binaural DSP."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from native_renderer import ABI_VERSION, find_native_library
from rosella_filterbank import DEFAULT_KERNEL_DATA, load_kernel_tables
from rosella_model import RosellaModel

BLOCK_SAMPLES = 512
INPUT_CHANNELS = 16
OUTPUT_CHANNELS = 2
HYBRID_BANDS = 77


class NativeBinauralDsp:
    def __init__(self, model: RosellaModel, *, library_path=None,
                 kernel_data: str | Path = DEFAULT_KERNEL_DATA):
        self.library_path = find_native_library(library_path)
        self._lib = ctypes.CDLL(str(self.library_path))
        self._bind()
        version = int(self._lib.ejoc_abi_version())
        if version != ABI_VERSION:
            raise RuntimeError(
                f"native ABI mismatch: expected {ABI_VERSION}, got {version}")
        self._handle = self._lib.ejoc_binaural_renderer_create()
        if not self._handle:
            raise RuntimeError("native binaural renderer creation failed")
        try:
            self._configure_kernels(kernel_data)
            self._configure_room(model)
        except Exception:
            self.close()
            raise

    def _bind(self):
        void_p = ctypes.c_void_p
        f64_p = ctypes.POINTER(ctypes.c_double)
        i16_p = ctypes.POINTER(ctypes.c_int16)
        u32_p = ctypes.POINTER(ctypes.c_uint32)
        self._lib.ejoc_abi_version.argtypes = []
        self._lib.ejoc_abi_version.restype = ctypes.c_uint32
        self._lib.ejoc_binaural_renderer_create.argtypes = []
        self._lib.ejoc_binaural_renderer_create.restype = void_p
        self._lib.ejoc_binaural_renderer_destroy.argtypes = [void_p]
        self._lib.ejoc_binaural_renderer_destroy.restype = None
        self._lib.ejoc_binaural_renderer_reset.argtypes = [void_p]
        self._lib.ejoc_binaural_renderer_reset.restype = ctypes.c_int
        self._lib.ejoc_binaural_renderer_last_error.argtypes = [void_p]
        self._lib.ejoc_binaural_renderer_last_error.restype = ctypes.c_char_p
        self._lib.ejoc_binaural_renderer_configure_kernels.argtypes = [
            void_p, f64_p, f64_p, i16_p, f64_p, ctypes.c_uint32, f64_p, f64_p]
        self._lib.ejoc_binaural_renderer_configure_kernels.restype = ctypes.c_int
        self._lib.ejoc_binaural_renderer_configure_room.argtypes = [
            void_p, ctypes.c_uint32, ctypes.c_uint32, u32_p, f64_p,
            u32_p, f64_p, ctypes.c_uint32, f64_p, f64_p, f64_p,
            ctypes.c_uint32, u32_p, f64_p, f64_p]
        self._lib.ejoc_binaural_renderer_configure_room.restype = ctypes.c_int
        self._lib.ejoc_binaural_renderer_process.argtypes = [
            void_p, f64_p, f64_p, f64_p, ctypes.c_double, f64_p]
        self._lib.ejoc_binaural_renderer_process.restype = ctypes.c_int

    def _raise(self, operation, status):
        message = self._lib.ejoc_binaural_renderer_last_error(self._handle)
        detail = (message or b"").decode("utf-8", "replace")
        raise RuntimeError(
            f"native binaural renderer {operation} failed ({status}): {detail}")

    @staticmethod
    def _f64_pointer(values):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def _configure_kernels(self, kernel_data):
        tables = load_kernel_tables(kernel_data)
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
        status = self._lib.ejoc_binaural_renderer_configure_kernels(
            self._handle,
            self._f64_pointer(qmf_analysis),
            self._f64_pointer(hybrid_low),
            hybrid_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            self._f64_pointer(hybrid_values),
            len(hybrid_values),
            self._f64_pointer(qmf_basis),
            self._f64_pointer(qmf_taps),
        )
        if status:
            self._raise("configure_kernels", status)

    def _configure_room(self, model: RosellaModel):
        if float(model.table_a_scalar) >= 0.5:
            raise NotImplementedError("alternate table-A room mode")
        bands = min(64, model.table_a_dimension)
        allpass_delays = np.ascontiguousarray(
            model.table_a_option_ids, dtype=np.uint32)
        allpass_gains = np.ascontiguousarray(
            model.table_a_option_values, dtype=np.float64)
        fdn_delays = np.ascontiguousarray(
            model.table_a_four_integers, dtype=np.uint32)
        fdn_matrix = np.ascontiguousarray(
            np.asarray(model.table_a_vector16, dtype=np.float64).reshape(
                4, 4, order="F"))

        filter8 = np.asarray(
            model.table_a_filter_8x64_padded, dtype=np.float64).reshape(20, 4, 2, 4)
        filter4 = np.asarray(
            model.table_a_filter_4x64_padded, dtype=np.float64).reshape(20, 4, 4)
        filter16 = np.asarray(
            model.table_a_filter_16x64_padded, dtype=np.float64).reshape(20, 4, 4, 4)
        feedback = np.empty((64, 4, 2), dtype=np.float64)
        output_taps = np.empty((64, 4), dtype=np.float64)
        output_matrix = np.empty((2, 64, 4, 2), dtype=np.float64)
        for band in range(64):
            group, lane = divmod(band, 4)
            feedback[band, :, 0] = filter8[group, :, 0, lane]
            feedback[band, :, 1] = filter8[group, :, 1, lane]
            output_taps[band] = filter4[group, :, lane]
            output_matrix[0, band, :, 0] = filter16[group, :, 0, lane]
            output_matrix[0, band, :, 1] = filter16[group, :, 1, lane]
            output_matrix[1, band, :, 0] = filter16[group, :, 2, lane]
            output_matrix[1, band, :, 1] = filter16[group, :, 3, lane]

        extra_count = int(model.table_a_extra)
        extra_delays = np.ascontiguousarray(
            model.table_a_extra_indices, dtype=np.uint32)
        extra_fields = np.empty((extra_count, 64, 2), dtype=np.float64)
        extra_source = np.asarray(
            model.table_a_extra_fields_padded, dtype=np.float64).reshape(
                extra_count, 20, 2, 4)
        for extra in range(extra_count):
            for band in range(64):
                group, lane = divmod(band, 4)
                extra_fields[extra, band] = extra_source[extra, group, :, lane]
        extra_matrices = np.empty((extra_count, 4, 4), dtype=np.float64)
        for extra in range(extra_count):
            extra_matrices[extra] = np.asarray(
                model.table_a_extra_vectors[extra], dtype=np.float64).reshape(
                    4, 4, order="F")

        null_u32 = ctypes.POINTER(ctypes.c_uint32)()
        null_f64 = ctypes.POINTER(ctypes.c_double)()
        status = self._lib.ejoc_binaural_renderer_configure_room(
            self._handle,
            bands,
            len(allpass_delays),
            allpass_delays.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self._f64_pointer(allpass_gains),
            fdn_delays.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self._f64_pointer(fdn_matrix),
            int(model.table_a_integer),
            self._f64_pointer(feedback),
            self._f64_pointer(output_taps),
            self._f64_pointer(output_matrix),
            extra_count,
            (extra_delays.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
             if extra_count else null_u32),
            self._f64_pointer(extra_fields) if extra_count else null_f64,
            self._f64_pointer(extra_matrices) if extra_count else null_f64,
        )
        if status:
            self._raise("configure_room", status)

    def reset(self):
        if not self._handle:
            raise RuntimeError("native binaural renderer is closed")
        status = self._lib.ejoc_binaural_renderer_reset(self._handle)
        if status:
            self._raise("reset", status)

    def process_block(self, pcm16, gains, room_sends, output_gain=1.0):
        if not self._handle:
            raise RuntimeError("native binaural renderer is closed")
        source = np.ascontiguousarray(pcm16, dtype=np.float64)
        gain_values = np.asarray(gains)
        sends = np.ascontiguousarray(room_sends, dtype=np.float64)
        if source.shape != (BLOCK_SAMPLES, INPUT_CHANNELS):
            raise ValueError(f"pcm16 block must be (512,16), got {source.shape}")
        if gain_values.shape != (INPUT_CHANNELS, OUTPUT_CHANNELS, HYBRID_BANDS):
            raise ValueError(f"gains must be (16,2,77), got {gain_values.shape}")
        direct = np.ascontiguousarray(
            gain_values, dtype=np.complex128).view(np.float64)
        if sends.shape != (INPUT_CHANNELS,):
            raise ValueError(f"room_sends must be (16,), got {sends.shape}")
        output = np.empty((BLOCK_SAMPLES, OUTPUT_CHANNELS), dtype=np.float64)
        status = self._lib.ejoc_binaural_renderer_process(
            self._handle,
            self._f64_pointer(source),
            self._f64_pointer(direct),
            self._f64_pointer(sends),
            float(output_gain),
            self._f64_pointer(output),
        )
        if status:
            self._raise("process", status)
        return output

    def close(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.ejoc_binaural_renderer_destroy(handle)
            self._handle = None
