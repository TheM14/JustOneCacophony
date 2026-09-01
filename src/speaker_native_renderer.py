"""ctypes bridge for the float64 native object-to-speaker renderer."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from native_renderer import ABI_VERSION, find_native_library
from oamd_bits import JocFieldState, frame_update
from speaker_layouts import SpeakerLayout, get_speaker_layout

FRAME_SAMPLES = 1536
INPUT_CHANNELS = 16
OBJECTS = 15
COORDINATES = 3


class NativeSpeakerRenderer:
    def __init__(self, layout: str | SpeakerLayout, library_path=None):
        self.layout = get_speaker_layout(layout) if isinstance(layout, str) else layout
        self.library_path = find_native_library(library_path)
        self._lib = ctypes.CDLL(str(self.library_path))
        self._bind()
        version = int(self._lib.ejoc_abi_version())
        if version != ABI_VERSION:
            raise RuntimeError(f"native ABI mismatch: expected {ABI_VERSION}, got {version}")
        channels = int(self._lib.ejoc_speaker_layout_channel_count(self.layout.speaker_bitfield))
        if channels != self.layout.channel_count:
            raise RuntimeError(
                f"native layout channel count mismatch: expected {self.layout.channel_count}, got {channels}"
            )
        self._handle = self._lib.ejoc_speaker_renderer_create(self.layout.speaker_bitfield)
        if not self._handle:
            raise RuntimeError(f"native speaker renderer rejected mask 0x{self.layout.speaker_bitfield:X}")
        self._state = JocFieldState()

    def _bind(self):
        void_p = ctypes.c_void_p
        self._lib.ejoc_abi_version.argtypes = []
        self._lib.ejoc_abi_version.restype = ctypes.c_uint32
        self._lib.ejoc_speaker_layout_channel_count.argtypes = [ctypes.c_uint32]
        self._lib.ejoc_speaker_layout_channel_count.restype = ctypes.c_uint32
        self._lib.ejoc_speaker_renderer_create.argtypes = [ctypes.c_uint32]
        self._lib.ejoc_speaker_renderer_create.restype = void_p
        self._lib.ejoc_speaker_renderer_destroy.argtypes = [void_p]
        self._lib.ejoc_speaker_renderer_destroy.restype = None
        self._lib.ejoc_speaker_renderer_reset.argtypes = [void_p]
        self._lib.ejoc_speaker_renderer_reset.restype = ctypes.c_int
        self._lib.ejoc_speaker_renderer_last_error.argtypes = [void_p]
        self._lib.ejoc_speaker_renderer_last_error.restype = ctypes.c_char_p
        self._lib.ejoc_speaker_renderer_process.argtypes = [
            void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.ejoc_speaker_renderer_process.restype = ctypes.c_int

    def _raise(self, operation, status):
        message = self._lib.ejoc_speaker_renderer_last_error(self._handle)
        detail = (message or b"").decode("utf-8", "replace")
        raise RuntimeError(f"native speaker renderer {operation} failed ({status}): {detail}")

    def reset(self):
        if not self._handle:
            raise RuntimeError("native speaker renderer is closed")
        status = self._lib.ejoc_speaker_renderer_reset(self._handle)
        if status:
            self._raise("reset", status)
        self._state = JocFieldState()

    def close(self):
        if self._handle:
            self._lib.ejoc_speaker_renderer_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _metadata_arrays(self, events):
        events = list(events)
        count = len(events)
        if not count:
            return count, None, None, None
        offsets = np.empty(count, dtype=np.uint32)
        ramps = np.empty(count, dtype=np.uint32)
        positions = np.empty((count, OBJECTS, COORDINATES), dtype=np.uint16)
        for event_index, (outer_offset, payload) in enumerate(events):
            update = frame_update(payload)
            self._state.apply(update["values"])
            offsets[event_index] = int(outer_offset) + int(update["block_offset_samples"])
            ramps[event_index] = int(update["ramp_duration_samples"])
            for object_index in range(1, OBJECTS + 1):
                positions[event_index, object_index - 1] = (
                    self._state.q[(object_index, "q1")],
                    self._state.q[(object_index, "q2")],
                    self._state.q[(object_index, "q3")],
                )
        return count, offsets, ramps, positions

    def process(self, objects16, events=()):
        if not self._handle:
            raise RuntimeError("native speaker renderer is closed")
        source = np.ascontiguousarray(objects16, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"objects16 must have shape [samples,16], got {source.shape}")
        if len(source) % 32:
            raise ValueError("sample count must be a multiple of 32")
        count, offsets, ramps, positions = self._metadata_arrays(events)
        output = np.empty((len(source), self.layout.channel_count), dtype=np.float64)
        null_u32 = ctypes.POINTER(ctypes.c_uint32)()
        null_u16 = ctypes.POINTER(ctypes.c_uint16)()
        null_u8 = ctypes.POINTER(ctypes.c_uint8)()
        null_f64 = ctypes.POINTER(ctypes.c_double)()
        status = self._lib.ejoc_speaker_renderer_process(
            self._handle,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(source),
            count,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if count else null_u32,
            ramps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if count else null_u32,
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)) if count else null_u16,
            null_u8,
            null_u8,
            null_f64,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status:
            self._raise("process", status)
        return output

    def render_frame(self, objects16, payload=None, metadata_offset=1473):
        source = np.asarray(objects16)
        if source.shape != (FRAME_SAMPLES, INPUT_CHANNELS):
            raise ValueError(f"frame must have shape (1536,16), got {source.shape}")
        events = () if payload is None else ((int(metadata_offset), bytes(payload)),)
        return self.process(source, events)
