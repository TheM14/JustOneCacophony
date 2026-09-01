"""ctypes bridge for the dependency-free MSVC C++ JOC DSP core.

Metadata parsing intentionally stays in Python. One C call consumes a complete
1536-sample frame, so Python is not involved in the hot 24-timeslot x 15-object
DSP loops.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

import numpy as np

from evo_unpack import unpack_evolution
from joc_decode import dequantize, diff_decode, parse_joc

FRAME_SAMPLES = 1536
MAX_OBJECTS = 15
MAX_DPOINTS = 2
CORE_CHANNELS = 5
MAX_BANDS = 23
ABI_VERSION = 1


class NativeBackendUnavailable(RuntimeError):
    pass


def native_library_filename():
    if sys.platform == "win32":
        return "eac3joc_core.dll"
    if sys.platform == "darwin":
        return "libeac3joc_core.dylib"
    if sys.platform.startswith("linux"):
        return "libeac3joc_core.so"
    raise NativeBackendUnavailable(f"unsupported native platform: {sys.platform}")


def _candidate_libraries():
    override = os.environ.get("EAC3JOC_NATIVE_LIBRARY")
    if override:
        yield Path(override).expanduser()
    root = Path(__file__).resolve().parent.parent
    yield root / "lib" / native_library_filename()


def find_native_library(explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise NativeBackendUnavailable(f"native library not found: {path}")
        return path
    checked = []
    for item in _candidate_libraries():
        path = item.resolve()
        checked.append(str(path))
        if path.is_file():
            return path
    raise NativeBackendUnavailable("native library not found; checked: " + "; ".join(checked))


def default_native_threads():
    override = os.environ.get("EAC3JOC_NATIVE_THREADS")
    if override is not None:
        value = int(override)
        if value < 1:
            raise ValueError("EAC3JOC_NATIVE_THREADS must be at least 1")
        return min(value, MAX_OBJECTS)
    return 2 if (os.cpu_count() or 1) >= 4 else 1

def _load_library(path):
    lib = ctypes.CDLL(str(path))
    float_p = ctypes.POINTER(ctypes.c_float)
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    double_p = ctypes.POINTER(ctypes.c_double)

    lib.ejoc_abi_version.argtypes = []
    lib.ejoc_abi_version.restype = ctypes.c_uint32
    lib.ejoc_build_info.argtypes = []
    lib.ejoc_build_info.restype = ctypes.c_char_p
    lib.ejoc_renderer_create.argtypes = []
    lib.ejoc_renderer_create.restype = ctypes.c_void_p
    lib.ejoc_renderer_destroy.argtypes = [ctypes.c_void_p]
    lib.ejoc_renderer_destroy.restype = None
    lib.ejoc_renderer_reset.argtypes = [ctypes.c_void_p]
    lib.ejoc_renderer_reset.restype = ctypes.c_int
    lib.ejoc_renderer_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.ejoc_renderer_set_threads.restype = ctypes.c_int
    lib.ejoc_renderer_thread_count.argtypes = [ctypes.c_void_p]
    lib.ejoc_renderer_thread_count.restype = ctypes.c_uint32
    lib.ejoc_renderer_last_error.argtypes = [ctypes.c_void_p]
    lib.ejoc_renderer_last_error.restype = ctypes.c_char_p
    lib.ejoc_renderer_process.argtypes = [
        ctypes.c_void_p,
        float_p,
        float_p,
        ctypes.c_uint32,
        u8_p,
        u8_p,
        u8_p,
        u8_p,
        double_p,
        ctypes.c_double,
        ctypes.c_float,
        ctypes.c_float,
        float_p,
    ]
    lib.ejoc_renderer_process.restype = ctypes.c_int
    abi = int(lib.ejoc_abi_version())
    if abi != ABI_VERSION:
        raise NativeBackendUnavailable(f"native ABI mismatch: library={abi}, Python={ABI_VERSION}")
    return lib


class NativeJocRenderer:
    """Stateful whole-frame native DSP renderer with the Python renderer API shape."""

    def __init__(self, output_scale=1.0, library_path=None, threads=None):
        self.output_scale = np.float32(output_scale)
        if not np.isfinite(self.output_scale):
            raise ValueError("output_scale must be finite")
        self.library_path = find_native_library(library_path)
        self._lib = _load_library(self.library_path)
        self._handle = self._lib.ejoc_renderer_create()
        if not self._handle:
            raise MemoryError("ejoc_renderer_create failed")
        requested_threads = default_native_threads() if threads is None else int(threads)
        if requested_threads < 1:
            raise ValueError("threads must be at least 1")
        result = self._lib.ejoc_renderer_set_threads(self._handle, requested_threads)
        if result:
            self._raise_native("set_threads", result)
        self.threads = int(self._lib.ejoc_renderer_thread_count(self._handle))
        self._n_bands = np.zeros(MAX_OBJECTS, dtype=np.uint8)
        self._n_dpoints = np.zeros(MAX_OBJECTS, dtype=np.uint8)
        self._slope_idx = np.zeros(MAX_OBJECTS, dtype=np.uint8)
        self._offset_ts = np.zeros((MAX_OBJECTS, MAX_DPOINTS), dtype=np.uint8)
        self._dq = np.zeros(
            (MAX_OBJECTS, MAX_DPOINTS, CORE_CHANNELS, MAX_BANDS),
            dtype=np.float64,
        )
        self._output = np.zeros((16, FRAME_SAMPLES), dtype=np.float32)

    @property
    def build_info(self):
        value = self._lib.ejoc_build_info()
        return value.decode("utf-8", "replace") if value else ""

    @staticmethod
    def decode_payload(payload_bytes):
        subs, _ = unpack_evolution(payload_bytes, loose=True)
        return NativeJocRenderer.decode_subpayloads(subs)

    @staticmethod
    def decode_subpayloads(subs):
        if 14 not in subs:
            raise ValueError("EMDF missing ID14/JOC")
        out = parse_joc(subs[14])
        mix_q = diff_decode(out)
        mix_dq = dequantize(out, mix_q)
        return out, mix_q, mix_dq

    def reset(self):
        self._require_open()
        result = self._lib.ejoc_renderer_reset(self._handle)
        if result:
            self._raise_native("reset", result)

    def close(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.ejoc_renderer_destroy(handle)
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

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("native renderer is closed")

    def _raise_native(self, operation, code):
        raw = self._lib.ejoc_renderer_last_error(self._handle)
        detail = raw.decode("utf-8", "replace") if raw else "unknown native error"
        raise RuntimeError(f"native {operation} failed ({code}): {detail}")

    def _pack_frame(self, out, mix_dq):
        if out["n_channels"] != CORE_CHANNELS:
            raise ValueError(f"native core requires 5 JOC channels, got {out['n_channels']}")
        if out["n_objects"] > MAX_OBJECTS:
            raise ValueError(f"native core supports at most 15 objects, got {out['n_objects']}")
        self._n_bands.fill(0)
        self._n_dpoints.fill(0)
        self._slope_idx.fill(0)
        self._offset_ts.fill(0)
        mask = 0
        for object_index, info in enumerate(out["objs"]):
            if not info["present"]:
                continue
            if info["sparse"]:
                raise ValueError("native core does not accept unvalidated Sparse JOC")
            bands = int(info["n_bands"])
            points = int(info["n_dpoints"])
            if bands > MAX_BANDS or points > MAX_DPOINTS:
                raise ValueError(f"native descriptor out of range: bands={bands}, points={points}")
            values = np.asarray(mix_dq[object_index], dtype=np.float64)
            expected = (points, CORE_CHANNELS, bands)
            if values.shape != expected:
                raise ValueError(f"object {object_index} dq shape {values.shape}, expected {expected}")
            mask |= 1 << object_index
            self._n_bands[object_index] = bands
            self._n_dpoints[object_index] = points
            self._slope_idx[object_index] = int(info["slope_idx"])
            offsets = info.get("offset_ts", ())
            self._offset_ts[object_index, :len(offsets)] = offsets
            self._dq[object_index, :points, :, :bands] = values
        return mask

    def render_frame(self, payload_bytes, bed5_pcm, lfe_pcm=None):
        subs, _ = unpack_evolution(payload_bytes, loose=True)
        return self.render_subpayloads(subs, bed5_pcm, lfe_pcm)

    def render_subpayloads(self, subs, bed5_pcm, lfe_pcm=None):
        self._require_open()
        out, _, mix_dq = self.decode_subpayloads(subs)
        object_mask = self._pack_frame(out, mix_dq)
        bed5 = np.ascontiguousarray(bed5_pcm, dtype=np.float32)
        if bed5.shape != (CORE_CHANNELS, FRAME_SAMPLES):
            raise ValueError(f"core PCM shape must be (5,1536), got {bed5.shape}")
        if lfe_pcm is None:
            lfe = None
            lfe_ptr = ctypes.POINTER(ctypes.c_float)()
        else:
            lfe = np.ascontiguousarray(lfe_pcm, dtype=np.float32)
            if lfe.shape != (FRAME_SAMPLES,):
                raise ValueError(f"LFE shape must be (1536,), got {lfe.shape}")
            lfe_ptr = lfe.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        result = self._lib.ejoc_renderer_process(
            self._handle,
            bed5.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            lfe_ptr,
            object_mask,
            self._n_bands.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._n_dpoints.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._slope_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._offset_ts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._dq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            float(out["clipgain"]),
            ctypes.c_float(0.0625),
            ctypes.c_float(self.output_scale),
            self._output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if result:
            self._raise_native("process", result)
        return self._output, None


def native_available(library_path=None):
    try:
        path = find_native_library(library_path)
        lib = _load_library(path)
        return True, str(path), (lib.ejoc_build_info() or b"").decode("utf-8", "replace")
    except Exception as exc:
        return False, None, str(exc)
