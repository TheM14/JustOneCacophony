"""Compile canonical SOFA data into the runtime directional HRTF field."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import zipfile

import numpy as np

from public_filterbank import (
    ANALYSIS_SYNTHESIS_LATENCY_SAMPLES,
    HYBRID_BANDS,
    QMF_HOP,
    PublicAnalysis77,
    PublicSynthesis77,
    filterbank_fingerprint,
    hybrid_band_center_frequencies_hz,
    project_hrir_to_hybrid_gains,
)
from sofa_canonical import (
    CanonicalHrtf,
    load_simple_free_field_hrir,
    time_align_hrtf,
)
from spherical_harmonics import (
    evaluate_real_spherical_harmonics,
    fit_real_spherical_harmonics,
    real_spherical_harmonics,
    spherical_voronoi_weights,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HRTF_CACHE_DIR = PROJECT_DIR / "output" / "hrtf-cache"
JOCHRTF_MAGIC = "JOC-HRTF-CACHE"
JOCHRTF_FORMAT_VERSION = 1
JOCHRTF_CACHE_SCHEMA = "joc-compiled-hrtf-v1"
COMPILER_VERSION = "joc-sofa-compiler-v1"
PHASE_POLICY_VERSION = "sofa-delay-exactly-once-v1"
SH_CONVENTION = "ACN/N3D real"
TARGET_SAMPLE_RATE_HZ = 48000.0
DEFAULT_ORDER = 5
DEFAULT_PROJECTION_RIDGE = 1.0e-3
DEFAULT_SH_RIDGE = 1.0e-5

_ARCHIVE_KEYS = {
    "metadata_json",
    "band_center_frequencies_hz",
    "coefficients",
    "delay_coefficients",
    "delay_bounds",
}
_METADATA_KEYS = {
    "magic",
    "format_version",
    "cache_schema",
    "cache_key_version",
    "compiler_version",
    "phase_policy_version",
    "sh_convention",
    "filterbank",
    "cache_key",
    "payload_sha256",
    "source_sha256",
    "source_display_name",
    "sample_rate_hz",
    "measurement_radius_m",
    "order",
    "projection_ridge",
    "spherical_harmonic_ridge",
    "delay_source",
    "fit_report",
}
_MAX_ARCHIVE_BYTES = 4 << 20
_MAX_METADATA_BYTES = 64 << 10
_MAX_DELAY_COEFFICIENT_ABS = TARGET_SAMPLE_RATE_HZ * 64.0
_MEMORY_CACHE_MAX_ENTRIES = 8
_MEMORY_CACHE: OrderedDict[str, "SofaHrtfField"] = OrderedDict()
_MEMORY_CACHE_LOCK = threading.RLock()


class HrtfCacheError(ValueError):
    """A compiled HRTF cache is damaged, stale, or incompatible."""


def adm_to_sofa_direction(position) -> np.ndarray:
    """ADM +X right,+Y front,+Z up to SOFA +X front,+Y left,+Z up."""
    value = np.asarray(position, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("ADM direction must contain three finite values")
    result = np.asarray([value[1], -value[0], value[2]], dtype=np.float64)
    length = float(np.linalg.norm(result))
    if length <= 1.0e-15:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return result / length


def sofa_to_adm_direction(position) -> np.ndarray:
    value = np.asarray(position, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("SOFA direction must contain three finite values")
    result = np.asarray([-value[1], value[0], value[2]], dtype=np.float64)
    length = float(np.linalg.norm(result))
    if length <= 1.0e-15:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    return result / length


def _group_coincident(directions, gains, delays):
    unit = np.asarray(directions, dtype=np.float64)
    keys = np.round(unit, 10)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    grouped_directions = unit[first]
    grouped_gains = np.zeros((len(first),) + gains.shape[1:], dtype=np.complex128)
    grouped_delays = np.zeros((len(first), 2), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    for measurement, group in enumerate(inverse):
        grouped_gains[group] += gains[measurement]
        grouped_delays[group] += delays[measurement]
    grouped_gains /= counts[:, None, None]
    grouped_delays /= counts[:, None]
    return grouped_directions, grouped_gains, grouped_delays


def _cache_key_payload(*, source_sha256: str, sample_rate_hz: float,
                       shell_radius_m: float, order: int,
                       projection_ridge: float, sh_ridge: float) -> dict:
    return {
        "source_sha256": str(source_sha256).upper(),
        "target_sample_rate_hz": float(sample_rate_hz).hex(),
        "shell_radius_m": float(shell_radius_m).hex(),
        "order": int(order),
        "projection_ridge": float(projection_ridge).hex(),
        "spherical_harmonic_ridge": float(sh_ridge).hex(),
        "compiler_version": COMPILER_VERSION,
        "phase_policy_version": PHASE_POLICY_VERSION,
        "sh_convention": SH_CONVENTION,
        "filterbank": filterbank_fingerprint(),
    }


def compiled_hrtf_cache_key(*, source_sha256: str, sample_rate_hz: float,
                            shell_radius_m: float, order: int,
                            projection_ridge: float,
                            sh_ridge: float) -> str:
    source_sha256 = _validate_sha256(source_sha256)
    order = _strict_integer(order, "compiled HRTF order")
    numeric = (
        float(sample_rate_hz), float(shell_radius_m),
        float(projection_ridge), float(sh_ridge))
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("compiled HRTF cache-key values must be finite")
    if numeric[0] <= 0.0 or numeric[1] <= 0.0:
        raise ValueError("compiled HRTF sample rate and radius must be positive")
    if numeric[2] < 0.0 or numeric[3] < 0.0:
        raise ValueError("compiled HRTF ridge values must be non-negative")
    payload = _cache_key_payload(
        source_sha256=source_sha256,
        sample_rate_hz=sample_rate_hz,
        shell_radius_m=shell_radius_m,
        order=order,
        projection_ridge=projection_ridge,
        sh_ridge=sh_ridge,
    )
    encoded = b"JOC-HRTF-CACHE-KEY-V1\0" + json.dumps(
        payload, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _safe_display_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HrtfCacheError("source_display_name must be a string or null")
    if not value:
        return None
    name = value
    if len(name) > 255 or Path(name).name != name or "/" in name or "\\" in name:
        raise HrtfCacheError("source_display_name must be a plain file name")
    return name


def _validate_sha256(value: str, label: str = "source_sha256") -> str:
    if not isinstance(value, str):
        raise HrtfCacheError(f"{label} must be a hexadecimal string")
    digest = value.upper()
    if re.fullmatch(r"[0-9A-F]{64}", digest) is None:
        raise HrtfCacheError(f"{label} must be a 64-digit hexadecimal digest")
    return digest


def _strict_integer(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _json_number(metadata: dict, label: str) -> float:
    value = metadata[label]
    if isinstance(value, bool) or type(value) not in (int, float):
        raise HrtfCacheError(f"compiled HRTF {label} must be a JSON number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise HrtfCacheError(
            f"compiled HRTF {label} is outside the supported numeric range") from exc
    if not math.isfinite(result):
        raise HrtfCacheError(f"compiled HRTF {label} must be finite")
    return result


def _json_bytes(metadata: dict) -> bytes:
    encoded = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise HrtfCacheError("compiled HRTF metadata is too large")
    return encoded


def _payload_sha256(centers, coefficients, delay_coefficients, delay_bounds) -> str:
    digest = hashlib.sha256(b"JOC-HRTF-CACHE-PAYLOAD-V1\0")
    for name, value, dtype in (
            ("band_center_frequencies_hz", centers, "<f8"),
            ("coefficients", coefficients, "<c16"),
            ("delay_coefficients", delay_coefficients, "<f8"),
            ("delay_bounds", delay_bounds, "<f8")):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def _strict_json_object(text: str) -> dict:
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise HrtfCacheError(f"duplicate compiled HRTF metadata key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise HrtfCacheError(f"non-finite JSON number in compiled HRTF metadata: {value}")

    try:
        value = json.loads(
            text, object_pairs_hook=pairs_hook, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise HrtfCacheError(f"invalid compiled HRTF metadata JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HrtfCacheError("compiled HRTF metadata must be a JSON object")
    return value


def _try_lock_stream(stream) -> bool:
    """Acquire one process-owned advisory lock without unlink races."""
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise


def _unlock_stream(stream) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _cache_write_lock(target: Path, timeout_seconds: float = 30.0):
    # The sidecar intentionally remains on disk.  The OS releases the held
    # lock on process exit, so a crashed writer cannot leave a stale owner or
    # trigger the unlink/recreate ABA race of sentinel-file locks.
    lock = target.with_name("." + target.name + ".lock")
    deadline = time.monotonic() + float(timeout_seconds)
    descriptor = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
    stream = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        while not acquired:
            acquired = _try_lock_stream(stream)
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise HrtfCacheError(
                    f"timed out waiting for cache writer: {target.name}")
            time.sleep(0.05)
        yield
    finally:
        if acquired:
            _unlock_stream(stream)
        stream.close()


def _memory_cache_get(key: str) -> "SofaHrtfField | None":
    with _MEMORY_CACHE_LOCK:
        value = _MEMORY_CACHE.pop(key, None)
        if value is not None:
            _MEMORY_CACHE[key] = value
        return value


def _memory_cache_put(key: str, value: "SofaHrtfField") -> None:
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE.pop(key, None)
        _MEMORY_CACHE[key] = value
        while len(_MEMORY_CACHE) > _MEMORY_CACHE_MAX_ENTRIES:
            _MEMORY_CACHE.popitem(last=False)


def _read_npy_member(payload: bytes, *, name: str) -> np.ndarray:
    """Validate an NPY header before NumPy is allowed to allocate its array."""
    expected = {
        "band_center_frequencies_hz.npy": ("<f8", (HYBRID_BANDS,)),
        "coefficients.npy": ("<c16", (36, 2, HYBRID_BANDS)),
        "delay_coefficients.npy": ("<f8", (36, 2)),
        "delay_bounds.npy": ("<f8", (2, 2)),
    }
    header = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(header)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                header)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                header)
        else:
            raise HrtfCacheError(
                f"compiled HRTF member uses unsupported NPY version {version}: {name}")
    except HrtfCacheError:
        raise
    except (EOFError, OSError, TypeError, ValueError) as exc:
        raise HrtfCacheError(
            f"compiled HRTF member has an invalid NPY header: {name}: {exc}") from exc

    dtype = np.dtype(dtype)
    if fortran_order:
        raise HrtfCacheError(
            f"compiled HRTF member must be C-contiguous: {name}")
    if name == "metadata_json.npy":
        if shape != () or dtype.kind != "U":
            raise HrtfCacheError(
                "compiled HRTF metadata must be a scalar Unicode string")
        if dtype.itemsize <= 0 or dtype.itemsize > 4 * _MAX_METADATA_BYTES:
            raise HrtfCacheError("compiled HRTF metadata is too large")
        payload_bytes = dtype.itemsize
    else:
        dtype_string, expected_shape = expected[name]
        if dtype.str != dtype_string or shape != expected_shape:
            raise HrtfCacheError(
                f"compiled HRTF {Path(name).stem} must be "
                f"{dtype_string}{expected_shape}, got {dtype.str}{shape}")
        payload_bytes = math.prod(expected_shape) * dtype.itemsize
    if header.tell() + payload_bytes != len(payload):
        raise HrtfCacheError(
            f"compiled HRTF member payload length does not match its NPY header: {name}")

    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError) as exc:
        raise HrtfCacheError(
            f"invalid compiled HRTF NPY member: {name}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise HrtfCacheError(f"compiled HRTF member is not an array: {name}")
    return value


def _read_validated_archive(stream) -> dict[str, np.ndarray]:
    size = os.fstat(stream.fileno()).st_size
    if size <= 0 or size > _MAX_ARCHIVE_BYTES:
        raise HrtfCacheError("compiled HRTF cache has an invalid file size")
    expected = {name + ".npy" for name in _ARCHIVE_KEYS}
    limits = {
        "metadata_json.npy": 4 * _MAX_METADATA_BYTES + 4096,
        "band_center_frequencies_hz.npy": 8192,
        "coefficients.npy": 200000,
        "delay_coefficients.npy": 8192,
        "delay_bounds.npy": 4096,
    }
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream, "r") as archive:
            members = archive.infolist()
            if (len(members) != len(expected)
                    or {member.filename for member in members} != expected):
                raise HrtfCacheError("compiled HRTF cache has an invalid member set")
            total_size = 0
            arrays = {}
            for member in members:
                if member.flag_bits & 0x1:
                    raise HrtfCacheError("encrypted compiled HRTF caches are unsupported")
                if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise HrtfCacheError("unsupported compiled HRTF compression method")
                if member.file_size > limits[member.filename]:
                    raise HrtfCacheError(
                        f"compiled HRTF member is unexpectedly large: {member.filename}")
                total_size += member.file_size
            if total_size > 512 << 10:
                raise HrtfCacheError("compiled HRTF cache expands beyond its size limit")
            for member in members:
                with archive.open(member, "r") as member_stream:
                    payload = member_stream.read(limits[member.filename] + 1)
                if len(payload) != member.file_size:
                    raise HrtfCacheError(
                        f"compiled HRTF member has an invalid expanded size: "
                        f"{member.filename}")
                arrays[Path(member.filename).stem] = _read_npy_member(
                    payload, name=member.filename)
            return arrays
    except HrtfCacheError:
        raise
    except (EOFError, RuntimeError, zipfile.BadZipFile, OSError, ValueError) as exc:
        raise HrtfCacheError(f"invalid compiled HRTF archive: {exc}") from exc


@dataclass(frozen=True)
class FieldEvaluation:
    aligned_gains: np.ndarray
    delay_samples: np.ndarray
    transfer_gains: np.ndarray


@dataclass(frozen=True)
class SofaHrtfField:
    source_sha256: str
    source_display_name: str | None
    sample_rate_hz: float
    measurement_radius_m: float
    order: int
    projection_ridge: float
    spherical_harmonic_ridge: float
    band_center_frequencies_hz: np.ndarray
    coefficients: np.ndarray
    delay_coefficients: np.ndarray
    delay_bounds: np.ndarray
    delay_source: str
    fit_report: dict
    cache_key: str
    format_version: int = JOCHRTF_FORMAT_VERSION

    def __post_init__(self):
        digest = _validate_sha256(self.source_sha256)
        display_name = _safe_display_name(self.source_display_name)
        rate = float(self.sample_rate_hz)
        radius = float(self.measurement_radius_m)
        projection_ridge = float(self.projection_ridge)
        sh_ridge = float(self.spherical_harmonic_ridge)
        order = _strict_integer(self.order, "field order")
        format_version = _strict_integer(self.format_version, "field format_version")
        if format_version != JOCHRTF_FORMAT_VERSION:
            raise HrtfCacheError(
                f"unsupported .jochrtf version {format_version}; rebuild it from the source SOFA")
        if not (math.isfinite(rate) and rate > 0.0
                and math.isfinite(radius) and radius > 0.0):
            raise ValueError("field sample rate and measurement radius must be positive")
        if abs(rate - TARGET_SAMPLE_RATE_HZ) > 1.0e-9:
            raise ValueError("runtime HRTF fields must use 48 kHz")
        if not (math.isfinite(projection_ridge) and projection_ridge >= 0.0
                and math.isfinite(sh_ridge) and sh_ridge >= 0.0):
            raise ValueError("field ridge values must be finite and non-negative")
        if order != DEFAULT_ORDER:
            raise ValueError("runtime HRTF fields must be fifth order")
        terms = (order + 1) ** 2
        centers = np.array(
            self.band_center_frequencies_hz, dtype="<f8", order="C", copy=True)
        coefficients = np.array(
            self.coefficients, dtype="<c16", order="C", copy=True)
        delay_coefficients = np.array(
            self.delay_coefficients, dtype="<f8", order="C", copy=True)
        delay_bounds = np.array(
            self.delay_bounds, dtype="<f8", order="C", copy=True)
        if centers.shape != (HYBRID_BANDS,):
            raise ValueError("field must contain 77 hybrid-band centers")
        if coefficients.shape != (terms, 2, HYBRID_BANDS):
            raise ValueError("field coefficient shape mismatch")
        if delay_coefficients.shape != (terms, 2) or delay_bounds.shape != (2, 2):
            raise ValueError("field delay coefficient shape mismatch")
        if np.any(delay_bounds[:, 0] > delay_bounds[:, 1]):
            raise ValueError("field delay bounds are reversed")
        if np.any(delay_bounds < 0.0):
            raise ValueError("field delay bounds must be non-negative")
        for value in (centers, coefficients, delay_coefficients, delay_bounds):
            if not np.isfinite(value).all():
                raise ValueError("field contains non-finite values")
            value.setflags(write=False)
        if np.any(centers < 0.0) or np.any(centers >= 0.5 * rate):
            raise ValueError("field band centers must lie in [0, Nyquist)")
        if np.max(np.abs(coefficients), initial=0.0) > 1.0e6:
            raise ValueError("field coefficients exceed the supported safety bound")
        if np.max(np.abs(delay_bounds), initial=0.0) > rate:
            raise ValueError("field delays exceed the supported one-second bound")
        if (np.max(np.abs(delay_coefficients), initial=0.0)
                > _MAX_DELAY_COEFFICIENT_ABS):
            raise ValueError("field delay coefficients exceed the supported safety bound")
        expected_key = compiled_hrtf_cache_key(
            source_sha256=digest,
            sample_rate_hz=rate,
            shell_radius_m=radius,
            order=order,
            projection_ridge=projection_ridge,
            sh_ridge=sh_ridge,
        )
        if str(self.cache_key).upper() != expected_key:
            raise HrtfCacheError("compiled HRTF cache key does not match its configuration")
        object.__setattr__(self, "source_sha256", digest)
        object.__setattr__(self, "source_display_name", display_name)
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "measurement_radius_m", radius)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "format_version", format_version)
        object.__setattr__(self, "projection_ridge", projection_ridge)
        object.__setattr__(self, "spherical_harmonic_ridge", sh_ridge)
        object.__setattr__(self, "band_center_frequencies_hz", centers)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "delay_coefficients", delay_coefficients)
        object.__setattr__(self, "delay_bounds", delay_bounds)
        object.__setattr__(self, "cache_key", expected_key)
        object.__setattr__(self, "fit_report", json.loads(
            _json_bytes(dict(self.fit_report)).decode("utf-8")))

    @classmethod
    def fit(cls, canonical: CanonicalHrtf, *, shell_radius_m: float | None = None,
            order: int = DEFAULT_ORDER, ridge: float = DEFAULT_SH_RIDGE,
            projection_ridge: float = DEFAULT_PROJECTION_RIDGE) -> "SofaHrtfField":
        order = _strict_integer(order, "field order")
        if order != DEFAULT_ORDER:
            raise ValueError("the runtime HRTF field is fixed at fifth order")
        if abs(float(canonical.sample_rate_hz) - TARGET_SAMPLE_RATE_HZ) > 1.0e-9:
            raise ValueError("SOFA HRTF fields must be compiled at 48 kHz")
        aligned = time_align_hrtf(canonical)
        shell_target = 1.0 if shell_radius_m is None else float(shell_radius_m)
        if not math.isfinite(shell_target) or shell_target <= 0.0:
            raise ValueError("shell_radius_m must be positive and finite")
        indices = canonical.shell_indices(shell_target)
        if len(indices) < (order + 1) ** 2:
            raise ValueError(
                f"fifth-order SH needs at least 36 measurements on one shell, got {len(indices)}")
        shell_radius = float(np.mean(canonical.measurement_radius_m[indices]))
        directions = canonical.unit_directions[indices]
        runtime_delay = np.asarray(aligned.runtime_delay_samples[indices], dtype=np.float64)
        centers = hybrid_band_center_frequencies_hz(canonical.sample_rate_hz)
        gains, projection_report = project_hrir_to_hybrid_gains(
            np.asarray(canonical.hrir[indices], dtype=np.float64),
            embedded_delay_samples=np.asarray(
                aligned.embedded_delay_removed_samples[indices], dtype=np.float64),
            sample_rate_hz=canonical.sample_rate_hz,
            ridge=projection_ridge)

        directions, gains, runtime_delay = _group_coincident(
            directions, gains, runtime_delay)
        if len(directions) < (order + 1) ** 2:
            raise ValueError("coincident-direction merging left fewer than 36 directions")
        weights = spherical_voronoi_weights(directions)
        coefficients = fit_real_spherical_harmonics(
            directions, gains, order=order, ridge=ridge, weights=weights)
        delay_coefficients = fit_real_spherical_harmonics(
            directions, runtime_delay, order=order, ridge=ridge, weights=weights)
        reconstructed_aligned = evaluate_real_spherical_harmonics(
            coefficients, directions, order=order)
        delay_reconstructed = evaluate_real_spherical_harmonics(
            delay_coefficients, directions, order=order)
        reference_phase = np.exp(
            -2j * np.pi * runtime_delay[..., None]
            * centers[None, None, :] / canonical.sample_rate_hz)
        reconstructed_phase = np.exp(
            -2j * np.pi * delay_reconstructed[..., None]
            * centers[None, None, :] / canonical.sample_rate_hz)
        reference_transfer = np.asarray(gains * reference_phase, dtype=np.complex128)
        reconstructed_transfer = np.asarray(
            reconstructed_aligned * reconstructed_phase, dtype=np.complex128)
        magnitude_reference = np.maximum(np.abs(reference_transfer), 1.0e-12)
        relative = np.abs(reconstructed_transfer - reference_transfer) / magnitude_reference
        magnitude_db_error = np.abs(
            20.0 * np.log10(np.maximum(np.abs(reconstructed_transfer), 1.0e-12))
            - 20.0 * np.log10(magnitude_reference))
        delay_error = delay_reconstructed - runtime_delay
        report = {
            "format": "SOFA FIR -> public 64-QMF/77-hybrid -> ACN/N3D real SH",
            "order": order,
            "terms": (order + 1) ** 2,
            "input_measurements": int(len(indices)),
            "unique_directions": int(len(directions)),
            "shell_radius_m": shell_radius,
            "spherical_harmonic_ridge": float(ridge),
            "projection": projection_report,
            "phase_policy_version": PHASE_POLICY_VERSION,
            "complex_relative_error_median": float(np.median(relative)),
            "complex_relative_error_p95": float(np.percentile(relative, 95.0)),
            "magnitude_error_db_median": float(np.median(magnitude_db_error)),
            "magnitude_error_db_p95": float(np.percentile(magnitude_db_error, 95.0)),
            "delay_error_samples_rms": float(np.sqrt(np.mean(delay_error * delay_error))),
            "delay_error_samples_max": float(np.max(np.abs(delay_error))),
            "delay_source": aligned.delay_source,
            "precision": "float64/complex128",
        }
        delay_bounds = np.stack(
            (np.min(runtime_delay, axis=0), np.max(runtime_delay, axis=0)), axis=1)
        cache_key = compiled_hrtf_cache_key(
            source_sha256=canonical.source_sha256,
            sample_rate_hz=canonical.sample_rate_hz,
            shell_radius_m=shell_radius,
            order=order,
            projection_ridge=float(projection_ridge),
            sh_ridge=float(ridge),
        )
        display_name = Path(canonical.source_path).name if canonical.source_path else None
        return cls(
            source_sha256=canonical.source_sha256,
            source_display_name=display_name,
            sample_rate_hz=canonical.sample_rate_hz,
            measurement_radius_m=shell_radius,
            order=order,
            projection_ridge=float(projection_ridge),
            spherical_harmonic_ridge=float(ridge),
            band_center_frequencies_hz=centers,
            coefficients=coefficients,
            delay_coefficients=delay_coefficients,
            delay_bounds=delay_bounds,
            delay_source=aligned.delay_source,
            fit_report=report,
            cache_key=cache_key,
        )

    def evaluate_sofa(self, direction_sofa) -> FieldEvaluation:
        direction = np.asarray(direction_sofa, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("SOFA direction must contain three finite values")
        length = float(np.linalg.norm(direction))
        direction = (np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                     if length <= 1.0e-15 else direction / length)
        basis = real_spherical_harmonics(direction, order=self.order)
        with np.errstate(over="ignore", invalid="ignore"):
            aligned = np.asarray(
                np.tensordot(basis, self.coefficients, axes=(0, 0)),
                dtype=np.complex128)
            delay = np.asarray(
                np.tensordot(basis, self.delay_coefficients, axes=(0, 0)),
                dtype=np.float64)
        if not np.isfinite(aligned).all() or not np.isfinite(delay).all():
            raise HrtfCacheError("HRTF field evaluation produced non-finite values")
        delay = np.clip(delay, self.delay_bounds[:, 0], self.delay_bounds[:, 1])
        phase = np.exp(
            -2j * np.pi * delay[:, None]
            * self.band_center_frequencies_hz[None, :] / self.sample_rate_hz)
        return FieldEvaluation(aligned, delay, np.asarray(aligned * phase, np.complex128))

    def evaluate_adm(self, direction_adm) -> FieldEvaluation:
        return self.evaluate_sofa(adm_to_sofa_direction(direction_adm))

    def render_impulse(self, direction_sofa, *, sample_count: int = 1024) -> np.ndarray:
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        evaluation = self.evaluate_sofa(direction_sofa)
        delay_slots = np.floor(
            evaluation.delay_samples / QMF_HOP).astype(np.int64)
        residual_delay = (
            evaluation.delay_samples - delay_slots.astype(np.float64) * QMF_HOP)
        residual_phase = np.exp(
            -2j * np.pi
            * residual_delay[:, None]
            * self.band_center_frequencies_hz[None, :]
            / self.sample_rate_hz)
        residual_gains = np.asarray(
            evaluation.aligned_gains * residual_phase, dtype=np.complex128)
        total = int(math.ceil(
            (ANALYSIS_SYNTHESIS_LATENCY_SAMPLES + sample_count + 512)
            / QMF_HOP) * QMF_HOP)
        impulse = np.zeros((total, 1), dtype=np.float64)
        impulse[0, 0] = 1.0
        base = PublicAnalysis77(1).process(impulse)[:, 0, :]
        hybrid = np.zeros(
            (len(base) + int(np.max(delay_slots, initial=0)), 2, HYBRID_BANDS),
            dtype=np.complex128)
        for ear in range(2):
            start_slot = int(delay_slots[ear])
            hybrid[start_slot:start_slot + len(base), ear] = (
                base * residual_gains[ear][None, :])
        raw = PublicSynthesis77(2).process(hybrid)
        start = ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
        return np.asarray(raw[start:start + sample_count].T, dtype=np.float64)

    def _metadata(self) -> dict:
        return {
            "magic": JOCHRTF_MAGIC,
            "format_version": JOCHRTF_FORMAT_VERSION,
            "cache_schema": JOCHRTF_CACHE_SCHEMA,
            "cache_key_version": 1,
            "compiler_version": COMPILER_VERSION,
            "phase_policy_version": PHASE_POLICY_VERSION,
            "sh_convention": SH_CONVENTION,
            "filterbank": filterbank_fingerprint(),
            "cache_key": self.cache_key,
            "payload_sha256": _payload_sha256(
                self.band_center_frequencies_hz,
                self.coefficients,
                self.delay_coefficients,
                self.delay_bounds),
            "source_sha256": self.source_sha256,
            "source_display_name": self.source_display_name,
            "sample_rate_hz": self.sample_rate_hz,
            "measurement_radius_m": self.measurement_radius_m,
            "order": self.order,
            "projection_ridge": self.projection_ridge,
            "spherical_harmonic_ridge": self.spherical_harmonic_ridge,
            "delay_source": self.delay_source,
            "fit_report": self.fit_report,
        }

    def save(self, path) -> Path:
        requested = Path(path).expanduser()
        if requested.suffix.lower() != ".jochrtf":
            raise ValueError("compiled HRTF cache path must end in .jochrtf")
        requested.parent.mkdir(parents=True, exist_ok=True)
        target = requested.parent.resolve() / requested.name
        if target.is_symlink():
            raise HrtfCacheError("refusing to replace a symlinked compiled HRTF cache")
        metadata_text = _json_bytes(self._metadata()).decode("utf-8")
        with _cache_write_lock(target):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    np.savez_compressed(
                        stream,
                        metadata_json=np.asarray(metadata_text),
                        band_center_frequencies_hz=np.asarray(
                            self.band_center_frequencies_hz, dtype="<f8"),
                        coefficients=np.asarray(self.coefficients, dtype="<c16"),
                        delay_coefficients=np.asarray(
                            self.delay_coefficients, dtype="<f8"),
                        delay_bounds=np.asarray(self.delay_bounds, dtype="<f8"),
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                self.load(
                    temporary,
                    expected_source_sha256=self.source_sha256,
                    expected_cache_key=self.cache_key)
                os.replace(temporary, target)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        return target

    @classmethod
    def load(cls, path, *, expected_source_sha256: str | None = None,
             expected_cache_key: str | None = None) -> "SofaHrtfField":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            with source.open("rb") as stream:
                arrays = _read_validated_archive(stream)
                metadata_array = arrays.pop("metadata_json")
                metadata_text = str(metadata_array.item())
                if len(metadata_text.encode("utf-8")) > _MAX_METADATA_BYTES:
                    raise HrtfCacheError("compiled HRTF metadata is too large")
                metadata = _strict_json_object(metadata_text)
        except HrtfCacheError:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
                OverflowError, RecursionError, zipfile.BadZipFile) as exc:
            raise HrtfCacheError(f"invalid compiled HRTF cache: {exc}") from exc
        try:
            version = _strict_integer(
                metadata.get("format_version", -1), "compiled HRTF version")
        except ValueError as exc:
            raise HrtfCacheError(str(exc)) from exc
        if version != JOCHRTF_FORMAT_VERSION:
            raise HrtfCacheError(
                f"unsupported .jochrtf version {version}; rebuild it from the source SOFA")
        if set(metadata) != _METADATA_KEYS:
            raise HrtfCacheError("compiled HRTF metadata has an invalid key set")
        try:
            cache_key_version = _strict_integer(
                metadata["cache_key_version"], "compiled HRTF cache_key_version")
            order = _strict_integer(metadata["order"], "compiled HRTF order")
        except ValueError as exc:
            raise HrtfCacheError(str(exc)) from exc
        if cache_key_version != 1:
            raise HrtfCacheError("compiled HRTF cache_key_version is incompatible")
        if order != DEFAULT_ORDER:
            raise HrtfCacheError("compiled HRTF order is incompatible")
        for name in (
                "sample_rate_hz", "measurement_radius_m", "projection_ridge",
                "spherical_harmonic_ridge"):
            _json_number(metadata, name)
        for name in ("cache_key", "payload_sha256", "source_sha256", "delay_source"):
            if not isinstance(metadata[name], str):
                raise HrtfCacheError(f"compiled HRTF {name} must be a string")
        if metadata["source_display_name"] is not None and not isinstance(
                metadata["source_display_name"], str):
            raise HrtfCacheError(
                "compiled HRTF source_display_name must be a string or null")
        if metadata.get("magic") != JOCHRTF_MAGIC:
            raise HrtfCacheError("compiled HRTF cache magic mismatch")
        expected_static = {
            "cache_schema": JOCHRTF_CACHE_SCHEMA,
            "cache_key_version": 1,
            "compiler_version": COMPILER_VERSION,
            "phase_policy_version": PHASE_POLICY_VERSION,
            "sh_convention": SH_CONVENTION,
            "filterbank": filterbank_fingerprint(),
        }
        for name, expected in expected_static.items():
            if metadata.get(name) != expected:
                raise HrtfCacheError(f"compiled HRTF {name} is incompatible")
        expected_arrays = {
            "band_center_frequencies_hz": ("<f8", (HYBRID_BANDS,)),
            "coefficients": ("<c16", (36, 2, HYBRID_BANDS)),
            "delay_coefficients": ("<f8", (36, 2)),
            "delay_bounds": ("<f8", (2, 2)),
        }
        for name, (dtype_string, shape) in expected_arrays.items():
            value = arrays[name]
            if value.dtype.str != dtype_string or value.shape != shape:
                raise HrtfCacheError(
                    f"compiled HRTF {name} must be {dtype_string}{shape}, "
                    f"got {value.dtype.str}{value.shape}")
            if not np.isfinite(value).all():
                raise HrtfCacheError(f"compiled HRTF {name} contains non-finite values")
            if not value.flags.c_contiguous:
                raise HrtfCacheError(f"compiled HRTF {name} must be C-contiguous")
        if not np.array_equal(
                arrays["band_center_frequencies_hz"],
                hybrid_band_center_frequencies_hz(float(metadata["sample_rate_hz"]))):
            raise HrtfCacheError("compiled HRTF band centers do not match the filterbank")
        payload_sha256 = _validate_sha256(
            metadata["payload_sha256"], "payload_sha256")
        actual_payload_sha256 = _payload_sha256(
            arrays["band_center_frequencies_hz"], arrays["coefficients"],
            arrays["delay_coefficients"], arrays["delay_bounds"])
        if payload_sha256 != actual_payload_sha256:
            raise HrtfCacheError("compiled HRTF payload hash mismatch")
        if not isinstance(metadata["fit_report"], dict):
            raise HrtfCacheError("compiled HRTF fit_report must be an object")
        if not isinstance(metadata["delay_source"], str):
            raise HrtfCacheError("compiled HRTF delay_source must be a string")
        try:
            field = cls(
                source_sha256=str(metadata["source_sha256"]),
                source_display_name=metadata["source_display_name"],
                sample_rate_hz=_json_number(metadata, "sample_rate_hz"),
                measurement_radius_m=_json_number(metadata, "measurement_radius_m"),
                order=order,
                projection_ridge=_json_number(metadata, "projection_ridge"),
                spherical_harmonic_ridge=_json_number(
                    metadata, "spherical_harmonic_ridge"),
                band_center_frequencies_hz=arrays["band_center_frequencies_hz"],
                coefficients=arrays["coefficients"],
                delay_coefficients=arrays["delay_coefficients"],
                delay_bounds=arrays["delay_bounds"],
                delay_source=metadata["delay_source"],
                fit_report=metadata["fit_report"],
                cache_key=str(metadata["cache_key"]),
                format_version=version,
            )
        except HrtfCacheError:
            raise
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise HrtfCacheError(
                f"invalid compiled HRTF metadata values: {exc}") from exc
        if expected_source_sha256 is not None:
            expected_digest = _validate_sha256(expected_source_sha256)
            if field.source_sha256 != expected_digest:
                raise HrtfCacheError("compiled HRTF source hash mismatch")
        if expected_cache_key is not None and field.cache_key != str(expected_cache_key).upper():
            raise HrtfCacheError("compiled HRTF configuration hash mismatch")
        return field

    def info(self) -> dict:
        return {
            "format": JOCHRTF_MAGIC,
            "format_version": self.format_version,
            "cache_key": self.cache_key,
            "source_sha256": self.source_sha256,
            "source_display_name": self.source_display_name,
            "sample_rate_hz": self.sample_rate_hz,
            "measurement_radius_m": self.measurement_radius_m,
            "order": self.order,
            "terms": (self.order + 1) ** 2,
            "bands": HYBRID_BANDS,
            "projection_ridge": self.projection_ridge,
            "spherical_harmonic_ridge": self.spherical_harmonic_ridge,
            "delay_source": self.delay_source,
            "fit_report": self.fit_report,
            "precision": "float64/complex128",
        }


def _cache_file_name(display_name: str | None, cache_key: str) -> str:
    stem = Path(display_name).stem if display_name else "hrtf"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "hrtf"
    return f"{safe}.{cache_key[:20]}.jochrtf"


def compile_sofa_hrtf(
        sofa: str | Path | CanonicalHrtf, *,
        target_sample_rate_hz: float = TARGET_SAMPLE_RATE_HZ,
        shell_radius_m: float = 1.0,
        order: int = DEFAULT_ORDER,
        projection_ridge: float = DEFAULT_PROJECTION_RIDGE,
        sh_ridge: float = DEFAULT_SH_RIDGE,
        cache_policy: str = "memory",
        cache_dir: str | Path | None = None) -> SofaHrtfField:
    """Compile SOFA in memory, optionally using a validated disposable cache."""
    policy = str(cache_policy).strip().lower()
    if policy not in {"none", "memory", "disk"}:
        raise ValueError("cache_policy must be none, memory, or disk")
    target_rate = float(target_sample_rate_hz)
    if (not math.isfinite(target_rate)
            or abs(target_rate - TARGET_SAMPLE_RATE_HZ) > 1.0e-9):
        raise ValueError("the public binaural runtime currently requires 48 kHz")
    radius = float(shell_radius_m)
    projection_ridge = float(projection_ridge)
    sh_ridge = float(sh_ridge)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("shell_radius_m must be positive and finite")
    if (not math.isfinite(projection_ridge) or projection_ridge < 0.0
            or not math.isfinite(sh_ridge) or sh_ridge < 0.0):
        raise ValueError("compiler ridge values must be finite and non-negative")
    order = _strict_integer(order, "compiler order")
    if order != DEFAULT_ORDER:
        raise ValueError("the runtime HRTF field is fixed at fifth order")
    if isinstance(sofa, CanonicalHrtf):
        canonical = (sofa if abs(sofa.sample_rate_hz - target_rate) <= 1.0e-9
                     else sofa.resampled(target_rate))
    else:
        canonical = load_simple_free_field_hrir(
            sofa, target_sample_rate_hz=target_rate)
    selected = canonical.shell_indices(radius)
    actual_radius = float(np.mean(canonical.measurement_radius_m[selected]))
    key = compiled_hrtf_cache_key(
        source_sha256=canonical.source_sha256,
        sample_rate_hz=target_rate,
        shell_radius_m=actual_radius,
        order=order,
        projection_ridge=projection_ridge,
        sh_ridge=sh_ridge,
    )
    cached = None
    if policy in {"memory", "disk"}:
        cached = _memory_cache_get(key)
        if policy == "memory" and cached is not None:
            return cached
    target: Path | None = None
    if policy == "disk":
        directory = (DEFAULT_HRTF_CACHE_DIR if cache_dir is None
                     else Path(cache_dir).expanduser().resolve())
        directory.mkdir(parents=True, exist_ok=True)
        display_name = Path(canonical.source_path).name if canonical.source_path else None
        target = directory / _cache_file_name(display_name, key)
        if target.is_file():
            try:
                field = SofaHrtfField.load(
                    target,
                    expected_source_sha256=canonical.source_sha256,
                    expected_cache_key=key)
                _memory_cache_put(key, field)
                return field
            except (HrtfCacheError, OSError):
                pass
        if cached is not None:
            cached.save(target)
            return cached
    field = SofaHrtfField.fit(
        canonical,
        shell_radius_m=actual_radius,
        order=order,
        ridge=sh_ridge,
        projection_ridge=projection_ridge)
    if field.cache_key != key:
        raise RuntimeError("internal compiled HRTF cache-key mismatch")
    if target is not None:
        field.save(target)
    if policy in {"memory", "disk"}:
        _memory_cache_put(key, field)
    return field
