"""Strict SimpleFreeFieldHRIR to canonical HRTF import.

The canonical representation keeps ``Data.IR`` and ``Data.Delay`` separate.
No importer operation silently bakes the SOFA delay into the stored FIRs.  A
caller must explicitly request :meth:`CanonicalHrtf.materialized_measurement`
when a time-domain FIR with ``Data.Delay`` applied exactly once is required.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import signal
from scipy.fft import next_fast_len


_SUPPORTED_VERSIONS = {"0.4", "1.0", "1.1"}
_FREE_FIELD_ROOM_TYPES = {"free field", "free-field", "anechoic", "hemi-anechoic"}
_LENGTH_UNITS = {
    "m": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "cm": 1.0e-2,
    "centimetre": 1.0e-2,
    "centimetres": 1.0e-2,
    "centimeter": 1.0e-2,
    "centimeters": 1.0e-2,
    "mm": 1.0e-3,
    "millimetre": 1.0e-3,
    "millimetres": 1.0e-3,
    "millimeter": 1.0e-3,
    "millimeters": 1.0e-3,
}
_ANGLE_UNITS = {
    "degree": np.deg2rad,
    "degrees": np.deg2rad,
    "radian": lambda value: np.asarray(value, dtype=np.float64),
    "radians": lambda value: np.asarray(value, dtype=np.float64),
}


class SofaImportError(ValueError):
    """The file is outside the deliberately narrow public SOFA contract."""


def _text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "strict")
    return str(value)


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(4 << 20), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest().upper()


@contextmanager
def _stable_hdf5_source(path: Path):
    """Read arrays and content identity from one stable open-file snapshot."""
    with path.open("rb") as stream:
        before = _sha256_stream(stream)
        with h5py.File(stream, "r") as file:
            yield file, before
        after = _sha256_stream(stream)
        if after != before:
            raise SofaImportError("SOFA file changed while it was being imported")


def _tokens(units: str) -> list[str]:
    return [token.strip().lower() for token in units.split(",") if token.strip()]


def _coordinate_attributes(dataset: h5py.Dataset, *, inherit=None) -> tuple[str, str]:
    source = dataset.attrs
    if "Type" not in source or "Units" not in source:
        if inherit is None or "Type" not in inherit.attrs or "Units" not in inherit.attrs:
            raise SofaImportError(f"{dataset.name} must declare Type and Units")
        source = inherit.attrs
    return _text(source["Type"]).strip().lower(), _text(source["Units"]).strip()


def coordinates_to_cartesian_m(values, coordinate_type: str, units: str,
                               *, variable: str) -> np.ndarray:
    """Convert a SOFA coordinate array to Cartesian metres without reshaping it."""
    data = np.asarray(values, dtype=np.float64)
    if data.shape[-1] != 3 or not np.isfinite(data).all():
        raise SofaImportError(f"{variable} must contain finite C=3 coordinates")
    kind = coordinate_type.strip().lower()
    unit_tokens = _tokens(units)
    if kind == "cartesian":
        if len(unit_tokens) == 1:
            factors = [_LENGTH_UNITS.get(unit_tokens[0])] * 3
        elif len(unit_tokens) == 3:
            factors = [_LENGTH_UNITS.get(token) for token in unit_tokens]
        else:
            factors = []
        if len(factors) != 3 or any(value is None for value in factors):
            raise SofaImportError(f"unsupported Cartesian units for {variable}: {units!r}")
        return data * np.asarray(factors, dtype=np.float64)
    if kind != "spherical" or len(unit_tokens) != 3:
        raise SofaImportError(
            f"unsupported coordinates for {variable}: Type={coordinate_type!r}, Units={units!r}")
    if unit_tokens[0] not in _ANGLE_UNITS or unit_tokens[1] not in _ANGLE_UNITS:
        raise SofaImportError(f"unsupported spherical angle units for {variable}: {units!r}")
    radius_factor = _LENGTH_UNITS.get(unit_tokens[2])
    if radius_factor is None:
        raise SofaImportError(f"unsupported spherical radius unit for {variable}: {units!r}")
    azimuth = _ANGLE_UNITS[unit_tokens[0]](data[..., 0])
    elevation = _ANGLE_UNITS[unit_tokens[1]](data[..., 1])
    radius = data[..., 2] * radius_factor
    if np.any(radius < 0.0):
        raise SofaImportError(f"{variable} contains a negative spherical radius")
    horizontal = np.cos(elevation)
    return np.stack(
        (radius * horizontal * np.cos(azimuth),
         radius * horizontal * np.sin(azimuth),
         radius * np.sin(elevation)),
        axis=-1,
    ).astype(np.float64, copy=False)


def _rows(file: h5py.File, name: str, measurements: int, *, inherit=None) -> np.ndarray:
    if name not in file:
        raise SofaImportError(f"missing required SOFA variable {name}")
    dataset = file[name]
    kind, units = _coordinate_attributes(dataset, inherit=inherit)
    result = coordinates_to_cartesian_m(dataset[...], kind, units, variable=name)
    if result.shape not in ((1, 3), (measurements, 3)):
        raise SofaImportError(
            f"{name} must have shape [I,C] or [M,C], got {result.shape}")
    return np.broadcast_to(result, (measurements, 3)).astype(np.float64, copy=True)


def _receiver_rows(file: h5py.File, measurements: int) -> np.ndarray:
    if "ReceiverPosition" not in file:
        raise SofaImportError("missing required SOFA variable ReceiverPosition")
    dataset = file["ReceiverPosition"]
    raw = np.asarray(dataset[...], dtype=np.float64)
    if raw.shape not in ((2, 3, 1), (2, 3, measurements)):
        raise SofaImportError(
            "ReceiverPosition must have shape [R=2,C=3,I=1 or M]")
    values = np.moveaxis(raw, 1, -1)  # [R,I/M,C]
    kind, units = _coordinate_attributes(dataset)
    cartesian = coordinates_to_cartesian_m(
        values, kind, units, variable="ReceiverPosition")
    cartesian = np.moveaxis(cartesian, 0, 1)  # [I/M,R,C]
    return np.broadcast_to(cartesian, (measurements, 2, 3)).astype(
        np.float64, copy=True)


def _emitter_is_origin(file: h5py.File, measurements: int) -> None:
    if "EmitterPosition" not in file:
        raise SofaImportError("missing required SOFA variable EmitterPosition")
    dataset = file["EmitterPosition"]
    raw = np.asarray(dataset[...], dtype=np.float64)
    if raw.shape not in ((1, 3, 1), (1, 3, measurements)):
        raise SofaImportError("SimpleFreeFieldHRIR v1 requires E=1 EmitterPosition[E,C,I/M]")
    values = np.moveaxis(raw, 1, -1)
    kind, units = _coordinate_attributes(dataset)
    cartesian = coordinates_to_cartesian_m(
        values, kind, units, variable="EmitterPosition")
    if np.max(np.abs(cartesian), initial=0.0) > 1.0e-9:
        raise SofaImportError("non-zero EmitterPosition needs a separate source-pose adapter")


def _sampling_rate(file: h5py.File) -> float:
    if "Data.SamplingRate" not in file:
        raise SofaImportError("missing Data.SamplingRate")
    dataset = file["Data.SamplingRate"]
    values = np.asarray(dataset[...], dtype=np.float64).reshape(-1)
    if values.size != 1 or not math.isfinite(float(values[0])) or values[0] <= 0.0:
        raise SofaImportError("Data.SamplingRate must contain one positive finite value")
    units = _text(dataset.attrs.get("Units", "")).strip().lower()
    if units not in {"hertz", "hz"}:
        raise SofaImportError(f"Data.SamplingRate Units must be hertz, got {units!r}")
    return float(values[0])


def _processing_label(file: h5py.File) -> str:
    parts = []
    for key in ("DatabaseName", "Title", "ListenerShortName", "Comment"):
        value = _text(file.attrs.get(key, "")).strip()
        if value and value not in parts:
            parts.append(value)
    return " | ".join(parts)


def _read_delay(file: h5py.File, measurements: int) -> np.ndarray:
    if "Data.Delay" not in file:
        raise SofaImportError("missing Data.Delay")
    delay = np.asarray(file["Data.Delay"][...], dtype=np.float64)
    if delay.shape not in ((1, 2), (measurements, 2)) or not np.isfinite(delay).all():
        raise SofaImportError("Data.Delay must have finite shape [I=1,R=2] or [M,R=2]")
    delay = np.broadcast_to(delay, (measurements, 2)).astype(np.float64, copy=True)
    if np.min(delay, initial=0.0) < -1.0e-9:
        raise SofaImportError("negative Data.Delay is outside the supported causal contract")
    delay[delay < 0.0] = 0.0
    return delay


def _readonly(array, dtype) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CanonicalHrtf:
    source_path: str
    source_sha256: str
    convention: str
    convention_version: str
    sofa_version: str
    source_sample_rate_hz: float
    sample_rate_hz: float
    source_position_cartesian_m: np.ndarray  # listener-local [M,3]
    listener_view: np.ndarray                 # world, normalized [M,3]
    listener_up: np.ndarray                   # world, orthonormal [M,3]
    receiver_position_cartesian_m: np.ndarray # listener-local, L/R [M,2,3]
    left_receiver_index: int
    right_receiver_index: int
    hrir: np.ndarray                          # canonical L/R [M,2,N]
    delay_samples: np.ndarray                 # canonical L/R [M,2], not applied
    measurement_radius_m: np.ndarray          # [M]
    processing_label: str
    resampling_label: str = "none"

    def __post_init__(self):
        object.__setattr__(self, "source_position_cartesian_m", _readonly(
            self.source_position_cartesian_m, np.float64))
        object.__setattr__(self, "listener_view", _readonly(self.listener_view, np.float64))
        object.__setattr__(self, "listener_up", _readonly(self.listener_up, np.float64))
        object.__setattr__(self, "receiver_position_cartesian_m", _readonly(
            self.receiver_position_cartesian_m, np.float64))
        object.__setattr__(self, "hrir", _readonly(self.hrir, np.float64))
        object.__setattr__(self, "delay_samples", _readonly(self.delay_samples, np.float64))
        object.__setattr__(self, "measurement_radius_m", _readonly(
            self.measurement_radius_m, np.float64))

    @property
    def measurements(self) -> int:
        return int(self.hrir.shape[0])

    @property
    def taps(self) -> int:
        return int(self.hrir.shape[2])

    @property
    def unit_directions(self) -> np.ndarray:
        return self.source_position_cartesian_m / self.measurement_radius_m[:, None]

    @property
    def shells_m(self) -> np.ndarray:
        return np.unique(np.round(self.measurement_radius_m, 9))

    def shell_indices(self, radius_m: float) -> np.ndarray:
        shell = float(self.shells_m[np.argmin(np.abs(self.shells_m - float(radius_m)))])
        return np.flatnonzero(np.isclose(
            self.measurement_radius_m, shell, atol=5.0e-7, rtol=0.0))

    def nearest_index(self, direction_sofa, radius_m: float = 1.0) -> tuple[int, float]:
        direction = np.asarray(direction_sofa, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("direction must contain three finite SOFA Cartesian values")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-15:
            raise ValueError("direction must be non-zero")
        direction = direction / norm
        indices = self.shell_indices(radius_m)
        dots = self.unit_directions[indices] @ direction
        local = int(np.argmax(dots))
        error = math.degrees(math.acos(float(np.clip(dots[local], -1.0, 1.0))))
        return int(indices[local]), float(error)

    def resampled(self, target_sample_rate_hz: float) -> "CanonicalHrtf":
        target = float(target_sample_rate_hz)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("target sample rate must be positive and finite")
        if abs(target - self.sample_rate_hz) <= 1.0e-9:
            return self
        ratio = target / self.sample_rate_hz
        fraction = Fraction(ratio).limit_denominator(100000)
        if abs(float(fraction) - ratio) > 1.0e-10:
            raise ValueError("sample-rate ratio cannot be represented safely")
        converted = signal.resample_poly(
            np.asarray(self.hrir, dtype=np.float64), fraction.numerator,
            fraction.denominator, axis=-1, window=("kaiser", 8.6), padtype="constant")
        converted = np.asarray(converted, dtype=np.float64)
        return replace(
            self,
            sample_rate_hz=target,
            hrir=converted,
            delay_samples=np.asarray(self.delay_samples * ratio, dtype=np.float64),
            resampling_label=(
                f"scipy.signal.resample_poly {self.sample_rate_hz:g}->{target:g} Hz "
                f"({fraction.numerator}/{fraction.denominator}, Kaiser beta=8.6)"),
        )

    def materialized_measurement(self, index: int, *, fractional_half_length: int = 48
                                 ) -> np.ndarray:
        """Return [L/R,taps] with SOFA Data.Delay applied exactly once."""
        index = int(index)
        if not 0 <= index < self.measurements:
            raise IndexError(index)
        ears = []
        for ear in range(2):
            ears.append(apply_fractional_delay(
                self.hrir[index, ear], float(self.delay_samples[index, ear]),
                half_length=fractional_half_length))
        length = max(map(len, ears))
        result = np.zeros((2, length), dtype=np.float64)
        for ear, value in enumerate(ears):
            result[ear, :len(value)] = value
        return result

    def info(self) -> dict:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "convention": self.convention,
            "convention_version": self.convention_version,
            "sofa_version": self.sofa_version,
            "source_sample_rate_hz": self.source_sample_rate_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "measurements": self.measurements,
            "taps": self.taps,
            "shells_m": [float(value) for value in self.shells_m],
            "source_receiver_order": [self.left_receiver_index, self.right_receiver_index],
            "canonical_ear_order": ["left", "right"],
            "data_delay_samples_min": float(np.min(self.delay_samples)),
            "data_delay_samples_max": float(np.max(self.delay_samples)),
            "data_delay_applied": False,
            "processing_label": self.processing_label,
            "resampling": self.resampling_label,
            "precision": "float64",
        }


def load_simple_free_field_hrir(path, *, target_sample_rate_hz: float | None = None
                                ) -> CanonicalHrtf:
    """Strictly import the supported SimpleFreeFieldHRIR subset."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with _stable_hdf5_source(source) as (file, source_sha256):
        if _text(file.attrs.get("Conventions", "")) != "SOFA":
            raise SofaImportError("Conventions must be SOFA")
        convention = _text(file.attrs.get("SOFAConventions", ""))
        if convention != "SimpleFreeFieldHRIR":
            raise SofaImportError(
                f"unsupported SOFAConventions={convention!r}; convert explicitly first")
        convention_version = _text(file.attrs.get("SOFAConventionsVersion", ""))
        if convention_version not in _SUPPORTED_VERSIONS:
            raise SofaImportError(
                f"unsupported SimpleFreeFieldHRIR version {convention_version!r}; "
                f"supported={sorted(_SUPPORTED_VERSIONS)}")
        if _text(file.attrs.get("DataType", "")) != "FIR":
            raise SofaImportError("DataType must be FIR")
        room_type = _text(file.attrs.get("RoomType", "")).strip().lower()
        if room_type not in _FREE_FIELD_ROOM_TYPES:
            raise SofaImportError(f"RoomType must explicitly be free-field, got {room_type!r}")
        if "Data.IR" not in file:
            raise SofaImportError("missing Data.IR")
        hrir_source = np.asarray(file["Data.IR"][...], dtype=np.float64)
        if hrir_source.ndim != 3 or hrir_source.shape[1] != 2 or min(hrir_source.shape) <= 0:
            raise SofaImportError("Data.IR must have shape [M,R=2,N]")
        if not np.isfinite(hrir_source).all():
            raise SofaImportError("Data.IR contains non-finite values")
        measurements = int(hrir_source.shape[0])
        sample_rate = _sampling_rate(file)
        delay_source = _read_delay(file, measurements)
        _emitter_is_origin(file, measurements)

        listener_position = _rows(file, "ListenerPosition", measurements)
        listener_view = _rows(file, "ListenerView", measurements)
        listener_up_raw = _rows(
            file, "ListenerUp", measurements,
            inherit=file["ListenerView"] if "ListenerView" in file else None)
        forward_norm = np.linalg.norm(listener_view, axis=1)
        if np.any(forward_norm <= 1.0e-12):
            raise SofaImportError("ListenerView must be non-zero")
        forward = listener_view / forward_norm[:, None]
        left = np.cross(listener_up_raw, forward)
        left_norm = np.linalg.norm(left, axis=1)
        if np.any(left_norm <= 1.0e-12):
            raise SofaImportError("ListenerUp must not be parallel to ListenerView")
        left /= left_norm[:, None]
        up = np.cross(forward, left)

        if "SourcePosition" not in file:
            raise SofaImportError("missing required SOFA variable SourcePosition")
        source_dataset = file["SourcePosition"]
        source_type, source_units = _coordinate_attributes(source_dataset)
        source_world = coordinates_to_cartesian_m(
            source_dataset[...], source_type, source_units, variable="SourcePosition")
        if source_world.shape != (measurements, 3):
            raise SofaImportError("SourcePosition must have shape [M,C=3]")
        relative = source_world - listener_position
        source_local = np.stack(
            (np.sum(relative * forward, axis=1),
             np.sum(relative * left, axis=1),
             np.sum(relative * up, axis=1)), axis=1)
        radii = np.linalg.norm(source_local, axis=1)
        if np.any(radii <= 1.0e-8) or not np.isfinite(radii).all():
            raise SofaImportError("every source measurement must have a positive radius")

        receiver = _receiver_rows(file, measurements)
        lateral_difference = receiver[:, 0, 1] - receiver[:, 1, 1]
        if np.all(lateral_difference > 1.0e-5):
            left_index, right_index = 0, 1
        elif np.all(lateral_difference < -1.0e-5):
            left_index, right_index = 1, 0
        else:
            raise SofaImportError(
                "ReceiverPosition does not identify one consistently-left and one "
                "consistently-right receiver")
        ear_order = [left_index, right_index]
        hrir = hrir_source[:, ear_order, :]
        delay = delay_source[:, ear_order]
        receiver = receiver[:, ear_order, :]
        processing_label = _processing_label(file)
        sofa_version = _text(file.attrs.get("Version", ""))

    canonical = CanonicalHrtf(
        source_path=str(source),
        source_sha256=source_sha256,
        convention=convention,
        convention_version=convention_version,
        sofa_version=sofa_version,
        source_sample_rate_hz=sample_rate,
        sample_rate_hz=sample_rate,
        source_position_cartesian_m=source_local,
        listener_view=forward,
        listener_up=up,
        receiver_position_cartesian_m=receiver,
        left_receiver_index=left_index,
        right_receiver_index=right_index,
        hrir=hrir,
        delay_samples=delay,
        measurement_radius_m=radii,
        processing_label=processing_label,
    )
    return (canonical if target_sample_rate_hz is None
            else canonical.resampled(target_sample_rate_hz))


def apply_fractional_delay(values, delay_samples: float, *, half_length: int = 48
                           ) -> np.ndarray:
    """Apply one causal non-negative delay to a real FIR using windowed sinc."""
    source = np.asarray(values, dtype=np.float64)
    delay = float(delay_samples)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise ValueError("fractional delay input must be a finite real vector")
    if not math.isfinite(delay) or delay < -1.0e-12:
        raise ValueError("fractional delay must be finite and non-negative")
    if delay < 1.0e-12:
        return source.copy()
    integer = int(math.floor(delay))
    fraction = delay - integer
    if fraction < 1.0e-12:
        return np.pad(source, (integer, 0)).astype(np.float64, copy=False)
    half = int(half_length)
    if half < 8:
        raise ValueError("fractional delay half_length must be at least 8")
    index = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.sinc(index - fraction) * np.kaiser(2 * half + 1, 8.6)
    kernel /= np.sum(kernel, dtype=np.float64)
    full = signal.fftconvolve(source, kernel, mode="full")
    causal = np.asarray(full[half:], dtype=np.float64)
    return np.pad(causal, (integer, 0)).astype(np.float64, copy=False)


def shift_signal_fft(values, shift_samples: float) -> np.ndarray:
    """Band-limited linear shift; positive is delay and negative is advance."""
    source = np.asarray(values, dtype=np.float64)
    shift = float(shift_samples)
    if source.ndim != 1 or not np.isfinite(source).all() or not math.isfinite(shift):
        raise ValueError("shift input and amount must be finite")
    if abs(shift) < 1.0e-12:
        return source.copy()
    guard = max(128, int(math.ceil(abs(shift))) + 64)
    needed = len(source) + 2 * guard
    fft_size = next_fast_len(needed)
    padded = np.zeros(fft_size, dtype=np.float64)
    padded[guard:guard + len(source)] = source
    bins = np.arange(fft_size // 2 + 1, dtype=np.float64)
    spectrum = np.fft.rfft(padded)
    spectrum *= np.exp(-2j * np.pi * bins * shift / fft_size)
    shifted = np.fft.irfft(spectrum, fft_size)
    return np.asarray(shifted[guard:guard + len(source)], dtype=np.float64)


def estimate_interaural_delay_samples(left, right, sample_rate_hz: float,
                                       *, low_hz: float = 200.0,
                                       high_hz: float = 1500.0) -> float:
    """Estimate L-minus-R delay by low-frequency circular phase coherence.

    A coarse-to-fine delay search avoids the phase-unwrapping branch failures
    that ordinary straight-line regression can exhibit for strongly filtered
    Far responses.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("ITD inputs must be equal-length vectors")
    fft_size = next_fast_len(max(4096, 4 * len(left)))
    left_spectrum = np.fft.rfft(left, fft_size)
    right_spectrum = np.fft.rfft(right, fft_size)
    frequency = np.fft.rfftfreq(fft_size, 1.0 / float(sample_rate_hz))
    selected = (frequency >= low_hz) & (frequency <= high_hz)
    if np.count_nonzero(selected) < 8:
        return 0.0
    cross = left_spectrum[selected] * np.conj(right_spectrum[selected])
    magnitude = np.abs(cross)
    maximum = float(np.max(magnitude, initial=0.0))
    if maximum <= 1.0e-20:
        return 0.0
    weighted_unit = cross / np.maximum(magnitude, 1.0e-30)
    weight = np.sqrt(magnitude / maximum)
    weighted_unit *= weight
    omega = 2.0 * np.pi * frequency[selected] / float(sample_rate_hz)
    limit = 0.0012 * float(sample_rate_hz)

    def best(candidates: np.ndarray) -> float:
        steering = np.exp(1j * omega[:, None] * candidates[None, :])
        score = np.abs(weighted_unit @ steering)
        return float(candidates[int(np.argmax(score))])

    coarse = np.arange(-limit, limit + 0.25, 0.5, dtype=np.float64)
    estimate = best(coarse)
    fine = np.arange(estimate - 0.6, estimate + 0.6001, 0.02, dtype=np.float64)
    return float(np.clip(best(fine), -limit, limit))

def _subsample_peak(values: np.ndarray) -> float:
    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    index = int(np.argmax(magnitude))
    if index == 0 or index + 1 >= len(magnitude):
        return float(index)
    y0, y1, y2 = (float(magnitude[index - 1]), float(magnitude[index]),
                  float(magnitude[index + 1]))
    denominator = y0 - 2.0 * y1 + y2
    correction = 0.0 if abs(denominator) < 1.0e-30 else 0.5 * (y0 - y2) / denominator
    return float(index + np.clip(correction, -0.5, 0.5))


@dataclass(frozen=True)
class TimeAlignedHrtf:
    canonical: CanonicalHrtf
    aligned_hrir: np.ndarray
    runtime_delay_samples: np.ndarray
    embedded_delay_removed_samples: np.ndarray
    delay_source: str

    def __post_init__(self):
        object.__setattr__(self, "aligned_hrir", _readonly(self.aligned_hrir, np.float64))
        object.__setattr__(self, "runtime_delay_samples", _readonly(
            self.runtime_delay_samples, np.float64))
        object.__setattr__(self, "embedded_delay_removed_samples", _readonly(
            self.embedded_delay_removed_samples, np.float64))


def time_align_hrtf(canonical: CanonicalHrtf) -> TimeAlignedHrtf:
    """Separate one delay representation before directional interpolation.

    Trusted non-zero ``Data.Delay`` is external to ``Data.IR`` and is therefore
    retained without de-rotating the FIR.  When ``Data.Delay`` is identically
    zero, ordinary measured HRIRs with a positive onset use their per-ear main
    peaks.  A zero-origin effective FIR is already expressed at one common
    time origin; its interaural phase is therefore retained in ``Data.IR``.

    These representations are mutually exclusive.  Runtime rendering must
    restore exactly the delay separated here and must not add any second ear
    delay or phase-group delay.
    """
    hrir = np.asarray(canonical.hrir, dtype=np.float64)
    if np.max(np.abs(canonical.delay_samples), initial=0.0) > 1.0e-12:
        return TimeAlignedHrtf(
            canonical=canonical,
            aligned_hrir=hrir.copy(),
            runtime_delay_samples=np.asarray(canonical.delay_samples, dtype=np.float64),
            embedded_delay_removed_samples=np.zeros_like(canonical.delay_samples),
            delay_source="Data.Delay (external; applied once at render time)",
        )

    measurements = canonical.measurements
    runtime = np.zeros((measurements, 2), dtype=np.float64)
    removed = np.zeros_like(runtime)
    used_peak = 0
    retained_embedded_phase = 0
    for measurement in range(measurements):
        peaks = np.asarray([
            _subsample_peak(hrir[measurement, 0]),
            _subsample_peak(hrir[measurement, 1]),
        ], dtype=np.float64)
        if float(np.max(peaks)) > 2.0:
            delays = peaks
            used_peak += 1
        else:
            # An effective response can have both ear FIRs beginning at sample
            # zero while still carrying the correct ITD in complex phase.  Do
            # not invent an external delay which SOFA did not author.
            delays = np.zeros(2, dtype=np.float64)
            retained_embedded_phase += 1
        runtime[measurement] = delays
        removed[measurement] = delays

    aligned = np.empty_like(hrir)
    for measurement in range(measurements):
        for ear in range(2):
            aligned[measurement, ear] = shift_signal_fft(
                hrir[measurement, ear], -float(removed[measurement, ear]))
    source = (
        f"embedded Data.IR arrival separation: peak={used_peak}, "
        f"zero-origin embedded phase retained={retained_embedded_phase}; "
        "positive onset restored once at render time")
    return TimeAlignedHrtf(
        canonical=canonical,
        aligned_hrir=aligned,
        runtime_delay_samples=runtime,
        embedded_delay_removed_samples=removed,
        delay_source=source,
    )
