"""Configured reference environment, geometry and scalar level calibration."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

from rosella_bank import RosellaBank
from rosella_direct import _profile_geometry, direct_and_room_send
from rosella_model import load_personalized_headphone
from rosella_room import RosellaRoomFir


DEFAULT_REFERENCE_CONFIG = Path(__file__).resolve().parents[1]/"local"/"binaural_reference.json"
REFERENCE_MODES = {"near": 1, "mid": 3, "far": 2}


@dataclass(frozen=True)
class ReferencePosition:
    direction: np.ndarray
    distance_m: float
    ear_distance_m: float
    centre_weight: float
    direct_gain: float
    room_send: float


class ReferenceEnvironment:
    def __init__(self, config=None, *, mode=None):
        self.config_path = Path(config or os.environ.get("BINAURAL_REFERENCE_CONFIG") or
                                DEFAULT_REFERENCE_CONFIG).expanduser().resolve()
        with self.config_path.open(encoding="utf-8") as stream:
            spec = json.load(stream)
        self.configuration_sha256 = hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if spec.get("schema") != "binaural-reference-v1":
            raise ValueError("unsupported reference configuration")
        path = Path(spec["model_path"]).expanduser()
        if not path.is_absolute():
            path = self.config_path.parent/path
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.model_sha256.lower() != str(spec["model_sha256"]).lower():
            raise ValueError("reference data hash mismatch")
        self.model = load_personalized_headphone(path)
        self.calibration_profile_index = int(spec["profile_index"])
        self.profile_id = str(spec["profile_id"])
        if self.calibration_profile_index not in (2, 3) or self.model.sample_rate != 48000:
            raise ValueError("unsupported reference profile")
        self.mode = ({v: k for k, v in REFERENCE_MODES.items()}[self.calibration_profile_index]
                     if mode is None else str(mode).lower())
        if self.mode not in REFERENCE_MODES:
            raise ValueError("reference mode must be near, mid, or far")
        self.profile_index = REFERENCE_MODES[self.mode]
        if int(self.model.header_integer_fields[0]) != 0:
            raise ValueError("reference-v1 requires a Cartesian mapping")
        self.profile = self.model.profiles[self.profile_index]
        self.compiled_receivers = spec.get("compiled_receiver_positions_m")
        self.room_slots = int(spec.get("room_impulse_slots", 4096))
        self.calibration_directions = int(spec.get("level_calibration_directions", 128))
        if not 128 <= self.room_slots <= 16384 or not 16 <= self.calibration_directions <= 2048:
            raise ValueError("reference configuration exceeds supported bounds")
        self.h0, self.coupling = map(float, self.model.header_float_scalars[:2])
        if self.h0 <= 0 or self.coupling <= 0:
            raise ValueError("invalid reference energy coupling")
        self.minimum_distance_m = float(self.profile.minimum_normalized_radius)*float(self.profile.distance_scale_m)
        self.reference_distance_m = self.map_position((0., 1., 0.)).distance_m

    def map_position(self, position, *, profile_index=None):
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("reference position must contain three finite values")
        extension = max(1., float(np.max(np.abs(position))))
        if extension > 50.1+1e-9:
            raise ValueError("position exceeds the finite metadata domain")
        index = self.profile_index if profile_index is None else int(profile_index)
        profile, direction, radius, _, _ = _profile_geometry(self.model, position/extension, index)
        distance = float(profile.distance_scale_m)*radius*extension
        minimum = float(profile.minimum_normalized_radius)*float(profile.distance_scale_m)
        regularized = max(distance, minimum)
        alpha = distance/regularized
        effective = self.h0*distance
        if index == REFERENCE_MODES["near"]:
            a, send = 1., 0.
        else:
            a = 1/math.sqrt(1+self.coupling*effective*effective)
            send = effective*a
        return ReferencePosition(np.asarray(direction), distance, regularized, alpha, a, send)

    def create_room(self):
        return RosellaRoomFir(self.model, impulse_slots=self.room_slots)

    @staticmethod
    def create_bank(source_channels):
        return RosellaBank(1, source_channels=source_channels, output_channels=2)

    def calibrate(self, field, spectrum_at):
        from public_filterbank import hybrid_gain_synthesis_dictionary
        from rosella_direct import _FIELD_CACHE
        # Discard stale cache entries when object ids are reused.
        _FIELD_CACHE.pop(id(self.model), None)
        count = self.calibration_directions
        i = np.arange(count, dtype=np.float64)
        z = 1-2*(i+.5)/count
        phi = i*np.pi*(3-math.sqrt(5))
        unit = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
        dictionary = hybrid_gain_synthesis_dictionary(field.fft_size)
        band = (field.frequencies >= 100) & (field.frequencies <= 16000)
        weights = 1/field.frequencies[band]
        reference_energy = 0.
        new_energy = 0.
        for direction in unit:
            adm = (-direction[1], direction[0], direction[2])
            state = self.map_position(adm, profile_index=self.calibration_profile_index)
            parameters = direct_and_room_send(self.model, adm, self.calibration_profile_index)
            vector = np.empty((2, 154), dtype=np.float64)
            vector[:, 0::2], vector[:, 1::2] = parameters.gains.real, parameters.gains.imag
            reference = np.fft.rfft(vector @ dictionary.T, field.fft_size)
            target = spectrum_at(state)*state.direct_gain
            reference_energy += float(np.sum(np.abs(reference[:, band])**2*weights))
            new_energy += float(np.sum(np.abs(target[:, band])**2*weights))
        if min(reference_energy, new_energy) <= 1e-24 or not math.isfinite(new_energy):
            raise ValueError("reference level calibration has no finite signal energy")
        return math.sqrt(reference_energy/new_energy)

    def info(self):
        return dict(profile_id=self.profile_id, model_sha256=self.model_sha256,
                    configuration_sha256=self.configuration_sha256,
                    mode=self.mode, selected_profile_index=self.profile_index,
                    calibration_profile_index=self.calibration_profile_index,
                    distance_semantics="reference-reproduction-metres", reference_distance_m=self.reference_distance_m,
                    regularized_distance_m=self.minimum_distance_m,
                    filterbank=1, room_impulse_slots=self.room_slots,
                    calibration_directions=self.calibration_directions)
