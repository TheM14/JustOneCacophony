"""Project-owned distance policy for the public SOFA renderer."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DistanceState:
    profile: str
    normalized_radius: float
    reference_distance_m: float
    physical_distance_m: float
    direction_adm: np.ndarray


class ReferenceDistanceProfileV1:
    """Reference behavior, not a claim about any external public standard."""

    DISTANCE_M = {
        "near": 1.00000465,
        "mid": 2.19327927,
        "far": 6.40177584,
    }
    MINIMUM_DISTANCE_M = 0.10
    # Distance profiles in the reference renderer are presentation presets,
    # not an instruction to attenuate already-authored programme PCM by 1/r.
    # Use an energy-normalized dry/room crossfade instead.  The coefficient is
    # an explicit project calibration target.
    ROOM_ENERGY_COUPLING_PER_M2 = 0.01318359375
    PUBLIC_ROOM_CALIBRATION_GAIN = 1.4
    # Public, project-owned room coupling; it is not a SOFA or Dolby constant.
    LATE_SEND = {
        "near": 0.06,
        "mid": 0.16,
        "far": 0.28,
    }

    @classmethod
    def validate_profile(cls, profile: str) -> str:
        value = str(profile).strip().lower()
        if value not in cls.DISTANCE_M:
            raise ValueError("distance profile must be near, mid, or far")
        return value

    @classmethod
    def map_adm_position(cls, position, profile: str) -> DistanceState:
        name = cls.validate_profile(profile)
        values = np.asarray(position, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("ADM position must contain three finite Cartesian values")
        radius = float(np.linalg.norm(values))
        direction = (values / radius if radius > 1.0e-15
                     else np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        reference = float(cls.DISTANCE_M[name])
        distance = max(float(cls.MINIMUM_DISTANCE_M), radius * reference)
        return DistanceState(
            profile=name,
            normalized_radius=radius,
            reference_distance_m=reference,
            physical_distance_m=distance,
            direction_adm=np.asarray(direction, dtype=np.float64),
        )

    @staticmethod
    def inverse_distance_gain(measurement_radius_m: float,
                              path_distance_m: float) -> float:
        radius = float(measurement_radius_m)
        distance = float(path_distance_m)
        if not (math.isfinite(radius) and math.isfinite(distance)):
            raise ValueError("measurement and path distances must be finite")
        if radius <= 0.0 or distance <= 0.0:
            raise ValueError("measurement and path distances must be positive")
        return radius / distance

    @classmethod
    def direct_level_gain(cls, state: DistanceState) -> float:
        """Programme-normalized direct level for a distance presentation.

        Near is the SOFA reference response.  Mid/Far use an equal-power dry
        coefficient rather than a physical free-field 1/r attenuation.  Room
        distance still changes through image-path lengths and late send.
        """
        if state.profile == "near":
            return 1.0
        distance = float(state.physical_distance_m)
        return 1.0 / math.sqrt(
            1.0 + cls.ROOM_ENERGY_COUPLING_PER_M2 * distance * distance)

    @classmethod
    def room_calibration_gain(cls, state: DistanceState) -> float:
        del state
        return float(cls.PUBLIC_ROOM_CALIBRATION_GAIN)

    @classmethod
    def late_send(cls, state: DistanceState) -> float:
        base = float(cls.LATE_SEND[state.profile])
        radial = math.sqrt(max(state.normalized_radius, 0.0))
        return base * min(max(radial, 0.25), 1.5)

    @classmethod
    def info(cls) -> dict:
        return {
            "name": "ReferenceDistanceProfileV1",
            "reference_distance_m": dict(cls.DISTANCE_M),
            "minimum_distance_m": cls.MINIMUM_DISTANCE_M,
            "direct_level_policy": (
                "Near unity; Mid/Far equal-power dry coefficient, never raw 1/r "
                "programme attenuation"),
            "room_energy_coupling_per_m2": cls.ROOM_ENERGY_COUPLING_PER_M2,
            "public_room_calibration_gain": cls.PUBLIC_ROOM_CALIBRATION_GAIN,
            "late_send": dict(cls.LATE_SEND),
            "standard_claim": False,
        }
