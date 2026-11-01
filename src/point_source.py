"""Listener-centred point-source coordinates for direct HRTF rendering."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from reference_distance import ReferenceDistanceProfileV1


@dataclass(frozen=True)
class PointSourcePosition:
    direction: np.ndarray
    distance_m: float
    normalized_radius: float


class PointSourceReference:
    def __init__(self, reference_distance_m=None, *, mode=None):
        self.mode = ReferenceDistanceProfileV1.validate_profile("mid" if mode is None else mode)
        self.reference_distance_m = float(ReferenceDistanceProfileV1.DISTANCE_M[self.mode]
                                         if reference_distance_m is None else reference_distance_m)
        if not math.isfinite(self.reference_distance_m) or self.reference_distance_m <= 0:
            raise ValueError("source reference distance must be finite and positive")

    def map_position(self, position, *, previous_direction=None):
        values = np.asarray(position, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("source position must contain three finite values")
        if np.max(np.abs(values)) > 50.1 + 1e-8:
            raise ValueError("source position exceeds the finite metadata domain")
        radius = float(np.linalg.norm(values))
        direction = values[[1, 0, 2]] * [1., -1., 1.]
        if radius > 1e-12:
            direction /= radius
        elif previous_direction is None:
            direction = np.array([1., 0., 0.])
        else:
            direction = np.asarray(previous_direction, dtype=np.float64).copy()
        return PointSourcePosition(direction, radius * self.reference_distance_m, radius)

    def info(self):
        return dict(reference_variant="direct-point-source-v1",
                    reference_distance_m=self.reference_distance_m,
                    mode=self.mode, profile_distances_m=dict(ReferenceDistanceProfileV1.DISTANCE_M),
                    distance_semantics="nominal-object-coordinate-scale",
                    distance_attenuation="none; preserve authored programme gain",
                    radial_adaptation="none; retain selected SOFA measurement shell",
                    coincident_source_direction="previous direction, otherwise front",
                    additional_absolute_propagation_delay=False)
