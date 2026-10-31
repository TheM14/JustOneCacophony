"""SOFA spatial metadata adaptation on a sample-timed trajectory."""
from __future__ import annotations

import numpy as np
from binaural_metadata import OamdPositionTimeline
from oamd_bits import frame_update
from variant_error import UnsupportedVariantError


DISTANCE_FACTORS = (1.1, 1.3, 1.6, 2., 2.5, 3.2, 4., 5., 6.3, 7.9, 10., 12.6,
                    15.8, 20., 25.1, 50.1)


class SofaPositionTimeline(OamdPositionTimeline):
    def __init__(self, object_count=15, bed_positions=None):
        super().__init__(object_count, bed_positions)
        self.distance_factors = np.ones(16, dtype=np.float64)

    def _targets(self):
        targets = super()._targets()
        for index, target in enumerate(targets, 1):
            if index in self.bed_positions or self.distance_factors[index] == 1:
                continue
            value = np.asarray(target, dtype=np.float64)
            boundary = float(np.max(np.abs(value)))
            if boundary <= 1e-12:
                raise UnsupportedVariantError("sofa", "undefined_outside_direction",
                                              "Outside object has no ray from the listener")
            targets[index-1] = tuple(value/boundary*self.distance_factors[index])
        return targets

    def submit_update(self, update, **timing):
        if update.get("program", {}).get("num_isf_objects", 0):
            raise UnsupportedVariantError("sofa", "isf_not_supported", "ISF spatial mapping is unsupported")
        for info in update.get("objects", ()):
            slot = int(info["object"])
            if info.get("at_infinity"):
                raise UnsupportedVariantError("sofa", "infinite_distance",
                                              "Infinite-distance rendering is unsupported")
            if info.get("use_screen_ref"):
                raise UnsupportedVariantError("sofa", "screen_mapping_required",
                                              "Screen-anchored objects require a calibrated mapping")
            if "distance_specified" in info:
                index = info.get("distance_factor_idx")
                if info["distance_specified"] and (index is None or not 0 <= index < 16):
                    raise ValueError("invalid outside distance index")
                self.distance_factors[slot] = DISTANCE_FACTORS[index] if info["distance_specified"] else 1.
        return super().submit_update(update, **timing)

    def submit_payload(self, payload, **timing):
        update = frame_update(payload, retain_spatial=True)
        self.submit_update(update, **timing)
        return update
