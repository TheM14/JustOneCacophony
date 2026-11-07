"""Static lookup data used by the OAMD metadata application path.

``metadata_params.json``
    ISF base coefficients and layout descriptors, applied as lookup tables.
    gain, size, priority, position, delta and refinement values are computed
    in code; the size tables are generated from the region geometry at runtime,
    and bed routing is derived from the speaker layouts.

The data file is parsed once per process.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "oamd"


@lru_cache(maxsize=1)
def metadata_params():
    """Return the parsed ``metadata_params.json`` mapping."""
    with (_DATA_DIR / "metadata_params.json").open(
            "r", encoding="utf-8") as handle:
        return json.load(handle)


