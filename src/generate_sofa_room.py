"""Generate a float64 FIR realization of the existing project room FDN."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from public_room import LateFdnConfig, SharedUnitaryFdn, ShoeboxRoomConfig


SCHEMA = "sofa-project-room-v1"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "sofa-room.npz"
ROOM_PATHS = {"legacy": DEFAULT_PATH, "compact": DEFAULT_PATH.with_name("sofa-room-compact.npz")}


def preset_config(preset):
    early, late = ShoeboxRoomConfig(), LateFdnConfig()
    provenance = {"preset": preset, "size_time_scale": 1.}
    if preset == "compact":
        dimensions = np.array([8.5, 11.3, 5.2])
        original = np.array(early.dimensions_m)

        def volume_over_area(d):
            return float(np.prod(d) / (2 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])))

        scale = volume_over_area(dimensions) / volume_over_area(original)
        early = replace(early, dimensions_m=tuple(dimensions), listener_position_m=tuple(dimensions / 2))
        late = replace(late, rt60_seconds=late.rt60_seconds * scale,
                       delay_seconds=tuple(v * scale for v in late.delay_seconds),
                       allpass_seconds=tuple(v * scale for v in late.allpass_seconds))
        provenance.update(size_time_scale=scale,
                          geometry_reference="Zahorik 2009, Table 1 model 9, L=11.3 W=8.5 H=5.2 m",
                          geometry_reference_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2730711/",
                          scaling="V/S ratio; same effective average wall absorption; original FDN algorithm",
                          reproduction_scope="geometry only; not a reproduction of the paper's BRIR model")
    elif preset != "legacy":
        raise ValueError("room preset must be compact or legacy")
    return early, late, provenance


def generate(path=None, *, preset="legacy"):
    started = time.perf_counter()
    early, config, provenance = preset_config(preset)
    fdn = SharedUnitaryFdn(config)
    length = int(np.ceil(2 * config.rt60_seconds * config.sample_rate_hz))
    probe_length = int(np.ceil(config.rt60_seconds * config.sample_rate_hz))
    impulse = np.zeros(length + probe_length)
    impulse[0] = 1.
    response = fdn.process(impulse)
    kernel = np.ascontiguousarray(response[:length].T, dtype="<f8")
    discarded = float(np.sum(response[length:] ** 2) / np.sum(kernel ** 2))
    if discarded > 1e-12:
        raise ValueError(f"FDN FIR tail exceeds the -120 dB energy budget: {discarded}")
    metadata = dict(schema=SCHEMA, sample_rate_hz=config.sample_rate_hz,
                    fdn_config=asdict(config), early_config=asdict(early), **provenance,
                    algorithm=fdn.info(), impulse_samples=length,
                    discarded_probe_samples=probe_length, discarded_energy_ratio=discarded,
                    kernel_sha256=hashlib.sha256(kernel.tobytes()).hexdigest(),
                    source_algorithm_sha256=hashlib.sha256(
                        Path(__file__).with_name("public_room.py").read_bytes()).hexdigest(),
                    generation_seconds=time.perf_counter() - started)
    path = Path(ROOM_PATHS[preset] if path is None else path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=np.array(SCHEMA), sample_rate_hz=np.int64(48000),
                        late_fir=kernel, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(dict(path=str(path), taps=length, bytes=path.stat().st_size,
                         preset=preset, dimensions_m=early.dimensions_m, rt60_parameter_s=config.rt60_seconds,
                         discarded_energy_ratio=discarded,
                         generation_seconds=metadata["generation_seconds"]), indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--preset", choices=("all", "legacy", "compact"), default="all")
    args = parser.parse_args()
    if args.preset == "all":
        if args.output is not None:
            parser.error("a custom output requires --preset legacy or compact")
        for name in ROOM_PATHS:
            generate(preset=name)
    else:
        generate(args.output, preset=args.preset)
