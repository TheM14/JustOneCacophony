"""Explicit offline export of a SOFA source to the compact cache format."""
import argparse
from pathlib import Path
from sofa_canonical import load_simple_free_field_hrir
from sofa_hrtf_field import compile_sofa_hrtf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sofa", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radius-m", type=float)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".jochrtf":
        parser.error("output must use .jochrtf")
    canonical = load_simple_free_field_hrir(args.sofa, target_sample_rate_hz=48000)
    if args.radius_m is None and len(canonical.shells_m) != 1:
        parser.error("multi-radius input requires --radius-m")
    radius = float(canonical.shells_m[0]) if args.radius_m is None else args.radius_m
    field = compile_sofa_hrtf(canonical, shell_radius_m=radius, cache_policy="none")
    field.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
