# Binaural rendering

[中文](binaural.md) · [README](../README.en.md)

JOC renders binaural audio from two kinds of input:

| Input | Renderer | Distance and room |
|---|---|---|
| `.personalized_headphone` (default) | Rosella DSP, filterbank and model parameters | the model's own Near/Mid/Far and room handling, unchanged |
| `.jochrtf` (reserved) | 77-band directional field, 64-QMF/hybrid analysis and synthesis | direct sound only by default; `--jochrtf-room on` adds the project room |

A `.jochrtf` is fitted offline from SOFA measurements by a separate compiler;
this repository loads and renders the result through `--compiled-hrtf-cache`.

```powershell
python main.py input.m4a --binaural
python main.py input.m4a --binaural --personalized-headphone --binaural-mode far --rosella-filterbank 2
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf --jochrtf-room on
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf --jochrtf-room on
```

Without an explicit input the renderer uses `HRTF/binaural.personalized_headphone`
from the project directory. A `.jochrtf` is never generated or guessed: it has to
be named. Binaural DSP runs in float64/complex128 and updates parameters every 128
samples.

## Point-source scale

The `.jochrtf` route keeps the Near/Mid/Far point-source scales of about 1 m,
2.193 m and 6.402 m, defaulting to Mid. For an ADM right/front/up position
`p = (x, y, z)`:

```text
q_SOFA    = (y, -x, z) / ||p||
d_nominal = reference_distance_m * ||p||
```

The reference distance only gives object coordinates a nominal metric scale: it
does not change the field response and adds no 1/r attenuation or propagation
delay. A coincident source keeps its last valid direction, or front when there is
no history.

## The `.jochrtf` room

Off by default. `--jochrtf-room on` enables the project's own room, the same algorithm
the earlier public route used, computed in code with no asset files:

- six first-order image reflections (`public_room.first_order_image_sources`, the
  original 18x18x14 m geometry);
- a shared FDN tail (four orthogonal Hadamard feedback lines, two all-pass diffusers,
  low-pass damping, RT60 0.85 s);
- early reflections scaled by measurement radius over path distance, late send scaled
  by mode and normalised radius.

It is not a measured BRIR and not an external standard room. With the switch off the
renderer skips room processing and outputs only the field's direct binaural sound; the
room parameters are fixed at construction, with no preset and no external file.

## Object input

Each reconstructed frame is `[1536,16]` PCM: LFE at slot 0 and fifteen
reconstructed object/bed slots at 1–15. The ID11 timeline feeds these directly,
without another decode or an intermediate loudspeaker render. The existing
1473-sample object compensation and 32-sample metadata alignment are retained.
Parameters update every 128 samples.

LFE keeps its separate low-pass treatment and no room send. Multi-position blocks,
ISF, explicit infinity and anchors requiring unavailable screen calibration are
rejected. `finish()` drains once. `--binaural-tail-seconds` bounds filterbank and
room draining.

## Compact representation

`.jochrtf` v1 is a pickle-free NumPy ZIP holding JSON metadata, 77 band centres,
`complex128[36,2,77]` directional coefficients, `float64[36,2]` delay coefficients
and `float64[2,2]` bounds. Loading checks the format version, shapes and scalar
ranges, rehashes every member, and recomputes the cache key to confirm that the
coefficients match their configuration; a mismatch rejects the file.

This repository owns the format, the cache key and the hash validation; the
fitting compiler lives in the separate checkout and shares the same format module
instead of duplicating it. The compact representation keeps its angular and
band-wise approximation error and is not equivalent to per-measurement precision.

Filterbank provenance is documented in [data notes](../data/README.en.md). See the
[MCA paper](https://doi.org/10.1109/TASLP.2023.3313908) for the correction model.
Algorithms are independently implemented; external measurements retain their own
licensing terms.
