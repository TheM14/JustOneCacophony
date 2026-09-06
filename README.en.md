# JustOneCacophony — JOC

[中文版](README.md)

> JustOneCacophony is an experimental/test implementation of E-AC-3 JOC for studying JOC parsing, reconstruction, rendering, and the associated mathematics.

The project can extract and parse EMDF, ID14 JOC parameters, and ID11 OAMD metadata from common E-AC-3 JOC streams. It combines those data with the core 5.1 PCM decoded by FFmpeg, reconstructs LFE plus 15 object channels, and writes ADM BWF, a WAV file for a selected speaker layout, or direct binaural stereo using a standard SOFA HRTF.

This is research code, not a complete, standards-compliant, or production-grade JOC decoder. It covers only the stream forms currently implemented. Unknown variants fail explicitly—because when the math goes wrong, all that may remain is the cacophony.

## Current features

- Scan common contiguous EMDF containers in E-AC-3 sync frames.
- Parse ID14 dense JOC parameters, Huffman data, differential matrices, and `joc_clipgain`.
- Parse ID11 OAMD position updates and build object trajectories.
- Reconstruct LFE plus 15 object channels through analysis QMF, parameter interpolation, the object matrix, and inverse QMF.
- Write a 25-channel ADM BWF: a 10-channel 7.1.2 bed (silent except for LFE) plus 15 objects.
- Render directly to `2.0`, `3.1`, `5.1`, `7.1`, `5.1.2`, `5.1.4`, `7.1.2`, `7.1.4`, `9.1.4`, or `9.1.6`.
- Run public SOFA binaural rendering directly from `pcm16 + ID11/OAMD`, without a temporary ADM BWF.
- Keep the binaural DSP in float64/complex128, including 961-sample latency compensation, cross-frame state, and the room tail.
- Use a shared float32/PCM24 WAV writer and explicit PCM24 clipping policy for direct outputs.
- Use the NumPy backend or an optional C++20 core through `ctypes`; `auto` falls back to Python when the native library is unavailable.
- Read or write metadata sidecars and produce metadata, timing, and output reports.

## Processing flow

```text
M4A / E-AC-3
  ├─ FFmpeg extracts E-AC-3 and decodes the core 5.1 PCM
  ├─ EMDF → ID14 JOC parameters → object matrix
  ├─ core PCM → analysis QMF → parameter interpolation → inverse QMF
  ├─ ID11 OAMD → object positions and timing
  └─ LFE + 15 objects
       ├─ 25ch ADM BWF
       ├─ speaker WAV for the selected layout
       └─ direct ID11 timeline + SOFA HRTF → binaural WAV
```

The Python and C++ backends follow the same mathematics for JOC object reconstruction and speaker rendering. The public SOFA binaural backend currently runs in Python; bitstream parsing, the OAMD timeline, and CLI behavior also remain in Python.

## Requirements

- Python 3.10+
- NumPy 1.24+
- h5py 3.8+
- SciPy 1.10+
- A standalone FFmpeg executable; `ffmpeg-python` is not required. FFmpeg is discovered through `PATH` by default or selected with `--ffmpeg`
- Optional: CMake and a C++20 toolchain to build the native core

Install the Python dependency in a project-specific environment:

```powershell
python -m pip install -r requirements.txt
```

If FFmpeg is not on `PATH`:

```powershell
python main.py input.m4a --ffmpeg C:\path\to\ffmpeg.exe
```

## Usage

Write a 25-channel ADM BWF by default:

```powershell
python main.py input.m4a
```

Select a backend or output path:

```powershell
python main.py input.eac3 -o output.adm.wav --backend python
python main.py input.m4a --backend native --native-threads 2
python main.py input.m4a --native-library lib/eac3joc_core.dll
```

Write a speaker-layout WAV directly:

```powershell
python main.py input.m4a --speaker-layout 2.0 --speaker-format float32
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24
python main.py input.m4a --speaker-layout 7.1.2 --speaker-output output.7.1.2.wav
```

Write binaural stereo directly (ordinary objects are Near/Mid/Far only; Mid is
the default). The HRTF input accepts three sources:

```powershell
# 1) SOFA (defaults to HRTF/binaural.sofa, or an explicit path)
python main.py input.m4a --binaural
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa

# 2) Rosella .personalized_headphone (defaults to HRTF/binaural.personalized_headphone)
python main.py input.m4a --binaural --personalized-headphone
python main.py input.m4a --binaural --personalized-headphone C:\HRTF\subject.personalized_headphone

# 3) .jochrtf compiled cache
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf

# Common options
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa `
  --binaural-mode near
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa `
  --hrtf-cache-policy disk
python main.py input.m4a --binaural --binaural-output output.binaural.wav
```

With none of the three specified, resolution tries, in order:
`HRTF/binaural.sofa`, the unique `.jochrtf` under `output/hrtf-cache`, then
`HRTF/binaural.personalized_headphone`; if none exist, an error asks for an
explicit path.

- `.sofa` is the portable source of truth; it can hold self-scanned or any
  generic HRTF data.
- `.personalized_headphone` is a model produced by Dolby's official
  personalization scan; its JSON parsing is implemented by this project
  (`src/rosella_model.py`) and does not invoke any Dolby software.
- `.jochrtf` is a project-internal cache compiled from SOFA; it is disposable,
  rebuildable, and written to `output/hrtf-cache` by default.

HRTF data lives under `HRTF/` (git-ignored): the default SOFA
`HRTF/binaural.sofa` and the default model
`HRTF/binaural.personalized_headphone`. Because the cache contains transformed
HRTF data, its use and redistribution remain subject to the source dataset's
terms. See [Binaural Rendering](docs/binaural.en.md) and
[Third-party notices](THIRD_PARTY_NOTICES.md) for format boundaries, formulas,
state, timing, and distribution considerations.

Speaker and binaural output share peak analysis, the WAV writer, and clipping policy. When PCM24 may clip in a non-interactive environment, select a policy explicitly:

```powershell
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action abort
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action float32
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action continue
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa `
  --binaural-format int24 --clip-action abort
```

Metadata and diagnostics:

```powershell
python main.py input.m4a --print-metadata summary
python main.py input.m4a --metadata-only --print-metadata frames
python main.py input.m4a --metadata-cache metadata_cache
python main.py input.m4a --metadata-dir metadata_cache
```

### Binaural render mode

`--binaural-mode off|near|mid|far` selects the binaural render mode; the default
is `mid`, and both outputs share this single option:

- **Direct binaural rendering** (`--binaural`): `off` is rejected (error);
  near/mid/far apply, defaulting to `mid`;
- **ADM BWF**: the low 3 binaural-render-mode bits of the last 15 JOC object
  entries in DBMD segment 10 carry `off=0/near=1/far=2/mid=3`, leaving the first
  10 bed entries unchanged; the default is `mid`, and `off` explicitly disables
  the binaural metadata hint.

```powershell
python main.py input.m4a --binaural-mode mid
python main.py input.m4a --binaural-mode off   # ADM BWF only: disable the DBMD hint
```

**The default `mid` is a human-specified rendering hint**; it is not original
binaural metadata extracted or recovered from the input E-AC-3 JOC bitstream,
nor does it represent the original mix's per-object binaural settings. The hint
does not change PCM, object trajectories, or direct speaker rendering. The
adjacent `.report.json` records `binaural_mode` (the mode name) and
`binaural_mode_value` (the ADM code; `null` for direct binaural output).

### OAMD time alignment

Object trajectories and direct speaker rendering both default to a metadata delay of `1473 samples`. This value describes the theoretical mapping between decoder-output PCM and OAMD updates. The speaker renderer retains its existing 32-sample control block, so the default update lands on effective block boundary `1472`:

```text
align32(1473) = 1472
```

Override the two paths with `--object-delay-samples` and `--speaker-metadata-offset`, respectively. The 1473-sample timing offset is distinct from the 640-value inverse-QMF filter/window state; 640 is a QMF state length, not a metadata delay.

The direct binaural path uses `--object-delay-samples`. Each ID11/OAMD event is
placed on an absolute sample timeline from its frame start, outer-subpayload
offset, and block offset, then shifted by that delay. Each 1536-sample input
frame is processed as three consecutive 512-sample blocks; the interpolated
position, direction, and profile are updated at each block's absolute starting
sample.

### Binaural calculation

See [Binaural Rendering Mathematics](docs/binaural.en.md) for QMF, hybrid processing, direction fields, distance, ITD, room processing, the 512-sample parameter updates above, and 961-sample latency compensation.

For all options:

```powershell
python main.py --help
```

Without `-o`, output still goes to `output/` at the repository root. The directory move intentionally preserves this behavior.

## Native core

The repository does not include native binaries by default. Download a prebuilt runtime for the current platform from a project Release, or build one locally, then place the runtime library under `lib/` at the repository root; create the directory if it is absent. To build it yourself, run CMake from the repository root:

```powershell
cmake -S native -B build/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/lib"
cmake --build build/cmake --config Release
cmake --install build/cmake --config Release
```

The runtime lookup order is:

1. `--native-library`;
2. `EAC3JOC_NATIVE_LIBRARY`;
3. the standard platform library name under `lib/`.

See the [native-core notes](docs/native.en.md) for ABI, state, and precision details.

## Repository layout

```text
JustOneCacophony/
├─ main.py               command-line entry point
├─ src/                  Python implementation modules
├─ native/               C/C++ acceleration core, C ABI, and required table data
├─ data/                 Python runtime table data
├─ lib/                  native runtime drop-in directory (create as needed)
├─ HRTF/                 user HRTF data directory (create as needed, git-ignored)
├─ output/               output directory (create as needed; the .jochrtf cache defaults to its hrtf-cache subdirectory)
├─ docs/                 math and native-core notes in both languages
├─ requirements.txt      Python dependency
├─ README.md             Chinese documentation
└─ README.en.md          English documentation
```

## Mathematical implementation

The main documented stages are:

- dense JOC differential reconstruction and dequantization;
- parameter-band mapping to 64 QMF subbands;
- cross-frame parameter interpolation;
- analysis/inverse QMF, surround delay, and FIR state;
- the 1217-sample LFE delay;
- OAMD Q15 coordinate conversion;
- equal-power panning over target-layout regions;
- layout-dependent position compensation and sample-wise gain ramps;
- float32 and PCM24 output quantization;
- SOFA canonical import, 64-QMF/77-hybrid projection, `36×2×77` fifth-order fields, exactly-once delay/phase, project early/late room behavior, and special LFE.

See the [mathematical notes](docs/math.en.md) for the equations used by the decoding and rendering process.

## Known limitations

- Only the common contiguous EMDF transport is covered. Fragmented transport across multiple audio-block skip fields is not covered.
- Dense JOC is the main path. The Sparse JOC branch should not be treated as supported.
- The speaker and SOFA binaural paths currently cover ordinary point objects; extent, spread, diffuse, divergence, channel lock, and similar controls are outside the supported scope.
- OAMD trim elements are boundary-checked and skipped; warp, balance, and trim parameters are not applied to raw object trajectories or speaker rendering.
- Multi-data-point streams, uncommon band configurations, and unusual OAMD scheduling have less coverage than common 12-band, single-data-point material.
- A speaker limiter is outside the current primary formula.
- The SOFA importer currently supports the strict `SimpleFreeFieldHRIR` FIR subset; other SOFA conventions require explicit adapters.
- The binaural runtime is fixed at 48 kHz, fifth order, and one measurement-radius shell at a time; the public binaural backend defaults to the native accelerator and falls back to Python when the native library is unavailable.
- ADM output, native binaries, speaker layouts, and binaural models still need broader interoperability checks across platforms, players, and real material.

## Documentation

- [Mathematical notes](docs/math.en.md) · [中文](docs/math.md)
- [Native-core notes](docs/native.en.md) · [中文](docs/native.md)
- [Binaural rendering](docs/binaural.en.md) · [中文](docs/binaural.md)
