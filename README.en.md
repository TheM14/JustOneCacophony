# JustOneCacophony — JOC

[中文版](README.md)

> JustOneCacophony is an experimental/test implementation of E-AC-3 JOC for studying JOC parsing, reconstruction, rendering, and the associated mathematics.

The project can extract and parse EMDF, ID14 JOC parameters, and ID11 OAMD metadata from common E-AC-3 JOC streams. It combines those data with the core 5.1 PCM decoded by FFmpeg, reconstructs LFE plus 15 object channels, and writes either ADM BWF or a WAV file for a selected speaker layout.

This is research code, not a complete, standards-compliant, or production-grade Dolby JOC decoder. It covers only the stream forms currently implemented. Unknown variants fail explicitly—because when the math goes wrong, all that may remain is the cacophony.

## Current features

- Scan common contiguous EMDF containers in E-AC-3 sync frames.
- Parse ID14 dense JOC parameters, Huffman data, differential matrices, and `joc_clipgain`.
- Parse ID11 OAMD position updates and build object trajectories.
- Reconstruct LFE plus 15 object channels through analysis QMF, parameter interpolation, the object matrix, and inverse QMF.
- Write a 25-channel ADM BWF: a 10-channel 7.1.2 bed (silent except for LFE) plus 15 objects.
- Render directly to `2.0`, `3.1`, `5.1`, `7.1`, `5.1.2`, `5.1.4`, `7.1.2`, `7.1.4`, `9.1.4`, or `9.1.6`.
- Write float32 or PCM24 WAV and require an explicit policy when PCM24 would clip.
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
       └─ speaker WAV for the selected layout
```

The Python and C++ backends follow the same documented mathematics. The native core handles the state-heavy DSP and speaker rendering; high-level bitstream parsing, ADM assembly, and CLI behavior remain in Python.

## Requirements

- Python 3.10+
- NumPy 1.24+
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

When PCM24 may clip in a non-interactive environment, select a policy explicitly:

```powershell
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action abort
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action float32
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action continue
```

Metadata and diagnostics:

```powershell
python main.py input.m4a --print-metadata summary
python main.py input.m4a --metadata-only --print-metadata frames
python main.py input.m4a --metadata-cache metadata_cache
python main.py input.m4a --metadata-dir metadata_cache
```

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
├─ data/                 runtime table data for Python
├─ lib/                  native runtime drop-in directory (create as needed)
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
- float32 and PCM24 output quantization.

See the [mathematical notes](docs/math.en.md) for the equations used by the decoding and rendering process.

## Known limitations

- Only the common contiguous EMDF transport is covered. Fragmented transport across multiple audio-block skip fields is not covered.
- Dense JOC is the main path. The Sparse JOC branch should not be treated as supported.
- The speaker path currently covers ordinary point objects; extent, spread, divergence, and similar modes are outside the supported scope.
- Multi-data-point streams, uncommon band configurations, and unusual OAMD scheduling have less coverage than common 12-band, single-data-point material.
- A speaker limiter is outside the current primary formula.
- ADM output, native binaries, and speaker layouts still need broader interoperability checks across platforms, players, and real material.

## Documentation

- [Mathematical notes](docs/math.en.md) · [中文](docs/math.md)
- [Native-core notes](docs/native.en.md) · [中文](docs/native.md)
