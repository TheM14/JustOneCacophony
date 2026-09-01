# JustOneCacophony native-core notes

[中文](native.md) · [Back to README](../README.en.md)

## 1. Responsibility boundary

`native/` contains only the state-heavy, frequently called DSP and speaker-rendering kernels. High-level EMDF/JOC/OAMD parsing, error reporting, ADM assembly, and the CLI remain in Python.

Python calls a C ABI through the standard-library `ctypes` module. The native core does not use pybind11, Cython, FFTW, MKL, or OpenMP. It is an optional acceleration path and does not expand the set of supported stream variants.

Main files:

```text
native/include/eac3joc_core.h       C ABI
native/src/eac3joc_core.cpp         JOC/QMF object reconstruction
native/src/speaker_renderer.cpp     object-to-speaker rendering
native/src/qmf_tables.h             QMF tables
native/src/speaker_layouts.h        layout tables
native/src/joc_huffman_tables.h     JOC Huffman tables
src/native_renderer.py              JOC ctypes bridge
src/speaker_native_renderer.py      speaker ctypes bridge
```

## 2. JOC rendering ABI

An opaque renderer owns all cross-frame state. Its main call is:

```c
int ejoc_renderer_process(
    ejoc_renderer_handle handle,
    const float* bed5_planar,      /* [5][1536] */
    const float* lfe,              /* [1536] or NULL */
    uint32_t object_mask,
    const uint8_t* n_bands,        /* [15] */
    const uint8_t* n_dpoints,      /* [15] */
    const uint8_t* slope_idx,      /* [15] */
    const uint8_t* offset_ts,      /* [15][2] */
    const double* dq,              /* [15][2][5][23] */
    double clipgain,
    float phase_new,
    float output_scale,
    float* output16_planar);       /* [16][1536] */
```

Python performs dense-JOC Huffman decoding, differential reconstruction, and dequantization before the call. Sparse JOC is not silently passed to the dense native path.

Thread control is exposed as:

```c
int ejoc_renderer_set_threads(ejoc_renderer_handle handle, uint32_t total_threads);
uint32_t ejoc_renderer_thread_count(ejoc_renderer_handle handle);
```

`total_threads` includes the calling thread. Frames must be submitted sequentially to one renderer instance; the instance may parallelize work across objects and analysis channels.

## 3. Cross-frame state

Each JOC renderer stores:

- analysis FIFO: `double[5][9][64]`;
- L/R/C analysis delay: `float[3][10][64]`;
- Ls/Rs QMF delay: `complex<double>[2][10][64]`;
- Ls/Rs band-0 FIR history: `complex<double>[2][20]`;
- previous matrix interpolation values: `double[15][5][64]`;
- inverse-QMF state: `double[15][640]`;
- LFE delay: `double[1217]`.

This state belongs to the renderer instance. Processing cannot be arbitrarily segmented or reordered without a corresponding state checkpoint.

## 4. FFT, QMF, and precision

The native core contains a fixed 64-point radix-2 complex FFT:

- analysis QMF uses a forward FFT followed by division by 64;
- inverse QMF uses the fixed reorder, rotation, and 640-value active-window state;
- no external FFT library is called.

The JOC path uses:

- float32 core-PCM input;
- double matrices, complex QMF, FFT, FIR, and cross-frame state;
- float32 phase and final gain;
- float32 16-channel object output.

## 5. Speaker-rendering ABI

The same shared library exports object-to-speaker rendering:

```c
uint32_t ejoc_speaker_layout_channel_count(uint32_t speaker_bitfield);

ejoc_speaker_renderer_handle
ejoc_speaker_renderer_create(uint32_t speaker_bitfield);

int ejoc_speaker_renderer_process(
    ejoc_speaker_renderer_handle handle,
    const float* objects16_interleaved,
    uint32_t sample_count,
    uint32_t metadata_count,
    const uint32_t* metadata_offsets,
    const uint32_t* ramp_durations,
    const uint16_t* positions_q15,
    const uint8_t* region_indices,
    const uint8_t* height_enabled,
    const double* object_gains,
    double* output_interleaved);
```

Input channel 0 is LFE and channels 1–15 are objects. Each metadata entry is an object-state snapshot. `sample_count` must be a multiple of 32; unfinished gain ramps remain in the handle and continue across calls.

The speaker path uses float32 object input, double coordinates/gains/accumulation, and interleaved double output. Quantization to float32 or PCM24 happens when the WAV is written.

Supported layouts:

```text
2.0  3.1  5.1  7.1  5.1.2  5.1.4  7.1.2  7.1.4  9.1.4  9.1.6
```

## 6. Building

The CMake definition is `native/CMakeLists.txt`. Run from the repository root:

```powershell
cmake -S native -B build/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/lib"
cmake --build build/cmake --config Release
cmake --install build/cmake --config Release
```

Platform runtime names:

```text
Windows  lib/eac3joc_core.dll
Linux    lib/libeac3joc_core.so
macOS    lib/libeac3joc_core.dylib
```

The MSVC configuration uses the static CRT. Other runtime dependencies depend on the platform and toolchain and should be checked independently before publishing a prebuilt library.

The repository does not include native binaries by default. A prebuilt Release runtime or a locally built runtime can be placed directly under `lib/`.

## 7. Runtime lookup and fallback

Lookup order:

1. explicit `--native-library`;
2. `EAC3JOC_NATIVE_LIBRARY`;
3. the standard platform filename under `lib/`.

`--backend auto` falls back to NumPy when loading fails, and `--backend python` skips native discovery. The current CLI also prints the failure and falls back for `--backend native`; this existing behavior should not be read as successful native execution.

## 8. Implementation boundaries

- The native layer accepts only dense-JOC data already parsed by Python.
- The ABI fixes a 1536-sample JOC frame, at most 15 objects, at most 23 parameter bands, and at most 2 data points.
- The shared library and Python bridge must report the same ABI version.
- Only the ABI and data types are specified across platforms; bit-identical float64 results are not guaranteed.
- Private table headers under `native/src/` serve the native side only. The current repository does not include the scripts that generated those headers.

See the [mathematical notes](math.en.md) for the related formulas.
