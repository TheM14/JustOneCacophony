# JustOneCacophony — C++ Core

[中文](README.md) · [Mathematics](docs/math.en.md) · [Binaural rendering](docs/binaural.en.md) · [SIMD dispatch](docs/simd.en.md)

The C++ implementation of JustOneCacophony: an execution core for E-AC-3 JOC
bitstream parsing, object reconstruction and rendering. It extracts EMDF, ID14 JOC
parameters and ID11 OAMD metadata from E-AC-3 syncframes, combines them with the
core 5.1 PCM decoded by FFmpeg to rebuild the LFE and 15 object signals, and writes
ADM BWF, a WAV for a chosen speaker layout, or a binaural WAV using a compiled HRTF
directional field.

This is research code, not a complete, standard-conformant or production JOC
decoder. It covers the bitstream forms it implements and reports an explicit error
on unknown variants instead of pretending everything is in harmony.

## Building

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Only MSVC (VS 2022, static CRT) is validated locally; Linux and macOS are built and
unit-tested by `.github/workflows/ci.yml`. Floating-point behaviour is part of the
byte-exact acceptance, so fast-math is never enabled: `/fp:precise` on MSVC,
`-fno-fast-math` elsewhere.

Windows Release builds target AVX2 by default (`JOC_ENABLE_AVX2`, ON, see
`CMakeLists.txt`). That switch is itself part of the byte-exact acceptance — the
SHA-256 of every rendered output is identical — and buys 12.4% on the SOFA binaural
kernel and 1.8% on Rosella. The price is a runtime requirement: such a `joc_core.dll`
executes AVX2 instructions and dies on an illegal instruction on pre-2013 x86. There
is no runtime dispatch, so a binary is one or the other:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DJOC_ENABLE_AVX2=OFF
```

gives a baseline-ISA (SSE2) build that runs on any x86-64. Non-MSVC builds never
receive the flag.

### Paths and encoding

Every path inside the library is **UTF-8**, converted only at the OS boundary
(`src/foundation/fs_utf8.*`): on Windows through `std::filesystem::path` (UTF-16
inside) into `_wfopen`/`CreateProcessW`, and as plain bytes elsewhere. Command line
arguments are re-parsed from `GetCommandLineW` + `CommandLineToArgvW` on Windows and
the console code page is set to UTF-8, so non-ASCII paths (Japanese, Chinese, ...)
work for the input, the ffmpeg child process and the output files alike; a
non-ASCII path regression case runs in `ctest`.

## Artifacts

| Artifact | Purpose |
|---|---|
| `joc_core.dll` | The engine: bitstream parsing, JOC/OAMD, DSP, speaker and binaural rendering, ADM BWF/WAV writing, file task, streaming surface |
| `joc_cli.exe` | Command line frontend for file tasks |
| `include/joc_core.h` | Engine, telemetry and file-task interface (pure C) |
| `include/joc_stream.h` | Embedder-facing streaming push/pull interface (pure C, self-contained) |

## Command line

The arguments match the reference Python CLI exactly: the input is positional,
**ADM BWF is the default output**, `--speaker-layout` or `--binaural` selects the
other two modes, and without `-o` the result lands in `output/`.

```powershell
# Default: ADM BWF (inherently 24-bit, so there is no format option)
joc_cli "07. Gold Forever (2021 Master).m4a"
#   -> output/07. Gold Forever (2021 Master).adm.wav

joc_cli input.m4a -o out/adm.wav                 # explicit output

# Speaker layout
joc_cli input.m4a --speaker-layout 5.1           # -> output/<name>.5.1.wav
joc_cli input.m4a --speaker-layout 7.1.4 --speaker-output out/714.wav --speaker-format int24

# Binaural (HRTF defaults to <exe>/HRTF/binaural.sofa, then <exe>/HRTF/binaural.personalized_headphone)
joc_cli input.m4a --binaural
joc_cli input.m4a --binaural --sofa-hrtf HRTF/other.sofa   # another SOFA
joc_cli input.m4a --binaural --personalized-headphone      # Rosella personalisation
#   -> output/<name>.binaural.wav

# Other common switches
joc_cli input.m4a --duration 30 --gain-db -3 --trajectory-mode dense64
joc_cli input.eac3 --metadata-only --print-metadata summary   # parse and print metadata only
```

`--speaker-format` / `--binaural-format` default to `float32`; an `int24` request that
would clip follows `--clip-action` (default `ask`; a non-interactive terminal must
pass `continue`, `float32` or `abort`). `--duration` is in **seconds**, and
`--object-delay-samples`, `--speaker-metadata-offset`, `--binaural-tail-seconds` and
`--binaural-tail-threshold` (1e-8, the binaural tail trim) keep the reference
defaults. A run always writes `<output>.report.json` (`--report-json` overrides it).

**Differences from the reference:** `--sofa-hrtf`, `--personalized-headphone`,
`--backend python` and the metadata sidecars (`--metadata-dir`, `--metadata-cache`,
`--metadata-backend sidecar`) are unavailable in this build and fail immediately
with an explanation instead of being ignored. This build adds `--bed` (pre-decoded
6-channel float32 PCM, which skips ffmpeg decoding), `--kernels`, `--work-dir`,
`--report-json`, `--dry-run` and `--quiet`.

## Library integration

The engine and the file task are exposed by `joc_core.h`: `joc_task_validate` /
`joc_task_execute` run a file task and report state events through a callback, and
`joc_task_result` carries frame counts, peak, byte count and SHA-256.

Players and decoder components use `joc_stream.h`: the caller pushes E-AC-3 bytes
and the matching core PCM (or already-rebuilt objects16) at its own pace and pulls
rendered PCM. Any chunking is allowed, and the result is byte-identical to the file
task.

```c
joc_stream_config config = {0};
config.struct_size = sizeof(config);
config.input = JOC_STREAM_IN_EAC3;      /* or JOC_STREAM_IN_PCM_OBJECTS16 */
config.output = JOC_STREAM_OUT_SPEAKER; /* or BINAURAL / PCM_OBJECTS16 */
config.speaker_layout_name = "5.1";
joc_stream* stream = NULL;
joc_stream_create(&config, &stream);
/* loop: joc_stream_push(...) / joc_stream_pull(...) */
joc_stream_flush(stream);
joc_stream_destroy(stream);
```

Contract: state is instance-private, so streams coexist; push and pull on one
instance must come from the same thread; rendering is stateful, so **this version
offers no seek** - repositioning means decoding from the start of the stream. The
kernel latency is 961 samples for speaker/binaural output and `joc_stream_flush`
drains the binaural room tail.

## Layout

```text
include/     public C ABI: joc_core.h (engine/file task), joc_stream.h (streaming),
             eac3joc_core.h (upstream ABI)
src/         implementation: eac3_transport, emdf, joc_bitstream, joc_core, oamd,
             timeline, speaker, binaural, hrtf, adm, io, telemetry, task, stream,
             api, cli, simd
tests/       unit tests (CTest, self-contained, no external data)
docs/        mathematics, the binaural rendering flow and SIMD dispatch
```

Eight files under `src/` are byte-identical copies of the upstream JustOneCacophony
native library and are never edited (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Compatibility note

`object_delay_samples` defaults to **1473**, preserving the behaviour of the existing
implementation; it is a configurable field and changing it changes the OAMD/ADM time
alignment. Upstream investigation suggests the value should be 0; this project keeps
the current default to stay byte-identical.

## License

MIT, see [LICENSE](LICENSE). Third-party provenance and patent boundaries are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
