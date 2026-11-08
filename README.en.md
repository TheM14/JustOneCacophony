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

The heaviest binaural loops are **runtime-dispatched**: one binary carries several
ISA variants and picks the widest one at startup from CPUID/XGETBV, with the output
byte-identical either way (see [SIMD dispatch](docs/simd.en.md)). `JOC_ENABLE_AVX2` /
`JOC_ENABLE_AVX512` (both ON by default) only decide whether those units are linked
in: carrying the AVX2 unit buys 12.4% on the .jochrtf binaural kernel and 1.8% on
Rosella, while a CPU without AVX2 still runs the baseline (SSE2) path. To produce a
baseline-only binary:

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
joc_cli input.m4a
#   -> output/input.adm.wav

joc_cli input.m4a -o out/adm.wav                 # explicit output

# Speaker layout
joc_cli input.m4a --speaker-layout 5.1           # -> output/<name>.5.1.wav
joc_cli input.m4a --speaker-layout 7.1.4 --speaker-output out/714.wav --speaker-format int24

# Binaural (defaults to <exe>/HRTF/binaural.personalized_headphone)
joc_cli input.m4a --binaural
joc_cli input.m4a --binaural --compiled-hrtf-cache HRTF/subject.jochrtf   # offline compiled field
joc_cli input.m4a --binaural --compiled-hrtf-cache HRTF/subject.jochrtf --jochrtf-room off
joc_cli input.m4a --binaural --personalized-headphone      # Rosella personalisation
#   -> output/<name>.binaural.wav
joc_cli input.m4a --binaural --personalized-headphone --rosella-filterbank 2   # captured bank

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

`--rosella-filterbank 1|2` picks the filterbank of the Rosella `.personalized_headphone`
path: `1` (default) is the public 13-tap / 6-slot bank, `2` is the captured 11-tap /
3-slot bank. Their latencies are 961 and 448 samples, and each keeps its own
coefficients, history and flush capacity. The option only applies to the Rosella
model; the `.jochrtf` path rejects it. See
[Rosella filterbank](docs/rosella_filterbank.en.md).

**Differences from the reference:** `--backend python` and the metadata sidecars
(`--metadata-dir`, `--metadata-cache`, `--metadata-backend sidecar`) are unavailable
in this build and fail immediately with an explanation instead of being ignored.
This build adds `--bed` (pre-decoded
6-channel float32 PCM, which skips ffmpeg decoding), `--kernels`, `--work-dir`,
`--report-json`, `--dry-run`, `--quiet` and `--rosella-filterbank`.

## Speaker layouts and channel order

`--speaker-layout` accepts the twelve layouts below, and the PCM/WAV channel order is the order in
the table:

| Layout | Channel order |
|---|---|
| 2.0 | L, R |
| 3.1 | L, R, C, LFE |
| 4.0 | L, R, Lrs, Rrs |
| 5.1 | L, R, C, LFE, Ls, Rs |
| 6.0 | L, R, Ls, Rs, Lrs, Rrs |
| 7.1 | L, R, C, LFE, Ls, Rs, Lrs, Rrs |
| 5.1.2 | L, R, C, LFE, Ls, Rs, Ltm, Rtm |
| 5.1.4 | L, R, C, LFE, Ls, Rs, Ltf, Rtf, Ltr, Rtr |
| 7.1.2 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltm, Rtm |
| 7.1.4 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf, Ltr, Rtr |
| 9.1.4 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Lw, Rw, Ltf, Rtf, Ltr, Rtr |
| 9.1.6 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Lw, Rw, Ltf, Rtf, Ltm, Rtm, Ltr, Rtr |

`4.0` is the quad layout this project adds (the traditional `FL FR BL BR`, here `L R Lrs Rrs`); it is
not part of the Dolby layout set. Every other layout and channel order follows the Dolby
specifications.

Multichannel WAV files use `WAVE_FORMAT_EXTENSIBLE` with `dwChannelMask = 0`: the file declares no
Windows speaker positions, and the PCM channel order is exactly the table above, never rearranged
into mask-bit order. Players, ASIO, DAWs and other devices have to map the channel numbers
themselves; this project guarantees the content and numbering of each PCM channel, not that a
downstream player recognises Dolby speaker positions on its own.

`--speaker-layout` accepts only the twelve names in the table; anything else fails as an unknown
layout.

Speaker rendering reads warp, trim and front/back balance from the OAMD metadata; the CLI has no
warp override. Warp 0/1 render directly; warp 2/3 render to a 7.1.x intermediate layout first,
then fold back to 5.1.x with PLIIx or Standard (Lo/Ro) coefficients, and calibration always
follows the final target layout. Source trajectories and binaural rendering do not pre-apply
these controls; see [mathematics §12](docs/math.en.md) for the parameters and evaluation order.

Speaker rendering also applies the OAMD object attributes: per-block gain/inactive,
zone/elevation, size/snap/divergence, the priority size budget, element 5 position
refinement, ISF basis matrices, and bed distribution. ADM export and the binaural
paths still place point objects and do not apply these speaker target coefficients.

Object position updates are scheduled from the complete `frame_state`: speaker and
binaural rendering both accept multiple position blocks and programmes with more than
16 objects and schedule every block at its own time and ramp. The binaural audio input
is still one LFE plus fifteen object sources; the extra position tracks are followed
but not panned. ADM export keeps the 15-track single-block position view and rejects
anything larger as an unsupported variant.

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

The QMF and binaural tables and the JOC Huffman tables under `src/` are written from
publicly documented standards; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records
their sources.

The ISF compressed base coefficients and layout descriptors are obtained from
lookup tables stored in the generated header (`src/oamd/isf_tables.h`), which is
compiled into the binary with the code, so nothing reads a data file at run
time. Gain, size, priority, position, divergence, delta and refinement values are
computed in code, while the size kernel is generated from the speaker layout
region geometry and bed routing is derived from those layouts.

## Mathematical implementation

The main documented stages are:

- dense JOC differential reconstruction and dequantization;
- parameter-band mapping to 64 QMF subbands;
- cross-frame parameter interpolation (hard cut when no previous state exists, zero-matrix publishing after an object disappears);
- the frame-level JOC state machine (normal, hold, hard cut, core passthrough);
- analysis/inverse QMF, surround delay, and FIR state;
- the 1217-sample LFE delay with its sample-wise clipgain ramp;
- OAMD Q15 coordinate conversion;
- equal-power panning over target-layout regions;
- OAMD metadata application (per-block gain and inactive, zone/elevation, size kernel, snap, divergence, position refinement, priority size budget, ISF matrices, bed distribution);
- layout-dependent position compensation and sample-wise gain ramps;
- float32 and PCM24 output quantization;
- Rosella and compiled `.jochrtf` binaural rendering, with exactly-once delay/phase and special LFE.

See the [mathematical notes](docs/math.en.md) for the equations used by the decoding
and rendering process.

## Compatibility note

### How `joc_clipgain` is applied

`joc_clipgain` is folded into the **analysis-side scalar**: `scalar = 1/16 x g`, with
`g = 2^(x-4)·(1 + y/32)` from the 8-bit field `x:3 | y:5`; the range is 0.0625..15.75. The x16 at the
output cancels the 1/16, so the net effect is a multiply by `g` and the clamp therefore sits **after**
the gain (`clamp(16·IQMF·g_interp)`). The scalar is kept in two per-frame slots: the first 256 samples
of a frame ramp linearly from the previous value, the rest stays constant, and the first frame after a
reset takes the new value directly. For `g = 1` the interpolation is the identity and
`clamp(x)·1 == clamp(x·1)`. LFE bypasses the analysis but applies the same full `g` as a sample-wise
ramp before its 1217-sample delay.

`object_delay_samples` defaults to **1473**, preserving the behaviour of the existing
implementation; it is a configurable field and changing it changes the OAMD/ADM time
alignment. Upstream investigation suggests the value should be 0; this project keeps
the current default to stay byte-identical.

### `joc_dmx_config_idx` 0..4

All five configurations are covered: $N=5,7,7,5,7$ with the phase switch
$\sigma(k)=(k\in\{3,4\})$ gating the Ls/Rs $-j$ rotation and the band-0 21-tap FIR together.
A 7-channel configuration needs an **8-channel** bed, mapped by channel code:

| $k$ | extra pair | decoder 8-channel order | JOC input (bed index) |
|---|---|---|---|
| 1 | Lb/Rb | FL FR FC LFE **BL BR** SL SR | 0 1 2 6 7 4 5 |
| 2 / 4 | Tfl/Tfr | FL FR FC LFE SL SR TFL TFR | 0 1 2 4 5 6 7 |

ffmpeg gets `-ac 6` only for the $N=5$ configurations; $N=7$ keeps the decoder's own layout
(`-ac 6` downmixes 5.1.2/7.1 into 5.1 and `-ac 8` remaps 5.1.2 into 7.1; both lose information).

**Cross-layer consistency** (section 11.2): the substreams' full-band channel count must equal $N$,
a non-zero $joc\_ext\_config\_idx$ is an unmapped variant, and a declared object count above 15 is
clamped to 15 and recorded as a diagnostic - those change the DSP input, so a mismatch is refused.
The extra pair's **identity** never reaches the DSP (section 10.1): when the dependent-substream
`chanmap` disagrees with $k$, the mismatch is recorded as a diagnostic and the frame still
renders, with the bed mapping following the channel codes the `chanmap` declares. The bed
layout the OAMD declares may hold more channels than the core substream codes (a 9.1.6 bed
with a 5.1 core, for example): only a core channel missing from that layout counts as a
mismatch. When one audio frame holds several EMDF containers carrying ID11/ID14, they
are scored by the association rule of section 8.2 and its Table 56: a container whose OAMD and JOC
payloads share a group id and whose ID14 parses completely wins, then one whose ID14 merely parses;
only a unique best score is used and the others are recorded as a diagnostic
(`emdf/extra_container_ignored`). The extra pair's 10-slot ADM skeleton placement ($k=1\to6/7$,
$k\in\{2,4\}\to8/9$) is recorded, but bed channels are not output channels under the current
contract.

### Frame status and the JOC decode state machine

Every frame carries a status; a non-zero status is reported as a diagnostic. A non-zero status
makes the frame invalid for rendering, so the renderer holds the previous matrices or passes the
core PCM through instead of decoding that frame's matrix: 2 unsupported downmix configuration,
3 object count clamped to 15, 4 unknown parameter-band index, 5 invalid `offset_ts`, and 6
trailing data after byte padding. A file task refuses a `joc_dmx_config_idx` outside 0..4
earlier, in the metadata scan, as an unsupported variant. Status 3 still yields the clamped
object list for diagnostics, while status 2 and 4 stop the parse at the failing header field.

Rendering then runs a per-frame state machine: normal, hold, hard cut, and core passthrough. An
unusable-JOC frame that arrives during normal rendering enters hold, where the previous frame-end
matrices keep being published; six consecutive unusable frames move the internal hold state on but
stay in hold, and a valid frame resynchronises the sequence number. A sequence number of 0 after
the renderer left its initial state, a sequence mismatch, or a change of the declared core channel
count is a hard cut: every object matrix is zeroed and each object's previous state becomes
the zero matrix, while a core channel count change also clears the LFE delay ring but keeps
the clipgain ramp state. Before the first valid frame the renderer passes the core PCM
through.

A frame with no JOC EMDF at all (for example when the encoder wrote none for the first frame)
always takes the passthrough path: the eight core planes L, R, C, LFE, Ls, Rs, (Lrs, Rrs) are
written to output slots 0..7 by identity, each delayed by 1217 samples, slots 8..15 stay zero,
and the 8-plane history keeps advancing. Those frames are counted separately in the statistics,
and E-AC-3 input without JOC metadata (ID11 + ID14) is outside the supported scope. While hold or
passthrough runs without fresh parameters, the analysis scalar and the LFE ramp keep the
currently held clipgain.

## License

MIT, see [LICENSE](LICENSE). Third-party provenance and patent boundaries are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
