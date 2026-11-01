# Binaural rendering

[中文](binaural.md) · [Back to README](../README.en.md)

JustOneCacophony's binaural backend supports two HRTF sources: the Rosella
`.personalized_headphone` model exported by Dolby's official personalization scan
(its JSON parsing is implemented by this project and invokes no Dolby software),
and an offline compiled `.jochrtf` directional-field cache. A `.jochrtf` file is
a disposable, reproducible JOC compiled HRTF cache, not an interchange format.

```text
.jochrtf
  -> 77-band directional field + delay model
  -> 64-QMF / 77-hybrid analysis
  -> per-object direct + optional early reflections
  -> optional shared unitary-FDN late room
  -> stereo float64
```

## Inputs

The binaural backend consumes a compiled directional field (a JOC compiled HRTF cache,
`.jochrtf`) plus the shared filterbank tables.  Both are read-only inputs: this library
performs no parsing or conversion of measurement data formats.

```powershell
joc_cli input.m4a --binaural `
    --compiled-hrtf-cache path\to\subject.jochrtf `
    --kernels data\rosella_kernels.npz `
    --binaural-mode mid --binaural-tail-seconds 5.0
```

The library exposes the same fields: `joc_task_config` for a file task,
`joc_stream_config` for streaming, where `hrtf_path`, `kernels_path`,
`binaural_mode` and `binaural_tail_seconds` configure the binaural path.

The `.jochrtf` file is an **input**, not a product of this library: compiling it from
SOFA data or measurements belongs to the toolchain and is decoupled from this
repository.  Loading validates the member set, dtypes and shapes, C-contiguity,
CRC-32 and a payload hash recomputed over the members (see `.jochrtf` below).

`binaural_mode` is `near`, `mid` or `far` and selects one of the backend's three
preset parameter sets; `binaural_tail_seconds` sets the room-tail length drained on
flush (default 5.0 s).

## Binaural render mode

`--binaural-mode off|near|mid|far` (default `mid`) is a **human-specified
rendering hint**, not original binaural metadata extracted or recovered from the
input E-AC-3 JOC bitstream:

- Direct binaural rendering (`--binaural`): near/mid/far apply, default `mid`;
  `off` is an error;
- ADM BWF: the low 3 binaural-render-mode bits of the last 15 JOC object entries
  in DBMD segment 10 carry `off=0/near=1/far=2/mid=3`, leaving the first 10 bed
  entries unchanged; the default is `mid`, and `off` explicitly disables the
  binaural metadata hint.

## Exactly-once delay and phase

The compiler recognizes three mutually exclusive representations:

1. Nonzero `Data.Delay` is external to `Data.IR`; the FIR is not de-rotated and
   runtime applies the delay once.
2. With `Data.Delay=0` and an ordinary positive-onset HRIR, each ear's main peak
   supplies arrival time. Compilation separates it and runtime restores it once.
   The current threshold is a peak index greater than two samples.
3. With `Data.Delay=0` and both FIRs at a shared sample-zero origin, no external
   delay is invented. The authored complex phase stays in the fifth-order field.

No path may add a second ear delay or phase-group delay.

## Public filterbank and directional field

The `.jochrtf` runtime is fixed at:

- 48 kHz;
- a 64-sample QMF hop;
- 64-QMF / 77 hybrid bands;
- 961 samples of analysis/synthesis latency;
- fifth order, 36 terms, ACN/N3D real spherical harmonics;
- float64 PCM, delay, SH, and room state; complex128 band transfers and spectra.

The `.personalized_headphone` path selects its bank rather than fixing it. The
default is the shared public bank described here (961 samples of latency); bank
2 is the captured Rosella bank (448 samples). Both keep their own coefficients,
history and high-band delay, and the runtime skips the selected bank's latency
and sizes the flush with it. See [Rosella filterbank](rosella_filterbank.en.md)
for the parameters, the CLI switch and the validation results. The SOFA and
`.jochrtf` banks described here are unchanged and never use the selector.

Real and imaginary unit gains for every hybrid band pass through the same
analysis/synthesis chain to form a 154-real-parameter impulse dictionary. The
compiler does not sample 77 FFT bins. Defaults are `1e-3` projection ridge and
`1e-5` SH ridge. Coincident directions are merged before a spherical-Voronoi
weighted ridge fit.

The fixed resource is `data/rosella_kernels.npz`, which implements publicly
standardized filter banks, computable from the following formulas.

The hybrid analysis kernels are defined in [3GPP TS 26.405 / ETSI TS 126 405](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf),
Section 5.2.2 (Table 1 $Q=8$/
$Q=4$ coefficients, delay 6):

$$G_q^p[n] = g^p[n]\cdot\exp\Bigl(j\,\frac{2\pi}{Q^p}\bigl(q+\tfrac12\bigr)(n-6)\Bigr),\qquad n=0,\dots,12$$

The QMF analysis table is the MPEG-4 AAC/SBR 64 complex QMF bank of
ISO/IEC 14496-3/AMD1:2003, subclause 4.B.18.2, stored as the polyphase
reordering of the public 640-tap prototype $c_0,\dots,c_{639}$:

$$A_{r,t} = \frac{(-1)^t}{128}\,c_{63-r+64t},\qquad r=0,\dots,63,\ t=0,\dots,9$$

The QMF synthesis table is the causal left inverse of the analysis polyphase
matrix $\mathbf{A}$, i.e. the solution of
$\mathbf{A}\,\mathbf{W}=\mathbf{P}$
($\mathbf{P}$ is the 577-sample delay permutation; total latency
$961 = 577 + 6\times64$), stored as a rank-4 factorization:

$$W_{b,l} = \sum_{r=1}^{4} t_{b,l,r}\,\mathbf{b}_{b,r}^{\top}$$

The hybrid synthesis table is the 77→64 recombination: identity for the high
bands, $Y_{3+b}=X_{16+b}$, and for the low bands(
$C_p$ is the
$8+4+4$ child partition):

$$Y_p = \sum_{q\in C_p}\Bigl(\mathrm{Re}X_q + j\,s_q\,\mathrm{Im}X_q\Bigr),\qquad s_q\in\{\pm1\}$$

The loader verifies the archive and every array by SHA-256; the table version,
all array hashes, and the 77 reference band-center values are part of the cache
key. Public availability of a standard does not by itself grant permission to
practice related patent claims. See
[`data/README.en.md`](../data/README.en.md) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the sources and the
rights boundary.

## `.jochrtf`

A `.jochrtf` file is a pickle-free compressed NumPy archive with an exact member set:

| key | dtype / shape |
|---|---|
| `metadata_json` | NumPy Unicode scalar containing JSON text (`dtype.kind == "U"`) |
| `band_center_frequencies_hz` | little-endian `float64[77]` |
| `coefficients` | little-endian `complex128[36,2,77]` |
| `delay_coefficients` | little-endian `float64[36,2]` |
| `delay_bounds` | little-endian `float64[2,2]` |

Metadata uses the `JOC-HRTF-CACHE` magic and records the schema, compiler and
phase-policy versions, ACN/N3D convention, filterbank hashes, SOFA content
SHA-256, sample rate, radius, order, both ridge values, payload hash, and fit
report. Every setting that changes compilation participates in the cache key.
Metadata never persists an absolute local `source_path`; it may keep a display
name only.

Before constructing a field, the loader uses `allow_pickle=False` and validates
ZIP members and expanded sizes, shapes, dtypes, byte order, contiguous layout,
finite values, delay bounds, band centers, payload hash, and cache key. The
writer uses a same-directory temporary file, `fsync`, a process-held OS file
lock, and atomic `os.replace`. Its hidden `.lock` sidecar may remain and does not
mean that a writer still owns the lock. Outdated, damaged, or mismatched
caches cannot hit. SOFA input rebuilds an invalid cache; an explicitly selected
cache reports the error.

Deleting a disk cache must not change the field or render produced from the same
SOFA and compiler configuration.

A `.jochrtf` file contains directional-field coefficients and delay data
transformed from the source HRIRs. Its reproducibility therefore does not make
it licence-free. Creating a cache does not enlarge the rights granted by the
source SOFA/HRTF dataset: use, copying, and redistribution remain subject to
that dataset's terms. If those terms are unclear, keep `.jochrtf` as a private
local cache and do not ship it with the program or another build artifact.
`source_sha256` is only a content-integrity identifier, not proof of provenance
or permission.

## JOC objects and room behavior

The production adapter retains the existing JOC schedule:

- `[1536,16]` input per frame;
- channel 0 is special LFE and channels 1..15 are JOC objects;
- ID11/OAMD positions use a sample-timed timeline;
- source parameters update every 512 samples;
- every object owns independent direct/early history while one late FDN is shared;
- `finish()` drains early/late tails; output gain is explicit, with no implicit
  limiter or programme loudness normalization.

Near/Mid/Far, equal-power direct level, six first-order shoebox image sources,
late sends, the unitary FDN, the 120–180 Hz cosine-squared LFE low-pass, and room
calibration are JOC project-defined behavior, not constants published by SOFA or
Dolby.

The binaural renderer is implemented inside this library: the filterbank, the SH
directional-field evaluation, the per-object early/direct histories and the shared
FDN all run in `joc_core` (the `ejoc_jochrtf_*` kernel), consuming a compiled
directional field and 512-sample metadata updates.

## Technical references and rights boundary

- [SOFA SimpleFreeFieldHRIR convention](https://www.sofaconventions.org/mediawiki/index.php/SimpleFreeFieldHRIR)
- [3GPP TS 26.405 / ETSI TS 126 405 (64-QMF/77-hybrid definition)](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf)
- [Dolby binaural render-mode workflow](https://professionalsupport.dolby.com/s/article/What-is-Binaural-Render-Mode-and-how-do-the-settings-affect-my-mix)
- [EP3090576A1](https://patents.google.com/patent/EP3090576A1/en), used only as
  architectural background for direct/early/late, subbands, and FDNs; it does
  not establish that any product uses a particular embodiment.

Public availability of a specification, source file, or patent document does
not by itself authorize copying its contents, redistribution of derivatives,
or practice of patent claims. These technical references grant no patent
licence and make no non-infringement representation. Anyone preparing a release
or product integration must assess the applicable data and software licences,
patent permissions, and freedom to operate. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the public-standard
provenance and rights boundary.
