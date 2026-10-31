# Binaural rendering

[中文](binaural.md) · [Back to README](../README.en.md)

## Routes

- SOFA: full complex local HRIR interpolation, adaptive-length filters and 128-sample partitioned convolution.
- `.jochrtf`: the existing compact 77-band/fifth-order field and delay representation.
- `.personalized_headphone`: the existing DSP, filterbank selection and Near/Mid/Far behavior.

SOFA and compact inputs use the same externally configured reference geometry, distance gains and environment state. SOFA never compiles or silently falls back to the compact format. The old compact loader/writer and compiler remain available.

```powershell
python main.py input.m4a --binaural
python main.py input.m4a --binaural --sofa-hrtf subject.sofa --reference-config reference.json
python main.py input.m4a --binaural --sofa-hrtf subject.sofa --sofa-interpolation lf_mca_zero
python main.py input.m4a --binaural --sofa-hrtf subject.sofa --binaural-mode near
python main.py input.m4a --binaural --sofa-hrtf subject.sofa --binaural-mode far
python src\compile_hrtf.py subject.sofa subject.jochrtf
python main.py input.m4a --binaural --compiled-hrtf-cache subject.jochrtf
python main.py input.m4a --binaural --personalized-headphone --binaural-mode far --rosella-filterbank 2
```

The default SOFA is `HRTF/binaural.sofa` under the project directory, independently of the current working directory. Use `--sofa-hrtf PATH` for other locations. A single measured shell is selected automatically; multi-shell data requires `--hrtf-radius-m`. The personalized default path is unchanged. All three binaural routes accept `--binaural-mode near|mid|far`, defaulting to mid in the CLI; off remains an ADM-only DBMD hint.

## Reference configuration

Resolution order: `--reference-config`, `BINAURAL_REFERENCE_CONFIG`, then local `local/binaural_reference.json`. Missing data, a hash mismatch, or an unsupported mapping is an error. Environment data and its specific identity are not supplied with the source. Example:

```json
{"schema":"binaural-reference-v1","profile_id":"reference-v1",
 "model_path":"C:/path/to/reference-data.json","model_sha256":"<64 hex digits>",
 "profile_index":3,"room_impulse_slots":4096,"level_calibration_directions":128,
 "compiled_receiver_positions_m":[[0,0.09,0],[0,-0.09,0]]}
```

The external data must be accepted by the existing parameter loader. Relative paths resolve against the configuration file. `profile_index` fixes the shared level-calibration anchor; `--binaural-mode` selects the rendering profile without changing that anchor. Compact files do not store receiver positions; they require explicit configured geometry. Full SOFA uses its own ear positions. A measured radius does not set the scene scale or room size.

Reference mapping uses the selected profile's quantization, axes, bounds and centre regularization. Outside positions extend the listener-to-boundary ray by the coded finite factor and are never clamped back inside. Near uses direct gain 1 and environment send 0. Mid/Far use `a=1/sqrt(1+c*(h0*r)^2)` and `b=h0*r*a`. Send gain is not final reverberant pressure. All modes share the same interpolation, radial adaptation, convolution and timing. There is no additional 1/r attenuation or absolute flight-time delay.

## Full HRIR model

The strict importer validates SimpleFreeFieldHRIR 0.4/1.0/1.1, FIR, units, listener orientation, ear order, zero emitter offset, finite data and free-field declarations. Resampling scales Data.Delay in samples. The radial adapter requires fixed symmetric receiver positions centred on the listener.

Map ADM right/front/up to SOFA front/left/up with `(y,-x,z)`. A 3D convex hull supplies local spherical triangles. Ray/planar barycentric weights are nonnegative and sum to one; exact measured nodes preserve the full complex response. Large triangles over coverage holes still have limited support.

Per-ear delay comes from phase regression at 200–1500 Hz. Interpolate aligned complex residuals and restore weighted delay, including Data.Delay once. The default `lf_logmag_phase` combines interpolated log magnitude with aligned complex phase. `lf_mca_zero` is an independent local MCA approximation with 41 continuous Gammatone power bands, PCHIP and a ±12 dB gain bound; it is not a bitwise reproduction of a third-party implementation.

Radial adaptation multiplies by the complex normalized rigid-sphere ratio `H_sphere(r)/H_sphere(r_measured)`, exactly one at the measured radius. Ear geometry determines sphere radius. Near the centre, the configured regularization and an equal-ear, zero-ITD, spherical-area-weighted energy response give a continuous blend. This is an approximate human-head radial model; angular interpolation scores do not validate arbitrary distances.

Calibration computes one global scalar per input at the configured anchor from equal-area directions and total binaural energy with logarithmic frequency weighting at 100–16000 Hz. All three modes share that scalar; no per-mode/ear/direction/frequency normalization is applied.

The FFT size is at least 2048, automatically increased to a power of two that fits the complete HRIR and Data.Delay, up to 16384. All resulting FIR taps are retained, with a declared common shift of half the FFT size. Rendering rebases one global recording origin, the rounded median of mean-ear fitted delays; the environment branch receives matching delay before common compensation. Original spectrum queries retain the dataset time reference. The shift is independent of target source distance.

Per-object convolution retains input history and independent old/new overlap during a 128-sample output crossfade. Environment computation batches 1536 samples without changing the 128-sample parameter timeline; extra computation padding is trimmed when draining. Source PCM, delays and accumulation use float64; spectra use complex128, at 48 kHz.

## Scheduling and limits

Each `[1536,16]` frame contains LFE followed by 15 already reconstructed objects/bed slots. Metadata retains the existing 1473-sample content offset and 32-sample alignment; new parameters update every 128 samples. A SOFA-only opt-in preserves negative elevation, outside distance and anchor fields. Legacy parsing and the personalized timeline retain their default behavior.

Multi-position blocks, ISF, explicit infinity and screen anchors requiring calibration return structured unsupported errors. Finite outside updates participate in timeline interpolation. `finish()` drains state only once; output gain is applied once without an implicit limiter.

## Compact files

Version-1 `.jochrtf` files retain their pickle-free ZIP schema, band centres, complex128 `[36,2,77]` coefficients, float64 `[36,2]` delay coefficients, `[2,2]` bounds, metadata, hash validation and atomic writer. Their inherent directional compression error remains. See [filterbank data](../data/README.en.md) for the public filterbank sources.

Mathematical sources: [Duda and Martens 1998](https://www.ece.ucdavis.edu/cipic/wp-content/uploads/sites/12/2015/04/cipic_JASA_Nov_1998.pdf), [MCA paper](https://doi.org/10.1109/TASLP.2023.3313908), [ETSI TS 103 420](https://www.etsi.org/deliver/etsi_ts/103400_103499/103420/01.02.01_60/ts_103420v010201p.pdf). HRTF algorithms are independently implemented; external data retain their own licence terms.
