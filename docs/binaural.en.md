# Binaural rendering

[中文](binaural.md) · [README](../README.en.md)

## Independent input routes

| Input | Rendering | Distance and environment |
|---|---|---|
| SOFA | Full complex directional HRIR interpolation, adaptive FIR length, 128-sample partitioned convolution | Direct by default; optional project room; no synthetic radial head filtering |
| `.jochrtf` | Existing 77-band, fifth-order field and delay representation | Direct by default; same optional room |
| `.personalized_headphone` | Existing DSP, filterbank and model parameters | Existing Near/Mid/Far and room processing unchanged |

SOFA never invokes the compact compiler, writes a compact cache or falls back to it. The compact format, validation and offline compiler remain available. SOFA and compact input use `DirectSofaBackend` with the object timeline.

```powershell
python main.py input.m4a --binaural
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --sofa-interpolation lf_mca_zero
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --source-reference-distance-m 2.193
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --sofa-room
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --sofa-room --sofa-room-preset legacy
python src\compile_hrtf.py C:\HRTF\subject.sofa C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --personalized-headphone --binaural-mode far --rosella-filterbank 2
```

The default SOFA is `HRTF/binaural.sofa` under the project directory, independent of the working directory. Use `--sofa-hrtf PATH` for other locations. A single measurement shell is selected automatically; multiple shells require `--hrtf-radius-m`. Radial interpolation is not implemented. The personalized default remains `HRTF/binaural.personalized_headphone`.

SOFA and compact inputs retain the Near/Mid/Far point-source scales, default Mid. Personalized presets and ADM DBMD hints retain their existing behavior.

## Point-source scale

Near/Mid/Far reference distances are 1.00000465, 2.19327927 and 6.40177584 m. `--source-reference-distance-m` overrides the selected scale. For ADM right/front/up coordinates `p=(x,y,z)`:

```text
q_SOFA = (y, -x, z) / ||p||
d_nominal = reference_distance_m * ||p||
H_render(ear, f) = H_selected_shell(ear, f, q_SOFA)
```

This defines nominal metric object coordinates, without turning a 1.2 m measurement into measured 2.193 m data or changing SOFA metadata. Direct-only rendering adds no inverse-distance attenuation, radial sphere ratio, synthetic centre response, room send, dry/wet coupling or distance-dependent flight time. Programme gain and the SOFA response scale are retained. The optional room uses the original project coupling; explicit output gain is applied once.

A coincident source has no defined direction. Retain that object's last valid direction, or use front when no history exists. No zero-ITD response is synthesized. Finite outside coordinates retain the metadata distance factor rather than being clamped into the unit box.

SOFA requires no personalized model or machine-local configuration. Direct-only rendering does not read a room asset. Reports record input identity, measured radius, interpolation and the room switch.

## Optional project room

`--sofa-room` is off by default. It reuses the original `public_room.py` and `reference_distance.py` algorithms with original and compact parameter sets: six first-order image reflections plus a shared four-line orthogonal Hadamard FDN with two all-pass diffusers and low-pass damping. The project listening preset aims to approximate an ideal listening environment dominated by direct sound. It is not a measured BRIR or a mandated standard room. No Rosella room table is required.

Near/Mid default to `compact`; Far defaults to `legacy` to retain its unit-coordinate
source range. Select either with `--sofa-room-preset compact|legacy`. Compact dimensions
come from [model 9 in Zahorik 2009, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2730711/).
The same project algorithm scales late delays, diffusion times and the RT60 parameter
by V/S. Only the geometry is borrowed; this does not reproduce the paper's BRIR model.

Full-HRIR rendering evaluates the selected SOFA at each reflected direction. Fractional delays use phase ramps across the full FFT grid, with extra delay `(L_path-L_direct)/c`; native `Data.Delay` is retained once. Direct, early and late gains follow the original project policy. Sources must lie inside the optional room geometry; outside positions are not silently clipped.

Both FDNs are excited by a unit impulse at 48 kHz and stored as stereo float64 FIRs.
`data/sofa-room-compact.npz` has 37,383 taps and an RT60 parameter of about 0.3894 s;
`data/sofa-room.npz` retains the original 81,600 taps and 0.85 s parameter. Omitted probe
tail energy is about `2.2e-13` in both. PCM FFT convolution realizes each time-invariant
system. Early reflections use the selected SOFA; late output retains the original
stereo FDN definition.

Late FIR alignment includes the convolution implementation shift and the selected SOFA's common recording-delay estimate. Enabled room processing uses 1536-sample batches while retaining 128-sample parameter updates; extra flush padding is discarded. Early reflections in `.jochrtf` retain their 77-band approximation. Regenerate both assets with `python src\generate_sofa_room.py`, or select one using `--preset compact|legacy`; the archives record parameters, provenance, coefficient hash and tail budget.

## Full HRIR and time reference

The importer validates `SimpleFreeFieldHRIR` versions 0.4/1.0/1.1, FIR data, two consistently identifiable left/right receivers, zero emitter offset, free-field metadata, units and finite numbers. Processing uses 48 kHz; resampling scales sample-valued `Data.Delay` accordingly. Receivers must remain fixed across measurements; angular interpolation no longer fits a symmetric spherical head.

BRIR and SOFA conventions other than `SimpleFreeFieldHRIR` are not implemented. Unsupported files are rejected. The backend has a response-type guard to announce BRIR and ignore `--sofa-room` once BRIR loading is implemented; current tests cover that policy interface, not a BRIR loader.

A 3-D convex hull of measured unit directions defines local triangles. Ray/triangle barycentric weights are nonnegative and sum to one. There is no longitude/latitude planar interpolation or extrapolation beyond directional coverage. Measured nodes retain their complex response.

Low-frequency alignment fits delay over 200–1500 Hz. Default `lf_logmag_phase` interpolates log magnitude and reconstructs phase from aligned complex residuals and interpolated delay. Optional `lf_mca_zero` applies auditory-band magnitude correction. Neither replaces the selected full FIR with a minimum-phase response.

`Data.Delay` is included once. A common half-FFT shift stores interpolation pre-ringing; output compensates only convolution/filterbank implementation latency. The file's common recording delay, relative ear delays and frequency-dependent phase remain intact. The fitted recording origin is no longer subtracted, and target distance adds no flight time. DC and Nyquist use real endpoints for a real output sequence.

Each object retains its own convolution history. Updates crossfade old/new outputs over 128 samples using complete input history for both filters. Filter updates are delayed by the common artificial FIR shift so removing that shift does not advance metadata changes. Full HRIR uses PCM convolution; QMF/hybrid is confined to explicit compact input. Arithmetic is float64/complex128, without implicit limiting.

## Object input

Each reconstructed frame is `[1536,16]` PCM: LFE at slot 0 and fifteen reconstructed object/bed slots at 1–15. The ID11 timeline feeds these directly, without another decode or an intermediate loudspeaker render. The existing 1473-sample object compensation and 32-sample metadata alignment are retained. Parameters update every 128 samples.

LFE retains separate low-pass treatment and no room send. SOFA metadata preserves negative elevation, finite outside factors and anchoring flags; personalized parsing and timing are unchanged. Multi-position blocks, ISF, explicit infinity and anchors requiring unavailable screen calibration are rejected. `finish()` drains once. The tail option bounds FIR/filterbank and optional room draining.

## Compact representation

`.jochrtf` v1 remains a pickle-free NumPy ZIP containing JSON metadata, 77 band centres, `complex128[36,2,77]` directional coefficients, `float64[36,2]` delay coefficients and `float64[2,2]` bounds. Its format, cache key, hashes, atomic writing and source-data licensing remain unchanged. Its angular/frequency approximation is not equivalent to full-HRIR precision.

Filterbank provenance is documented in [data notes](../data/README.en.md). See the [MCA paper](https://doi.org/10.1109/TASLP.2023.3313908) for the correction model. Algorithms are independently implemented; external measurements retain their own licensing terms.
