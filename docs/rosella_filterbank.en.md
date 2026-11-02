# Rosella filterbank

[中文](rosella_filterbank.md) · [Back to README](../README.en.md)

The Rosella `.personalized_headphone` path can run either of two 64-QMF /
77-hybrid filterbanks: the 13-tap public bank (the default) or the 11-tap Rosella
bank. The SOFA and `.jochrtf` paths always use
the public bank and never consult the selector.

## Which bank runs

`--rosella-filterbank 1|2` selects the bank of the Rosella path; the library
form is `RosellaBinauralRenderer(..., filterbank=...)`, where omitting the
argument means `1`.

| Value | Implementation | Low-band hybrid | High-band delay | Latency |
| --- | --- | --- | --- | ---: |
| `1` (default) | `rosella_filterbank.py` | 13 taps | 6 slots | 961 samples |
| `2` | `rosella_captured_filterbank.py` | 11 taps | 3 slots | 448 samples |

Both banks expose the same four stateful half-steps (QMF analysis, hybrid
analysis, hybrid synthesis, QMF synthesis), so the choice switches the
coefficients, the high-band delay, the history layout, the startup skip and the
flush capacity together. The analysis and synthesis stages keep their own state
and never share it. Any other value is rejected, and the option is an error when
the binaural input is not a Rosella model: a SOFA or `.jochrtf` render must not
silently ignore it.

## Bank parameters

| Parameter | 11-tap Rosella bank | 13-tap public bank |
| --- | --- | --- |
| QMF prototype | 640 asymmetric coefficients | polyphase reorder of the public 640-tap prototype |
| Low-band hybrid filter | 11 taps | 13 taps |
| High-band hybrid delay | 3 slots | 6 slots |
| Analysis/synthesis latency | 448 samples | 961 samples |

The selector does not change room coefficients, direct HRTF calculation, phase
mapping or output EQ. Bank `1` is the default.

## Implementation

`rosella_captured_filterbank.py` executes the four half-steps of the 11-tap
bank in float64/complex128:

- QMF analysis keeps a 9-slot ring history, reads `(position + tap + 1) mod 10`,
  and writes `spectrum[2b] = F[b]/64`, `spectrum[2b+1] = conj(F[63-b])/64`;
- hybrid analysis keeps a 10-slot history, transforms the three low bands with
  8/4/4-point transforms, reorders them with a fixed map, conjugates bands 1 and
  3, and passes the high bands through delayed by 3 slots;
- QMF synthesis keeps a 9-slot history and accumulates 10 prototype-matched
  lags, taking the imaginary part on odd lags.

`data/rosella_captured_kernels.npz` stores the 11-tap bank's prototype
(`float32[64,10]`), rotation (`complex64[64]`) and hybrid coefficients
(`complex64[22]`); they are promoted to float64/complex128 on load and contain
no personalized HRTF or room tuning.

The startup skip and the flush capacity both come from the selected bank (961 or
448 samples). `.report.json` records the actual choice in
`binaural_renderer_backend`, as `filterbank`, `filterbank_name` and
`latency_compensated_samples`.

The public filterbank formulas and `.jochrtf` direction field are described in
`docs/binaural.md`; the 11-tap bank only affects the Rosella
path.
