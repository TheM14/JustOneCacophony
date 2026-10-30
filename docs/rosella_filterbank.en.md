# Rosella filterbank

The Rosella renderer uses a separate 64-band QMF and 77-band hybrid filterbank.
The SOFA compiler and renderer continue to use `PublicFilterbank`.

## Why the banks differ

The reference headphone renderer selects its analysis and synthesis callbacks
during preparation, after initialization. Capturing only the initial callbacks
identified the wrong filterbank variant.

Runtime buffer captures identified these parameters:

| Parameter | Rosella bank | Public bank |
| --- | --- | --- |
| QMF prototype | 640 asymmetric coefficients | Separate prototype |
| Low-band hybrid filter | 11 taps | 13 taps |
| High-band hybrid delay | 3 slots | 6 slots |
| Analysis/synthesis latency | 448 samples | 961 samples |

The wrong variant produced a frequency-dependent room-tail error. With the
captured room send as input, the existing room recursion already matched the
reference. The fix changes the filterbank, not the room coefficients, direct
HRTF calculation, phase mapping, or output EQ.

## Implementation

`RosellaFilterbank` keeps separate analysis, hybrid, and synthesis histories for
each channel. QMF data uses `[slot, channel, 64]` order. Hybrid data uses
`[slot, channel, 77]` order. The half-step methods preserve state between calls.

`rosella_filterbank_tables.h` stores the captured float32 prototype, rotation,
and hybrid coefficients as hexadecimal literals. Processing uses double
precision. The tables do not contain the personalized HRTF or room tuning.

The runtime removes 448 startup samples and uses the same latency when it
calculates flush capacity. Its rendered buffer is resized to the current block
length, including a short final block.

## Validation

Run the self-contained tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`rosella_filterbank` checks deterministic analysis, hybrid, folding, and synthesis
vectors from an independent NumPy implementation. It also checks empty and
unequal chunks, reset, channel isolation, and impulse latency. The tests need no
reference binary, audio fixture, or personalized headphone file.

Before integration with the upstream bed-layout changes, four captured-input
tests gave the following room PCM residuals. No gain fitting was used for this
table. Residual is `10 * log10(sum(abs(reference - candidate)^2) /
sum(abs(reference)^2))`.

| Input | Room PCM residual |
| --- | ---: |
| Left-front click | -133.94 dB |
| Left-front log sweep | -135.40 dB |
| Left-front noise | -133.50 dB |
| Right-front signal | -133.34 dB |

The old bank produced room residuals near -10 to -11 dB on these captures.
A 30-second music replay using the reference decoder's object PCM reached
correlation 0.9999999503 and a -70.02 dB residual after delay alignment and one
global gain fit. A full-file render through this project's decoder reached
correlation 0.99988816 after alignment and gain fitting.

These are separate experiments, not bit-exactness claims. A fixed gain
difference of approximately 1.0004885 remains in the captured chain. The patch
does not compensate for that difference. Decoder differences and the remaining
music residual are outside this fix.
