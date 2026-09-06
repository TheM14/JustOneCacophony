# Python runtime tables

[中文](README.md)

This directory contains static production tables. It does not contain user HRTFs.

`tables.npz` contains the JOC core decoding tables:

```text
analysis_window                         float64[10,64]
qmf5_window                             float64[640]
joc_huff_code_coarse_generic            int64[95,2]
joc_huff_code_fine_generic              int64[191,2]
joc_huff_code_coarse_coeff_sparse       int64[95,2]
joc_huff_code_fine_coeff_sparse         int64[191,2]
joc_huff_code_5ch_pos_index_sparse      int64[4,2]
joc_huff_code_7ch_pos_index_sparse      int64[6,2]
```

`src/joc_qmf.py` loads the QMF tables, while `src/joc_decode.py` loads the JOC Huffman trees. Python does not read C/C++ headers under `native/`.

The corresponding native data are stored in `native/src/qmf_tables.h` and `native/src/joc_huffman_tables.h`. Changes on either side should update the other and be checked for value-by-value agreement.

## Binaural rendering tables

`rosella_kernels.npz` contains the fixed 64-QMF/77-hybrid tables used by the
public SOFA binaural path:

```text
format_version                  little-endian int32[1]
qmf_analysis_coefficients       float32[64,10]
hybrid_analysis_low_kernel      float32[3,2,13,16,2]
hybrid_synthesis_indices        int16[154,4]
hybrid_synthesis_values         float32[154]
qmf_synthesis_basis             float64[64,4,128]
qmf_synthesis_taps              float64[64,10,4]
```

The float32 table values are promoted to float64 when loaded.
`src/public_filterbank.py` verifies the archive and every array by SHA-256.
Those hashes, the table version, and the 77 reference band-center values all
participate in the `.jochrtf` cache key. The full analysis/synthesis latency is
961 samples.

The packaged tables implement publicly standardized filter banks, computable
from the following formulas.

The 64-QMF → 77-hybrid structure, the 13-tap low-band prototypes, and their
half-bin complex modulation are defined in
[3GPP TS 26.405 / ETSI TS 126 405](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf),
Section 5.2.2 (Table 1 $Q=8$/$Q=4$ coefficients, delay 6):

$$G_q^p[n] = g^p[n]\cdot\exp\!\Bigl(j\,\frac{2\pi}{Q^p}\bigl(q+\tfrac12\bigr)(n-6)\Bigr),\qquad n=0,\dots,12$$

The 64-band QMF analysis is the MPEG-4 AAC/SBR 64 complex QMF analysis bank of
ISO/IEC 14496-3/AMD1:2003, subclause 4.B.18.2; the packaged $64\times10$ table
is the polyphase reordering of the public 640-tap prototype $c_0,\dots,c_{639}$:

$$A_{r,t} = \frac{(-1)^t}{128}\,c_{63-r+64t},\qquad r=0,\dots,63,\ t=0,\dots,9$$

The QMF synthesis table is the causal left inverse of the analysis polyphase
matrix $\mathbf{A}$, i.e. the solution of $\mathbf{A}\,\mathbf{W}=\mathbf{P}$
($\mathbf{P}$ is the 577-sample delay permutation; total latency
$961 = 577 + 6\times64$), stored as a rank-4 factorization:

$$W_{b,l} = \sum_{r=1}^{4} t_{b,l,r}\,\mathbf{b}_{b,r}^{\top}$$

The hybrid synthesis table is the 77→64 recombination: identity for the high
bands, $Y_{3+b}=X_{16+b}$, and for the low bands ($C_p$ is the $8+4+4$ child
partition):

$$Y_p = \sum_{q\in C_p}\Bigl(\operatorname{Re}X_q + j\,s_q\,\operatorname{Im}X_q\Bigr),\qquad s_q\in\{\pm1\}$$

The same values also appear in other public implementations of these standards
(for example FFmpeg's `aacps_tablegen.h` and `aacsbrdata.h`).

Public availability of a standard does not by itself grant permission to
practice related patent claims.

SOFA is the user-visible source of truth. A `.jochrtf` file is a disposable JOC
compiled HRTF cache that can be rebuilt from SOFA. The cache contains
transformed source-HRTF data and remains subject to the source SOFA/HRTF
dataset's licence and redistribution restrictions.
