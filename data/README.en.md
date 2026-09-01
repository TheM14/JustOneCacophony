# Python runtime tables

[中文](README.md)

`tables.npz` contains the static table data used by the Python path:

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
