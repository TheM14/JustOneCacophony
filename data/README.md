# Python 运行时表

[English](README.en.md)

`tables.npz` 集中保存 Python 路径使用的静态表数据：

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

`src/joc_qmf.py` 读取 QMF 表，`src/joc_decode.py` 读取 JOC Huffman 树。Python 不读取 `native/` 下的 C/C++ 头文件。

原生侧对应数据分别位于 `native/src/qmf_tables.h` 与 `native/src/joc_huffman_tables.h`。修改任何一侧时，应同步更新另一侧并进行逐值一致性检查。
