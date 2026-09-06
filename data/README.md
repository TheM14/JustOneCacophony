# Python 运行时表

[English](README.en.md)

本目录保存 Python 生产路径使用的静态表数据，不保存用户 HRTF。

`tables.npz` 保存 JOC 核心解码表：

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

## 双耳渲染表

`rosella_kernels.npz` 保存公开 SOFA 双耳路径使用的 64-QMF/77-hybrid 固定表：

```text
format_version                  little-endian int32[1]
qmf_analysis_coefficients       float32[64,10]
hybrid_analysis_low_kernel      float32[3,2,13,16,2]
hybrid_synthesis_indices        int16[154,4]
hybrid_synthesis_values         float32[154]
qmf_synthesis_basis             float64[64,4,128]
qmf_synthesis_taps              float64[64,10,4]
```

float32 表值载入后提升为 float64。`src/public_filterbank.py` 在读取时校验 archive
及每个数组的 SHA-256；这些 hash、table version 和 77 个 band-center 参考值共同进入
`.jochrtf` cache key。analysis/synthesis 全链 latency 为 961 samples。

打包表实现的是公开标准化的滤波器组，各表可由如下公式计算。

64-QMF → 77-hybrid 结构、13-tap 低带 prototype 与半 bin 复调制定义于
[3GPP TS 26.405 / ETSI TS 126 405](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf)
第 5.2.2 节（Table 1 的 $Q=8$/$Q=4$ 系数，delay 6）：

$$G_q^p[n] = g^p[n]\cdot\exp\!\Bigl(j\,\frac{2\pi}{Q^p}\bigl(q+\tfrac12\bigr)(n-6)\Bigr),\qquad n=0,\dots,12$$

64-band QMF analysis 即 ISO/IEC 14496-3/AMD1:2003 第 4.B.18.2 节的 MPEG-4
AAC/SBR 64 complex QMF bank；打包的 $64\times10$ 表是公开 640-tap prototype
$c_0,\dots,c_{639}$ 的多相重排：

$$A_{r,t} = \frac{(-1)^t}{128}\,c_{63-r+64t},\qquad r=0,\dots,63,\ t=0,\dots,9$$

QMF synthesis 表为上述 analysis 多相矩阵 $\mathbf{A}$ 的因果左逆，即求解
$\mathbf{A}\,\mathbf{W}=\mathbf{P}$（$\mathbf{P}$ 为 577-sample 延迟置换；
全链 $961 = 577 + 6\times64$），以 rank-4 分解形式存储：

$$W_{b,l} = \sum_{r=1}^{4} t_{b,l,r}\,\mathbf{b}_{b,r}^{\top}$$

hybrid synthesis 表为 77→64 重组：高频带恒等 $Y_{3+b}=X_{16+b}$；低频带
（$C_p$ 为 $8+4+4$ 子带划分）：

$$Y_p = \sum_{q\in C_p}\Bigl(\operatorname{Re}X_q + j\,s_q\,\operatorname{Im}X_q\Bigr),\qquad s_q\in\{\pm1\}$$

相同数值可在 FFmpeg（`aacps_tablegen.h`、`aacsbrdata.h`）等公开实现中查到。

标准可公开获取不等于获准实施相关专利。

`.sofa` 是用户可见的 source of truth；`.jochrtf` 是可删除、可从 SOFA 重建的
JOC compiled HRTF cache。cache 含有源 HRTF 的变换数据，仍受源 SOFA/HRTF
数据集的许可与再分发限制约束。
