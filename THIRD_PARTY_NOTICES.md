# Third-party notices / 第三方通知

本仓库包含 JustOneCacophony 原生库的逐字节副本（下称"复用文件"），以及遵循公开标准实现的
滤波器组表。本文件记录这些来源、公开标准依据与权利边界。

## 复用文件

以下文件是本项目（JustOneCacophony，MIT）原生 C++ 库的逐字节副本，不修改、不追加注释：

| 本仓库路径 | 上游路径 | SHA-256（前 16 位） |
|---|---|---|
| `src/joc_core/eac3joc_core.cpp` | `native/src/eac3joc_core.cpp` | `1978eea64a2616fa` |
| `src/joc_core/qmf_tables.h` | `native/src/qmf_tables.h` | `c205ea187e956e87` |
| `src/speaker/speaker_renderer.cpp` | `native/src/speaker_renderer.cpp` | `96f36f40daf86eec` |
| `src/speaker/speaker_layouts.h` | `native/src/speaker_layouts.h` | `51e24c11be09787f` |
| `src/binaural/binaural_renderer.cpp` | `native/src/binaural_renderer.cpp` | `d579803f0f5a6699` |
| `src/joc_bitstream/joc_huffman_tables.h` | `native/src/joc_huffman_tables.h` | `698498b3778d88db` |
| `include/eac3joc_core.h` | `native/include/eac3joc_core.h` | `7392f48dfd840656` |

上游修订：`6bc2c2885666bb151bb66af93472199af9a99b81`。

## 派生文件

`src/binaural/sofa_binaural_renderer.cpp` 是上游 `native/src/sofa_binaural_renderer.cpp` 的派生
实现：同一 `ejoc_sofa_binaural_*` C ABI，增加表提升、结果记忆化、输入校验与 SIMD 派发。它
**不属于**逐字节副本，也不受"不得修改"约束，但来源固定为上述上游修订，对应上游源文件的
SHA-256 为 `81b485e4c71907672ab308ddc382284acf29161cee93d7906785a4cce58941c7`，输出与原
实现逐位相同（见该文件头部的验证记录）。

## 公开标准来源

64-QMF → 77-hybrid 结构与 13-tap 低带 prototype 定义于
[3GPP TS 26.405 / ETSI TS 126 405](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf)
第 5.2.2 节（Table 1 的 $Q=8$/$Q=4$ 系数，delay 6）：

$$G_q^p[n] = g^p[n]\cdot\exp\Bigl(j\,\frac{2\pi}{Q^p}\bigl(q+\tfrac12\bigr)(n-6)\Bigr)$$

64-band QMF analysis 即 ISO/IEC 14496-3/AMD1:2003 第 4.B.18.2 节的 MPEG-4
AAC/SBR 64 complex QMF bank；打包的 $64\times10$ 表是公开 640-tap prototype 的多相重排：

$$A_{r,t} = \frac{(-1)^t}{128}\,c_{63-r+64t}$$

QMF synthesis 表为 analysis 多相矩阵 $\mathbf{A}$ 的因果左逆
$\mathbf{A}\,\mathbf{W}=\mathbf{P}$（$\mathbf{P}$ 为 577-sample 延迟置换；
全链 $961 = 577 + 6\times64$），rank-4 分解存储：

$$W_{b,l} = \sum_{r=1}^{4} t_{b,l,r}\,\mathbf{b}_{b,r}^{\top}$$

hybrid synthesis 表为 77→64 重组：高频带恒等 $Y_{3+b}=X_{16+b}$，低频带：

$$Y_p = \sum_{q\in C_p}\Bigl(\mathrm{Re}X_q + j\,s_q\,\mathrm{Im}X_q\Bigr),\qquad s_q\in\{\pm1\}$$

相同数值可在 FFmpeg（`aacps_tablegen.h`、`aacsbrdata.h`）等公开实现中查到。

## HRTF 数据与 `.jochrtf`

`.jochrtf` 含有特定源 SOFA/HRTF 数据集的变换系数与 delay；其使用、复制与再分发仍受源
数据集许可约束，权限不明确时应作为私有 cache 保存。本仓库不分发任何 HRTF 数据集。

## 专利说明

标准可公开获取不等于获准实施相关专利。
