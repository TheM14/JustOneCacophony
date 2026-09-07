# Third-party notices / 第三方通知

本文件记录 `data/rosella_kernels.npz`（`src/public_filterbank.py` 使用的滤波器组表）
的公开标准来源，以及 HRTF 数据与专利的边界说明。

## 公开标准来源

64-QMF → 77-hybrid 结构与 13-tap 低带 prototype 定义于
[3GPP TS 26.405 / ETSI TS 126 405](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf)
第 5.2.2 节（Table 1 的 $Q=8$/
$Q=4$ 系数，delay 6）：

$$G_q^p[n] = g^p[n]\cdot\exp\!\Bigl(j\,\frac{2\pi}{Q^p}\bigl(q+\tfrac12\bigr)(n-6)\Bigr)$$

64-band QMF analysis 即 ISO/IEC 14496-3/AMD1:2003 第 4.B.18.2 节的 MPEG-4
AAC/SBR 64 complex QMF bank；打包的 $64\times10$ 表是公开 640-tap prototype 的
多相重排：

$$A_{r,t} = \frac{(-1)^t}{128}\,c_{63-r+64t}$$

QMF synthesis 表为 analysis 多相矩阵 $\mathbf{A}$ 的因果左逆
$\mathbf{A}\,\mathbf{W}=\mathbf{P}$（
$\mathbf{P}$ 为 577-sample 延迟置换；
全链 $961 = 577 + 6\times64$），rank-4 分解存储：

$$W_{b,l} = \sum_{r=1}^{4} t_{b,l,r}\,\mathbf{b}_{b,r}^{\top}$$

hybrid synthesis 表为 77→64 重组：高频带恒等 $Y_{3+b}=X_{16+b}$，低频带：

$$Y_p = \sum_{q\in C_p}\Bigl(\mathrm{Re}X_q + j\,s_q\,\mathrm{Im}X_q\Bigr),\qquad s_q\in\{\pm1\}$$

相同数值可在 FFmpeg（`aacps_tablegen.h`、`aacsbrdata.h`）等公开实现中查到。

## HRTF 数据与 `.jochrtf`

`.jochrtf` 含有特定源 SOFA/HRTF 数据集的变换系数与 delay；其使用、复制与再分发
仍受源数据集许可约束，权限不明确时应作为私有 cache 保存。

## 专利说明

标准可公开获取不等于获准实施相关专利。
