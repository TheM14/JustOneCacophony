# Rosella 滤波器组

[English](rosella_filterbank.en.md) · [返回 README](../README.md)

Rosella `.personalized_headphone` 路径可以在两套 64-QMF / 77-hybrid 滤波器组之间选择：
13-tap 公开滤波器组（默认）与 11-tap Rosella 滤波器组。SOFA 与 `.jochrtf` 路径
始终使用公开滤波器组，不经过该选择。

## 选择哪一套

`--rosella-filterbank 1|2` 选择 Rosella 路径的滤波器组；库接口是
`RosellaBinauralRenderer(..., filterbank=...)`，省略参数等于 `1`。

| 取值 | 实现 | 低频 hybrid | 高频 slot delay | 延迟 |
| --- | --- | --- | --- | ---: |
| `1`（默认） | `rosella_filterbank.py` | 13 taps | 6 slots | 961 samples |
| `2` | `rosella_captured_filterbank.py` | 11 taps | 3 slots | 448 samples |

两套实现暴露同一组四个有状态的 half-step（QMF 分析、hybrid 分析、hybrid 合成、
QMF 合成），因此切换会同时改变系数、高频延迟、历史缓冲布局、起始延迟跳过量和
flush 容量；分析与合成两级各自持有状态，互不共享。其它取值直接报错；非 Rosella
输入（SOFA、`.jochrtf`）给出该选项也报错，不会静默忽略。

## 两套参数

| 参数 | 11-tap Rosella | 13-tap 公开版 |
| --- | --- | --- |
| QMF prototype | 640 个非对称系数 | 公开 640-tap prototype 的多相重排 |
| 低频 hybrid 滤波 | 11 taps | 13 taps |
| 高频 hybrid delay | 3 slots | 6 slots |
| analysis/synthesis 延迟 | 448 samples | 961 samples |

该选项不改变 room 系数、direct HRTF、相位映射或输出 EQ。默认选择 `1`。

## 实现

`rosella_captured_filterbank.py` 用 float64/complex128 执行 11-tap 滤波器组的四个 half-step：

- QMF 分析保留 9 slot 环形历史，按 `(position + tap + 1) mod 10` 读取，输出
  `spectrum[2b] = F[b]/64`、`spectrum[2b+1] = conj(F[63-b])/64`；
- hybrid 分析保留 10 slot 历史，低频三个 band 分别做 8/4/4 点变换后按固定顺序重排，
  band 1 与 band 3 取共轭；高频带原样延迟 3 slot；
- QMF 合成保留 9 slot 历史，用 10 个 lag 的 prototype 匹配累加，奇数 lag 取虚部。

`data/rosella_captured_kernels.npz` 保存 11-tap 滤波器组的 prototype（`float32[64,10]`）、
rotation（`complex64[64]`）与 hybrid 系数（`complex64[22]`）；载入后提升为
float64/complex128，不包含个性化 HRTF 或 room 调参。

运行时的起始延迟跳过与 flush 容量都取自所选滤波器组：`961` 或 `448` samples，
`.report.json` 的 `binaural_renderer_backend` 用 `filterbank`、`filterbank_name` 与
`latency_compensated_samples` 记录实际选择。

`docs/binaural.md` 描述的公开滤波器组公式与 `.jochrtf` 方向和公开版一致；11-tap 版本
只作用于 Rosella 路径。
