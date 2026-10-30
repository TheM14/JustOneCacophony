# Rosella 滤波器组

[English](rosella_filterbank.en.md) · [返回 README](../README.md)

Rosella `.personalized_headphone` 路径可以在两套 64-QMF / 77-hybrid 滤波器组之间选择：
公开滤波器组（默认）与从参考渲染器捕获的 Rosella 滤波器组。SOFA 与 `.jochrtf` 路径
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

## 为什么两套不同

参考头戴渲染器在初始化之后的准备阶段才选定 analysis/synthesis 回调，只捕获初始化
时的回调会拿到错误的滤波器组变体。运行时缓冲捕获得到的参数是：

| 参数 | Rosella 捕获版 | 公开版 |
| --- | --- | --- |
| QMF prototype | 640 个非对称系数 | 公开 640-tap prototype 的多相重排 |
| 低频 hybrid 滤波 | 11 taps | 13 taps |
| 高频 hybrid delay | 3 slots | 6 slots |
| analysis/synthesis 延迟 | 448 samples | 961 samples |

用错变体会在 room 尾音上引入随频率变化的误差；把捕获到的 room send 作为输入时，
现有 room 递推已经能对上参考，因此修复只更换滤波器组，不改 room 系数、direct HRTF、
相位映射或输出 EQ。公开版是 `1`，捕获版是 `2`；默认保留公开版，便于与历史输出对齐。

## 实现

`rosella_captured_filterbank.py` 用 float64/complex128 复现捕获版的四个 half-step：

- QMF 分析保留 9 slot 环形历史，按 `(position + tap + 1) mod 10` 读取，输出
  `spectrum[2b] = F[b]/64`、`spectrum[2b+1] = conj(F[63-b])/64`；
- hybrid 分析保留 10 slot 历史，低频三个 band 分别做 8/4/4 点变换后按固定顺序重排，
  band 1 与 band 3 取共轭；高频带原样延迟 3 slot；
- QMF 合成保留 9 slot 历史，用 10 个 lag 的 prototype 匹配累加，奇数 lag 取虚部。

`data/rosella_captured_kernels.npz` 保存捕获版的 prototype（`float32[64,10]`）、
rotation（`complex64[64]`）与 hybrid 系数（`complex64[22]`）；载入后提升为
float64/complex128。这些数值与 C++ 内核编译进去的
`src/hrtf/rosella_filterbank_tables.h` 相同，不包含个性化 HRTF 或 room 调参。

运行时的起始延迟跳过与 flush 容量都取自所选滤波器组：`961` 或 `448` samples，
`.report.json` 的 `binaural_renderer_backend` 用 `filterbank`、`filterbank_name` 与
`latency_compensated_samples` 记录实际选择。

## 验证

本地回归脚本放在 `tests/`（该目录不随仓库跟踪，与其余本地验证脚本一致），覆盖：

- 捕获版四个 half-step 的参考向量（与 C++ 内核 `tests/rosella_filterbank_vectors.h`
  同一组、由独立 NumPy 实现生成），逐样本误差小于 `1e-10`；
- 两套滤波器组的脉冲往返延迟分别落在 `961` 与 `448` samples；
- 分块处理（含 0 长度与不等长切分）与 `reset()` 不变性；
- 两套滤波器组互不干扰：交叉处理一个 bank 不改变另一个 bank 的状态；
- 越界取值被拒绝。

`docs/binaural.md` 描述的公开滤波器组公式与 `.jochrtf` 方向和公开版一致；捕获版只
作用于 Rosella 路径。
