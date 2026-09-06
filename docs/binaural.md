# 双耳渲染

[English](binaural.en.md) · [返回 README](../README.md)

JustOneCacophony 的双耳后端支持三种 HRTF 来源：`SimpleFreeFieldHRIR` SOFA、
杜比官方软件个性化扫描导出的 Rosella `.personalized_headphone`（JSON 解析由本项目
自行实现，不调用杜比软件），以及从 SOFA 编译出的 `.jochrtf` 缓存。SOFA 在模型加载
时编译成内存方向场；`.jochrtf` 只是可删除、可重建的 JOC compiled HRTF cache，
不是交换格式，也不是使用 SOFA 的前置步骤。

```text
SOFA FIR
  -> CanonicalHrtf
  -> 48 kHz / 单 radius shell / delay-phase policy
  -> 64-QMF / 77-hybrid projection
  -> 五阶 ACN/N3D 实球谐场
  -> 逐对象 direct + early reflections
  -> shared unitary-FDN late room
  -> stereo float64
```

## 输入接口

CLI 有三个互斥的 HRTF 输入来源；都不指定时按默认规则自动选择：

```powershell
# 1) SOFA：缺省取 HRTF/binaural.sofa，也可显式指定
python main.py input.m4a --binaural
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa

# 2) Rosella .personalized_headphone：缺省取 HRTF/binaural.personalized_headphone
python main.py input.m4a --binaural --personalized-headphone
python main.py input.m4a --binaural --personalized-headphone C:\HRTF\subject.personalized_headphone

# 3) .jochrtf：显式读取预编译 cache
python main.py input.m4a --binaural `
  --compiled-hrtf-cache C:\HRTF\subject.jochrtf

# 可选：SOFA 透明生成/复用磁盘 cache
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa `
  --hrtf-cache-policy disk
```

默认选择顺序：`HRTF/binaural.sofa` → `output/hrtf-cache` 下唯一的 `.jochrtf` →
`HRTF/binaural.personalized_headphone`；三者都没有时报错并提示显式指定。
`output/hrtf-cache` 下有多个 `.jochrtf` 时同样报错，要求显式选择。

`.personalized_headphone` 的 JSON 解析由本项目自行实现（`src/rosella_model.py`），
不调用任何杜比软件。

`--hrtf-cache-policy` 可取 `none`、`memory`、`disk`。默认是 `memory`；`none` 和
`memory` 都不会创建磁盘文件。`disk` 默认写入 `output/hrtf-cache`，也可用
`--hrtf-cache-dir` 指定。`--hrtf-radius-m` 选择距离目标最近的 measurement shell。

Python API 使用显式 factory：

```python
from sofa_binaural_backend import SofaBinauralBackend

renderer = SofaBinauralBackend.from_sofa(
    "subject.sofa",
    source_count=16,
    default_profile="mid",
    cache_policy="memory",
)

cached = SofaBinauralBackend.from_compiled_cache(
    "subject.jochrtf",
    source_count=16,
    default_profile="mid",
)
```

文件工厂不会按“未知后缀”猜格式：SOFA 和 `.jochrtf` 始终走不同 loader。

## 双耳渲染模式

`--binaural-mode off|near|mid|far`（默认 `mid`）是**人为指定的渲染提示**，不是
从输入 E-AC-3 JOC 码流提取或还原的原始双耳元数据：

- 直接双耳渲染（`--binaural`）：near/mid/far 生效，默认 `mid`；`off` 报错；
- ADM BWF：DBMD segment 10 中后 15 个 JOC 对象的 binaural render mode 写
  `off=0/near=1/far=2/mid=3`，前 10 个 bed 保持不变，默认 `mid`；`off` 用于显式
  关闭双耳元数据提示。

## Canonical SOFA 契约

当前 strict importer 接受：

- `Conventions=SOFA`；
- `SOFAConventions=SimpleFreeFieldHRIR`，version `0.4`、`1.0` 或 `1.1`；
- `DataType=FIR`，`Data.IR[M,2,N]`；
- 单一正有限 `Data.SamplingRate`，单位为 hertz/Hz；
- spherical 或 Cartesian `SourcePosition`；
- 单值或 per-measurement 的 `ListenerPosition/View/Up`；
- 两个能由 listener-local lateral 坐标唯一识别左右的 receiver；
- 单一且零偏移的 emitter；
- causal `Data.Delay[I,2]` 或 `[M,2]`；
- 明确的 free-field/anechoic `RoomType`。

receiver 左右顺序由几何决定，不能假定 `Data.IR` 的 receiver index。SOFA listener
坐标为 $+X$ front、$+Y$ left、$+Z$ up；ADM 坐标为 $+X$ right、$+Y$ front、
$+Z$ up，转换为：

$$\bigl(x_{\mathrm{SOFA}},\ y_{\mathrm{SOFA}},\ z_{\mathrm{SOFA}}\bigr) = \bigl(y_{\mathrm{ADM}},\ -x_{\mathrm{ADM}},\ z_{\mathrm{ADM}}\bigr)$$

`CanonicalHrtf` 将 `Data.IR` 与 `Data.Delay` 分开保存。只有时域 baseline 才调用
`materialized_measurement()` 将 delay 应用一次；运行时 SH 路径不先 materialize。
非 48 kHz HRIR 使用 float64 `scipy.signal.resample_poly` 规范化，delay samples 按
相同比例缩放。

GeneralFIR、BRIR、TF、多 emitter、多义 receiver 或非 free-field 数据需要单独的
convention adapter，不能只通过 reshape 进入核心 importer。

## Delay/phase：exactly once

编译器只允许三种互斥语义：

1. 非零 `Data.Delay` 是 `Data.IR` 外部 delay；FIR 不去旋，运行时应用一次。
2. `Data.Delay=0` 且 HRIR 有普通正 onset：以每耳 main peak 分离 arrival，拟合后
   在运行时恢复一次；当前阈值为 peak index 大于 2 samples。
3. `Data.Delay=0` 且双耳 FIR 共享 sample-0 起点：不发明外部 delay，原 complex
   phase 直接进入五阶场。

任何路径都不能再叠加第二套 ear delay 或 phase-group delay。

## 公开 filterbank 与方向场

运行时固定为：

- 48 kHz；
- 64-sample QMF hop；
- 64-QMF / 77-hybrid；
- analysis/synthesis latency 961 samples；
- 五阶、36 项、ACN/N3D real spherical harmonics；
- PCM、delay、SH、room state 为 float64；频带传递和频域状态为 complex128。

每个 hybrid band 的 real/imaginary 单位增益都通过同一套 analysis/synthesis 链生成
脉冲字典，共 154 个实参数；编译不是直接读取 77 个 FFT bin。默认 projection
ridge 为 `1e-3`，SH ridge 为 `1e-5`。同方向 measurement 先合并，再用球面 Voronoi
面积权重做 ridge fit。

固定表位于 `data/rosella_kernels.npz`，实现公开标准化的滤波器组，各表可由如下
公式计算。

hybrid 分析核定义于 [3GPP TS 26.405 / ETSI TS 126 405](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf)
第 5.2.2 节（Table 1 的 $Q=8$/$Q=4$ 系数，delay 6）：

$$G_q^p[n] = g^p[n]\cdot\exp\!\Bigl(j\,\frac{2\pi}{Q^p}\bigl(q+\tfrac12\bigr)(n-6)\Bigr),\qquad n=0,\dots,12$$

QMF analysis 表即 MPEG-4 AAC/SBR（ISO/IEC 14496-3/AMD1:2003 第 4.B.18.2 节）
的 64 complex QMF bank；打包的 $64\times10$ 表是公开 640-tap prototype
$c_0,\dots,c_{639}$ 的多相重排：

$$A_{r,t} = \frac{(-1)^t}{128}\,c_{63-r+64t},\qquad r=0,\dots,63,\ t=0,\dots,9$$

QMF synthesis 表为上述 analysis 多相矩阵 $\mathbf{A}$ 的因果左逆，即求解
$\mathbf{A}\,\mathbf{W}=\mathbf{P}$（$\mathbf{P}$ 为 577-sample 延迟置换；
全链 $961 = 577 + 6\times64$），以 rank-4 分解形式存储：

$$W_{b,l} = \sum_{r=1}^{4} t_{b,l,r}\,\mathbf{b}_{b,r}^{\top}$$

hybrid synthesis 表为 77→64 重组：高频带恒等 $Y_{3+b}=X_{16+b}$；低频带
（$C_p$ 为 $8+4+4$ 子带划分）：

$$Y_p = \sum_{q\in C_p}\Bigl(\operatorname{Re}X_q + j\,s_q\,\operatorname{Im}X_q\Bigr),\qquad s_q\in\{\pm1\}$$

loader 校验 archive 和每个数组的 SHA-256；table version、所有数组 hash 与
77 个 band-center 参考值都属于 cache key。标准可公开获取不等于获准实施相关
专利；更多来源信息见 [`data/README.md`](../data/README.md) 与
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

## `.jochrtf`

`.jochrtf` 是无 pickle 的压缩 NumPy archive，固定包含：

| key | dtype / shape |
|---|---|
| `metadata_json` | 含 JSON 文本的 NumPy Unicode scalar（`dtype.kind == "U"`） |
| `band_center_frequencies_hz` | little-endian `float64[77]` |
| `coefficients` | little-endian `complex128[36,2,77]` |
| `delay_coefficients` | little-endian `float64[36,2]` |
| `delay_bounds` | little-endian `float64[2,2]` |

metadata magic 固定为 `JOC-HRTF-CACHE`，并记录 schema/compiler/phase-policy、
ACN/N3D、filterbank table hashes、SOFA content SHA-256、采样率、radius、order、
两个 ridge、payload hash 和 fit report。cache key 覆盖所有会改变编译结果的字段。
metadata 不保存本机绝对 `source_path`，仅可保存 source display name。

loader 使用 `allow_pickle=False`，并在构造对象前检查 ZIP 成员集、解压大小、shape、
dtype、端序、连续布局、有限值、delay bounds、band centers、payload hash 和 cache
key。writer 使用同目录临时文件、`fsync`、进程持有的 OS 文件锁和原子
`os.replace`；对应的隐藏 `.lock` sidecar 可保留，但不代表仍有 writer 持锁。
旧版本、损坏或配置不匹配的 cache 不能命中；从 SOFA 启动时会重建，显式 cache
入口则直接报错。

删除磁盘 cache 后，从同一 SOFA 和同一编译配置得到的场与渲染结果不得改变。

`.jochrtf` 包含由源 HRIR 变换得到的方向场系数与 delay 数据，因此“可以重建”不表示
它不受数据许可约束。生成 cache 不会扩大源 SOFA/HRTF 数据集授予的权利；cache 的
使用、复制和再分发仍须遵守源数据集条款。不能确认条款时，应把 `.jochrtf` 作为本地
私有 cache，不随程序或构建产物发布。`source_sha256` 只用于内容一致性校验，不是许可
或来源证明。

## JOC 对象与房间

生产适配器继续使用现有 JOC 调度：

- 每帧输入 `[1536,16]`；
- channel 0 是 special LFE，channel 1..15 是 JOC objects；
- ID11/OAMD position 使用 sample-timed timeline；
- 每 512 samples 更新方向/profile；
- 每个对象拥有独立 direct/early history，late FDN 全局共享；
- `finish()` 排空 early/late tail；输出增益显式应用，不隐含 limiter 或节目响度归一化。

Near/Mid/Far、equal-power direct level、六面 shoebox 一阶 image source、late send、
unitary FDN、LFE 120–180 Hz cosine-squared 低通及 room calibration 都是 JOC
项目定义行为，不是 SOFA 或 Dolby 公布常数。

公开 SOFA 双耳渲染在 `--backend auto/native` 下默认走 C++20 原生核
（`lib/eac3joc_core.dll` 的 `ejoc_sofa_binaural_*` 接口：filterbank、SH 方向场求值、
逐对象 early/direct 历史与共享 FDN 全部在原生侧执行，Python 只做 SOFA 编译与每
512-sample 的元数据更新）；原生库不可用时自动回退 Python/NumPy 参考实现，两者
逐值一致（差异 < 1e-9）。`--backend python` 强制使用 Python 后端。

## 技术引用与权利边界

- [SOFA SimpleFreeFieldHRIR convention](https://www.sofaconventions.org/mediawiki/index.php/SimpleFreeFieldHRIR)
- [3GPP TS 26.405 / ETSI TS 126 405（64-QMF/77-hybrid 定义）](https://www.etsi.org/deliver/etsi_ts/126400_126499/126405/06.00.00_60/ts_126405v060000p.pdf)
- [Dolby binaural render mode workflow](https://professionalsupport.dolby.com/s/article/What-is-Binaural-Render-Mode-and-how-do-the-settings-affect-my-mix)
- [EP3090576A1](https://patents.google.com/patent/EP3090576A1/en)，仅作 direct/early/late、
  subband 与 FDN 架构背景，不证明某个产品使用特定实施例。

规范、源码或专利文献可公开获取，不等于获准复制其内容、再分发派生产物或实施其中的
专利权利要求。本项目的技术引用本身不授予专利许可，也不作不侵权保证；准备发布或集成
到产品的一方应自行审查适用的数据许可、软件许可、专利许可及 freedom-to-operate。
公开标准来源与权利边界见
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。
