# JustOneCacophony — JOC

[English](README.en.md)

> JustOneCacophony 是一个 E-AC-3 JOC 的实验性 / 测试实现，用于研究 JOC 的解析、重建、渲染以及相关数学过程。

项目可以从常见 E-AC-3 JOC 码流中提取并解析 EMDF、ID14 JOC 参数和 ID11 OAMD 元数据，结合 FFmpeg 解码出的核心 5.1 PCM 重建 LFE 与 15 路对象 PCM，并输出 ADM BWF、指定扬声器布局的 WAV，或使用标准 SOFA HRTF 直接输出双耳 WAV。

这是研究代码，不是完整、标准兼容或生产级的 JOC 解码器。它只覆盖当前已实现的码流形态；遇到未知变体时会明确报错，而不是假装一切都很和谐——如果哪里算错了，它可能就真的只剩 cacophony 了。

## 当前功能

- 扫描 E-AC-3 同步帧中的常见连续 EMDF 容器；
- 解析 ID14 dense JOC 参数、Huffman 数据、差分矩阵与 `joc_clipgain`；
- 解析 ID11 OAMD 位置更新并生成对象轨迹；
- 通过 analysis QMF、参数插值、对象矩阵和 inverse QMF 重建 LFE + 15 路对象 PCM；
- 输出 25 声道 ADM BWF：10 声道 7.1.2 bed（除 LFE 外静音）+ 15 个对象；
- 直接渲染 `2.0`、`3.1`、`5.1`、`7.1`、`5.1.2`、`5.1.4`、`7.1.2`、`7.1.4`、`9.1.4`、`9.1.6`；
- 从 `pcm16 + ID11/OAMD` 直接运行公开 SOFA 双耳渲染，不生成临时 ADM BWF；
- 双耳 DSP 全程使用 float64/complex128，并保留 961-sample latency compensation、跨帧状态和 room 尾声；
- 直接输出统一支持 float32 或 PCM24 WAV，并在 PCM24 削波前提供明确处理策略；
- 使用 NumPy 后端，或通过 `ctypes` 调用可选的 C++20 原生核；`auto` 模式在原生库不可用时回退到 Python；
- 读取或写入 metadata sidecar，并生成元数据、运行时间和输出摘要。

## 处理流程

```text
M4A / E-AC-3
  ├─ FFmpeg 提取 E-AC-3 并解码核心 5.1 PCM
  ├─ EMDF → ID14 JOC 参数 → 对象矩阵
  ├─ 核心 PCM → analysis QMF → 参数插值 → inverse QMF
  ├─ ID11 OAMD → 对象位置与时间轨迹
  └─ LFE + 15 objects
       ├─ 25ch ADM BWF
       ├─ 指定布局的扬声器 WAV
       └─ ID11 直接时间轴 + SOFA HRTF → 双耳 WAV
```

Python 与 C++ 后端在 JOC 对象重建、扬声器渲染和公开 SOFA 双耳渲染中使用同一组
数学过程；native 双耳后端与 Python 参考实现逐值一致（差异 < 1e-9）。位流解析、
OAMD 时间轴和命令行逻辑在 Python 中。

## 环境

- Python 3.10+
- NumPy 1.24+
- h5py 3.8+
- SciPy 1.10+
- 独立的 FFmpeg 可执行程序；不需要 `ffmpeg-python`。默认从 `PATH` 查找，也可通过 `--ffmpeg` 指定可执行文件路径。启动时会探测 `ffmpeg -h decoder=eac3`：缺 E-AC-3 解码器或 `-drc_scale` 直接报错，缺 `-target_level` 只在使用 `--eac3-target-level` 时报错
- 可选：支持 C++20 的 CMake 工具链，用于自行构建原生核

建议在项目专用虚拟环境中安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果 FFmpeg 不在 `PATH` 中：

```powershell
python main.py input.m4a --ffmpeg C:\path\to\ffmpeg.exe
```

## 使用方法

默认输出 25 声道 ADM BWF：

```powershell
python main.py input.m4a
```

选择后端或输出路径：

```powershell
python main.py input.eac3 -o output.adm.wav --backend python
python main.py input.m4a --backend native --native-threads 2
python main.py input.m4a --native-library lib/eac3joc_core.dll
```

直接输出扬声器 WAV：

```powershell
python main.py input.m4a --speaker-layout 2.0 --speaker-format float32
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24
python main.py input.m4a --speaker-layout 7.1.2 --speaker-output output.7.1.2.wav
```

直接输出双耳渲染 WAV（普通对象仅 Near/Mid/Far，默认 Mid）。HRTF 输入支持三种来源：

```powershell
# 1) SOFA（缺省取 HRTF/binaural.sofa，也可显式指定）
python main.py input.m4a --binaural
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa

# 2) Rosella .personalized_headphone（缺省取 HRTF/binaural.personalized_headphone）
python main.py input.m4a --binaural --personalized-headphone
python main.py input.m4a --binaural --personalized-headphone C:\HRTF\subject.personalized_headphone

# 3) .jochrtf 编译缓存
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf

# 常用选项
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa `
  --binaural-mode near
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa `
  --hrtf-cache-policy disk
python main.py input.m4a --binaural --binaural-output output.binaural.wav
```

三者都不指定时的自动选择顺序：`HRTF/binaural.sofa` → `output/hrtf-cache` 下唯一的
`.jochrtf` → `HRTF/binaural.personalized_headphone`；都没有则报错并提示显式指定。

- `.sofa` 是可移植的 source of truth；可以是自行扫描或任何来源的通用 HRTF 数据。
- `.personalized_headphone` 是杜比官方软件个性化扫描得到的模型，其 JSON 解析由
  本项目自行实现（`src/rosella_model.py`），不调用杜比软件。
- `.jochrtf` 是从 SOFA 编译出的项目内部 cache，可删除、可从 SOFA 重建，默认写在
  `output/hrtf-cache`。

HRTF 数据统一放在 `HRTF/`（git 忽略）：默认 SOFA `HRTF/binaural.sofa`、默认模型
`HRTF/binaural.personalized_headphone`。cache 含有源 HRTF 的变换数据，使用与再分发
仍受源数据许可约束；格式边界、计算公式、状态、时间轴及发布注意事项见
[双耳渲染](docs/binaural.md) 和 [第三方通知](THIRD_PARTY_NOTICES.md)。

扬声器和双耳输出共享峰值检查、writer 与削波策略。在非交互环境请求 PCM24 且可能削波时，需要显式选择处理方式：

```powershell
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action abort
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action float32
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action continue
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa `
  --binaural-format int24 --clip-action abort
```

元数据与诊断：

```powershell
python main.py input.m4a --print-metadata summary
python main.py input.m4a --metadata-only --print-metadata frames
python main.py input.m4a --metadata-cache metadata_cache
python main.py input.m4a --metadata-dir metadata_cache
```

### E-AC-3 解码级动态范围与电平

FFmpeg 解码 E-AC-3 时默认施加码流 `dynrng` 动态范围压缩（`-drc_scale 1`）。核心 5.1 PCM 是 JOC 对象重建的输入，而 `dynrng` 属于回放期增益，会被线性继承到全部对象与成品（ADM／扬声器／双耳），因此本工具默认按**全动态范围**解码：

```powershell
python main.py input.m4a                            # 默认：-drc_scale 0，全动态范围
python main.py input.m4a --eac3-drc-scale 1         # 复现消费者回放（码流作者意图）
python main.py input.m4a --eac3-drc-scale 0.5       # 施加一半
python main.py input.m4a --eac3-target-level -27    # 按码流 dialnorm 归一化电平
```

- `--eac3-drc-scale`（`0`～`6`，默认 `0`）对应 FFmpeg 的 `-drc_scale`：每个 E-AC-3 block 的增益为 `dynrng 因子 ^ 该值`。`0` 关闭 DRC；`1` 为码流作者意图；`>1` 非对称（响处全压、轻处增强）。
- `--eac3-target-level`（`-31`～`0`，默认 `0` 不施加）对应 FFmpeg 的 `-target_level`：按每帧 dialnorm 施加静态增益，约 `target_level - dialnorm` dB，与 `--eac3-drc-scale` 相互独立、可叠加。dialnorm 是逐码流属性（实测 Apple Music Atmos 流约 `-18`～`-19` dB，故 `-27` 约等于衰减 `8`～`9` dB）。
- 电平变化是预期的：与 FFmpeg 默认值相比，实测曲目峰值变化 `0`～`-2.15` dB、RMS `0`～`-1.69` dB（方向取决于码流 `dynrng`），`.report.json` 的 `output_clip.peak` 与 int24 削波判定会随之变化。
- `--gain-db` 是**重建之后**的静态增益（双耳路径 float64），与解码级 DRC 不是一回事；解码级 DRC 是按 block 时变的，不要用 `--gain-db` 去抵消它。
- `.report.json` 的 `ffmpeg` 字段记录 FFmpeg 版本与实际下发的解码选项（`version`、`eac3_decode_options`）。

### 双耳渲染模式

`--binaural-mode off|near|mid|far` 选择双耳渲染模式，默认 `mid`，两种输出共用这一个选项：

- **直接双耳渲染**（`--binaural`）：`off` 不可用（报错），near/mid/far 生效，默认 `mid`；
- **ADM BWF**：DBMD segment 10 中后 15 个 JOC 对象的 binaural render mode 写
  `off=0/near=1/far=2/mid=3`，前 10 个 bed 保持不变，默认 `mid`；`off` 用于显式
  关闭双耳元数据提示。

```powershell
python main.py input.m4a --binaural-mode mid
python main.py input.m4a --binaural-mode off   # 仅 ADM BWF：关闭 DBMD 双耳提示
```

**默认 `mid` 是本工具人为指定的渲染提示**，不是从输入 E-AC-3 JOC 码流中提取或
还原的原始双耳元数据，也不代表原始混音中各对象的双耳设置。该提示不改变 PCM、
对象轨迹或直接扬声器渲染。输出旁的 `.report.json` 用 `binaural_mode`（模式名）
和 `binaural_mode_value`（ADM 编码值，直接双耳输出时为 `null`）记录。

### OAMD 时间对齐

对象轨迹和直接扬声器渲染的 metadata delay 默认均为 `1473 samples`。该值描述 decoder 输出 PCM 与 OAMD 更新之间的理论时间映射；扬声器 renderer 仍使用现有的 32-sample control block，因此默认更新的实际 block boundary 为 `1472`：

```text
align32(1473) = 1472
```

可分别用 `--object-delay-samples` 和 `--speaker-metadata-offset` 覆盖默认值。这里的 1473 不应与 inverse-QMF 的 640 项 filter/window state 混淆；后者是 QMF 状态长度，不是 metadata delay。

直接双耳路径使用 `--object-delay-samples`。每个 ID11/OAMD event 先按 frame start、
outer subpayload offset 与 block offset 落到绝对 sample timeline，再加该 delay；每个
1536-sample 输入帧按三个连续 512-sample block 处理，并在每块的绝对起始 sample
查询插值后的位置、更新方向和 profile。

### 双耳计算

双耳路径的 QMF、hybrid、方向场、距离、ITD、room、上述 512-sample 参数更新和 961-sample 延迟补偿见[双耳渲染数学](docs/binaural.md)。

更多参数可查看：

```powershell
python main.py --help
```

未指定 `-o` 时，输出仍写入仓库根目录的 `output/`。这是文件移动后特意保持的原有行为。

## 原生核

仓库默认不附带原生二进制。可以从项目 Release 下载适合当前平台的预构建运行库，或自行构建，然后把运行库直接放入仓库根目录的 `lib/`；若该目录不存在，创建即可。自行构建时可从仓库根目录使用 CMake：

```powershell
cmake -S native -B build/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/lib"
cmake --build build/cmake --config Release
cmake --install build/cmake --config Release
```

运行时查找顺序为：

1. `--native-library`；
2. `EAC3JOC_NATIVE_LIBRARY`；
3. `lib/` 下当前平台的标准库文件名。

详细 ABI、状态与精度说明见[原生核说明](docs/native.md)。

## 目录结构

```text
JustOneCacophony/
├─ main.py               命令行启动入口
├─ src/                  Python 实现模块
├─ native/               C/C++ 加速核、C ABI 与必要表数据
├─ data/                 Python 运行时表数据
├─ lib/                  原生运行库投放目录（按需创建）
├─ HRTF/                 用户 HRTF 数据目录（按需创建，git 忽略）
├─ output/               输出目录（按需创建；.jochrtf 缓存默认在其 hrtf-cache 子目录）
├─ docs/                 数学与原生核文档（中英文）
├─ requirements.txt      Python 依赖
├─ README.md             中文说明
└─ README.en.md          English documentation
```

## 数学实现

核心过程包括：

- dense JOC 差分还原与去量化；
- 参数带到 64 个 QMF 子带的映射；
- 跨帧参数插值；
- analysis / inverse QMF、环绕声道延迟与 FIR 状态；
- LFE 1217-sample 延迟；
- OAMD Q15 坐标转换；
- 基于目标布局 region 的等功率声像；
- 布局位置补偿与逐样本增益斜坡；
- float32 与 PCM24 输出量化；
- SOFA canonical importer、64-QMF/77-hybrid 投影、`36×2×77` 五阶方向 field、exactly-once delay/phase、项目 early/late room 与 special LFE。

解码与渲染过程使用的公式见[数学说明](docs/math.md)。

## 已知限制

- 当前只覆盖常见 continuous EMDF transport；跨多个 audio-block skip field 的碎片化 transport 尚未覆盖。
- Dense JOC 是当前主要路径；Sparse JOC 分支不应视为受支持能力。
- 扬声器与 SOFA 双耳路径当前只覆盖普通点对象；extent、spread、diffuse、divergence、channel lock 等对象控制不在支持范围内。
- OAMD trim element 会按声明边界校验并跳过；warp、balance 和 trim 参数不应用于当前原始对象轨迹或扬声器渲染。
- 多数据点、少见参数带配置和特殊 OAMD 调度的覆盖度低于常见 12-band、单数据点素材。
- 扬声器 limiter 不属于当前实现的主公式。
- SOFA importer 当前严格支持 `SimpleFreeFieldHRIR` FIR；其它 SOFA convention 需要显式 adapter。
- 双耳 runtime 固定 48 kHz、五阶和一次选择一个 measurement-radius shell；公开双耳默认走 native 加速，原生库不可用时自动回退 Python。
- ADM 输出、原生库、扬声器布局和双耳模型仍需在更多平台、播放器与真实素材上确认互操作性。

## 文档

- [数学说明](docs/math.md) · [English](docs/math.en.md)
- [原生核说明](docs/native.md) · [English](docs/native.en.md)
- [双耳渲染](docs/binaural.md) · [English](docs/binaural.en.md)
