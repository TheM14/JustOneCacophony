# JustOneCacophony — JOC

[English](README.en.md)

> JustOneCacophony 是一个 E-AC-3 JOC 的实验性 / 测试实现，用于研究 JOC 的解析、重建、渲染以及相关数学过程。

项目可以从常见 E-AC-3 JOC 码流中提取并解析 EMDF、ID14 JOC 参数和 ID11 OAMD 元数据，结合 FFmpeg 解码出的核心 5.1 PCM 重建 LFE 与 15 路对象 PCM，并输出 ADM BWF 或指定扬声器布局的 WAV。

这是研究代码，不是完整、标准兼容或生产级的 Dolby JOC 解码器。它只覆盖当前已实现的码流形态；遇到未知变体时会明确报错，而不是假装一切都很和谐——如果哪里算错了，它可能就真的只剩 cacophony 了。

## 当前功能

- 扫描 E-AC-3 同步帧中的常见连续 EMDF 容器；
- 解析 ID14 dense JOC 参数、Huffman 数据、差分矩阵与 `joc_clipgain`；
- 解析 ID11 OAMD 位置更新并生成对象轨迹；
- 通过 analysis QMF、参数插值、对象矩阵和 inverse QMF 重建 LFE + 15 路对象 PCM；
- 输出 25 声道 ADM BWF：10 声道 7.1.2 bed（除 LFE 外静音）+ 15 个对象；
- 直接渲染 `2.0`、`3.1`、`5.1`、`7.1`、`5.1.2`、`5.1.4`、`7.1.2`、`7.1.4`、`9.1.4`、`9.1.6`；
- 输出 float32 或 PCM24 WAV，并在 PCM24 削波前提供明确处理策略；
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
       └─ 指定布局的扬声器 WAV
```

Python 与 C++ 后端使用同一组已记录的数学过程。原生核只处理状态密集的 DSP 和扬声器渲染，高层位流解析、ADM 组装与命令行逻辑仍在 Python 中。

## 环境

- Python 3.10+
- NumPy 1.24+
- 独立的 FFmpeg 可执行程序；不需要 `ffmpeg-python`。默认从 `PATH` 查找，也可通过 `--ffmpeg` 指定可执行文件路径
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

在非交互环境请求 PCM24 且可能削波时，需要显式选择处理方式：

```powershell
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action abort
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action float32
python main.py input.m4a --speaker-layout 5.1 --speaker-format int24 --clip-action continue
```

元数据与诊断：

```powershell
python main.py input.m4a --print-metadata summary
python main.py input.m4a --metadata-only --print-metadata frames
python main.py input.m4a --metadata-cache metadata_cache
python main.py input.m4a --metadata-dir metadata_cache
```

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
- float32 与 PCM24 输出量化。

解码与渲染过程使用的公式见[数学说明](docs/math.md)。

## 已知限制

- 当前只覆盖常见 continuous EMDF transport；跨多个 audio-block skip field 的碎片化 transport 尚未覆盖。
- Dense JOC 是当前主要路径；Sparse JOC 分支不应视为受支持能力。
- 扬声器路径当前只覆盖普通点对象；extent、spread、divergence 等对象模式不在支持范围内。
- 多数据点、少见参数带配置和特殊 OAMD 调度的覆盖度低于常见 12-band、单数据点素材。
- 扬声器 limiter 不属于当前实现的主公式。
- ADM 输出、原生库和扬声器布局仍需在更多平台、播放器与真实素材上确认互操作性。

## 文档

- [数学说明](docs/math.md) · [English](docs/math.en.md)
- [原生核说明](docs/native.md) · [English](docs/native.en.md)
