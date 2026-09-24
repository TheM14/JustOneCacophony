# JustOneCacophony — C++ Core

[English](README.en.md) · [数学说明](docs/math.md) · [双耳渲染](docs/binaural.md) · [SIMD 派发](docs/simd.md)

JustOneCacophony 的 C++ 实现：E-AC-3 JOC 码流解析、对象重建与渲染的执行内核。它从 E-AC-3
同步帧中提取 EMDF、ID14 JOC 参数与 ID11 OAMD 元数据，结合 FFmpeg 解码出的核心 5.1 PCM
重建 LFE 与 15 路对象 PCM，并输出 ADM BWF、指定扬声器布局的 WAV，或用编译好的 HRTF 方向场
直接输出双耳 WAV。

这是研究代码，不是完整、标准兼容或生产级的 JOC 解码器。它只覆盖已实现的码流形态，遇到未知
变体时明确报错，而不是假装一切都很和谐。

## 构建

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows（MSVC / VS 2022，静态 CRT）、Linux 与 macOS 由 `.github/workflows/ci.yml` 同时构建并跑
单元测试；本地只验证 MSVC。浮点行为是逐字节验收的一部分，因此不启用 fast-math：MSVC 用
`/fp:precise`，其他编译器用 `-fno-fast-math`。

Windows Release 默认带 `/arch:AVX2`（`JOC_ENABLE_AVX2`，默认 ON，见 `CMakeLists.txt`）。这个
开关是逐字节验收过的：所有渲染产物的 SHA-256 完全一致，换来 SOFA 双耳内核 12.4%、Rosella
1.8% 的提升。代价是运行要求——这样的 `joc_core.dll` 会执行 AVX2 指令，在 2013 年以前的 x86 上
直接非法指令退出。没有运行时派发，一个二进制只能二选一，所以：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DJOC_ENABLE_AVX2=OFF
```

得到基线 ISA（SSE2）、任何 x86-64 都能跑的产物。非 MSVC 构建永远不会带上这个开关。

### 路径与编码

库内部所有路径都是 **UTF-8**，只在系统边界转换（`src/foundation/fs_utf8.*`）：Windows 上经
`std::filesystem::path`（内部 UTF-16）落到 `_wfopen`/`CreateProcessW`，其他平台直接是字节。
命令行参数在 Windows 上由 `GetCommandLineW` + `CommandLineToArgvW` 重新解析，控制台设为
UTF-8，因此日文/中文等非 ASCII 路径（含 ffmpeg 子进程与输出文件）都能正常工作；单元测试里有
一条非 ASCII 路径的回归用例守在 `ctest` 里。

## 产物

| 产物 | 说明 |
|---|---|
| `joc_core.dll` | 执行内核：码流解析、JOC/OAMD、DSP、扬声器与双耳渲染、ADM BWF/WAV 落盘、文件任务、流式接口 |
| `joc_cli.exe` | 文件任务命令行前端 |
| `include/joc_core.h` | 引擎、遥测与文件任务接口（纯 C） |
| `include/joc_stream.h` | 面向嵌入者的流式 push/pull 接口（纯 C，仅包含它即可） |

## 命令行

参数与上游 Python CLI 完全一致：输入是位置参数，**默认输出 25 通道 ADM BWF**，用
`--speaker-layout` 或 `--binaural` 切换到另外两种模式；未指定 `-o` 时产物落在 `output/`。

```powershell
# 默认：ADM BWF（本身就是 24-bit，没有也不需要格式参数）
joc_cli "07. Gold Forever (2021 Master).m4a"
#   -> output/07. Gold Forever (2021 Master).adm.wav

joc_cli input.m4a -o out/adm.wav                 # 指定输出

# 扬声器布局
joc_cli input.m4a --speaker-layout 5.1           # -> output/<名称>.5.1.wav
joc_cli input.m4a --speaker-layout 7.1.4 --speaker-output out/714.wav --speaker-format int24

# 双耳（HRTF 默认取 <exe>/HRTF/binaural.sofa，其次 <exe>/HRTF/binaural.personalized_headphone）
joc_cli input.m4a --binaural
joc_cli input.m4a --binaural --sofa-hrtf HRTF/other.sofa   # 换一个 SOFA
joc_cli input.m4a --binaural --personalized-headphone      # Rosella 个性化模型
#   -> output/<名称>.binaural.wav

# 其它常用开关
joc_cli input.m4a --duration 30 --gain-db -3 --trajectory-mode dense64
joc_cli input.eac3 --metadata-only --print-metadata summary   # 只解析并打印元数据
```

`--speaker-format` / `--binaural-format` 默认 `float32`；`int24` 若会削波按 `--clip-action`
处理（默认 `ask`，非交互终端下需显式给出 `continue`/`float32`/`abort`）。`--duration` 以**秒**
为单位；`--object-delay-samples`、`--speaker-metadata-offset`、`--binaural-tail-seconds`、
`--binaural-tail-threshold`（默认 1e-8，双耳尾音裁切阈值）等默认值与上游一致。命令总是写出
`<输出>.report.json`（`--report-json` 可改路径）。

**与上游参数的差异**：`--sofa-hrtf`、`--personalized-headphone`、`--backend python` 以及
metadata sidecar（`--metadata-dir`/`--metadata-cache`/`--metadata-backend sidecar`）在本构建中
不可用，给出时立刻报错并说明原因，而不是静默忽略。本构建额外提供 `--bed`（已解码的 6 通道
float32 PCM，给出后不调用 ffmpeg 解码）、`--kernels`（滤波器组表路径）、`--work-dir`、
`--report-json`、`--dry-run`、`--quiet`。

## 库集成

引擎与文件任务使用 `joc_core.h`：`joc_task_validate` / `joc_task_execute` 跑一个文件任务并
通过回调返回状态事件，`joc_task_result` 给出帧数、峰值、字节数与 SHA-256。

播放器或解码组件使用 `joc_stream.h`：调用方按自己的节奏推入 E-AC-3 字节与对应的核心 PCM
（或已重建的 objects16），再拉取渲染后的 PCM；分块粒度任意，输出与文件任务逐字节一致。

```c
joc_stream_config config = {0};
config.struct_size = sizeof(config);
config.input = JOC_STREAM_IN_EAC3;      /* 或 JOC_STREAM_IN_PCM_OBJECTS16 */
config.output = JOC_STREAM_OUT_SPEAKER; /* 或 BINAURAL / PCM_OBJECTS16 */
config.speaker_layout_name = "5.1";
joc_stream* stream = NULL;
joc_stream_create(&config, &stream);
/* 循环：joc_stream_push(...) / joc_stream_pull(...) */
joc_stream_flush(stream);
joc_stream_destroy(stream);
```

契约：状态实例私有、可并存；同一实例的 push/pull 必须在同一线程；渲染是有状态的，
因此**本版本不提供 seek**——定位需要从流起点重新解码。扬声器/双耳通路的内核延迟为
961 样本，`joc_stream_flush` 负责排空双耳房间尾音。

## 目录

```text
include/     公共 C ABI：joc_core.h（引擎/文件任务）、joc_stream.h（流式）、eac3joc_core.h（上游 ABI）
src/         实现：eac3_transport、emdf、joc_bitstream、joc_core、oamd、timeline、speaker、
             binaural、hrtf、adm、io、telemetry、task、stream、api、cli、simd
tests/       单元测试（CTest，自足，不需要外部素材）
docs/        数学说明、双耳渲染流程与 SIMD 派发
```

`src/` 下 8 个文件是 JustOneCacophony 原生库的逐字节副本，永不修改（见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。

## 兼容性说明

`object_delay_samples` 默认 **1473**，与既有实现的行为保持一致；它是可配置字段，改动它会
改变 OAMD/ADM 时间对齐。上游调查认为该值应为 0，本项目为保持逐字节等价暂不改默认值。

## 许可

MIT，见 [LICENSE](LICENSE)；第三方来源与专利边界见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
