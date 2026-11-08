# JustOneCacophony — C++ Core

[English](README.en.md) · [数学说明](docs/math.md) · [双耳渲染](docs/binaural.md) · [SIMD 派发](docs/simd.md)

JustOneCacophony 的 C++ 实现：E-AC-3 JOC 码流解析、对象重建与渲染的执行内核。它从 E-AC-3
同步帧中提取 EMDF、ID14 JOC 参数与 ID11 OAMD 元数据，结合 FFmpeg 解码出的核心 PCM（N=5 或 7）
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

最重的双耳循环由**运行时派发**：同一份二进制里装多套 ISA 代码，启动时按 CPUID/XGETBV 选最宽
的一套，输出逐字节不变（见 [SIMD 与运行时派发](docs/simd.md)）。`JOC_ENABLE_AVX2` /
`JOC_ENABLE_AVX512`（默认都 ON）只决定是否把这些单元编进二进制：带上 AVX2 单元让 `.jochrtf` 双耳
内核快 12.4%、Rosella 快 1.8%，而没有 AVX2 的 CPU 仍走基线（SSE2）路径。要产出只带基线 ISA
的二进制：

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
joc_cli input.m4a
#   -> output/input.adm.wav

joc_cli input.m4a -o out/adm.wav                 # 指定输出

# 扬声器布局
joc_cli input.m4a --speaker-layout 5.1           # -> output/<名称>.5.1.wav
joc_cli input.m4a --speaker-layout 7.1.4 --speaker-output out/714.wav --speaker-format int24

# 双耳（默认取 <exe>/HRTF/binaural.personalized_headphone）
joc_cli input.m4a --binaural
joc_cli input.m4a --binaural --personalized-headphone      # Rosella 个性化模型
joc_cli input.m4a --binaural --compiled-hrtf-cache HRTF/subject.jochrtf   # 离线编译的 .jochrtf
joc_cli input.m4a --binaural --compiled-hrtf-cache HRTF/subject.jochrtf --jochrtf-room off
#   -> output/<名称>.binaural.wav
joc_cli input.m4a --binaural --personalized-headphone --rosella-filterbank 2   # 新版 Rosella 滤波器组

# 其它常用开关
joc_cli input.m4a --duration 30 --gain-db -3 --trajectory-mode dense64
joc_cli input.eac3 --metadata-only --print-metadata summary   # 只解析并打印元数据
```

`--speaker-format` / `--binaural-format` 默认 `float32`；`int24` 若会削波按 `--clip-action`
处理（默认 `ask`，非交互终端下需显式给出 `continue`/`float32`/`abort`）。`--duration` 以**秒**
为单位；`--object-delay-samples`、`--speaker-metadata-offset`、`--binaural-tail-seconds`、
`--binaural-tail-threshold`（默认 1e-8，双耳尾音裁切阈值）等默认值与上游一致。命令总是写出
`<输出>.report.json`（`--report-json` 可改路径）。

`--rosella-filterbank 1|2` 选择 Rosella 个性化模型使用的滤波器组：`1`（默认）是 13 taps /
6 slots 的公开滤波器组，`2` 是 11 taps / 3 slots 的捕获版。两者延迟分别为 961 / 448 samples，
系数、历史缓冲与 flush 容量各自独立；该选项只作用于 `.personalized_headphone`，
`.jochrtf` 路径给出时直接报错。详见 [Rosella 滤波器组](docs/rosella_filterbank.en.md)。

**与上游参数的差异**：`--backend python` 与 metadata sidecar（`--metadata-dir`/
`--metadata-cache`/`--metadata-backend sidecar`）在本构建中不可用，给出时立刻报错并说明原因，
而不是静默忽略。本构建额外提供 `--bed`（已解码的 6 通道 float32 PCM，给出后不调用 ffmpeg
解码）、`--kernels`（滤波器组表路径）、`--work-dir`、`--report-json`、`--dry-run`、
`--quiet`、`--rosella-filterbank`。

## 扬声器布局与声道顺序

`--speaker-layout` 只接受下表 12 个布局，PCM/WAV 的声道顺序就是表里的顺序：

| 布局 | 声道顺序 |
|---|---|
| 2.0 | L, R |
| 3.1 | L, R, C, LFE |
| 4.0 | L, R, Lrs, Rrs |
| 5.1 | L, R, C, LFE, Ls, Rs |
| 6.0 | L, R, Ls, Rs, Lrs, Rrs |
| 7.1 | L, R, C, LFE, Ls, Rs, Lrs, Rrs |
| 5.1.2 | L, R, C, LFE, Ls, Rs, Ltm, Rtm |
| 5.1.4 | L, R, C, LFE, Ls, Rs, Ltf, Rtf, Ltr, Rtr |
| 7.1.2 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltm, Rtm |
| 7.1.4 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf, Ltr, Rtr |
| 9.1.4 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Lw, Rw, Ltf, Rtf, Ltr, Rtr |
| 9.1.6 | L, R, C, LFE, Ls, Rs, Lrs, Rrs, Lw, Rw, Ltf, Rtf, Ltm, Rtm, Ltr, Rtr |

`4.0` 是本项目补充的 Quad 布局（传统 Quad 的 `FL FR BL BR`），声道为 `L R Lrs Rrs`，不在杜比
文档的布局列表里；除它以外的布局及声道顺序均依据杜比相关规范定义。

多声道 WAV 使用 `WAVE_FORMAT_EXTENSIBLE`，`dwChannelMask = 0`：文件不声明 Windows 扬声器位置，
PCM 声道顺序严格按上表，不按掩码位序重排。播放器、ASIO、DAW 或其他设备需要按声道编号自行完成
输出映射；本项目保证各 PCM 声道的内容和编号符合上表，不保证下游播放器自动识别杜比扬声器位置。

`--speaker-layout` 只接受上表 12 个名字；其它名字按未知布局报错。

扬声器渲染从 OAMD 元数据读取 warp、trim 和 front/back balance；命令行不提供 warp 覆盖。
warp 0/1 直接渲染；warp 2/3 先渲染到 7.1.x 中间布局，再按 PLIIx 或 Standard（Lo/Ro）
折回 5.1.x，校准参数始终按最终目标布局选择。原始对象轨迹与双耳渲染不预处理这些控制，
参数与求值顺序见[数学说明 §12](docs/math.md)。

扬声器渲染还会应用 OAMD 的对象属性：每块增益/inactive、zone/elevation、size/snap/divergence、
priority 尺寸预算、element 5 位置细化、ISF 基矩阵和 bed distribute。ADM 导出与双耳路径仍按点
对象放置，不应用这些扬声器目标系数。

对象位置的调度按完整 `frame_state` 进行：扬声器与双耳渲染都接受多 position block 和超过 16
个对象的节目，按各块的时间与 ramp 逐个调度；双耳渲染的音频输入仍是 1 LFE + 15 对象，多出的
位置轨道只跟踪不参与声像。ADM 导出仍使用 15 轨、单 block 的对象位置视图，超出时按变体报错。

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

`src/` 下的 QMF/双耳表与 JOC Huffman 表按公开标准实现；来源记录见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

ISF 压缩基系数与布局描述以查表方式提供：表数据写在生成头文件 `src/oamd/isf_tables.h` 中，
与代码一起编译进二进制，运行时不读取数据文件。gain、size、priority、位置、divergence
与差分/细化值都在代码中计算，size 核由扬声器布局的 region 几何生成，bed 路由由布局表推导。

## 数学实现

主要文档化阶段：

- dense JOC 差分还原与去量化；
- 参数带到 64 个 QMF 子带的映射；
- 跨帧参数插值（无上一帧时硬切，对象消失后按零矩阵发布）；
- 帧级 JOC 状态机（正常 / 保持 / 硬切 / 核心直通）；
- analysis / inverse QMF、环绕声道延迟与 FIR 状态；
- LFE 1217-sample 延迟与逐点 clipgain 斜坡；
- OAMD Q15 坐标转换；
- 基于目标布局 region 的等功率声像；
- OAMD 元数据应用（每块增益与 inactive、zone/elevation、size 核、snap、divergence、位置细化、priority 尺寸预算、ISF 矩阵、bed 分配）；
- 布局位置补偿与逐样本增益斜坡；
- float32 与 PCM24 输出量化；
- Rosella 与编译好的 `.jochrtf` 双耳渲染，含 exactly-once delay/phase 与 special LFE。

解码与渲染所用的公式见[数学说明](docs/math.md)。

## 兼容性说明

### `joc_clipgain` 的施加方式

`joc_clipgain` 在**分析端**并入分析标量：`标量 = 1/16 × g`（`g = 2^(x−4)·(1 + y/32)`，8 bit 字段
`x:3 | y:5`，值域 0.0625..15.75）。出口的 ×16 抵消 1/16，净效果是 `×g`，因此 clamp 处在增益**之后**
（`clamp(16·IQMF·g_interp)`）。标量按帧保存在两个槽里：本帧前 256 个采样从上一帧值线性过渡、之后为
常数，复位后的第一帧直接取新值。`g = 1` 时插值恒等，且 `clamp(x)·1 == clamp(x·1)`。LFE 走独立
通路，但使用同一个完整 `g` 的逐点斜坡，并乘在 1217-sample 延迟之前。

`object_delay_samples` 默认 **1473**，与既有实现的行为保持一致；它是可配置字段，改动它会
改变 OAMD/ADM 时间对齐。上游调查认为该值应为 0，本项目为保持逐字节等价暂不改默认值。

### `joc_dmx_config_idx` 0..4

覆盖全部五个档位：$N=5,7,7,5,7$，相位开关 $\sigma(k)=(k\in\{3,4\})$ 同时门控 Ls/Rs 的
$-j$ 旋转与 band-0 21-tap FIR。$N=7$ 档位必须解出 **8 通道**床，并按通道码映射：

| $k$ | 额外那一对 | 解码器给出 8 通道的顺序 | JOC 输入序（床下标） |
|---|---|---|---|
| 1 | Lb/Rb | FL FR FC LFE **BL BR** SL SR | 0 1 2 6 7 4 5 |
| 2 / 4 | Tfl/Tfr | FL FR FC LFE SL SR TFL TFR | 0 1 2 4 5 6 7 |

ffmpeg 侧只在 $N=5$ 档位传 `-ac 6`；$N=7$ 档位不传 `-ac`，保留解码器自身的布局
（`-ac 6` 会把 5.1.2/7.1 下混成 5.1，`-ac 8` 会把 5.1.2 重映射成 7.1，两者都会丢信息）。

**跨层一致性校验**（§11.2）：子流声明的全频段通道数必须等于 $N$，$joc\_ext\_config\_idx$ 非零
时报未映射变体，对象数超过 15 时截断到 15 并记入 diagnostics——这几项会改变 DSP 输入，不一致就
拒绝。额外那一对的**身份**只进元数据、不进 DSP（§10.1）：依赖子流 `chanmap` 与 $k$ 不一致、或
OAMD 声明的 bed 布局可以比核心子流实际编码的声道多（例如 9.1.6 bed 只编 5.1 核心），只有核心
声道在 bed 布局里找不到位置才记 diagnostics 并照常解码，床映射按 `chanmap` 的通道码走。
同一音频帧里出现多个含 ID11/ID14 的 EMDF 容器时，按规格 §8.2 + Table 56 的关联规则打分：
OAMD 与 JOC 载荷 groupid 相同且 ID14 能完整解析的优先，其次只看 ID14 能否解析；唯一最高分才用，
其余记入 diagnostics（`emdf/extra_container_ignored`）。额外那一对在 10 槽 ADM 骨架上的落位
（$k=1\to6/7$、$k\in\{2,4\}\to8/9$）会被记录，但床通道按现行契约不作为输出通道。

### 帧状态与 JOC 解码状态机

每个帧都带一个状态，非零状态按诊断报告。非零状态使该帧不能用于渲染，渲染器因此保持上一帧矩阵或输出
核心 PCM 直通，而不还原该帧的矩阵：2 不支持的下混配置，3 对象数截断到 15，4 未知参数带索引，
5 非法 `offset_ts`，6 字节对齐后存在尾随数据。文件任务对 `joc_dmx_config_idx` 超出 0..4 的
输入更早在元数据扫描阶段按不支持变体拒绝。状态 3 仍给出截断后的对象列表（供诊断），状态 2、4
在出错的头字段处停止解析。

渲染随后按帧级状态机运行：正常、保持、硬切、核心直通。正常渲染期间出现 JOC 参数不可用的帧时
进入保持，继续发布上一帧末的矩阵；连续六个不可用帧只把内部保持状态推进一档、仍保持；出现有效
帧后重新对齐序号。渲染器离开初始状态后序号回到 0、序号不匹配、或声明的核心声道数变化都
触发硬切：所有对象矩阵清零、每个对象的 previous 状态置零，核心声道数变化还清空 LFE 延迟环
但保留 clipgain 斜坡状态。首个有效帧之前，渲染器输出核心 PCM 直通。

音频帧内完全没有 JOC EMDF 时（例如编码器没写首帧元数据），该帧一律走直通路径：8 路核心平面
L、R、C、LFE、Ls、Rs、(Lrs, Rrs) 按 identity 写入输出槽 0..7，每路延迟 1217 采样，槽 8..15
保持零，8 平面历史照常推进。这类帧在统计里单独计数；不含 JOC 元数据（ID11 + ID14）的 E-AC-3
输入不在支持范围。保持或直通期间没有新参数时，分析标量与 LFE 斜坡沿用当前保持的 clipgain。

## 许可

MIT，见 [LICENSE](LICENSE)；第三方来源与专利边界见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
