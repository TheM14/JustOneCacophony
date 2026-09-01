# JustOneCacophony 原生核说明

[English](native.en.md) · [返回 README](../README.md)

## 1. 职责边界

`native/` 只承载状态密集、调用频繁的 DSP 与扬声器渲染核。EMDF/JOC/OAMD 高层解析、错误报告、ADM 组装和 CLI 保留在 Python 中。

Python 通过标准库 `ctypes` 调用 C ABI；原生核不使用 pybind11、Cython、FFTW、MKL 或 OpenMP。它是可选加速路径，不扩大项目所支持的码流范围。

主要文件：

```text
native/include/eac3joc_core.h       C ABI
native/src/eac3joc_core.cpp         JOC/QMF 对象重建
native/src/speaker_renderer.cpp     对象到扬声器渲染
native/src/qmf_tables.h             QMF 表
native/src/speaker_layouts.h        布局表
native/src/joc_huffman_tables.h     JOC Huffman 表
src/native_renderer.py              JOC ctypes 桥
src/speaker_native_renderer.py      扬声器 ctypes 桥
```

## 2. JOC 渲染 ABI

一个 opaque renderer 保存所有跨帧状态。主要调用为：

```c
int ejoc_renderer_process(
    ejoc_renderer_handle handle,
    const float* bed5_planar,      /* [5][1536] */
    const float* lfe,              /* [1536] or NULL */
    uint32_t object_mask,
    const uint8_t* n_bands,        /* [15] */
    const uint8_t* n_dpoints,      /* [15] */
    const uint8_t* slope_idx,      /* [15] */
    const uint8_t* offset_ts,      /* [15][2] */
    const double* dq,              /* [15][2][5][23] */
    double clipgain,
    float phase_new,
    float output_scale,
    float* output16_planar);       /* [16][1536] */
```

Dense JOC 的 Huffman 解码、差分还原和去量化先在 Python 中完成。Sparse JOC 不会被静默送入 dense 原生路径。

线程接口为：

```c
int ejoc_renderer_set_threads(ejoc_renderer_handle handle, uint32_t total_threads);
uint32_t ejoc_renderer_thread_count(ejoc_renderer_handle handle);
```

`total_threads` 包含调用线程。单个 renderer 实例必须顺序提交帧；实例内部可以按对象和 analysis channel 并行。

## 3. 跨帧状态

每个 JOC renderer 独立保存：

- analysis FIFO：`double[5][9][64]`；
- L/R/C analysis delay：`float[3][10][64]`；
- Ls/Rs QMF delay：`complex<double>[2][10][64]`；
- Ls/Rs band-0 FIR history：`complex<double>[2][20]`；
- 矩阵插值 previous：`double[15][5][64]`；
- inverse-QMF state：`double[15][640]`；
- LFE delay：`double[1217]`。

这些状态属于 renderer 实例，不能在无 checkpoint 的情况下任意分段或乱序处理。

## 4. FFT、QMF 与精度

原生核包含固定 64 点 radix-2 complex FFT：

- analysis QMF 使用 forward FFT 后除以 64；
- inverse QMF 使用固定重排、旋转和 640 项有效窗状态；
- 不调用外部 FFT 库。

JOC 路径的数值类型为：

- 核心 PCM 输入：float32；
- 矩阵、复 QMF、FFT、FIR 和跨帧状态：double；
- phase 与最终 gain：float32；
- 16 声道对象输出：float32。

## 5. 扬声器渲染 ABI

同一个共享库还导出对象到扬声器布局的渲染接口：

```c
uint32_t ejoc_speaker_layout_channel_count(uint32_t speaker_bitfield);

ejoc_speaker_renderer_handle
ejoc_speaker_renderer_create(uint32_t speaker_bitfield);

int ejoc_speaker_renderer_process(
    ejoc_speaker_renderer_handle handle,
    const float* objects16_interleaved,
    uint32_t sample_count,
    uint32_t metadata_count,
    const uint32_t* metadata_offsets,
    const uint32_t* ramp_durations,
    const uint16_t* positions_q15,
    const uint8_t* region_indices,
    const uint8_t* height_enabled,
    const double* object_gains,
    double* output_interleaved);
```

输入声道 0 为 LFE，1–15 为对象。每个 metadata entry 是一份对象状态快照。`sample_count` 必须是 32 的倍数；未完成的增益斜坡保存在 handle 中并跨调用继续。

扬声器路径使用 float32 对象输入、double 坐标/增益/累加与 interleaved double 输出；写 WAV 时才量化为 float32 或 PCM24。

支持的布局为：

```text
2.0  3.1  5.1  7.1  5.1.2  5.1.4  7.1.2  7.1.4  9.1.4  9.1.6
```

## 6. 构建

CMake 定义位于 `native/CMakeLists.txt`。从仓库根目录运行：

```powershell
cmake -S native -B build/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/lib"
cmake --build build/cmake --config Release
cmake --install build/cmake --config Release
```

平台运行库文件名：

```text
Windows  lib/eac3joc_core.dll
Linux    lib/libeac3joc_core.so
macOS    lib/libeac3joc_core.dylib
```

MSVC 配置使用静态 CRT。其他运行时依赖由平台和工具链决定，发布预构建库前应对产物独立检查。

仓库默认不附带原生二进制。预构建的 Release 运行库或自行构建的运行库均可直接放入 `lib/`。

## 7. 运行时查找与回退

查找顺序为：

1. 显式 `--native-library`；
2. `EAC3JOC_NATIVE_LIBRARY`；
3. `lib/` 下当前平台的标准文件名。

`--backend auto` 在加载失败时回退到 NumPy；`--backend python` 跳过原生探测。`--backend native` 当前也会打印失败原因后回退，这是现有 CLI 行为，不应理解为原生库已成功使用。

## 8. 实现边界

- 原生层只接收 Python 已解析的 dense JOC 数据。
- ABI 固定了 1536-sample JOC 帧、最多 15 个对象、最多 23 个参数带和最多 2 个数据点。
- 共享库与 Python 桥需要 ABI version 一致。
- 跨平台只约定 ABI 与数据类型，不保证 float64 结果逐位一致。
- `native/src/` 中的私有表头只服务于原生侧；当前仓库不包含重新生成这些头文件的脚本。

相关公式见[数学说明](math.md)。
