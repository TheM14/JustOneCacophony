# SIMD 与运行时派发

[English](simd.en.md) · [返回 README](../README.md)

双耳通路里最重的那几段循环（QMF 分析/合成、混合分析的低频拼接、混合域路径渲染、
ROOM 的 FFT、球谐对齐）都有一份运行时分派的向量实现：同一份二进制里装多套 ISA 代码，
启动时问一次 CPU，然后选最宽的那套跑。**输出逐字节不变**——这是硬约束，不是目标。

```text
JOC_SIMD=auto|scalar|sse2|avx2|avx512|neon      强制某一层（验收用）
JOC_SIMD_LOG=1                                  打印每个 kernel 实际生效的 ISA
```

## 为什么必须"按编译单元分 ISA"

MSVC 没有 `__attribute__((target("avx2")))` 这类函数级多版本能力，一个 .cpp 只能有
一个 `/arch`。所以每个 ISA 一个编译单元，各自带自己的 `/arch:AVX2` / `/arch:AVX512`
（GCC/Clang 是 `-mavx2` / `-mavx512f`），由 `dispatch.cpp` 在运行时填函数表。
基线单元（含派发器本身、CPU 探测、标量参考实现）**不带任何 `/arch`**，它们只使用
x86-64 架构保证的 SSE2。

历史坑：早先的 `JOC_ENABLE_AVX2` 是**全局**的，一旦打开，连"给老 CPU 用"的基线路径
都带 AVX2 指令。现在这个选项只决定是否把 AVX2 单元编进二进制。

## 目录布局

一个扁平目录，**ISA 写在文件名里，不写进子目录**——这是 FLAC 的做法
（`src/libFLAC/lpc.c` 旁边就是 `lpc_intrin_sse2.c` / `lpc_intrin_avx2.c` /
`lpc_intrin_neon.c`，CPU 探测单独放在 `cpu.c`）。

```text
src/simd/
    simd.h                     唯一契约头：Isa / Kernel / Kernels / 维度常量
    cpu_probe.{h,cpp}          "这台机器能不能跑 ISA X"：CPUID+XGETBV / __builtin_cpu_supports / getauxval
    dispatch.cpp               策略：JOC_SIMD 解析、回退阶梯、填函数表、日志
    kernels_scalar.cpp         参考实现（Isa::scalar，永远编译、永远可选中）
    kernels_intrin_avx2.cpp    /arch:AVX2     -mavx2
    kernels_intrin_avx512.cpp  /arch:AVX512   -mavx512f
    kernels_intrin_neon.cpp    AArch64 默认  -march=armv8-a+simd
```

`simd.h` 的头注释里有一份同样的模块地图，改布局时两处一起改。

模块放在 `src/simd/` 而不是 `src/dsp/simd/`：这些 kernel 被 `foundation/`、`hrtf/`、
`binaural/` 三个模块共用，是横切的一层，不是 DSP 的子模块（更何况 `src/dsp/` 里除了
`simd/` 空无一物）。

## 三条规则

1. **只有 `kernels_intrin_*.cpp` 拿更宽的编译开关。** `CMakeLists.txt` 用
   `set_source_files_properties` 逐个点名，其余目标一律留在架构保证的 ISA 上。
   任何不属于这个命名却带了宽指令的单元都会被 `devtools/vec/isa_audit.ps1` 判失败。
2. **ISA 单元里不许有动态初始化。** 它们和基线代码链进同一个镜像，全局构造函数会在
   派发器看 CPU 之前就跑宽指令。常量表没问题（落在 `.rdata`）。
3. **逐位一致靠的是 lane 的划分，不是 ISA。** lane 里只能放**互相独立**的输出；单个输出
   的舍入序列、乘加分离（绝不用 FMA）、求和顺序都保持 `kernels_scalar.cpp` 原样。
   只改变 double **存放顺序**的布局改造（基函数表转秩小序、抽头表按项序、蝶形因子表
   按级连续化、ROOM 谱平面改频带主序）是允许的。

## 运行时怎么选

`dispatch.cpp` 先解析 `JOC_SIMD`（强制一个本机不支持的层会打印诊断并回退，而不是假装
选中然后崩），再走阶梯 `avx512 → avx2 → sse2 → neon`，每个 kernel 单独挑"已编进本
二进制 **且** 本机可跑"的最宽实现，挑不到就落到标量参考实现。所以 AVX-512 单元在
不支持它的 CPU 上只是不被选中，不影响启动。

x86 的探测要 CPUID 与 XGETBV **同时**成立：CPUID 说明硅片有这个能力，XCR0 说明操作
系统会保存对应寄存器状态，缺一个就不能用（否则上下文切换会踩坏别的线程）。AArch64
不需要探测，ASIMD 是架构基线。

`sse2` 是一个有意保留的档位但**没有单独的单元**：128 位 SSE2 寄存器就是标量 double
已经在用的寄存器，SSE2 加宽不了双精度运算，手写只会多出搬运指令，所以它选中的是基线
单元。

## 效果

30 s 参考用例，同一二进制只切 `JOC_SIMD`（DSP 阶段 `t_render_dsp`）：

| 用例 | `scalar` | `auto` | DSP 加速 | 端到端墙钟 |
|---|---|---|---|---|
| 双耳 Rosella | 1.256 s | **0.640 s** | **1.96×** | 1.690 → **0.941 s** |
| 双耳 SOFA | 1.560 s | **0.654 s** | **2.39×** | 1.859 → **0.849 s** |
| 扬声器 5.1 / 9.1.6 / ADM | — | — | 1.00× | 0 回归（不走这些 kernel） |

单看被向量化的循环，红利更大：合成基函数 5.89×、13 抽头低频拼接 4.50×、
SOFA QMF 合成 4.27×、128 点 FFT 2.24×。整条流水线到不了 8×，是因为相当一部分时间在
参数设置、直通拷贝、写盘这些没有独立工作项的代码上，以及 SOFA 每次采样的 33 M 次
sin/cos 按逐位契约不能向量化。

## 验证

```powershell
$env:JOC_SIMD_LOG='1'                                # 本机每个 kernel 实际选中的 ISA
$env:JOC_SIMD='scalar'                               # 强制参考路径：哈希必须一动不动
pwsh -NoProfile -File devtools\vec\isa_audit.ps1     # 反汇编全部 .obj：0 个无守卫的宽指令
pwsh -NoProfile -File devtools\vec\sha_matrix.ps1    # 5 档 × 2 渲染，逐字节比对参考摘要
cmd /c devtools\vec\build_kernel_probe.bat           # kernel 级逐字节摘要（8 个 kernel）
```

## 加一个新的 ISA

1. 写 `kernels_intrin_<isa>.cpp`，实现能实现的槽位，其余留 `nullptr`——派发器会逐
   kernel 回退（AVX-512 单元里的 `qmf_synthesis_basis` 就是这么处理的）。
2. 在 `CMakeLists.txt` 里加进 `JOC_SIMD_SOURCES`、`JOC_SIMD_HAVE_<ISA>=1` 和它的编译
   开关，并在 `simd.h` / `dispatch.cpp` 里补上 `Isa`、`isa_rank`、`isa_compiled`、
   `isa_supported` 与 `JOC_SIMD` 名字表。
3. 验证：kernel 摘要必须与标量逐字节相同，参考渲染在每个档位上的 SHA-256 都不能变。
