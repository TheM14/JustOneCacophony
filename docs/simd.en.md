# SIMD and runtime dispatch

[中文](simd.md) · [Back to README](../README.en.md)

The heaviest loops in the binaural path (QMF analysis and synthesis, the hybrid
analysis low join, hybrid-domain path rendering, the ROOM FFT, the spherical
harmonic alignment) each have a runtime-dispatched vector implementation: one
binary carries several instruction-set variants, asks the CPU once at startup and
runs the widest one. **The output is byte-identical either way** — that is a hard
constraint, not a goal.

```text
JOC_SIMD=auto|scalar|sse2|avx2|avx512|neon      pin one tier (used for verification)
JOC_SIMD_LOG=1                                  report the ISA each kernel actually got
```

## Why the split has to happen per translation unit

MSVC has no function-level attribute like `__attribute__((target("avx2")))`: one
`.cpp` file gets one `/arch`. So every ISA is its own translation unit with its own
`/arch:AVX2` or `/arch:AVX512` (GCC/Clang: `-mavx2` / `-mavx512f`), and
`dispatch.cpp` fills the function table at run time. The baseline units — the
dispatcher itself, the CPU probe and the scalar reference — carry **no** `/arch` at
all and stay on the SSE2 that x86-64 guarantees.

A trap from the history of this tree: `JOC_ENABLE_AVX2` used to be global, so
turning it on put AVX2 instructions into the very code paths that exist for older
CPUs. It now only selects whether the AVX2 unit is compiled in.

## Directory layout

One flat directory, **the instruction set in the file name and never in a
subdirectory** — that is how FLAC does it (`lpc.c` sits next to
`lpc_intrin_sse2.c`, `lpc_intrin_avx2.c` and `lpc_intrin_neon.c`, with the CPU
probe in its own `cpu.c`).

```text
src/simd/
    simd.h                     the contract: Isa / Kernel / Kernels / dimensions
    cpu_probe.{h,cpp}          "can this machine run ISA X": CPUID+XGETBV / __builtin_cpu_supports / getauxval
    dispatch.cpp               policy: JOC_SIMD, the fallback ladder, the table, the log
    kernels_scalar.cpp         the reference (Isa::scalar; always built, always selectable)
    kernels_intrin_avx2.cpp    /arch:AVX2     -mavx2
    kernels_intrin_avx512.cpp  /arch:AVX512   -mavx512f
    kernels_intrin_neon.cpp    AArch64 default
```

The module lives at `src/simd/`, not `src/dsp/simd/`: `foundation/`, `hrtf/` and
`binaural/` all call into these kernels, so it is a cross-cutting layer rather than
a submodule of the DSP code (and `src/dsp/` held nothing else).

The header comment of `simd.h` carries the same module map; keep the two in sync
when the layout changes.

## Three rules

1. **Only `kernels_intrin_*.cpp` gets a wider flag.** `CMakeLists.txt` names those
   files explicitly with `set_source_files_properties`; every other target stays on
   the architecture's guaranteed ISA. A unit that goes wide without matching that
   name fails `devtools/vec/isa_audit.ps1`.
2. **No dynamic initialisation inside an ISA unit.** Those objects are linked into
   the same image as the baseline, so a global constructor would execute a wide
   instruction before the dispatcher has looked at the CPU. Constant tables are
   fine — they land in `.rdata`.
3. **Bit-exactness comes from the lane assignment, not from the ISA.** A lane may
   only carry mutually independent outputs; the rounding sequence of a single
   output, the separation of multiply and add (never an FMA) and the summation
   order all stay exactly as `kernels_scalar.cpp` wrote them. Layout changes that
   only reorder stored doubles (rank-minor basis tables, term-ordered tap tables,
   stage-contiguous twiddle tables, band-major ROOM planes) are allowed.

## How the choice is made

`dispatch.cpp` parses `JOC_SIMD` first (forcing a tier this build or this machine
does not have prints a diagnostic and falls back, rather than pretending and
crashing), then walks `avx512 → avx2 → sse2 → neon` and picks, per kernel, the
widest implementation that is both compiled into this binary **and** runnable
here, falling back to the scalar reference. An AVX-512 unit therefore costs
nothing on a CPU without AVX-512; it is simply never selected.

On x86 the probe requires CPUID *and* XGETBV to agree: CPUID says the silicon can
do it, XCR0 says the OS saves the registers it needs. Either one alone is not
enough — using AVX without OS state support corrupts other threads across a
context switch. AArch64 needs no probe; ASIMD is the architectural baseline.

`sse2` is a selectable tier with **no unit of its own**, on purpose: a 128-bit SSE2
register is the register a scalar double already occupies, SSE2 cannot widen
double-precision arithmetic, and hand-written SSE2 would only add moves. The tier
resolves to the baseline unit.

## Effect

30-second reference cases, one binary with only `JOC_SIMD` switched (DSP stage,
`t_render_dsp`):

| Case | `scalar` | `auto` | DSP speed-up | End-to-end wall clock |
|---|---|---|---|---|
| Binaural Rosella | 1.256 s | **0.640 s** | **1.96×** | 1.690 → **0.941 s** |
| Binaural SOFA | 1.560 s | **0.654 s** | **2.39×** | 1.859 → **0.849 s** |
| Speaker 5.1 / 9.1.6 / ADM | — | — | 1.00× | no regression (these kernels are not on those paths) |

The vectorised loops themselves gain more: synthesis basis 5.89×, 13-tap low join
4.50×, SOFA QMF synthesis 4.27×, 128-point FFT 2.24×. The whole pipeline stops
short of 8× because a good part of the time goes to parameter setup, straight
copies and file writing — none of which has independent work items — and because
SOFA's 33 M sin/cos calls per sample cannot be vectorised under a byte-exactness
contract.

## Verifying a change

```powershell
$env:JOC_SIMD_LOG='1'                                # per-kernel ISA on this machine
$env:JOC_SIMD='scalar'                               # force the reference: hashes must not move
pwsh -NoProfile -File devtools\vec\isa_audit.ps1     # disassemble every .obj: 0 unguarded wide instructions
pwsh -NoProfile -File devtools\vec\sha_matrix.ps1    # 5 tiers x 2 renders against the reference digests
cmd /c devtools\vec\build_kernel_probe.bat           # per-kernel byte digests (8 kernels)
```

## Adding an ISA

1. Write `kernels_intrin_<isa>.cpp`, implementing the slots you have and leaving the
   rest `nullptr` — the dispatcher falls back per kernel (that is how
   `qmf_synthesis_basis` is handled in the AVX-512 unit).
2. Add it to `JOC_SIMD_SOURCES` in `CMakeLists.txt` with `JOC_SIMD_HAVE_<ISA>=1` and
   its flag, and extend `Isa`, `isa_rank`, `isa_compiled`, `isa_supported` and the
   `JOC_SIMD` name table in `simd.h` / `dispatch.cpp`.
3. Verify: the kernel digests must match the scalar unit byte for byte, and the
   reference renders must keep their SHA-256 on every tier.
