#pragma once

// "Can this CPU and this operating system actually execute ISA X?"
//
// Its own unit and free of policy: it reads CPUID / XGETBV / HWCAP, caches the
// answer, and knows nothing about JOC_SIMD, the kernel table or which
// implementation is preferred.  This is the one place that can fault the process
// if it is wrong, so it stays small enough to review on its own.
//
// Compiled with the baseline ISA like every other non-intrinsic unit (see
// CMakeLists.txt): it runs before any ISA unit is reached, so it must not execute
// a wide instruction itself.

namespace joc::simd {

// One flag per ISA the dispatcher can ask about.  A flag means "usable here":
// both the silicon and the OS state-management agree, not merely that CPUID
// advertises the feature.
struct CpuFeatures {
    bool sse2 = false;
    bool avx2 = false;
    bool avx512 = false;
    bool neon = false;
};

// Probed once, on first use.  Never throws and never reads the environment (the
// JOC_SIMD override is the dispatcher's business).
const CpuFeatures& cpu_features() noexcept;

}  // namespace joc::simd
