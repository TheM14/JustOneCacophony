// CPU feature probe; see cpu_probe.h.
//
// Nothing here is ISA-specific code: the x86 path uses CPUID / XGETBV (MSVC) or
// the compiler's own runtime probe (GCC/Clang, so no inline assembly), and the
// AArch64 path reads the aux vector.  That is what lets this unit stay baseline
// and run before the dispatcher has decided anything.

#include "simd/cpu_probe.h"

#if defined(_M_X64)
#include <immintrin.h>
#include <intrin.h>
#elif defined(__x86_64__)
#include <cpuid.h>
#endif

#if defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <sys/auxv.h>
#endif

namespace joc::simd {
namespace {

// ------------------------------------------------------------------- x86-64 --
#if defined(_M_X64) || defined(__x86_64__)

#if defined(_M_X64)

// CPUID tells us what the silicon can do; XCR0 tells us whether the OS saves the
// state the wider registers need.  Both have to agree, or the first AVX
// instruction after a context switch corrupts another thread's registers.
CpuFeatures probe_x86() noexcept {
    CpuFeatures features;
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    __cpuid(regs, 1);
    features.sse2 = (regs[3] & (1 << 26)) != 0;
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || max_leaf < 7) {
        return features;
    }
    const unsigned long long xcr0 = _xgetbv(0);
    const bool ymm_state = (xcr0 & 0x06ull) == 0x06ull;  // XMM + YMM saved
    const bool zmm_state = (xcr0 & 0xe6ull) == 0xe6ull;  // + opmask / ZMM / hi16
    if (!ymm_state) {
        return features;
    }
    __cpuidex(regs, 7, 0);
    features.avx2 = (regs[1] & (1 << 5)) != 0;
    features.avx512 = zmm_state && (regs[1] & (1 << 16)) != 0;  // AVX512F
    return features;
}

#else  // GCC/Clang on x86-64

// The compiler runtime performs the same CPUID + XGETBV probe (libgcc's cpuinfo
// checks XCR0 before it reports AVX), which keeps this file free of inline
// assembly.
CpuFeatures probe_x86() noexcept {
    CpuFeatures features;
    features.sse2 = __builtin_cpu_supports("sse2") != 0;
    features.avx2 = __builtin_cpu_supports("avx2") != 0;
    features.avx512 = __builtin_cpu_supports("avx512f") != 0;
    return features;
}

#endif

CpuFeatures probe() noexcept { return probe_x86(); }

// ----------------------------------------------------------------- AArch64 --
#elif defined(__aarch64__) || defined(_M_ARM64)

CpuFeatures probe() noexcept {
    CpuFeatures features;
#if defined(__linux__)
    // ASIMD is architectural for AArch64; the aux vector only confirms it.
    features.neon = (getauxval(AT_HWCAP) & (1u << 1)) != 0u;
#else
    features.neon = true;
#endif
    return features;
}

// ------------------------------------------------------------ anything else --
#else

CpuFeatures probe() noexcept { return CpuFeatures{}; }

#endif

}  // namespace

const CpuFeatures& cpu_features() noexcept {
    static const CpuFeatures features = probe();
    return features;
}

}  // namespace joc::simd
