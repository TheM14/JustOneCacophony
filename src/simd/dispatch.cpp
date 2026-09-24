// Kernel dispatch: CPU features in, kernel table out.
//
// This file holds policy only -- which ISA is asked for, which ladder is walked,
// which slot ends up with which implementation.  The question "can this machine
// run ISA X at all" belongs to cpu_probe.cpp, and the implementations themselves
// belong to the one ISA unit per ISA (kernels_scalar.cpp / kernels_intrin_*.cpp).
//
// JOC_SIMD=auto|scalar|sse2|avx2|avx512|neon.  A forced ISA that this build or
// this machine cannot provide is reported on stderr and then ignored -- forcing a
// path the CPU lacks would fault instead of verifying anything.
//
// This unit is compiled with the baseline ISA: it runs before anything else in
// any binary that links the kernels, so it must not execute an ISA-specific
// instruction itself.

#include "simd/simd.h"

#include "simd/cpu_probe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Which ISA units the build system put into this binary.  An ISA that was not
// compiled in has no entry point to reference at all.
#if !defined(JOC_SIMD_HAVE_SSE2)
#define JOC_SIMD_HAVE_SSE2 0
#endif
#if !defined(JOC_SIMD_HAVE_AVX2)
#define JOC_SIMD_HAVE_AVX2 0
#endif
#if !defined(JOC_SIMD_HAVE_AVX512)
#define JOC_SIMD_HAVE_AVX512 0
#endif
#if !defined(JOC_SIMD_HAVE_NEON)
#define JOC_SIMD_HAVE_NEON 0
#endif

namespace joc::simd {

// Implemented by the per-ISA units; referenced only when compiled in.  The file
// name carries the ISA (FLAC's `*_intrin_<isa>.c` convention), and the CMake list
// gives exactly those units their wider /arch or -m flag.
const Kernels& kernels_scalar() noexcept;
#if JOC_SIMD_HAVE_AVX2
const Kernels& kernels_intrin_avx2() noexcept;
#endif
#if JOC_SIMD_HAVE_AVX512
const Kernels& kernels_intrin_avx512() noexcept;
#endif
#if JOC_SIMD_HAVE_NEON
const Kernels& kernels_intrin_neon() noexcept;
#endif

namespace {

constexpr std::size_t kKernelCount = static_cast<std::size_t>(Kernel::count);

// The probe itself lives in cpu_probe.cpp; all this file needs from it is the
// cached answer, turned into the two questions the ladder asks: "was it compiled
// into this binary" (JOC_SIMD_HAVE_*) and "can this machine run it".

struct Resolution {
    Kernels kernels;
    Isa active[kKernelCount] = {};
    Isa selected = Isa::scalar;
    bool forced = false;
};

int isa_rank(Isa isa) noexcept {
    switch (isa) {
        case Isa::scalar: return 0;
        case Isa::sse2: return 1;
        case Isa::neon: return 1;
        case Isa::avx2: return 2;
        case Isa::avx512: return 3;
    }
    return 0;
}

bool isa_compiled(Isa isa) noexcept {
    switch (isa) {
        case Isa::scalar: return true;
        // The x86-64 baseline *is* SSE2: the baseline units are compiled for it
        // and there is no separate unit, because a 128-bit SSE2 register is
        // exactly the register a scalar double already uses -- SSE2 cannot widen
        // a double-precision operation, so "sse2" selects the reference kernels.
        case Isa::sse2: return JOC_SIMD_HAVE_SSE2 != 0;
        case Isa::avx2: return JOC_SIMD_HAVE_AVX2 != 0;
        case Isa::avx512: return JOC_SIMD_HAVE_AVX512 != 0;
        case Isa::neon: return JOC_SIMD_HAVE_NEON != 0;
    }
    return false;
}

const Kernels& kernels_of(Isa isa) noexcept {
    switch (isa) {
        case Isa::scalar:
        case Isa::sse2: return kernels_scalar();
#if JOC_SIMD_HAVE_AVX2
        case Isa::avx2: return kernels_intrin_avx2();
#endif
#if JOC_SIMD_HAVE_AVX512
        case Isa::avx512: return kernels_intrin_avx512();
#endif
#if JOC_SIMD_HAVE_NEON
        case Isa::neon: return kernels_intrin_neon();
#endif
        default: break;
    }
    return kernels_scalar();
}

bool parse_isa(const char* text, Isa* out) noexcept {
    struct Entry {
        const char* name;
        Isa isa;
    };
    static const Entry kEntries[] = {
        {"scalar", Isa::scalar}, {"sse2", Isa::sse2},           {"avx2", Isa::avx2},
        {"avx512", Isa::avx512}, {"neon", Isa::neon},
    };
    for (const Entry& entry : kEntries) {
        if (std::strcmp(text, entry.name) == 0) {
            *out = entry.isa;
            return true;
        }
    }
    return false;
}

Resolution resolve() noexcept {
    Resolution resolution;
    Isa best = Isa::scalar;
    bool forced = false;

    if (const char* requested = std::getenv("JOC_SIMD");
        requested != nullptr && requested[0] != '\0' && std::strcmp(requested, "auto") != 0) {
        Isa wanted = Isa::scalar;
        if (!parse_isa(requested, &wanted)) {
            std::fprintf(stderr,
                         "joc: JOC_SIMD=%s is not one of auto|scalar|sse2|avx2|avx512|neon; "
                         "using auto\n",
                         requested);
        } else if (!isa_compiled(wanted)) {
            std::fprintf(stderr, "joc: JOC_SIMD=%s is not part of this build; using auto\n",
                         requested);
        } else if (!isa_supported(wanted)) {
            std::fprintf(stderr, "joc: JOC_SIMD=%s is not supported by this CPU or OS; using auto\n",
                         requested);
        } else {
            best = wanted;
            forced = true;
        }
    }

    if (!forced) {
        // Widest runnable ISA, in descending order.
        static const Isa kLadder[] = {Isa::avx512, Isa::avx2, Isa::sse2, Isa::neon};
        for (const Isa candidate : kLadder) {
            if (isa_compiled(candidate) && isa_supported(candidate)) {
                best = candidate;
                break;
            }
        }
    }
    resolution.selected = best;
    resolution.forced = forced;

    // Per kernel: widest compiled + runnable implementation at or below the
    // selected ISA, falling back to the scalar reference, which is always there.
    static const Isa kCandidates[] = {Isa::avx512, Isa::avx2, Isa::sse2, Isa::neon, Isa::scalar};
    for (std::size_t index = 0u; index < kKernelCount; ++index) {
        const Kernel kernel = static_cast<Kernel>(index);
        resolution.active[index] = Isa::scalar;
        for (const Isa candidate : kCandidates) {
            if (isa_rank(candidate) > isa_rank(best) || !isa_compiled(candidate)
                || !isa_supported(candidate)) {
                continue;
            }
            const Kernels& set = kernels_of(candidate);
            bool resolved = false;
            switch (kernel) {
                case Kernel::fft_butterflies:
                    if (set.fft_butterflies != nullptr) {
                        resolution.kernels.fft_butterflies = set.fft_butterflies;
                        resolved = true;
                    }
                    break;
                case Kernel::qmf_synthesis_basis:
                    if (set.qmf_synthesis_basis != nullptr) {
                        resolution.kernels.qmf_synthesis_basis = set.qmf_synthesis_basis;
                        resolved = true;
                    }
                    break;
                case Kernel::hybrid_low_join:
                    if (set.hybrid_low_join != nullptr) {
                        resolution.kernels.hybrid_low_join = set.hybrid_low_join;
                        resolved = true;
                    }
                    break;
                case Kernel::render_hybrid_path:
                    if (set.render_hybrid_path != nullptr) {
                        resolution.kernels.render_hybrid_path = set.render_hybrid_path;
                        resolved = true;
                    }
                    break;
                case Kernel::complex_axpy:
                    if (set.complex_axpy != nullptr) {
                        resolution.kernels.complex_axpy = set.complex_axpy;
                        resolved = true;
                    }
                    break;
                case Kernel::qmf_analysis_taps:
                    if (set.qmf_analysis_taps != nullptr) {
                        resolution.kernels.qmf_analysis_taps = set.qmf_analysis_taps;
                        resolved = true;
                    }
                    break;
                case Kernel::complex_multiply:
                    if (set.complex_multiply != nullptr) {
                        resolution.kernels.complex_multiply = set.complex_multiply;
                        resolved = true;
                    }
                    break;
                case Kernel::count: break;
            }
            if (resolved) {
                resolution.active[index] = candidate;
                break;
            }
        }
    }

    if (std::getenv("JOC_SIMD_LOG") != nullptr) {
        std::fprintf(stderr, "joc: simd selected=%s forced=%d\n", isa_name(resolution.selected),
                     resolution.forced ? 1 : 0);
        for (std::size_t index = 0u; index < kKernelCount; ++index) {
            std::fprintf(stderr, "joc: simd kernel[%u]=%s\n", static_cast<unsigned>(index),
                         isa_name(resolution.active[index]));
        }
    }
    return resolution;
}

const Resolution& resolution() noexcept {
    static const Resolution resolved = resolve();
    return resolved;
}

}  // namespace

const char* isa_name(Isa isa) noexcept {
    switch (isa) {
        case Isa::scalar: return "scalar";
        case Isa::sse2: return "sse2";
        case Isa::avx2: return "avx2";
        case Isa::avx512: return "avx512";
        case Isa::neon: return "neon";
    }
    return "?";
}

bool isa_supported(Isa isa) noexcept {
    const CpuFeatures& cpu = cpu_features();
    switch (isa) {
        case Isa::scalar:
            return true;
        case Isa::sse2:
            return cpu.sse2;
        case Isa::avx2:
            return cpu.avx2;
        case Isa::avx512:
            return cpu.avx512;
        case Isa::neon:
            return cpu.neon;
    }
    return false;
}

Isa selected_isa() noexcept { return resolution().selected; }

Isa active_isa(Kernel kernel) noexcept {
    const std::size_t index = static_cast<std::size_t>(kernel);
    if (index >= kKernelCount) {
        return Isa::scalar;
    }
    return resolution().active[index];
}

const Kernels& kernels() noexcept { return resolution().kernels; }

}  // namespace joc::simd
