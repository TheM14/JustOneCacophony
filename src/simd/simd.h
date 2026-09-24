#pragma once

#include <cstddef>

// Runtime-dispatched SIMD kernels for the DSP core.
//
// Module layout (the ISA lives in the file name, never in a subdirectory):
//
//   simd.h                    this contract -- the only header a caller includes
//   cpu_probe.{h,cpp}         "can this CPU and OS run ISA X", no policy
//   dispatch.cpp              JOC_SIMD + the ladder + the kernel table
//   kernels_scalar.cpp        the reference implementation, Isa::scalar
//   kernels_intrin_avx2.cpp   /arch:AVX2     -mavx2    -- exactly the
//   kernels_intrin_avx512.cpp /arch:AVX512   -mavx512f -- `kernels_intrin_*`
//   kernels_intrin_neon.cpp   AArch64 default           -- units get a wider flag
//
// The scalar reference in kernels_scalar.cpp fixes both the data layout and the
// arithmetic: a vector path may only compute *independent* outputs in parallel
// lanes.  It may never reassociate an accumulation, contract a multiply into an
// FMA, or change the order in which one output's terms are summed, so every ISA
// path reproduces the scalar bits exactly and JOC_SIMD=scalar is a valid
// reference for any of them.  Kernel-level layout changes (AoS -> SoA inside a
// kernel, for instance) are fine as long as each individual output value is
// still produced by the same sequence of roundings.
//
// MSVC has no function-level ISA attribute, so each ISA lives in its own
// translation unit compiled with its own /arch (or -m) flag and the dispatcher
// picks one at run time: CPUID + XGETBV on x86-64, the architectural AArch64
// baseline (NEON/ASIMD needs no probing) elsewhere.
//
// The ISA units are linked into the same binaries as the baseline code, so they
// must stay free of objects with dynamic initialisation: a global constructor
// would execute ISA-specific instructions before the dispatcher has had a chance
// to look at the CPU.  Constant tables are fine -- they live in .rdata.
namespace joc::simd {

enum class Isa {
    scalar = 0,
    sse2,
    avx2,
    avx512,
    neon,
};

// One entry per dispatched kernel.
enum class Kernel {
    // Power-of-two radix-2 DIT butterfly cascade, in place, over `size` interleaved
    // (re, im) pairs already in bit-reversed order.  `twiddle` is the caller's own
    // factor table, read as interleaved pairs: stage s (length 2^(s+1)) takes its
    // 2^s factors starting at stage_begin[s], a *complex* index into `twiddle`.
    fft_butterflies = 0,
    // QMF synthesis basis application, see kSynthesis* below.
    qmf_synthesis_basis,
    // Hybrid analysis low join, see kHybrid* below.
    hybrid_low_join,
    // Hybrid-domain path rendering, see kPathBands below.
    render_hybrid_path,
    // Scalar-times-vector accumulate over interleaved complexes.
    complex_axpy,
    // QMF analysis polyphase accumulate, see kQmfAnalysisBands below.
    qmf_analysis_taps,
    // Element-wise complex product of two whole runs.
    complex_multiply,
    count,
};

// Both filterbanks apply the same map: for every output row and every one of the
// 64 bands, four ranks each accumulate 128 taps.
inline constexpr std::size_t kSynthesisBands = 64;
inline constexpr std::size_t kSynthesisRanks = 4;
inline constexpr std::size_t kSynthesisTaps = 128;

// ... and both derive the same 16 low hybrid bands from 78 terms: 3 parents x 2
// components x 13 lags.
inline constexpr std::size_t kHybridTerms = 78;
inline constexpr std::size_t kHybridOutputs = 32;
// Rows the callers stage at a time: enough to keep the kernel busy, small enough
// that the staged values stay in the first-level cache.
inline constexpr std::size_t kHybridJoinBlock = 32;

// Bands in one ear of a rendered path (the SOFA renderer's 77-band hybrid domain).
inline constexpr std::size_t kPathBands = 77;

// The cascade packs whole groups into a vector, so it needs at least one full
// packing: a size of 8 covers a 512-bit stage.  Smaller transforms are rare enough
// (and short enough) that the caller's portable loop is the right answer there.
inline constexpr std::size_t kMinVectorFftSize = 8;

// Bands in one QMF analysis row (the polyphase accumulator of both filterbanks).
inline constexpr std::size_t kQmfAnalysisBands = 64;

const char* isa_name(Isa isa) noexcept;

// Whether this CPU *and* the OS state it has to save can execute the ISA.  The
// answer never changes, so it is computed once.
bool isa_supported(Isa isa) noexcept;

// The ISA the dispatcher settled on, after applying a JOC_SIMD override.  This
// is the widest ISA considered, not necessarily the one that serves every
// kernel: active_isa() answers that per kernel.
Isa selected_isa() noexcept;

// Which ISA actually implements `kernel` after the per-kernel fallback.
Isa active_isa(Kernel kernel) noexcept;

// The dispatched kernels; a slot is never null.
struct Kernels {
    void (*fft_butterflies)(double* data, std::size_t size, const double* twiddle,
                            const std::size_t* stage_begin) noexcept = nullptr;

    // values: [rows][kSynthesisTaps] real/imaginary of the 64 bands of one row.
    // basis:  the same weights the caller holds, reordered rank-minor, so that
    //         (band, tap) addresses its kSynthesisRanks weights contiguously:
    //         basis[(band * kSynthesisTaps + tap) * kSynthesisRanks + rank].
    // out:    [rows][kSynthesisBands][kSynthesisRanks], rank-minor as well.
    //
    // Each of the four ranks is an independent dot product over the row, so the
    // four of them are what shares a vector; every lane keeps the tap order and
    // the two roundings of the caller's `sum += values[tap] * weight`.
    void (*qmf_synthesis_basis)(const double* values, const double* basis, double* out,
                                std::size_t rows) noexcept = nullptr;

    // Hybrid analysis low join.  Every output row accumulates kHybridTerms values
    // into kHybridOutputs outputs (16 bands x a real/imaginary pair, component
    // minor):
    //     out[row][output] = sum over term of values[row][term] * kernel[term][output]
    // values is [rows][kHybridTerms] in the caller's own term order and kernel is
    // the caller's taps regrouped to that same order, so that one term's 32
    // weights are contiguous.  Terms whose value is exactly zero are skipped, as
    // both callers do: adding a zero product to a lane can only leave it alone
    // (no lane's running sum can be a negative zero, since it starts at +0 and
    // sums without ever producing one).
    void (*hybrid_low_join)(const double* values, const double* kernel, double* out,
                            std::size_t rows) noexcept = nullptr;

    // One rendered path.  For each ear e and band b, with the ear's history sample
    // h and the path's transfer t:
    //     out[e][b].re += h.re * t[e][b].re * scale - h.im * t[e][b].im * scale
    //     out[e][b].im += h.re * t[e][b].im * scale + h.im * t[e][b].re * scale
    // `history0` and `history1` are the two ears' [kPathBands] interleaved complex
    // rows (they come from different places in the history ring); `transfer` and
    // `out` are [2][kPathBands] interleaved complexes.  The scale multiplies each
    // product separately, exactly as the caller writes it, so the bands -- which
    // are independent accumulations into independent outputs -- are what the lanes
    // carry.
    void (*render_hybrid_path)(double* out, const double* transfer, const double* history0,
                               const double* history1, double scale) noexcept = nullptr;

    // `count` interleaved complexes, accumulated in place:
    //     out[i] += field[i] * scale
    // Every element is an independent accumulation of one product, so the lanes
    // carry neighbouring elements and each one keeps the caller's multiply-then-add.
    void (*complex_axpy)(double* out, const double* field, double scale,
                         std::size_t count) noexcept = nullptr;

    // QMF analysis polyphase accumulate: `rows` rows of kQmfAnalysisBands bands,
    //     out[row][band] += source[row][band] * coefficients[band]
    // Neighbouring bands are neighbouring outputs, so they are what fills a vector;
    // each band accumulates its own product once, in the caller's order.
    void (*qmf_analysis_taps)(double* out, const double* source, const double* coefficients,
                              std::size_t rows) noexcept = nullptr;

    // `count` interleaved complexes, multiplied element by element:
    //     out[i] = left[i] * right[i]
    // with the caller's `re * re - im * im` and `re * im + im * re`, two roundings
    // per component.  Neighbouring complexes are independent products.
    void (*complex_multiply)(double* out, const double* left, const double* right,
                             std::size_t count) noexcept = nullptr;
};

const Kernels& kernels() noexcept;

// Convenience wrappers.
inline void fft_butterflies(double* data, std::size_t size, const double* twiddle,
                            const std::size_t* stage_begin) noexcept {
    kernels().fft_butterflies(data, size, twiddle, stage_begin);
}

inline void qmf_synthesis_basis(const double* values, const double* basis, double* out,
                                std::size_t rows) noexcept {
    kernels().qmf_synthesis_basis(values, basis, out, rows);
}

inline void hybrid_low_join(const double* values, const double* kernel, double* out,
                            std::size_t rows) noexcept {
    kernels().hybrid_low_join(values, kernel, out, rows);
}

inline void render_hybrid_path(double* out, const double* transfer, const double* history0,
                               const double* history1, double scale) noexcept {
    kernels().render_hybrid_path(out, transfer, history0, history1, scale);
}

inline void complex_axpy(double* out, const double* field, double scale,
                         std::size_t count) noexcept {
    kernels().complex_axpy(out, field, scale, count);
}

inline void qmf_analysis_taps(double* out, const double* source, const double* coefficients,
                              std::size_t rows) noexcept {
    kernels().qmf_analysis_taps(out, source, coefficients, rows);
}

inline void complex_multiply(double* out, const double* left, const double* right,
                             std::size_t count) noexcept {
    kernels().complex_multiply(out, left, right, count);
}

}  // namespace joc::simd
