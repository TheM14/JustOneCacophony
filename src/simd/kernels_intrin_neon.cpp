// AArch64 NEON (ASIMD) implementation of the dispatched kernels.
//
// CMake gives exactly the `kernels_intrin_*.cpp` units their ISA flag, so no
// baseline unit can inherit one by accident.
//
// Unlike the x86 units this one needs no feature probe -- ASIMD is architectural
// for AArch64 -- but it is still only reached through the dispatcher, so
// JOC_SIMD=scalar, sse2 or neon all stay available on the same binary.
//
// A NEON register holds two doubles, so a vector carries two lanes where an AVX2
// vector carries four.  The lanes are the same independent work items as in the
// AVX2 unit, and each lane performs exactly the scalar reference's operations in
// the scalar reference's order: separate multiplies and adds, never a fused
// multiply-add, and never a reassociated sum.
//
// Nothing in this file may have dynamic initialisation (it is linked into the
// same image as the baseline code and would run before the dispatcher).
//
// NOTE: the unit tests do not exercise this unit.  It is built by the AArch64 CI
// job, and its bit-exactness is argued from the shared kernel contract (simd.h)
// rather than measured here.

#include "simd/simd.h"

#include <arm_neon.h>

#include <cstddef>

#include "foundation/fft.h"

namespace joc::simd {
namespace {

constexpr std::size_t kFftSize = dsp::kQmfFftSize;
static_assert(kFftSize == 128u, "the QMF transform is the 128-point case");

// ------------------------------------------------------------ radix-2 FFT ----
//
// Two butterflies per iteration, held as separate real/imaginary vectors: vld2q
// loads two interleaved complexes and deinterleaves them, vst2q puts them back.
// `sum = even + odd * w` and `difference = even - odd * w` are then two add/sub
// pairs, and the complex product is four multiplies and two add/subs, which is
// what the scalar kernel writes.
inline float64x2x2_t complex_mul(float64x2_t odd_real, float64x2_t odd_imag,
                                 float64x2_t weight_real, float64x2_t weight_imag) {
    float64x2x2_t result;
    result.val[0] = vsubq_f64(vmulq_f64(odd_real, weight_real), vmulq_f64(odd_imag, weight_imag));
    result.val[1] = vaddq_f64(vmulq_f64(odd_real, weight_imag), vmulq_f64(odd_imag, weight_real));
    return result;
}

// Stages 0 and 1 hold fewer than two complexes per group, so their butterflies are
// packed from neighbouring groups; the factor depends only on the offset inside a
// group, so the two lanes share it.
void fft_butterflies_neon(double* buffer, std::size_t size, const double* twiddle,
                          const std::size_t* stage_begin) noexcept {
    // Stage 0: length 2, one butterfly per group.  The two lanes take the two
    // halves of two neighbouring groups, which the transposes select.
    {
        const double* table = twiddle + 2u * stage_begin[0];
        const float64x2_t weight_real = vdupq_n_f64(table[0]);
        const float64x2_t weight_imag = vdupq_n_f64(table[1]);
        for (std::size_t index = 0u; index < 2u * size; index += 8u) {
            const float64x2_t first = vld1q_f64(buffer + index);
            const float64x2_t second = vld1q_f64(buffer + index + 2u);
            const float64x2_t third = vld1q_f64(buffer + index + 4u);
            const float64x2_t fourth = vld1q_f64(buffer + index + 6u);
            const float64x2_t even_real = vtrn1q_f64(first, third);
            const float64x2_t even_imag = vtrn2q_f64(first, third);
            const float64x2_t odd_real = vtrn1q_f64(second, fourth);
            const float64x2_t odd_imag = vtrn2q_f64(second, fourth);
            const float64x2x2_t rotated =
                complex_mul(odd_real, odd_imag, weight_real, weight_imag);
            const float64x2_t sum_real = vaddq_f64(even_real, rotated.val[0]);
            const float64x2_t sum_imag = vaddq_f64(even_imag, rotated.val[1]);
            const float64x2_t difference_real = vsubq_f64(even_real, rotated.val[0]);
            const float64x2_t difference_imag = vsubq_f64(even_imag, rotated.val[1]);
            // sum.* are the two groups' even-position results and difference.* their
            // odd-position ones; a transpose of the two component vectors puts each
            // complex back together, in the order the groups appear.
            vst1q_f64(buffer + index, vtrn1q_f64(sum_real, sum_imag));
            vst1q_f64(buffer + index + 2u, vtrn1q_f64(difference_real, difference_imag));
            vst1q_f64(buffer + index + 4u, vtrn2q_f64(sum_real, sum_imag));
            vst1q_f64(buffer + index + 6u, vtrn2q_f64(difference_real, difference_imag));
        }
    }

    std::size_t stage = 1u;
    for (std::size_t half = 2u; half < size; half <<= 1u, ++stage) {
        const std::size_t length = half << 1u;
        const double* table = twiddle + 2u * stage_begin[stage];
        for (std::size_t start = 0u; start < size; start += length) {
            for (std::size_t offset = 0u; offset < half; offset += 2u) {
                double* even_values = buffer + (start + offset) * 2u;
                double* odd_values = even_values + half * 2u;
                const float64x2x2_t even = vld2q_f64(even_values);
                const float64x2x2_t odd = vld2q_f64(odd_values);
                const float64x2x2_t weight = vld2q_f64(table + offset * 2u);
                const float64x2x2_t rotated =
                    complex_mul(odd.val[0], odd.val[1], weight.val[0], weight.val[1]);
                float64x2x2_t sum;
                float64x2x2_t difference;
                sum.val[0] = vaddq_f64(even.val[0], rotated.val[0]);
                sum.val[1] = vaddq_f64(even.val[1], rotated.val[1]);
                difference.val[0] = vsubq_f64(even.val[0], rotated.val[0]);
                difference.val[1] = vsubq_f64(even.val[1], rotated.val[1]);
                vst2q_f64(even_values, sum);
                vst2q_f64(odd_values, difference);
            }
        }
    }
}

// --------------------------------------------------- QMF synthesis basis -----
//
// The four ranks of a band are four independent dot products over the same row, so
// two registers carry four lanes.  `Rows` rows run side by side because a band's
// chain is 128 dependent adds: the cure is more independent chains, not a shorter
// chain.
template <int Rows>
void synthesize_rows_neon(const double* values, const double* basis, double* out) noexcept {
    constexpr std::size_t kRanks = kSynthesisRanks;
    constexpr std::size_t kTaps = kSynthesisTaps;
    constexpr std::size_t kRowOut = kSynthesisBands * kRanks;
    for (std::size_t band = 0u; band < kSynthesisBands; ++band) {
        const double* table = basis + band * kTaps * kRanks;
        float64x2_t low[Rows];
        float64x2_t high[Rows];
        for (int row = 0; row < Rows; ++row) {
            low[row] = vdupq_n_f64(0.0);
            high[row] = vdupq_n_f64(0.0);
        }
        for (std::size_t tap = 0u; tap < kTaps; ++tap) {
            const float64x2_t weight_low = vld1q_f64(table + tap * kRanks);
            const float64x2_t weight_high = vld1q_f64(table + tap * kRanks + 2u);
            for (int row = 0; row < Rows; ++row) {
                const float64x2_t factor = vdupq_n_f64(values[row * kTaps + tap]);
                low[row] = vaddq_f64(low[row], vmulq_f64(factor, weight_low));
                high[row] = vaddq_f64(high[row], vmulq_f64(factor, weight_high));
            }
        }
        for (int row = 0; row < Rows; ++row) {
            vst1q_f64(out + row * kRowOut + band * kRanks, low[row]);
            vst1q_f64(out + row * kRowOut + band * kRanks + 2u, high[row]);
        }
    }
}

void qmf_synthesis_basis_neon(const double* values, const double* basis, double* out,
                              std::size_t rows) noexcept {
    constexpr std::size_t kTaps = kSynthesisTaps;
    constexpr std::size_t kRowOut = kSynthesisBands * kSynthesisRanks;
    std::size_t row = 0u;
    for (; row + 4u <= rows; row += 4u) {
        synthesize_rows_neon<4>(values + row * kTaps, basis, out + row * kRowOut);
    }
    for (; row < rows; ++row) {
        synthesize_rows_neon<1>(values + row * kTaps, basis, out + row * kRowOut);
    }
}

// ------------------------------------------------------ Hybrid low join -----
//
// One term against all 32 outputs, two outputs per register: the outputs are
// independent accumulations over the same terms, which is what the lanes carry.
void hybrid_low_join_neon(const double* values, const double* kernel, double* out,
                          std::size_t rows) noexcept {
    constexpr std::size_t kTerms = kHybridTerms;
    constexpr std::size_t kOutputs = kHybridOutputs;
    static_assert(kOutputs == 32u, "sixteen 128-bit accumulators cover one row");
    for (std::size_t row = 0u; row < rows; ++row) {
        const double* source = values + row * kTerms;
        double* destination = out + row * kOutputs;
        float64x2_t partial[16];
        for (int chunk = 0; chunk < 16; ++chunk) {
            partial[chunk] = vdupq_n_f64(0.0);
        }
        for (std::size_t term = 0u; term < kTerms; ++term) {
            const double value = source[term];
            if (value == 0.0) {
                continue;
            }
            const double* weights = kernel + term * kOutputs;
            const float64x2_t factor = vdupq_n_f64(value);
            for (int chunk = 0; chunk < 16; ++chunk) {
                partial[chunk] = vaddq_f64(
                    partial[chunk], vmulq_f64(factor, vld1q_f64(weights + chunk * 2)));
            }
        }
        for (int chunk = 0; chunk < 16; ++chunk) {
            vst1q_f64(destination + chunk * 2, partial[chunk]);
        }
    }
}

// --------------------------------------------------- Rendered hybrid path ---
//
// Two bands per iteration, real and imaginary held apart and reloaded from memory
// in the caller's own layout: `out += h * t * scale` for the real part and
// `out += h * swapped(t) * scale` for the imaginary one, with the scale applied to
// each product separately.
void render_hybrid_path_neon(double* out, const double* transfer, const double* history0,
                             const double* history1, double scale) noexcept {
    const float64x2_t factor = vdupq_n_f64(scale);
    const double* histories[2] = {history0, history1};
    for (std::size_t ear = 0u; ear < 2u; ++ear) {
        const double* source = histories[ear];
        const double* taps = transfer + ear * kPathBands * 2u;
        double* destination = out + ear * kPathBands * 2u;
        std::size_t band = 0u;
        for (; band + 2u <= kPathBands; band += 2u) {
            const float64x2x2_t history = vld2q_f64(source + band * 2u);
            const float64x2x2_t tap = vld2q_f64(taps + band * 2u);
            float64x2x2_t output = vld2q_f64(destination + band * 2u);
            // The caller scales each product on its own before combining, so the
            // scale is a separate multiply of every product -- not a multiply of the
            // difference and the sum, which would round differently.
            const float64x2_t real = vsubq_f64(
                vmulq_f64(vmulq_f64(history.val[0], tap.val[0]), factor),
                vmulq_f64(vmulq_f64(history.val[1], tap.val[1]), factor));
            const float64x2_t imaginary = vaddq_f64(
                vmulq_f64(vmulq_f64(history.val[0], tap.val[1]), factor),
                vmulq_f64(vmulq_f64(history.val[1], tap.val[0]), factor));
            output.val[0] = vaddq_f64(output.val[0], real);
            output.val[1] = vaddq_f64(output.val[1], imaginary);
            vst2q_f64(destination + band * 2u, output);
        }
        for (; band < kPathBands; ++band) {
            const double source_real = source[band * 2u];
            const double source_imag = source[band * 2u + 1u];
            const double tap_real = taps[band * 2u];
            const double tap_imag = taps[band * 2u + 1u];
            destination[band * 2u] +=
                source_real * tap_real * scale - source_imag * tap_imag * scale;
            destination[band * 2u + 1u] +=
                source_real * tap_imag * scale + source_imag * tap_real * scale;
        }
    }
}

// Scalar times a run of interleaved complexes, two complexes per iteration.
void complex_axpy_neon(double* out, const double* field, double scale,
                       std::size_t count) noexcept {
    const float64x2_t factor = vdupq_n_f64(scale);
    std::size_t index = 0u;
    for (; index + 2u <= count; index += 2u) {
        const float64x2_t values = vld1q_f64(field + index * 2u);
        const float64x2_t accumulated = vld1q_f64(out + index * 2u);
        vst1q_f64(out + index * 2u, vaddq_f64(accumulated, vmulq_f64(values, factor)));
    }
    for (; index < count; ++index) {
        out[index * 2u] += field[index * 2u] * scale;
        out[index * 2u + 1u] += field[index * 2u + 1u] * scale;
    }
}

// QMF analysis polyphase accumulate, two bands per iteration.
void qmf_analysis_taps_neon(double* out, const double* source, const double* coefficients,
                            std::size_t rows) noexcept {
    for (std::size_t row = 0u; row < rows; ++row) {
        double* destination = out + row * kQmfAnalysisBands;
        const double* values = source + row * kQmfAnalysisBands;
        for (std::size_t band = 0u; band < kQmfAnalysisBands; band += 2u) {
            vst1q_f64(destination + band,
                      vaddq_f64(vld1q_f64(destination + band),
                                vmulq_f64(vld1q_f64(values + band),
                                          vld1q_f64(coefficients + band))));
        }
    }
}

// Element-wise complex product of two runs, two complexes per iteration.
void complex_multiply_neon(double* out, const double* left, const double* right,
                           std::size_t count) noexcept {
    std::size_t index = 0u;
    for (; index + 2u <= count; index += 2u) {
        const float64x2x2_t first = vld2q_f64(left + index * 2u);
        const float64x2x2_t second = vld2q_f64(right + index * 2u);
        float64x2x2_t result;
        result.val[0] = vsubq_f64(vmulq_f64(first.val[0], second.val[0]),
                                  vmulq_f64(first.val[1], second.val[1]));
        result.val[1] = vaddq_f64(vmulq_f64(first.val[0], second.val[1]),
                                  vmulq_f64(first.val[1], second.val[0]));
        vst2q_f64(out + index * 2u, result);
    }
    for (; index < count; ++index) {
        const std::size_t offset = index * 2u;
        const double left_real = left[offset];
        const double left_imag = left[offset + 1u];
        const double right_real = right[offset];
        const double right_imag = right[offset + 1u];
        out[offset] = left_real * right_real - left_imag * right_imag;
        out[offset + 1u] = left_real * right_imag + left_imag * right_real;
    }
}

const Kernels kNeon{
    &fft_butterflies_neon,
    &qmf_synthesis_basis_neon,
    &hybrid_low_join_neon,
    &render_hybrid_path_neon,
    &complex_axpy_neon,
    &qmf_analysis_taps_neon,
    &complex_multiply_neon,
};

}  // namespace

const Kernels& kernels_intrin_neon() noexcept { return kNeon; }

}  // namespace joc::simd
