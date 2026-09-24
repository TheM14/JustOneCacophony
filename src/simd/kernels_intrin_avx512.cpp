// AVX-512F implementation of the dispatched kernels.  Compiled with /arch:AVX512
// (or -mavx512f) and only ever called after the dispatcher has confirmed AVX-512F
// *and* that the OS saves the opmask/ZMM state.
//
// CMake gives exactly the `kernels_intrin_*.cpp` units a wider flag, so no
// baseline unit can inherit one by accident.
//
// Nothing in this file may have dynamic initialisation: it is linked into the
// same image as the baseline code and would run before the dispatcher.

#include "simd/simd.h"

#include <immintrin.h>

#include <cstddef>

namespace joc::simd {
namespace {

// Four interleaved complexes per vector: a 512-bit register is four 128-bit
// lanes, each holding one (re, im) pair, so the AVX2 argument applies one lane
// wider -- movedup/permute broadcast the twiddle components inside every lane.
// AVX-512 never widened VADDSUBPD to 512 bits, so the real lanes take the
// difference and the imaginary ones the sum through merge masking: that is still
// one exact subtract and one exact add per output, with the same two roundings
// per component and no reassociation.
inline __m512d mul_quad(__m512d odd, __m512d twiddle) noexcept {
    const __m512d real = _mm512_mul_pd(odd, _mm512_movedup_pd(twiddle));
    const __m512d cross =
        _mm512_mul_pd(_mm512_permute_pd(odd, 0x55), _mm512_permute_pd(twiddle, 0xff));
    const __m512d upper = _mm512_mask_add_pd(real, 0xaa, real, cross);
    return _mm512_mask_sub_pd(upper, 0x55, real, cross);
}

// Stages 0 and 1 hold fewer than four complexes in a group, so their butterflies
// are packed from neighbouring groups instead; the factor depends only on the
// offset inside a group, so one broadcast serves every packed lane.
//
// VSHUFF64X2 takes a two-bit selector per output lane, but its first two output
// lanes can only name lanes of its first operand and its last two only lanes of
// its second (verified against the instruction).  These are the immediates the
// packings need, written as the four lane selections they spell out.
constexpr int kHalvesEven = 0x88;  // (low.l0, low.l2, high.l0, high.l2) = x0,x2,x4,x6
constexpr int kHalvesOdd = 0xdd;   // (low.l1, low.l3, high.l1, high.l3) = x1,x3,x5,x7
constexpr int kGroupsEven = 0x44;  // (low.l0, low.l1, high.l0, high.l1)
constexpr int kGroupsOdd = 0xee;   // (low.l2, low.l3, high.l2, high.l3)
constexpr int kSwapMiddle = 0xd8;  // (a.l0, a.l2, a.l1, a.l3)

void pack_halves(double* buffer, __m512d factor) noexcept {
    const __m512d low = _mm512_loadu_pd(buffer);
    const __m512d high = _mm512_loadu_pd(buffer + 8u);
    const __m512d even = _mm512_shuffle_f64x2(low, high, kHalvesEven);
    const __m512d odd = _mm512_shuffle_f64x2(low, high, kHalvesOdd);
    const __m512d rotated = mul_quad(odd, factor);
    const __m512d sum = _mm512_add_pd(even, rotated);
    const __m512d difference = _mm512_sub_pd(even, rotated);
    // `sum` holds the results for the even positions of the four packed groups and
    // `difference` those for the odd ones, and the buffer wants them interleaved
    // again.  One shuffle cannot do that, so each store pairs the two halves and
    // then swaps the two middle lanes of the pair.
    const __m512d pairs_low = _mm512_shuffle_f64x2(sum, difference, kGroupsEven);
    const __m512d pairs_high = _mm512_shuffle_f64x2(sum, difference, kGroupsOdd);
    _mm512_storeu_pd(buffer, _mm512_shuffle_f64x2(pairs_low, pairs_low, kSwapMiddle));
    _mm512_storeu_pd(buffer + 8u, _mm512_shuffle_f64x2(pairs_high, pairs_high, kSwapMiddle));
}

void pack_groups(double* buffer, __m512d factor) noexcept {
    const __m512d low = _mm512_loadu_pd(buffer);
    const __m512d high = _mm512_loadu_pd(buffer + 8u);
    const __m512d even = _mm512_shuffle_f64x2(low, high, kGroupsEven);
    const __m512d odd = _mm512_shuffle_f64x2(low, high, kGroupsOdd);
    const __m512d rotated = mul_quad(odd, factor);
    const __m512d sum = _mm512_add_pd(even, rotated);
    const __m512d difference = _mm512_sub_pd(even, rotated);
    _mm512_storeu_pd(buffer, _mm512_shuffle_f64x2(sum, difference, kGroupsEven));
    _mm512_storeu_pd(buffer + 8u, _mm512_shuffle_f64x2(sum, difference, kGroupsOdd));
}

void fft_butterflies_avx512(double* buffer, std::size_t size, const double* twiddle,
                            const std::size_t* stage_begin) noexcept {
    // Stage 0: length 2, one butterfly per group, one factor for all of them.
    {
        const __m128d pair = _mm_loadu_pd(twiddle + 2u * stage_begin[0]);
        const __m512d factor = _mm512_broadcast_f64x4(_mm256_broadcast_pd(&pair));
        for (std::size_t index = 0u; index < 2u * size; index += 16u) {
            pack_halves(buffer + index, factor);
        }
    }

    // Stage 1: length 4, two butterflies per group and two factors.
    {
        const __m512d factor =
            _mm512_broadcast_f64x4(_mm256_loadu_pd(twiddle + 2u * stage_begin[1]));
        for (std::size_t index = 0u; index < 2u * size; index += 16u) {
            pack_groups(buffer + index, factor);
        }
    }

    std::size_t stage = 2u;
    for (std::size_t half = 4u; half < size; half <<= 1u, ++stage) {
        const std::size_t length = half << 1u;
        const double* table = twiddle + 2u * stage_begin[stage];
        for (std::size_t start = 0u; start < size; start += length) {
            for (std::size_t offset = 0u; offset < half; offset += 4u) {
                const std::size_t even_index = (start + offset) * 2u;
                const std::size_t odd_index = even_index + half * 2u;
                const std::size_t factor_index = offset * 2u;
                const __m512d even = _mm512_loadu_pd(buffer + even_index);
                const __m512d odd = _mm512_loadu_pd(buffer + odd_index);
                const __m512d factor = _mm512_loadu_pd(table + factor_index);
                const __m512d rotated = mul_quad(odd, factor);
                _mm512_storeu_pd(buffer + even_index, _mm512_add_pd(even, rotated));
                _mm512_storeu_pd(buffer + odd_index, _mm512_sub_pd(even, rotated));
            }
        }
    }
}

// Scalar times a run of interleaved complexes: four complexes (eight doubles) per
// vector, every element an independent accumulation.
void complex_axpy_avx512(double* out, const double* field, double scale,
                         std::size_t count) noexcept {
    const __m512d factor = _mm512_set1_pd(scale);
    std::size_t index = 0u;
    for (; index + 4u <= count; index += 4u) {
        const std::size_t offset = index * 2u;
        _mm512_storeu_pd(out + offset,
                         _mm512_add_pd(_mm512_loadu_pd(out + offset),
                                       _mm512_mul_pd(_mm512_loadu_pd(field + offset), factor)));
    }
    for (; index < count; ++index) {
        out[index * 2u] += field[index * 2u] * scale;
        out[index * 2u + 1u] += field[index * 2u + 1u] * scale;
    }
}

// QMF analysis polyphase accumulate, eight bands per vector.
void qmf_analysis_taps_avx512(double* out, const double* source, const double* coefficients,
                              std::size_t rows) noexcept {
    for (std::size_t row = 0u; row < rows; ++row) {
        double* destination = out + row * kQmfAnalysisBands;
        const double* values = source + row * kQmfAnalysisBands;
        for (std::size_t band = 0u; band < kQmfAnalysisBands; band += 8u) {
            _mm512_storeu_pd(destination + band,
                             _mm512_add_pd(_mm512_loadu_pd(destination + band),
                                           _mm512_mul_pd(_mm512_loadu_pd(values + band),
                                                         _mm512_loadu_pd(coefficients + band))));
        }
    }
}

// Hybrid analysis low join: 32 outputs are exactly four 512-bit vectors, so one
// broadcast of the term's value feeds four multiplies and four adds.
void hybrid_low_join_avx512(const double* values, const double* kernel, double* out,
                            std::size_t rows) noexcept {
    constexpr std::size_t kTerms = kHybridTerms;
    constexpr std::size_t kOutputs = kHybridOutputs;
    static_assert(kOutputs == 32u, "four 512-bit accumulators cover one row");
    for (std::size_t row = 0u; row < rows; ++row) {
        const double* source = values + row * kTerms;
        double* destination = out + row * kOutputs;
        __m512d sum0 = _mm512_setzero_pd();
        __m512d sum1 = _mm512_setzero_pd();
        __m512d sum2 = _mm512_setzero_pd();
        __m512d sum3 = _mm512_setzero_pd();
        for (std::size_t term = 0u; term < kTerms; ++term) {
            const double value = source[term];
            if (value == 0.0) {
                continue;
            }
            const double* weights = kernel + term * kOutputs;
            const __m512d factor = _mm512_set1_pd(value);
            sum0 = _mm512_add_pd(sum0, _mm512_mul_pd(factor, _mm512_loadu_pd(weights)));
            sum1 = _mm512_add_pd(sum1, _mm512_mul_pd(factor, _mm512_loadu_pd(weights + 8u)));
            sum2 = _mm512_add_pd(sum2, _mm512_mul_pd(factor, _mm512_loadu_pd(weights + 16u)));
            sum3 = _mm512_add_pd(sum3, _mm512_mul_pd(factor, _mm512_loadu_pd(weights + 24u)));
        }
        _mm512_storeu_pd(destination, sum0);
        _mm512_storeu_pd(destination + 8u, sum1);
        _mm512_storeu_pd(destination + 16u, sum2);
        _mm512_storeu_pd(destination + 24u, sum3);
    }
}

// One rendered path, four bands per vector.  AVX-512 has no 512-bit addsub, so the
// two pairings are built with merge masking -- still one exact subtract and one
// exact add per output, and the same separate scale multiply per product.
void render_hybrid_path_avx512(double* out, const double* transfer, const double* history0,
                               const double* history1, double scale) noexcept {
    const __m512d factor = _mm512_set1_pd(scale);
    const double* histories[2] = {history0, history1};
    for (std::size_t ear = 0u; ear < 2u; ++ear) {
        const double* source = histories[ear];
        const double* taps = transfer + ear * kPathBands * 2u;
        double* destination = out + ear * kPathBands * 2u;
        std::size_t band = 0u;
        for (; band + 4u <= kPathBands; band += 4u) {
            const __m512d history = _mm512_loadu_pd(source + band * 2u);
            const __m512d tap = _mm512_loadu_pd(taps + band * 2u);
            const __m512d product = _mm512_mul_pd(
                _mm512_mul_pd(history, tap), factor);
            const __m512d crossed = _mm512_mul_pd(
                _mm512_mul_pd(history, _mm512_permute_pd(tap, 0x55)), factor);
            const __m512d swapped_product = _mm512_permute_pd(product, 0x55);
            const __m512d swapped_crossed = _mm512_permute_pd(crossed, 0x55);
            // The real component is the difference of the two products and sits in
            // the even element of each complex, the imaginary one is the sum of the
            // crossed pair and sits in the odd element -- so the subtract is masked
            // to the even lanes and the blend keeps the sum on the odd ones.
            const __m512d difference =
                _mm512_mask_sub_pd(product, 0x55, product, swapped_product);
            const __m512d total = _mm512_add_pd(crossed, swapped_crossed);
            const __m512d result = _mm512_mask_blend_pd(0xaa, difference, total);
            _mm512_storeu_pd(destination + band * 2u,
                             _mm512_add_pd(_mm512_loadu_pd(destination + band * 2u), result));
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

// Element-wise complex product, four complexes per vector.  AVX-512 has no 512-bit
// addsub, so the real difference is masked into the even lanes and the imaginary
// sum into the odd ones.
void complex_multiply_avx512(double* out, const double* left, const double* right,
                             std::size_t count) noexcept {
    std::size_t index = 0u;
    for (; index + 4u <= count; index += 4u) {
        const std::size_t offset = index * 2u;
        const __m512d first = _mm512_loadu_pd(left + offset);
        const __m512d second = _mm512_loadu_pd(right + offset);
        const __m512d product = _mm512_mul_pd(first, second);
        const __m512d crossed = _mm512_mul_pd(_mm512_permute_pd(first, 0x55), second);
        const __m512d difference =
            _mm512_mask_sub_pd(product, 0x55, product, _mm512_permute_pd(product, 0x55));
        const __m512d total = _mm512_add_pd(crossed, _mm512_permute_pd(crossed, 0x55));
        _mm512_storeu_pd(out + offset, _mm512_mask_blend_pd(0xaa, difference, total));
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

const Kernels kAvx512{
    &fft_butterflies_avx512,
    nullptr,  // qmf_synthesis_basis: four ranks do not fill a 512-bit vector
    &hybrid_low_join_avx512,
    &render_hybrid_path_avx512,
    &complex_axpy_avx512,
    &qmf_analysis_taps_avx512,
    &complex_multiply_avx512,
};

}  // namespace

const Kernels& kernels_intrin_avx512() noexcept { return kAvx512; }

}  // namespace joc::simd
