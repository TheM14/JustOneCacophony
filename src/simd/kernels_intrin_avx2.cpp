// AVX2 implementation of the dispatched kernels.  Compiled with /arch:AVX2 (or
// -mavx2) and only ever called after the dispatcher has confirmed that this CPU
// and the OS state it saves can execute AVX2 code.
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

// Two interleaved complexes per vector.
//
// The scalar kernel computes, for one butterfly,
//     odd.re = o.re * w.re - o.im * w.im
//     odd.im = o.re * w.im + o.im * w.re
// with separate multiplies and one rounding per operation.  In a 256-bit vector
// the even element of every 128-bit half is a real part, so duplicating the even
// elements of the twiddle broadcasts w.re onto both components of a complex and
// permuting the odd data elements in lines up `o.im`, `o.re` against `w.im`.
// addsub then subtracts on the real lanes and adds on the imaginary ones, which
// is the same two roundings: `o.im * w.re + o.re * w.im` is the scalar's
// `o.re * w.im + o.im * w.re` with the summands exchanged, and floating-point
// addition is commutative.
inline __m256d mul_pair(__m256d odd, __m256d twiddle) noexcept {
    const __m256d real = _mm256_mul_pd(odd, _mm256_movedup_pd(twiddle));
    const __m256d cross = _mm256_mul_pd(_mm256_permute_pd(odd, 0x05), _mm256_permute_pd(twiddle, 0x0f));
    return _mm256_addsub_pd(real, cross);
}

// The radix-2 DIT cascade over a bit-reversed buffer of any power-of-two size at
// or above kMinVectorFftSize.
//
// Stage s pairs element (start + offset) with (start + offset + 2^s) for every
// group of 2^(s+1) elements, and the butterflies of a stage are mutually
// independent: none of them reads a slot another one writes.  They are therefore
// free to share a vector, and each one still accumulates nothing -- every output
// is a single add or subtract of two products, in lanes whose operations are
// independent of each other.
//
// Stage 0 is the one case whose group holds a single butterfly, so the pair is
// packed from two neighbouring groups instead: their factors are identical
// (offset is always 0), and a permute2f128 moves the even and odd halves into
// two vector registers.
void fft_butterflies_avx2(double* data, std::size_t size, const double* twiddle,
                          const std::size_t* stage_begin) noexcept {
    {
        const auto* pair = reinterpret_cast<const __m128d*>(twiddle + 2u * stage_begin[0]);
        const __m256d factor = _mm256_broadcast_pd(pair);
        for (std::size_t index = 0u; index < 2u * size; index += 8u) {
            const __m256d low = _mm256_loadu_pd(data + index);
            const __m256d high = _mm256_loadu_pd(data + index + 4u);
            const __m256d even = _mm256_permute2f128_pd(low, high, 0x20);
            const __m256d odd = _mm256_permute2f128_pd(low, high, 0x31);
            const __m256d rotated = mul_pair(odd, factor);
            const __m256d sum = _mm256_add_pd(even, rotated);
            const __m256d difference = _mm256_sub_pd(even, rotated);
            _mm256_storeu_pd(data + index, _mm256_permute2f128_pd(sum, difference, 0x20));
            _mm256_storeu_pd(data + index + 4u, _mm256_permute2f128_pd(sum, difference, 0x31));
        }
    }

    std::size_t stage = 1u;
    for (std::size_t half = 2u; half < size; half <<= 1u, ++stage) {
        const std::size_t length = half << 1u;
        const double* table = twiddle + 2u * stage_begin[stage];
        for (std::size_t start = 0u; start < size; start += length) {
            for (std::size_t offset = 0u; offset < half; offset += 2u) {
                const std::size_t even_index = (start + offset) * 2u;
                const std::size_t odd_index = even_index + half * 2u;
                const std::size_t factor_index = offset * 2u;
                const __m256d even = _mm256_loadu_pd(data + even_index);
                const __m256d odd = _mm256_loadu_pd(data + odd_index);
                const __m256d factor = _mm256_loadu_pd(table + factor_index);
                const __m256d rotated = mul_pair(odd, factor);
                _mm256_storeu_pd(data + even_index, _mm256_add_pd(even, rotated));
                _mm256_storeu_pd(data + odd_index, _mm256_sub_pd(even, rotated));
            }
        }
    }
}

// QMF synthesis basis, four ranks of one band per vector.
//
// The four ranks are four dot products over the same 128 taps, so they are four
// independent accumulators and nothing has to be reassociated to fill a vector.
// Each lane performs exactly the caller's `sum += values[tap] * weight`, in tap
// order, with a separate multiply and add, and the rank-minor table makes the
// four weights one contiguous load.
//
// A band's chain is 128 dependent adds, so the loop is latency-bound long before
// it is throughput-bound; the cure is more independent chains, not a shorter
// chain (that would reassociate).  `Rows` output rows share one weight load and
// run their chains side by side -- the caller's rows are consecutive output
// channels, which do read the same weights.
template <int Rows>
void synthesize_rows(const double* values, const double* basis, double* out) noexcept {
    constexpr std::size_t kRanks = kSynthesisRanks;
    constexpr std::size_t kTaps = kSynthesisTaps;
    constexpr std::size_t kRowOut = kSynthesisBands * kRanks;
    for (std::size_t band = 0u; band < kSynthesisBands; ++band) {
        const double* table = basis + band * kTaps * kRanks;
        __m256d sum[Rows];
        for (int row = 0; row < Rows; ++row) {
            sum[row] = _mm256_setzero_pd();
        }
        for (std::size_t tap = 0u; tap < kTaps; ++tap) {
            const __m256d weight = _mm256_loadu_pd(table + tap * kRanks);
            for (int row = 0; row < Rows; ++row) {
                sum[row] = _mm256_add_pd(
                    sum[row],
                    _mm256_mul_pd(_mm256_broadcast_sd(values + row * kTaps + tap), weight));
            }
        }
        for (int row = 0; row < Rows; ++row) {
            _mm256_storeu_pd(out + row * kRowOut + band * kRanks, sum[row]);
        }
    }
}

void qmf_synthesis_basis_avx2(const double* values, const double* basis, double* out,
                              std::size_t rows) noexcept {
    constexpr std::size_t kTaps = kSynthesisTaps;
    constexpr std::size_t kRowOut = kSynthesisBands * kSynthesisRanks;
    std::size_t row = 0u;
    for (; row + 4u <= rows; row += 4u) {
        synthesize_rows<4>(values + row * kTaps, basis, out + row * kRowOut);
    }
    for (; row < rows; ++row) {
        synthesize_rows<1>(values + row * kTaps, basis, out + row * kRowOut);
    }
}

// Hybrid analysis low join, one term against all 32 outputs.
//
// The 32 outputs of a row are independent accumulations over the same 78 terms,
// so they are what fills the four vectors: one broadcast of the term's value,
// then a multiply and an add per vector.  Every lane keeps the term order and the
// two roundings of the caller's `out += value * weight`.  Terms that are exactly
// zero are skipped, which is what the caller does and what leaves its sums
// unchanged (a lane's running sum is never a negative zero).
void hybrid_low_join_avx2(const double* values, const double* kernel, double* out,
                          std::size_t rows) noexcept {
    constexpr std::size_t kTerms = kHybridTerms;
    constexpr std::size_t kOutputs = kHybridOutputs;
    static_assert(kOutputs == 32u, "eight 256-bit accumulators cover one row's outputs");
    for (std::size_t row = 0u; row < rows; ++row) {
        const double* source = values + row * kTerms;
        double* destination = out + row * kOutputs;
        __m256d sum0 = _mm256_setzero_pd();
        __m256d sum1 = _mm256_setzero_pd();
        __m256d sum2 = _mm256_setzero_pd();
        __m256d sum3 = _mm256_setzero_pd();
        __m256d sum4 = _mm256_setzero_pd();
        __m256d sum5 = _mm256_setzero_pd();
        __m256d sum6 = _mm256_setzero_pd();
        __m256d sum7 = _mm256_setzero_pd();
        for (std::size_t term = 0u; term < kTerms; ++term) {
            const double value = source[term];
            if (value == 0.0) {
                continue;
            }
            const double* weights = kernel + term * kOutputs;
            const __m256d factor = _mm256_broadcast_sd(&value);
            sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(factor, _mm256_loadu_pd(weights)));
            sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(factor, _mm256_loadu_pd(weights + 4u)));
            sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(factor, _mm256_loadu_pd(weights + 8u)));
            sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(factor, _mm256_loadu_pd(weights + 12u)));
            sum4 = _mm256_add_pd(sum4, _mm256_mul_pd(factor, _mm256_loadu_pd(weights + 16u)));
            sum5 = _mm256_add_pd(sum5, _mm256_mul_pd(factor, _mm256_loadu_pd(weights + 20u)));
            sum6 = _mm256_add_pd(sum6, _mm256_mul_pd(factor, _mm256_loadu_pd(weights + 24u)));
            sum7 = _mm256_add_pd(sum7, _mm256_mul_pd(factor, _mm256_loadu_pd(weights + 28u)));
        }
        _mm256_storeu_pd(destination, sum0);
        _mm256_storeu_pd(destination + 4u, sum1);
        _mm256_storeu_pd(destination + 8u, sum2);
        _mm256_storeu_pd(destination + 12u, sum3);
        _mm256_storeu_pd(destination + 16u, sum4);
        _mm256_storeu_pd(destination + 20u, sum5);
        _mm256_storeu_pd(destination + 24u, sum6);
        _mm256_storeu_pd(destination + 28u, sum7);
    }
}

// One rendered path, two bands per vector.
//
// A band's real and imaginary parts are adjacent, so a vector holds two whole
// bands and the cross terms are one swap away -- the same shape the FFT kernel
// uses.  Two products are needed per component here (`h * t` and `h * swapped(t)`),
// each scaled on its own, and the real part is their difference while the
// imaginary part is their sum: addsub produces both pairings and a blend keeps
// one lane of each.  Bands are independent accumulations into independent
// outputs, so this is still nothing but cross-band parallelism.
void render_hybrid_path_avx2(double* out, const double* transfer, const double* history0,
                             const double* history1, double scale) noexcept {
    const __m256d factor = _mm256_set1_pd(scale);
    const double* histories[2] = {history0, history1};
    for (std::size_t ear = 0u; ear < 2u; ++ear) {
        const double* source = histories[ear];
        const double* taps = transfer + ear * kPathBands * 2u;
        double* destination = out + ear * kPathBands * 2u;
        std::size_t band = 0u;
        for (; band + 2u <= kPathBands; band += 2u) {
            const __m256d history = _mm256_loadu_pd(source + band * 2u);
            const __m256d tap = _mm256_loadu_pd(taps + band * 2u);
            const __m256d product = _mm256_mul_pd(_mm256_mul_pd(history, tap), factor);
            const __m256d crossed = _mm256_mul_pd(
                _mm256_mul_pd(history, _mm256_permute_pd(tap, 0x05)), factor);
            const __m256d difference =
                _mm256_addsub_pd(product, _mm256_permute_pd(product, 0x05));
            const __m256d sum = _mm256_add_pd(crossed, _mm256_permute_pd(crossed, 0x05));
            // `difference` carries the real parts in lanes 0 and 2, `sum` the
            // imaginary ones in lanes 1 and 3 (four elements, so the blend takes a
            // four-bit selector).
            const __m256d result = _mm256_blend_pd(difference, sum, 0xa);
            _mm256_storeu_pd(destination + band * 2u,
                             _mm256_add_pd(_mm256_loadu_pd(destination + band * 2u), result));
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

// Scalar times a run of interleaved complexes, accumulated in place.  The elements
// are independent, so two complexes (one 256-bit vector) go together and each
// component keeps the caller's own multiply-then-add.
void complex_axpy_avx2(double* out, const double* field, double scale,
                       std::size_t count) noexcept {
    const __m256d factor = _mm256_set1_pd(scale);
    std::size_t index = 0u;
    for (; index + 2u <= count; index += 2u) {
        const std::size_t offset = index * 2u;
        _mm256_storeu_pd(out + offset,
                         _mm256_add_pd(_mm256_loadu_pd(out + offset),
                                       _mm256_mul_pd(_mm256_loadu_pd(field + offset), factor)));
    }
    for (; index < count; ++index) {
        out[index * 2u] += field[index * 2u] * scale;
        out[index * 2u + 1u] += field[index * 2u + 1u] * scale;
    }
}

// QMF analysis polyphase accumulate: the 64 bands of a row are 64 independent
// accumulations of one product each, and the coefficient row is reused by every
// row, so the rows stream past a first-level-cache-resident set of coefficients.
void qmf_analysis_taps_avx2(double* out, const double* source, const double* coefficients,
                            std::size_t rows) noexcept {
    for (std::size_t row = 0u; row < rows; ++row) {
        double* destination = out + row * kQmfAnalysisBands;
        const double* values = source + row * kQmfAnalysisBands;
        for (std::size_t band = 0u; band < kQmfAnalysisBands; band += 4u) {
            _mm256_storeu_pd(destination + band,
                             _mm256_add_pd(_mm256_loadu_pd(destination + band),
                                           _mm256_mul_pd(_mm256_loadu_pd(values + band),
                                                         _mm256_loadu_pd(coefficients + band))));
        }
    }
}

// Element-wise complex product of two runs, two complexes per vector.  Both
// operands take part in every lane, so both products of the scalar formula are
// formed and addsub pairs the real difference with the imaginary sum: the same
// four multiplies and the same two roundings per component as `a * b`.
void complex_multiply_avx2(double* out, const double* left, const double* right,
                           std::size_t count) noexcept {
    std::size_t index = 0u;
    for (; index + 2u <= count; index += 2u) {
        const std::size_t offset = index * 2u;
        const __m256d first = _mm256_loadu_pd(left + offset);
        const __m256d second = _mm256_loadu_pd(right + offset);
        // (re*re, im*im) and (im*re, re*im): the first gives the real difference,
        // the second the imaginary sum.
        const __m256d product = _mm256_mul_pd(first, second);
        const __m256d crossed = _mm256_mul_pd(_mm256_permute_pd(first, 0x05), second);
        const __m256d difference =
            _mm256_addsub_pd(product, _mm256_permute_pd(product, 0x05));
        const __m256d total = _mm256_add_pd(crossed, _mm256_permute_pd(crossed, 0x05));
        _mm256_storeu_pd(out + offset, _mm256_blend_pd(difference, total, 0xa));
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

const Kernels kAvx2{
    &fft_butterflies_avx2,
    &qmf_synthesis_basis_avx2,
    &hybrid_low_join_avx2,
    &render_hybrid_path_avx2,
    &complex_axpy_avx2,
    &qmf_analysis_taps_avx2,
    &complex_multiply_avx2,
};

}  // namespace

const Kernels& kernels_intrin_avx2() noexcept { return kAvx2; }

}  // namespace joc::simd
