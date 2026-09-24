// Scalar reference implementations of the dispatched DSP kernels.
//
// This unit is compiled with the baseline ISA of the target (no /arch flag on
// MSVC, no -m flag elsewhere) and is the ground truth every vector path is
// compared against: JOC_SIMD=scalar selects exactly these functions, and their
// output must be bit-identical to any ISA path's.
//
// The arithmetic below is deliberately written with std::complex<double>, which
// on MSVC expands to the plain `re * rr - im * ri` / `re * ri + im * rr` pair
// (no helper call, no scaling trick), so the vector paths have an unambiguous
// two-roundings-per-component target to reproduce.

#include "simd/simd.h"

#include <complex>

#include "foundation/fft.h"

namespace joc::simd {
namespace {

using Complex = dsp::Complex;

// Exactly the stage loop of FftPlan::apply, over a table this kernel does not own,
// for any power-of-two size.  The bit-reversal permutation stays with the caller:
// both call sites already write their input in permuted order.
void fft_butterflies_scalar(double* raw_data, std::size_t size, const double* raw_twiddle,
                            const std::size_t* stage_begin) noexcept {
    auto* data = reinterpret_cast<Complex*>(raw_data);
    const auto* twiddle = reinterpret_cast<const Complex*>(raw_twiddle);
    std::size_t stage = 0u;
    for (std::size_t length = 2u; length <= size; length <<= 1u, ++stage) {
        const Complex* table = twiddle + stage_begin[stage];
        for (std::size_t start = 0u; start < size; start += length) {
            for (std::size_t offset = 0u; offset < length / 2u; ++offset) {
                const Complex even = data[start + offset];
                const Complex odd = data[start + offset + length / 2u] * table[offset];
                data[start + offset] = even + odd;
                data[start + offset + length / 2u] = even - odd;
            }
        }
    }
}

// The reference for the synthesis basis: one accumulator per output, taps in
// increasing order, one rounding per multiply and per add.  This is the shape
// both callers write, with the basis rows reordered rank-minor -- the products
// and their order are the caller's.
void qmf_synthesis_basis_scalar(const double* values, const double* basis, double* out,
                                std::size_t rows) noexcept {
    for (std::size_t row = 0u; row < rows; ++row) {
        const double* source = values + row * kSynthesisTaps;
        double* destination = out + row * kSynthesisBands * kSynthesisRanks;
        for (std::size_t band = 0u; band < kSynthesisBands; ++band) {
            const double* table = basis + band * kSynthesisTaps * kSynthesisRanks;
            for (std::size_t rank = 0u; rank < kSynthesisRanks; ++rank) {
                double sum = 0.0;
                for (std::size_t tap = 0u; tap < kSynthesisTaps; ++tap) {
                    sum += source[tap] * table[tap * kSynthesisRanks + rank];
                }
                destination[band * kSynthesisRanks + rank] = sum;
            }
        }
    }
}

// The reference for the hybrid low join: one accumulator per output, terms in the
// caller's order, one rounding per multiply and per add, zero terms skipped.
void hybrid_low_join_scalar(const double* values, const double* kernel, double* out,
                            std::size_t rows) noexcept {
    for (std::size_t row = 0u; row < rows; ++row) {
        const double* source = values + row * kHybridTerms;
        double* destination = out + row * kHybridOutputs;
        for (std::size_t output = 0u; output < kHybridOutputs; ++output) {
            destination[output] = 0.0;
        }
        for (std::size_t term = 0u; term < kHybridTerms; ++term) {
            const double value = source[term];
            if (value == 0.0) {
                continue;
            }
            const double* weights = kernel + term * kHybridOutputs;
            for (std::size_t output = 0u; output < kHybridOutputs; ++output) {
                destination[output] += value * weights[output];
            }
        }
    }
}

// The reference for one rendered path: each band of each ear accumulates its own
// value, in the caller's order, with the scale applied to each product.
void render_hybrid_path_scalar(double* out, const double* transfer, const double* history0,
                               const double* history1, double scale) noexcept {
    const double* histories[2] = {history0, history1};
    for (std::size_t ear = 0u; ear < 2u; ++ear) {
        const double* source = histories[ear];
        const double* taps = transfer + ear * kPathBands * 2u;
        double* destination = out + ear * kPathBands * 2u;
        for (std::size_t band = 0u; band < kPathBands; ++band) {
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

// Neighbouring complexes and neighbouring bands are independent outputs, so the
// references for the two accumulates are plain element-wise loops.
void complex_axpy_scalar(double* out, const double* field, double scale,
                         std::size_t count) noexcept {
    for (std::size_t index = 0u; index < count * 2u; ++index) {
        out[index] += field[index] * scale;
    }
}

void qmf_analysis_taps_scalar(double* out, const double* source, const double* coefficients,
                              std::size_t rows) noexcept {
    for (std::size_t row = 0u; row < rows; ++row) {
        double* destination = out + row * kQmfAnalysisBands;
        const double* values = source + row * kQmfAnalysisBands;
        for (std::size_t band = 0u; band < kQmfAnalysisBands; ++band) {
            destination[band] += values[band] * coefficients[band];
        }
    }
}

void complex_multiply_scalar(double* out, const double* left, const double* right,
                             std::size_t count) noexcept {
    auto* destination = reinterpret_cast<Complex*>(out);
    const auto* first = reinterpret_cast<const Complex*>(left);
    const auto* second = reinterpret_cast<const Complex*>(right);
    for (std::size_t index = 0u; index < count; ++index) {
        destination[index] = first[index] * second[index];
    }
}

const Kernels kScalar{
    &fft_butterflies_scalar,
    &qmf_synthesis_basis_scalar,
    &hybrid_low_join_scalar,
    &render_hybrid_path_scalar,
    &complex_axpy_scalar,
    &qmf_analysis_taps_scalar,
    &complex_multiply_scalar,
};

}  // namespace

const Kernels& kernels_scalar() noexcept { return kScalar; }

}  // namespace joc::simd
