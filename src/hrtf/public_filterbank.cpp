#include "hrtf/public_filterbank.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "foundation/fft.h"
#include "simd/simd.h"

namespace joc::hrtf {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kQmfLength = dsp::kQmfFftSize;
constexpr int kQmfTaps = 10;
constexpr int kSynthesisRank = 4;
constexpr int kSynthesisTaps = 10;

// ----------------------------------------------------------- filterbank -----

// One shared forward plan for the 128-point QMF transform.  The analysis bank runs
// it 2 * slots * channels times per chunk, so the twiddle recurrence is built once
// instead of being re-derived inside every butterfly.
const dsp::FftPlan& qmf_fft_plan() {
    static const dsp::FftPlan plan(dsp::kQmfFftSize, false);
    return plan;
}

// Public 64-band complex QMF analysis (public_filterbank.QmfAnalysis).
class QmfAnalysis {
public:
    static_assert(static_cast<std::size_t>(kQmfBands) == simd::kQmfAnalysisBands,
                  "the dispatched accumulate is written for this band count");
    QmfAnalysis(const Kernels& kernels, std::size_t channels)
        : channels_(channels), coefficients_(kernels.qmf_analysis) {
        history_.assign(9u * channels_ * kQmfBands, 0.0);
        // The polyphase MAC consumes one coefficient per band, so the shipped
        // [band][tap] layout makes its inner loop a stride-10 gather.  Transposing
        // once here turns that into a contiguous AXPY.  The coefficient values and
        // the accumulation order are untouched, so the sums are bit-identical.
        coefficients_by_lag_.resize(static_cast<std::size_t>(kQmfTaps) * kQmfBands);
        for (int band = 0; band < kQmfBands; ++band) {
            for (int tap = 0; tap < kQmfTaps; ++tap) {
                coefficients_by_lag_[static_cast<std::size_t>(tap) * kQmfBands +
                                     static_cast<std::size_t>(band)] =
                    coefficients_[static_cast<std::size_t>(band) * kQmfTaps +
                                  static_cast<std::size_t>(tap)];
            }
        }
        premultiply_.resize(kQmfBands);
        post_.resize(kQmfBands);
        even_post_.resize(kQmfBands);
        for (int band = 0; band < kQmfBands; ++band) {
            const double phase = static_cast<double>(band);
            premultiply_[static_cast<std::size_t>(band)] =
                std::polar(1.0, -kPi * phase / 128.0);
            post_[static_cast<std::size_t>(band)] =
                std::polar(1.0, -3.0 * (phase + 0.5) * kPi / 128.0);
            even_post_[static_cast<std::size_t>(band)] =
                Complex(0.0, band % 2 == 0 ? 1.0 : -1.0);
        }
    }

    void reset() { std::fill(history_.begin(), history_.end(), 0.0); }

    // samples: [slots*64, channels]; output: [slots, channels, 64] complex.
    void process(const std::vector<double>& samples, std::size_t slots,
                 std::vector<Complex>* output) {
        const std::size_t joined_slots = 9u + slots;
        const std::size_t history_size = 9u * channels_ * kQmfBands;
        const std::size_t joined_size = joined_slots * channels_ * kQmfBands;
        // The joined window is filled completely -- the history lands in its first
        // 9 * channels * 64 entries and the new samples in the rest -- so it is a
        // reusable scratch buffer rather than a fresh zero-filled allocation.  The
        // history tail is taken by index instead of from end(), because the buffer may
        // be longer than the window this call uses.
        if (joined_.size() < joined_size) {
            joined_.resize(joined_size);
        }
        std::copy(history_.begin(), history_.end(), joined_.begin());
        std::copy(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(slots * channels_ * kQmfBands),
                  joined_.begin() + static_cast<std::ptrdiff_t>(history_size));

        // The two polyphase accumulators are read before they are written, so their
        // zero fill is load-bearing and stays; only the per-call allocation goes.
        const std::size_t accumulator_size = slots * channels_ * kQmfBands;
        if (even_.size() < accumulator_size) {
            even_.resize(accumulator_size);
        }
        if (odd_.size() < accumulator_size) {
            odd_.resize(accumulator_size);
        }
        std::fill(even_.begin(), even_.begin() + static_cast<std::ptrdiff_t>(accumulator_size), 0.0);
        std::fill(odd_.begin(), odd_.begin() + static_cast<std::ptrdiff_t>(accumulator_size), 0.0);
        // The ten lags are ten accumulate passes over the same 64 bands with one
        // shared coefficient row; the bands are independent accumulations of a
        // single product each, so they are what the dispatched kernel puts in its
        // lanes, and every band keeps the caller's own multiply-then-add.
        //
        // Slots are processed in blocks, with the lag loop inside: one lag pass
        // touches every source row once, so running the ten passes over the whole
        // chunk re-reads the joined window ten times -- at 1536 slots that is
        // hundreds of megabytes per chunk and the loop ends up bound by memory, not
        // by arithmetic.  A block's ten lag passes instead slide over a window of
        // (block + 9) rows that stays in the second-level cache.  Lags still run in
        // ascending order inside a block, which is the order each output's sum is
        // formed in, so nothing about the arithmetic changes.
        constexpr std::size_t kSlotBlock = 32;
        for (std::size_t first = 0u; first < slots; first += kSlotBlock) {
            const std::size_t block = std::min(kSlotBlock, slots - first);
            for (int lag = 0; lag < kQmfTaps; ++lag) {
                std::vector<double>& target = (lag % 2 == 0) ? even_ : odd_;
                const double* row =
                    coefficients_by_lag_.data() + static_cast<std::size_t>(lag) * kQmfBands;
                const std::size_t source_slot = 9u - static_cast<std::size_t>(lag) + first;
                simd::qmf_analysis_taps(
                    target.data() + first * channels_ * kQmfBands,
                    joined_.data() + source_slot * channels_ * kQmfBands, row,
                    block * channels_);
            }
        }
        std::copy(joined_.begin() + static_cast<std::ptrdiff_t>(joined_size - history_size),
                  joined_.begin() + static_cast<std::ptrdiff_t>(joined_size), history_.begin());

        // Every output element is assigned below, so the size is all that has to be
        // established; a resize of an already correctly sized buffer touches nothing.
        output->resize(slots * channels_ * kQmfBands);
        std::array<Complex, dsp::kQmfFftSize> even_spectrum{};
        std::array<Complex, dsp::kQmfFftSize> odd_spectrum{};
        for (std::size_t slot = 0u; slot < slots; ++slot) {
            for (std::size_t channel = 0u; channel < channels_; ++channel) {
                const double* even_values = even_.data() + (slot * channels_ + channel) * kQmfBands;
                const double* odd_values = odd_.data() + (slot * channels_ + channel) * kQmfBands;
                transform(even_values, &even_spectrum);
                transform(odd_values, &odd_spectrum);
                Complex* destination =
                    output->data() + (slot * channels_ + channel) * kQmfBands;
                for (int band = 0; band < kQmfBands; ++band) {
                    destination[band] = odd_spectrum[static_cast<std::size_t>(band)] +
                                        even_spectrum[static_cast<std::size_t>(band)] *
                                            even_post_[static_cast<std::size_t>(band)];
                }
            }
        }
    }

private:
    void transform(const double* values, std::array<Complex, dsp::kQmfFftSize>* spectrum) {
        for (int band = 0; band < kQmfBands; ++band) {
            (*spectrum)[static_cast<std::size_t>(band)] =
                Complex(values[band], 0.0) * premultiply_[static_cast<std::size_t>(band)];
        }
        for (int index = kQmfBands; index < dsp::kQmfFftSize; ++index) {
            (*spectrum)[static_cast<std::size_t>(index)] = Complex(0.0, 0.0);
        }
        dsp::fft_radix2(spectrum, qmf_fft_plan());
        for (int band = 0; band < kQmfBands; ++band) {
            (*spectrum)[static_cast<std::size_t>(band)] *= post_[static_cast<std::size_t>(band)];
        }
    }

    std::size_t channels_;
    std::vector<double> coefficients_;  // [64][10]
    std::vector<double> coefficients_by_lag_;  // [10][64], the same values transposed
    std::vector<double> history_;       // [9][channels][64]
    std::vector<double> joined_;        // scratch, [9 + slots][channels][64]
    std::vector<double> even_;          // scratch, [slots][channels][64], zeroed per call
    std::vector<double> odd_;           // scratch, [slots][channels][64], zeroed per call
    std::vector<Complex> premultiply_;
    std::vector<Complex> post_;
    std::vector<Complex> even_post_;
};

// Sparse 64-QMF to 77-hybrid analysis (public_filterbank.HybridAnalysis).
class HybridAnalysis {
public:
    HybridAnalysis(const Kernels& kernels, std::size_t channels)
        : channels_(channels), low_kernel_(kernels.hybrid_low) {
        history_.assign(12u * channels_ * 3u * 2u, 0.0);
        high_history_.assign(6u * channels_ * 61u, Complex(0.0, 0.0));
        // The dispatched join walks one term at a time and adds its 32 weights to
        // 32 outputs, so the shipped [tap][band][component] table is regrouped to
        // the term order the caller accumulates in.  Same weights, same order.
        const std::size_t outputs = simd::kHybridOutputs;
        low_by_term_.resize(simd::kHybridTerms * outputs);
        for (int lag = 0; lag < 13; ++lag) {
            for (int point = 0; point < 3; ++point) {
                for (int input = 0; input < 2; ++input) {
                    const std::size_t term =
                        (static_cast<std::size_t>(lag) * 3u + static_cast<std::size_t>(point)) * 2u +
                        static_cast<std::size_t>(input);
                    const std::size_t source = (static_cast<std::size_t>(point) * 2u +
                                                static_cast<std::size_t>(input)) * 13u +
                                               static_cast<std::size_t>(lag);
                    for (std::size_t output = 0u; output < outputs; ++output) {
                        low_by_term_[term * outputs + output] =
                            low_kernel_[source * outputs + output];
                    }
                }
            }
        }
        low_values_.resize(simd::kHybridJoinBlock * simd::kHybridTerms);
        low_out_.resize(simd::kHybridJoinBlock * outputs);
    }

    void reset() {
        std::fill(history_.begin(), history_.end(), 0.0);
        std::fill(high_history_.begin(), high_history_.end(), Complex(0.0, 0.0));
    }

    // qmf: [slots, channels, 64]; output: [slots, channels, 77] complex.
    void process(const std::vector<Complex>& qmf, std::size_t slots,
                 std::vector<Complex>* output) {
        const std::size_t joined_slots = 12u + slots;
        const std::size_t history_size = 12u * channels_ * 6u;
        const std::size_t joined_size = joined_slots * channels_ * 3u * 2u;
        // Both the joined window and the pending high-band history are written in full
        // before they are read, so they are reused scratch buffers; the history tail is
        // taken by index because the buffer can be longer than this call's window.
        if (joined_.size() < joined_size) {
            joined_.resize(joined_size);
        }
        std::copy(history_.begin(), history_.end(), joined_.begin());
        for (std::size_t slot = 0u; slot < slots; ++slot) {
            for (std::size_t channel = 0u; channel < channels_; ++channel) {
                const Complex* source = qmf.data() + (slot * channels_ + channel) * kQmfBands;
                double* destination =
                    joined_.data() + ((12u + slot) * channels_ + channel) * 6u;
                for (int band = 0; band < 3; ++band) {
                    destination[static_cast<std::size_t>(band) * 2u] = source[band].real();
                    destination[static_cast<std::size_t>(band) * 2u + 1u] = source[band].imag();
                }
            }
        }
        // The low bands are accumulated in a register block and written straight into
        // the output, and the high bands are written by the pass below; between them
        // every one of the 77 bands is assigned, so only the size has to be set.
        output->resize(slots * channels_ * kHybridBands);
        // The thirteen taps are summed in a per-output register block and the low
        // bands are written straight into the output.  Keeping a separate low plane
        // and then copying it into the output re-streams tens of megabytes per chunk
        // for nothing, and only the first kHybridLow bands are ever touched.  The
        // join itself is dispatched (see src/simd/simd.h): the 32 outputs of a
        // row are 32 independent accumulations over the same 78 terms, which is what
        // shares a vector.  Every lane keeps the caller's term order -- lag, then
        // point, then input -- and its two roundings, and skips exactly the terms
        // this loop skips.  Rows are staged in blocks so the gathered values do not
        // spill out of the first-level cache.
        const std::size_t hybrid_rows = slots * channels_;
        const std::size_t block = simd::kHybridJoinBlock;
        const std::size_t terms = simd::kHybridTerms;
        for (std::size_t first = 0u; first < hybrid_rows; first += block) {
            const std::size_t count = std::min(block, hybrid_rows - first);
            for (std::size_t index = 0u; index < count; ++index) {
                const std::size_t row = first + index;
                const std::size_t slot = row / channels_;
                const std::size_t channel = row % channels_;
                double* staged = low_values_.data() + index * terms;
                for (int lag = 0; lag < 13; ++lag) {
                    const std::size_t source_slot = 12u - static_cast<std::size_t>(lag) + slot;
                    const double* source =
                        joined_.data() + (source_slot * channels_ + channel) * 6u;
                    for (int point = 0; point < 3; ++point) {
                        for (int input = 0; input < 2; ++input) {
                            staged[(static_cast<std::size_t>(lag) * 3u +
                                    static_cast<std::size_t>(point)) * 2u +
                                   static_cast<std::size_t>(input)] =
                                source[static_cast<std::size_t>(point) * 2u +
                                       static_cast<std::size_t>(input)];
                        }
                    }
                }
            }
            simd::hybrid_low_join(low_values_.data(), low_by_term_.data(),
                                       low_out_.data(), count);
            for (std::size_t index = 0u; index < count; ++index) {
                Complex* destination = output->data() + (first + index) * kHybridBands;
                const double* values = low_out_.data() + index * simd::kHybridOutputs;
                for (int band = 0; band < kHybridLow; ++band) {
                    destination[band] = Complex(values[static_cast<std::size_t>(band) * 2u],
                                                values[static_cast<std::size_t>(band) * 2u + 1u]);
                }
            }
        }
        std::copy(joined_.begin() + static_cast<std::ptrdiff_t>(joined_size - history_size),
                  joined_.begin() + static_cast<std::ptrdiff_t>(joined_size), history_.begin());

        // The high bands pass through unchanged but delayed by the six slots of
        // history the reference concatenates in front of them.  Only the last six
        // entries of that concatenation survive into high_history_, so a six-entry
        // register replaces the (6 + slots) plane and its full copy.  Note the
        // output reads the concatenation at index `slot`, not `6 + slot`, so the
        // first six output slots come from the history: that offset is part of the
        // current output and is preserved verbatim.
        for (std::size_t slot = 0u; slot < slots; ++slot) {
            for (std::size_t channel = 0u; channel < channels_; ++channel) {
                Complex* destination = output->data() +
                                       (slot * channels_ + channel) * kHybridBands + kHybridLow;
                if (slot < 6u) {
                    const Complex* source =
                        high_history_.data() + (slot * channels_ + channel) * 61u;
                    for (int band = 0; band < 61; ++band) {
                        destination[band] = source[band];
                    }
                } else {
                    const Complex* source =
                        qmf.data() + ((slot - 6u) * channels_ + channel) * kQmfBands;
                    for (int band = 3; band < kQmfBands; ++band) {
                        destination[static_cast<std::size_t>(band - 3)] = source[band];
                    }
                }
            }
        }
        // Every entry of the pending high-band history is written here, so it is a
        // reusable scratch buffer; the copy into the live history is kept as it was.
        if (next_high_history_.size() < 6u * channels_ * 61u) {
            next_high_history_.resize(6u * channels_ * 61u);
        }
        for (std::size_t entry = 0u; entry < 6u; ++entry) {
            const std::size_t combined = slots + entry;
            for (std::size_t channel = 0u; channel < channels_; ++channel) {
                Complex* destination =
                    next_high_history_.data() + (entry * channels_ + channel) * 61u;
                if (combined < 6u) {
                    const Complex* source =
                        high_history_.data() + (combined * channels_ + channel) * 61u;
                    for (int band = 0; band < 61; ++band) {
                        destination[band] = source[band];
                    }
                } else {
                    const Complex* source =
                        qmf.data() + ((combined - 6u) * channels_ + channel) * kQmfBands;
                    for (int band = 3; band < kQmfBands; ++band) {
                        destination[static_cast<std::size_t>(band - 3)] = source[band];
                    }
                }
            }
        }
        std::copy(next_high_history_.begin(), next_high_history_.end(), high_history_.begin());
    }

private:
    std::size_t channels_;
    std::vector<double> low_kernel_;   // [3][2][13][16][2]
    std::vector<double> low_by_term_;  // [78][32], the same weights in the caller's term order
    std::vector<double> low_values_;   // scratch, [block][78]
    std::vector<double> low_out_;      // scratch, [block][32]
    std::vector<double> history_;      // [12][channels][3][2]
    std::vector<Complex> high_history_;  // [6][channels][61]
    std::vector<double> joined_;         // scratch, [12 + slots][channels][3][2]
    std::vector<Complex> next_high_history_;  // scratch, [6][channels][61]
};

// Instantaneous sparse 77-hybrid to 64-QMF synthesis map.
class HybridSynthesis {
public:
    explicit HybridSynthesis(const Kernels& kernels) {
        const std::size_t rows = kernels.hybrid_indices.size() / 4u;
        mapping_.reserve(rows);
        for (std::size_t index = 0u; index < rows; ++index) {
            Entry entry;
            for (int field = 0; field < 4; ++field) {
                entry.index[static_cast<std::size_t>(field)] =
                    kernels.hybrid_indices[index * 4u + static_cast<std::size_t>(field)];
            }
            entry.gain = kernels.hybrid_values[index];
            mapping_.push_back(entry);
        }
    }

    // hybrid: [slots, channels, 77]; output: [slots, channels, 64] complex.
    // The sparse map moves a real or imaginary part of one band into a real or
    // imaginary part of another, so the two components are accumulated apart.
    void process(const std::vector<Complex>& hybrid, std::size_t slots, std::size_t channels,
                 std::vector<Complex>* output) const {
        const std::size_t rows = slots * channels;
        std::vector<double> real(rows * kQmfBands, 0.0);
        std::vector<double> imaginary(rows * kQmfBands, 0.0);
        for (std::size_t slot = 0u; slot < slots; ++slot) {
            for (std::size_t channel = 0u; channel < channels; ++channel) {
                const std::size_t row = slot * channels + channel;
                const Complex* source = hybrid.data() + row * kHybridBands;
                for (const Entry& entry : mapping_) {
                    const double value = entry.index[1] == 0u ? source[entry.index[0]].real()
                                                             : source[entry.index[0]].imag();
                    if (value == 0.0) {
                        continue;
                    }
                    double* destination =
                        (entry.index[3] == 0u ? real.data() : imaginary.data()) + row * kQmfBands;
                    destination[entry.index[2]] += value * entry.gain;
                }
            }
        }
        // Every output element is assigned from the two accumulators below, so the
        // zero fill that `assign` performed was dead; only the size is needed.
        output->resize(rows * kQmfBands);
        for (std::size_t index = 0u; index < output->size(); ++index) {
            (*output)[index] = Complex(real[index], imaginary[index]);
        }
    }

private:
    struct Entry {
        std::size_t index[4] = {0u, 0u, 0u, 0u};
        double gain = 0.0;
    };
    std::vector<Entry> mapping_;
};

// Rank-4 64-band synthesis.
class QmfSynthesis {
public:
    QmfSynthesis(const Kernels& kernels, std::size_t channels)
        : channels_(channels), basis_(kernels.qmf_basis), taps_(kernels.qmf_taps) {
        history_.assign(9u * channels_ * kQmfBands * kSynthesisRank, 0.0);
        // The dispatched basis kernel reads the four ranks of one (band, tap) as
        // one vector, so the shipped [band][rank][tap] table is reordered once
        // here.  The weights are the same doubles, only their order differs.
        const std::size_t bands = static_cast<std::size_t>(kQmfBands);
        const std::size_t ranks = static_cast<std::size_t>(kSynthesisRank);
        const std::size_t taps = dsp::kQmfFftSize;
        basis_by_tap_.resize(bands * taps * ranks);
        for (std::size_t band = 0u; band < bands; ++band) {
            for (std::size_t tap = 0u; tap < taps; ++tap) {
                for (std::size_t rank = 0u; rank < ranks; ++rank) {
                    basis_by_tap_[(band * taps + tap) * ranks + rank] =
                        basis_[(band * ranks + rank) * taps + tap];
                }
            }
        }
    }

    void reset() { std::fill(history_.begin(), history_.end(), 0.0); }

    // qmf: [slots, channels, 64]; output: [slots*64, channels] real.
    void process(const std::vector<Complex>& qmf, std::size_t slots, std::vector<double>* output) {
        const std::size_t rows = slots * channels_;
        // [row][band][component] staging for the basis application.  Both staging
        // planes and the joined window are reusable scratch: every element of each is
        // written before it is read, so the buffers are sized once and kept instead of
        // being allocated and zero-filled on every call.
        const std::size_t flat_size = rows * dsp::kQmfFftSize;
        if (flat_.size() < flat_size) {
            flat_.resize(flat_size);
        }
        for (std::size_t row = 0u; row < rows; ++row) {
            for (int band = 0; band < kQmfBands; ++band) {
                flat_[row * dsp::kQmfFftSize + static_cast<std::size_t>(band) * 2u] =
                    qmf[row * kQmfBands + static_cast<std::size_t>(band)].real();
                flat_[row * dsp::kQmfFftSize + static_cast<std::size_t>(band) * 2u + 1u] =
                    qmf[row * kQmfBands + static_cast<std::size_t>(band)].imag();
            }
        }
        // The sums are written straight into the joined window: the destination index
        // is known up front, the summation order is untouched, and the application
        // itself is dispatched -- the four ranks of a band are four independent dot
        // products over the same 128 values, so they share a vector while every lane
        // keeps the tap order and the two roundings of `sum +=`.
        const std::size_t history_size = 9u * channels_ * kQmfBands * kSynthesisRank;
        const std::size_t joined_size = history_size + rows * kQmfBands * kSynthesisRank;
        if (joined_.size() < joined_size) {
            joined_.resize(joined_size);
        }
        std::copy(history_.begin(), history_.end(), joined_.begin());
        simd::qmf_synthesis_basis(flat_.data(), basis_by_tap_.data(),
                                       joined_.data() + history_size, rows);
        std::copy(joined_.begin() + static_cast<std::ptrdiff_t>(joined_size - history_size),
                  joined_.begin() + static_cast<std::ptrdiff_t>(joined_size), history_.begin());

        output->assign(rows * kQmfBands, 0.0);
        for (int lag = 0; lag < kSynthesisTaps; ++lag) {
            for (std::size_t slot = 0u; slot < slots; ++slot) {
                const std::size_t source_slot = 9u - static_cast<std::size_t>(lag) + slot;
                for (std::size_t channel = 0u; channel < channels_; ++channel) {
                    const double* source =
                        joined_.data() +
                        (source_slot * channels_ + channel) * kQmfBands * kSynthesisRank;
                    double* destination =
                        output->data() + (slot * channels_ + channel) * kQmfBands;
                    for (int band = 0; band < kQmfBands; ++band) {
                        double sum = 0.0;
                        for (int rank = 0; rank < kSynthesisRank; ++rank) {
                            sum += source[static_cast<std::size_t>(band) * kSynthesisRank +
                                          static_cast<std::size_t>(rank)] *
                                   taps_[(static_cast<std::size_t>(band) * kSynthesisTaps +
                                          static_cast<std::size_t>(lag)) * kSynthesisRank +
                                         static_cast<std::size_t>(rank)];
                        }
                        destination[band] += sum;
                    }
                }
            }
        }
    }

private:
    std::size_t channels_;
    std::vector<double> basis_;   // [64][4][128]
    std::vector<double> basis_by_tap_;  // [64][128][4], the same weights transposed
    std::vector<double> taps_;    // [64][10][4]
    std::vector<double> history_; // [9][channels][64][4]
    std::vector<double> flat_;    // scratch, [rows][128], fully written per call
    std::vector<double> joined_;  // scratch, [9 + slots][channels][64][4]
};

// public_filterbank.PublicAnalysis77.process: [N, channels] -> [N/64, channels, 77].
void analysis_77(const std::vector<double>& samples, std::size_t slots,
                 std::size_t channels, QmfAnalysis& qmf,
                 HybridAnalysis& hybrid_analysis, std::vector<Complex>* hybrid) {
    std::vector<double> hops(slots * channels * kQmfHop, 0.0);
    for (std::size_t slot = 0u; slot < slots; ++slot) {
        for (std::size_t channel = 0u; channel < channels; ++channel) {
            for (int index = 0; index < kQmfHop; ++index) {
                hops[(slot * channels + channel) * kQmfHop + static_cast<std::size_t>(index)] =
                    samples[(slot * kQmfHop + static_cast<std::size_t>(index)) * channels + channel];
            }
        }
    }
    std::vector<Complex> qmf_bands;
    qmf.process(hops, slots, &qmf_bands);
    hybrid_analysis.process(qmf_bands, slots, hybrid);
}

// public_filterbank.PublicSynthesis77.process: [slots, channels, 77] -> [slots*64, channels].
void synthesis_77(const std::vector<Complex>& hybrid, std::size_t slots,
                  std::size_t channels, const HybridSynthesis& synthesis,
                  QmfSynthesis& qmf, std::vector<double>* time) {
    std::vector<Complex> qmf_bands;
    synthesis.process(hybrid, slots, channels, &qmf_bands);
    std::vector<double> samples;
    qmf.process(qmf_bands, slots, &samples);
    // The reference transposes (slots, channels, 64) to sample-major output.
    time->assign(samples.size(), 0.0);
    for (std::size_t slot = 0u; slot < slots; ++slot) {
        for (std::size_t channel = 0u; channel < channels; ++channel) {
            for (int band = 0; band < kQmfBands; ++band) {
                (*time)[(slot * kQmfHop + static_cast<std::size_t>(band)) * channels + channel] =
                    samples[(slot * channels + channel) * kQmfBands + static_cast<std::size_t>(band)];
            }
        }
    }
}

}  // namespace

struct PublicFilterbank::Impl {
    Impl(const Kernels& kernels, std::size_t channels)
        : channels(channels), qmf(kernels, channels), hybrid_analysis(kernels, channels),
          hybrid_synthesis(kernels), qmf_synthesis(kernels, channels) {}

    std::size_t channels;
    QmfAnalysis qmf;
    HybridAnalysis hybrid_analysis;
    HybridSynthesis hybrid_synthesis;
    QmfSynthesis qmf_synthesis;
};

PublicFilterbank::PublicFilterbank(const Kernels& kernels, std::size_t channels)
    : impl_(std::make_unique<Impl>(kernels, channels)) {}

PublicFilterbank::~PublicFilterbank() = default;

void PublicFilterbank::reset() {
    impl_->qmf.reset();
    impl_->hybrid_analysis.reset();
    impl_->qmf_synthesis.reset();
}

void PublicFilterbank::analyze_full_rate(const std::vector<double>& samples, std::size_t slots,
                                         std::vector<Complex>* hybrid) {
    analysis_77(samples, slots, impl_->channels, impl_->qmf, impl_->hybrid_analysis, hybrid);
}

void PublicFilterbank::synthesize_full_rate(const std::vector<Complex>& hybrid, std::size_t slots,
                                            std::vector<double>* time) {
    synthesis_77(hybrid, slots, impl_->channels, impl_->hybrid_synthesis, impl_->qmf_synthesis,
                 time);
}

void PublicFilterbank::analyze_qmf(const std::vector<double>& hops, std::size_t slots,
                                   std::vector<Complex>* qmf) {
    impl_->qmf.process(hops, slots, qmf);
}

void PublicFilterbank::analyze_hybrid(const std::vector<Complex>& qmf, std::size_t slots,
                                      std::vector<Complex>* hybrid) {
    impl_->hybrid_analysis.process(qmf, slots, hybrid);
}

void PublicFilterbank::synthesize_hybrid(const std::vector<Complex>& hybrid, std::size_t slots,
                                         std::vector<Complex>* qmf) {
    impl_->hybrid_synthesis.process(hybrid, slots, impl_->channels, qmf);
}

void PublicFilterbank::synthesize_qmf(const std::vector<Complex>& qmf, std::size_t slots,
                                      std::vector<double>* time) {
    impl_->qmf_synthesis.process(qmf, slots, time);
}

}  // namespace joc::hrtf


