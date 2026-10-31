
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "foundation/fft.h"
#include "foundation/status.h"

// Uniform partitioned convolution, ported from the reference
// partitioned_convolution.py.  One input history is shared by the outputs of a
// source; a filter replacement can crossfade the output over a number of blocks
// while both the old and the new filter keep their own overlap tail.
namespace joc::hrtf {

class UniformPartitionedConvolver {
public:
    UniformPartitionedConvolver() = default;
    UniformPartitionedConvolver(std::size_t outputs, std::size_t max_filter_taps,
                                std::size_t block_size);

    void configure(std::size_t outputs, std::size_t max_filter_taps, std::size_t block_size);

    std::size_t outputs() const { return outputs_; }
    std::size_t block_size() const { return block_size_; }
    std::size_t fft_size() const { return fft_size_; }
    std::size_t max_filter_taps() const { return max_filter_taps_; }
    bool transition_active() const { return target_active_; }

    // `filters` is [outputs][taps] row-major and `taps` <= max_filter_taps.  A zero
    // fade applies the filter immediately and rebuilds the overlap tail from the
    // blocks already processed, so an impulse in flight is not dropped.
    Status set_filter(const double* filters, std::size_t taps, std::size_t fade_blocks);

    // `block` is block_size doubles; `out` is block_size*outputs, sample-major.
    Status process_block(const double* block, double* out);

    void reset();

private:
    void partition(const double* filters, std::size_t taps,
                   std::vector<std::complex<double>>* out);
    void convolution_block(const std::vector<std::complex<double>>& spectra,
                           const std::vector<double>& overlap, double* out,
                           double* next_overlap);
    void overlap_for_previous_block(const std::vector<std::complex<double>>& spectra,
                                    double* overlap);
    void order_history();
    void real_spectrum(const double* time, std::vector<std::complex<double>>* out) const;
    void real_signal(const std::vector<std::complex<double>>& spectrum, double* out) const;

    std::size_t outputs_ = 0;
    std::size_t max_filter_taps_ = 0;
    std::size_t block_size_ = 0;
    std::size_t fft_size_ = 0;
    std::size_t bins_ = 0;
    std::size_t partitions_ = 0;

    std::unique_ptr<dsp::FftPlan> forward_plan_;
    std::unique_ptr<dsp::FftPlan> inverse_plan_;
    mutable std::vector<std::complex<double>> fft_buffer_;

    std::vector<std::complex<double>> history_;      // [partitions][bins]
    std::vector<std::complex<double>> spectra_;      // [outputs][partitions][bins]
    std::vector<std::complex<double>> target_spectra_;
    std::vector<double> overlap_;                    // [outputs][block]
    std::vector<double> target_overlap_;             // [outputs][block]
    std::vector<std::complex<double>> work_;         // [bins] scratch
    std::vector<double> time_work_;                  // [fft_size] scratch
    std::vector<std::complex<double>> ordered_;      // [partitions][bins] scratch

    std::ptrdiff_t write_index_ = -1;
    std::uint64_t processed_blocks_ = 0;
    bool target_active_ = false;
    std::int64_t fade_samples_ = 0;
    std::int64_t fade_position_ = 0;
};

class MultiSourcePartitionedConvolver {
public:
    MultiSourcePartitionedConvolver() = default;
    MultiSourcePartitionedConvolver(std::size_t source_count, std::size_t block_size,
                                    std::size_t max_filter_taps,
                                    std::size_t output_channels = 2);

    void configure(std::size_t source_count, std::size_t block_size,
                   std::size_t max_filter_taps, std::size_t output_channels = 2);

    std::size_t source_count() const { return convolvers_.size(); }
    std::size_t block_size() const { return block_size_; }
    std::size_t output_channels() const { return output_channels_; }
    std::size_t max_filter_taps() const { return max_filter_taps_; }
    const UniformPartitionedConvolver& convolver(std::size_t source) const {
        return convolvers_[source];
    }

    Status set_filter(std::size_t source, const double* filters, std::size_t taps,
                      std::size_t fade_blocks);

    // `sources` is block_size*source_count (sample-major); `out` is
    // block_size*output_channels and is overwritten.
    Status process_block(const double* sources, double* out);

    void reset();

private:
    std::vector<UniformPartitionedConvolver> convolvers_;
    std::size_t block_size_ = 0;
    std::size_t max_filter_taps_ = 0;
    std::size_t output_channels_ = 2;
    std::vector<double> source_block_;
    std::vector<double> partial_;
};

}  // namespace joc::hrtf
