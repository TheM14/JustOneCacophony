
#include "hrtf/partitioned_convolution.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace joc::hrtf {
namespace {

using joc::dsp::Complex;

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

Status argument_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kDsp, message);
}

}  // namespace

UniformPartitionedConvolver::UniformPartitionedConvolver(std::size_t outputs,
                                                         std::size_t max_filter_taps,
                                                         std::size_t block_size) {
    configure(outputs, max_filter_taps, block_size);
}

void UniformPartitionedConvolver::configure(std::size_t outputs, std::size_t max_filter_taps,
                                            std::size_t block_size) {
    outputs_ = outputs;
    max_filter_taps_ = max_filter_taps;
    block_size_ = block_size;
    fft_size_ = 2 * block_size;
    bins_ = block_size + 1;
    partitions_ = block_size == 0 ? 0 : (max_filter_taps + block_size - 1) / block_size;
    history_.assign(partitions_ * bins_, Complex{});
    spectra_.assign(outputs_ * partitions_ * bins_, Complex{});
    target_spectra_.assign(outputs_ * partitions_ * bins_, Complex{});
    overlap_.assign(outputs_ * block_size, 0.0);
    target_overlap_.assign(outputs_ * block_size, 0.0);
    work_.assign(bins_, Complex{});
    fft_buffer_.assign(fft_size_, Complex{});
    time_work_.assign(fft_size_, 0.0);
    ordered_.assign(partitions_ * bins_, Complex{});
    if (is_power_of_two(fft_size_)) {
        forward_plan_ = std::make_unique<dsp::FftPlan>(fft_size_, false);
        inverse_plan_ = std::make_unique<dsp::FftPlan>(fft_size_, true);
    } else {
        forward_plan_.reset();
        inverse_plan_.reset();
    }
    write_index_ = -1;
    processed_blocks_ = 0;
    target_active_ = false;
    fade_samples_ = 0;
    fade_position_ = 0;
}

void UniformPartitionedConvolver::reset() {
    std::fill(history_.begin(), history_.end(), Complex{});
    std::fill(overlap_.begin(), overlap_.end(), 0.0);
    std::fill(target_overlap_.begin(), target_overlap_.end(), 0.0);
    std::fill(target_spectra_.begin(), target_spectra_.end(), Complex{});
    write_index_ = -1;
    processed_blocks_ = 0;
    target_active_ = false;
    fade_samples_ = 0;
    fade_position_ = 0;
}

void UniformPartitionedConvolver::real_spectrum(const double* time,
                                                std::vector<Complex>* out) const {
    // numpy.fft.rfft of a real block: the unnormalised forward transform, first
    // block_size+1 bins of the size-2*block_size complex transform.
    std::fill(fft_buffer_.begin(), fft_buffer_.end(), Complex{});
    for (std::size_t i = 0; i < fft_size_; ++i) {
        fft_buffer_[i] = Complex(time[i], 0.0);
    }
    if (forward_plan_) {
        dsp::fft_radix2(&fft_buffer_, *forward_plan_);
    } else {
        std::vector<Complex> transformed;
        dsp::fft_any(fft_buffer_, false, &transformed);
        fft_buffer_ = std::move(transformed);
    }
    out->assign(fft_buffer_.begin(), fft_buffer_.begin() + static_cast<std::ptrdiff_t>(bins_));
}

void UniformPartitionedConvolver::real_signal(const std::vector<Complex>& spectrum,
                                              double* out) const {
    // numpy.fft.irfft: rebuild the Hermitian half, inverse transform (1/N), real part.
    std::fill(fft_buffer_.begin(), fft_buffer_.end(), Complex{});
    for (std::size_t bin = 0; bin < bins_; ++bin) {
        fft_buffer_[bin] = spectrum[bin];
    }
    for (std::size_t bin = 1; bin + 1 < bins_; ++bin) {
        fft_buffer_[fft_size_ - bin] = std::conj(spectrum[bin]);
    }
    if (inverse_plan_) {
        dsp::fft_radix2(&fft_buffer_, *inverse_plan_);
    } else {
        std::vector<Complex> transformed;
        dsp::fft_any(fft_buffer_, true, &transformed);
        fft_buffer_ = std::move(transformed);
    }
    for (std::size_t i = 0; i < fft_size_; ++i) {
        out[i] = fft_buffer_[i].real();
    }
}

void UniformPartitionedConvolver::partition(const double* filters, std::size_t taps,
                                            std::vector<Complex>* out) {
    std::vector<double> padded(fft_size_);
    out->assign(outputs_ * partitions_ * bins_, Complex{});
    for (std::size_t partition_index = 0; partition_index < partitions_; ++partition_index) {
        const std::size_t start = partition_index * block_size_;
        if (start >= taps) {
            break;
        }
        const std::size_t stop = std::min(start + block_size_, taps);
        for (std::size_t output = 0; output < outputs_; ++output) {
            std::fill(padded.begin(), padded.end(), 0.0);
            const double* row = filters + output * taps;
            std::memcpy(padded.data(), row + start, sizeof(double) * (stop - start));
            real_spectrum(padded.data(), &work_);
            std::copy(work_.begin(), work_.end(),
                      out->begin() + static_cast<std::ptrdiff_t>(
                                          (output * partitions_ + partition_index) * bins_));
        }
    }
}

void UniformPartitionedConvolver::order_history() {
    // index 0 is the block just written; index p is p blocks ago.
    for (std::size_t partition_index = 0; partition_index < partitions_; ++partition_index) {
        std::ptrdiff_t index = 0;
        if (write_index_ >= 0) {
            const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(partitions_);
            index = (write_index_ - static_cast<std::ptrdiff_t>(partition_index) + count) % count;
        }
        std::copy(history_.begin() + index * static_cast<std::ptrdiff_t>(bins_),
                  history_.begin() + (index + 1) * static_cast<std::ptrdiff_t>(bins_),
                  ordered_.begin() + static_cast<std::ptrdiff_t>(partition_index * bins_));
    }
}

void UniformPartitionedConvolver::convolution_block(const std::vector<Complex>& spectra,
                                                    const std::vector<double>& overlap,
                                                    double* out, double* next_overlap) {
    order_history();
    for (std::size_t output = 0; output < outputs_; ++output) {
        std::fill(work_.begin(), work_.end(), Complex{});
        for (std::size_t partition_index = 0; partition_index < partitions_; ++partition_index) {
            const Complex* input = ordered_.data() + partition_index * bins_;
            const Complex* filter =
                spectra.data() + (output * partitions_ + partition_index) * bins_;
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                work_[bin] += input[bin] * filter[bin];
            }
        }
        real_signal(work_, time_work_.data());
        for (std::size_t sample = 0; sample < block_size_; ++sample) {
            out[output * block_size_ + sample] =
                time_work_[sample] + overlap[output * block_size_ + sample];
            next_overlap[output * block_size_ + sample] = time_work_[block_size_ + sample];
        }
    }
}

void UniformPartitionedConvolver::overlap_for_previous_block(
    const std::vector<Complex>& spectra, double* overlap) {
    if (processed_blocks_ == 0) {
        std::fill(overlap, overlap + outputs_ * block_size_, 0.0);
        return;
    }
    order_history();
    for (std::size_t output = 0; output < outputs_; ++output) {
        std::fill(work_.begin(), work_.end(), Complex{});
        for (std::size_t partition_index = 0; partition_index < partitions_; ++partition_index) {
            const Complex* input = ordered_.data() + partition_index * bins_;
            const Complex* filter =
                spectra.data() + (output * partitions_ + partition_index) * bins_;
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                work_[bin] += input[bin] * filter[bin];
            }
        }
        real_signal(work_, time_work_.data());
        std::memcpy(overlap + output * block_size_, time_work_.data() + block_size_,
                    sizeof(double) * block_size_);
    }
}

Status UniformPartitionedConvolver::set_filter(const double* filters, std::size_t taps,
                                               std::size_t fade_blocks) {
    if ((filters == nullptr && taps != 0) || block_size_ == 0 || outputs_ == 0 ||
        taps > max_filter_taps_) {
        return argument_error("filter update exceeds the configured tap budget");
    }
    for (std::size_t i = 0; i < outputs_ * taps; ++i) {
        if (!std::isfinite(filters[i])) {
            return argument_error("filter update contains a non-finite value");
        }
    }
    partition(filters, taps, &target_spectra_);
    if (fade_blocks == 0) {
        spectra_ = target_spectra_;
        overlap_for_previous_block(spectra_, overlap_.data());
        target_active_ = false;
        fade_samples_ = 0;
        fade_position_ = 0;
        return Status::success();
    }
    overlap_for_previous_block(target_spectra_, target_overlap_.data());
    target_active_ = true;
    fade_samples_ = static_cast<std::int64_t>(fade_blocks * block_size_);
    fade_position_ = 0;
    return Status::success();
}

Status UniformPartitionedConvolver::process_block(const double* block, double* out) {
    if (block == nullptr || out == nullptr) {
        return argument_error("block pointer is null");
    }
    for (std::size_t i = 0; i < block_size_; ++i) {
        if (!std::isfinite(block[i])) {
            return argument_error("input block contains a non-finite value");
        }
    }
    std::fill(time_work_.begin(), time_work_.end(), 0.0);
    std::memcpy(time_work_.data(), block, sizeof(double) * block_size_);
    real_spectrum(time_work_.data(), &work_);
    write_index_ = (write_index_ + 1) % static_cast<std::ptrdiff_t>(partitions_);
    std::copy(work_.begin(), work_.end(),
              history_.begin() + write_index_ * static_cast<std::ptrdiff_t>(bins_));

    std::vector<double> current(outputs_ * block_size_);
    std::vector<double> next_current_overlap(outputs_ * block_size_);
    convolution_block(spectra_, overlap_, current.data(), next_current_overlap.data());
    if (!target_active_) {
        overlap_ = next_current_overlap;
        ++processed_blocks_;
        for (std::size_t sample = 0; sample < block_size_; ++sample) {
            for (std::size_t output = 0; output < outputs_; ++output) {
                out[sample * outputs_ + output] = current[output * block_size_ + sample];
            }
        }
        return Status::success();
    }

    std::vector<double> target(outputs_ * block_size_);
    std::vector<double> next_target_overlap(outputs_ * block_size_);
    convolution_block(target_spectra_, target_overlap_, target.data(),
                      next_target_overlap.data());
    const double fade = static_cast<double>(fade_samples_);
    for (std::size_t sample = 0; sample < block_size_; ++sample) {
        const double position =
            static_cast<double>(fade_position_ + static_cast<std::int64_t>(sample) + 1);
        const double amount =
            std::clamp(fade > 0.0 ? position / fade : 1.0, 0.0, 1.0);
        for (std::size_t output = 0; output < outputs_; ++output) {
            const double dry = current[output * block_size_ + sample];
            const double wet = target[output * block_size_ + sample];
            out[sample * outputs_ + output] = dry * (1.0 - amount) + wet * amount;
        }
    }
    fade_position_ += static_cast<std::int64_t>(block_size_);
    overlap_ = next_current_overlap;
    target_overlap_ = next_target_overlap;
    if (fade_position_ >= fade_samples_) {
        spectra_ = target_spectra_;
        overlap_ = target_overlap_;
        target_active_ = false;
        fade_samples_ = 0;
        fade_position_ = 0;
    }
    ++processed_blocks_;
    return Status::success();
}

MultiSourcePartitionedConvolver::MultiSourcePartitionedConvolver(
    std::size_t source_count, std::size_t block_size, std::size_t max_filter_taps,
    std::size_t output_channels) {
    configure(source_count, block_size, max_filter_taps, output_channels);
}

void MultiSourcePartitionedConvolver::configure(std::size_t source_count,
                                                std::size_t block_size,
                                                std::size_t max_filter_taps,
                                                std::size_t output_channels) {
    block_size_ = block_size;
    max_filter_taps_ = max_filter_taps;
    output_channels_ = output_channels;
    convolvers_.clear();
    convolvers_.resize(source_count);
    for (UniformPartitionedConvolver& convolver : convolvers_) {
        convolver.configure(output_channels, max_filter_taps, block_size);
    }
    source_block_.assign(block_size, 0.0);
    partial_.assign(block_size * output_channels, 0.0);
}

void MultiSourcePartitionedConvolver::reset() {
    for (UniformPartitionedConvolver& convolver : convolvers_) {
        convolver.reset();
    }
}

Status MultiSourcePartitionedConvolver::set_filter(std::size_t source, const double* filters,
                                                   std::size_t taps, std::size_t fade_blocks) {
    if (source >= convolvers_.size()) {
        return argument_error("source index is out of range");
    }
    return convolvers_[source].set_filter(filters, taps, fade_blocks);
}

Status MultiSourcePartitionedConvolver::process_block(const double* sources, double* out) {
    if (sources == nullptr || out == nullptr) {
        return argument_error("block pointer is null");
    }
    std::fill(out, out + block_size_ * output_channels_, 0.0);
    for (std::size_t source = 0; source < convolvers_.size(); ++source) {
        for (std::size_t sample = 0; sample < block_size_; ++sample) {
            source_block_[sample] = sources[sample * convolvers_.size() + source];
        }
        const Status status =
            convolvers_[source].process_block(source_block_.data(), partial_.data());
        if (!status.ok()) {
            return status;
        }
        for (std::size_t i = 0; i < partial_.size(); ++i) {
            out[i] += partial_[i];
        }
    }
    return Status::success();
}

}  // namespace joc::hrtf
