#include "binaural/sofa_room.h"

#include <algorithm>
#include <cmath>

#include "foundation/fft.h"

namespace joc::binaural {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

Status argument_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kRender, message);
}

// Inverse real transform of one half spectrum on a power-of-two grid.
void irfft_row(const std::vector<std::complex<double>>& half, std::size_t fft_size,
               std::vector<double>* out, std::size_t out_offset, std::size_t out_stride) {
    std::vector<std::complex<double>> buffer(fft_size, std::complex<double>(0.0, 0.0));
    const std::size_t bins = half.size();
    for (std::size_t bin = 0; bin < bins; ++bin) {
        buffer[bin] = half[bin];
    }
    for (std::size_t bin = 1; bin + 1 < bins; ++bin) {
        buffer[fft_size - bin] = std::conj(half[bin]);
    }
    dsp::fft_radix2(&buffer, true);
    for (std::size_t sample = 0; sample < fft_size; ++sample) {
        (*out)[out_offset + (sample % out_stride)] = buffer[sample].real();
    }
}

// Real block (zero padded to `fft_size`) to its first `fft_size/2+1` bins.
void rfft_row(const std::vector<double>& time, std::size_t fft_size,
              std::vector<std::complex<double>>* out) {
    std::vector<std::complex<double>> buffer(fft_size, std::complex<double>(0.0, 0.0));
    for (std::size_t sample = 0; sample < fft_size && sample < time.size(); ++sample) {
        buffer[sample] = std::complex<double>(time[sample], 0.0);
    }
    dsp::fft_radix2(&buffer, false);
    out->assign(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(fft_size / 2u + 1u));
}

}  // namespace

Status SofaRoom::configure(std::size_t source_count, const PointSourceReference& reference,
                           int implementation_latency, int recording_origin,
                           const std::string& preset) {
    reference_ = reference;
    RoomPreset generated;
    const Status status = build_room_preset(preset, &generated);
    if (!status.ok()) {
        return status;
    }
    preset_ = preset;
    config_ = generated.early;
    rt60_seconds_ = generated.rt60_seconds;
    alignment_samples_ = static_cast<double>(implementation_latency) +
                         static_cast<double>(std::max(recording_origin, 0));
    const std::size_t taps = generated.late_fir[0].size();
    const std::size_t aligned = static_cast<std::size_t>(alignment_samples_) + taps;
    std::vector<double> kernel(2u * aligned, 0.0);
    for (std::size_t row = 0; row < 2u; ++row) {
        std::copy(generated.late_fir[row].begin(), generated.late_fir[row].end(),
                  kernel.begin() + static_cast<std::ptrdiff_t>(
                                       row * aligned + static_cast<std::size_t>(alignment_samples_)));
    }
    convolver_.configure(2u, aligned, kSofaRoomBatchSamples);
    const Status filtered = convolver_.set_filter(kernel.data(), aligned, 0u);
    if (!filtered.ok()) {
        return filtered;
    }
    current_.assign(source_count, 0.0);
    target_.assign(source_count, 0.0);
    pending_mono_.clear();
    pending_direct_.clear();
    pending_blocks_ = 0;
    maximum_early_delay_samples_ = 0.0;
    valid_ = true;
    return Status::success();
}

void SofaRoom::distance_state(const PointSourcePosition& state, DistanceState* out) const {
    out->profile = reference_.mode();
    out->normalized_radius = state.normalized_radius;
    out->reference_distance_m = reference_.reference_distance_m();
    out->physical_distance_m = std::max(kMinimumSourceDistanceM, state.distance_m);
    // Listener frame (+x front, +y left) back to ADM (+x right, +y front).
    out->direction_adm = {-state.direction[1], state.direction[0], state.direction[2]};
}

void SofaRoom::reset() {
    if (!valid_) {
        return;
    }
    std::fill(current_.begin(), current_.end(), 0.0);
    std::fill(target_.begin(), target_.end(), 0.0);
    pending_mono_.clear();
    pending_direct_.clear();
    pending_blocks_ = 0;
    maximum_early_delay_samples_ = 0.0;
    convolver_.reset();
}

Status SofaRoom::set_source(std::size_t source, const PointSourcePosition* state, double gain,
                            bool first, bool special_lfe) {
    if (source >= current_.size()) {
        return argument_error("room source index is out of range");
    }
    double value = 0.0;
    if (!special_lfe && state != nullptr) {
        DistanceState distance;
        distance_state(*state, &distance);
        value = gain * kPublicRoomCalibrationGain * late_send(distance);
    }
    target_[source] = value;
    if (first) {
        current_[source] = value;
    }
    return Status::success();
}

Status SofaRoom::radians_paths(const PointSourcePosition& state,
                               std::vector<EarlyReflection>* paths) {
    DistanceState distance;
    distance_state(state, &distance);
    const Status status = first_order_image_sources(distance.direction_adm,
                                                    distance.physical_distance_m, 48000.0,
                                                    config_, paths);
    if (!status.ok()) {
        return status;
    }
    for (const EarlyReflection& path : *paths) {
        maximum_early_delay_samples_ =
            std::max(maximum_early_delay_samples_, path.extra_delay_samples);
    }
    return Status::success();
}

double SofaRoom::early_delay_bound_samples() const {
    double extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double listener = config_.listener_position_m[static_cast<std::size_t>(axis)];
        const double dimension = config_.dimensions_m[static_cast<std::size_t>(axis)];
        extent = std::max(extent, std::max(listener, dimension - listener));
    }
    return 2.0 * extent * 48000.0 / config_.speed_of_sound_m_s;
}

std::size_t SofaRoom::filter_budget(const hrtf::FullHrirField& field) const {
    const double needed = static_cast<double>(field.fft_size()) +
                          std::ceil(early_delay_bound_samples()) + 64.0;
    std::size_t length = 1;
    while (static_cast<double>(length) < needed) {
        length <<= 1u;
    }
    return length;
}

double SofaRoom::direct_gain(const PointSourcePosition& state) const {
    DistanceState distance;
    distance_state(state, &distance);
    return direct_level_gain(distance);
}

double SofaRoom::early_gain(const EarlyReflection& reflection, double measurement_radius_m) const {
    return kPublicRoomCalibrationGain * reflection.reflection_gain * measurement_radius_m /
           reflection.path_distance_m;
}

Status SofaRoom::full_hrir(hrtf::FullHrirField& field, const PointSourcePosition& state,
                           std::size_t output_taps, std::vector<double>* hrir) {
    if (hrir == nullptr || output_taps < field.fft_size() || output_taps % 2u != 0) {
        return argument_error("room HRIR output is invalid");
    }
    const std::size_t fft_size = field.fft_size();
    const std::size_t field_bins = field.frequency_bins();
    const std::size_t bins = output_taps / 2u + 1u;
    std::vector<double> omega(bins);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        omega[bin] = 2.0 * kPi * static_cast<double>(bin) / static_cast<double>(output_taps);
    }
    std::vector<std::complex<double>> spectrum(2u * bins, std::complex<double>(0.0, 0.0));
    std::vector<double> impulse(fft_size);
    std::vector<double> padded(output_taps, 0.0);
    std::vector<std::complex<double>> half(field_bins);
    std::vector<std::complex<double>> bins_out;

    const auto accumulate = [&](const std::array<double, 3>& direction, double extra_delay,
                                double gain) -> Status {
        std::vector<std::complex<double>> value;
        const Status status = field.spectrum(direction.data(), &value);
        if (!status.ok()) {
            return status;
        }
        for (int ear = 0; ear < 2; ++ear) {
            for (std::size_t bin = 0; bin < field_bins; ++bin) {
                half[bin] = value[static_cast<std::size_t>(ear) * field_bins + bin];
            }
            irfft_row(half, fft_size, &impulse, 0u, fft_size);
            std::fill(padded.begin(), padded.end(), 0.0);
            for (std::size_t sample = 0; sample < fft_size; ++sample) {
                padded[(sample + fft_size / 2u) % fft_size] = impulse[sample];
            }
            rfft_row(padded, output_taps, &bins_out);
            for (std::size_t bin = 0; bin < bins; ++bin) {
                const std::complex<double> rotation =
                    extra_delay == 0.0
                        ? std::complex<double>(1.0, 0.0)
                        : std::exp(std::complex<double>(0.0, -omega[bin] * extra_delay));
                spectrum[static_cast<std::size_t>(ear) * bins + bin] +=
                    bins_out[bin] * rotation * gain;
            }
        }
        return Status::success();
    };

    Status status = accumulate(state.direction, 0.0, direct_gain(state));
    if (!status.ok()) {
        return status;
    }
    std::vector<EarlyReflection> paths;
    status = radians_paths(state, &paths);
    if (!status.ok()) {
        return status;
    }
    const double radius = field.measurement_radius_m();
    for (const EarlyReflection& path : paths) {
        // ADM (+x right, +y front) back to the listener frame (+x front, +y left).
        const std::array<double, 3> direction = {path.direction_adm[1], -path.direction_adm[0],
                                                 path.direction_adm[2]};
        status = accumulate(direction, path.extra_delay_samples, early_gain(path, radius));
        if (!status.ok()) {
            return status;
        }
    }

    hrir->assign(2u * output_taps, 0.0);
    std::vector<std::complex<double>> row(bins);
    for (int ear = 0; ear < 2; ++ear) {
        for (std::size_t bin = 0; bin < bins; ++bin) {
            row[bin] = spectrum[static_cast<std::size_t>(ear) * bins + bin];
        }
        irfft_row(row, output_taps, hrir, static_cast<std::size_t>(ear) * output_taps,
                  output_taps);
    }
    return Status::success();
}

Status SofaRoom::process(const double* block, std::size_t sources, const double* direct,
                         std::vector<double>* out, bool* produced) {
    if (!valid_ || block == nullptr || direct == nullptr || out == nullptr ||
        produced == nullptr) {
        return argument_error("room process arguments are invalid");
    }
    *produced = false;
    const std::size_t offset = pending_mono_.size();
    pending_mono_.resize(offset + 128u, 0.0);
    for (std::size_t sample = 0; sample < 128u; ++sample) {
        const double amount = static_cast<double>(sample + 1u) / 128.0;
        double mono = 0.0;
        for (std::size_t source = 0; source < sources; ++source) {
            const double send =
                current_[source] + (target_[source] - current_[source]) * amount;
            mono += block[sample * sources + source] * send;
        }
        pending_mono_[offset + sample] = mono;
    }
    pending_direct_.insert(pending_direct_.end(), direct, direct + 256u);
    current_ = target_;
    ++pending_blocks_;
    if (pending_blocks_ * 128u < kSofaRoomBatchSamples) {
        return Status::success();
    }
    std::vector<double> late(kSofaRoomBatchSamples * 2u, 0.0);
    const Status status = convolver_.process_block(pending_mono_.data(), late.data());
    if (!status.ok()) {
        return status;
    }
    out->assign(pending_direct_.size(), 0.0);
    for (std::size_t index = 0; index < pending_direct_.size(); ++index) {
        (*out)[index] = pending_direct_[index] + late[index];
    }
    pending_mono_.clear();
    pending_direct_.clear();
    pending_blocks_ = 0;
    *produced = true;
    return Status::success();
}

}  // namespace joc::binaural
