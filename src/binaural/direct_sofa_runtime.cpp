#include "binaural/direct_sofa_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "foundation/fft.h"
#include "hrtf/sofa.h"
#include "hrtf/sofa_field.h"

namespace joc::binaural {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

Status runtime_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kRender, message);
}

Status render_fail(const char* message) {
    return Status::fail(JOC_ERR_HRTF_FORMAT, joc::stage::kDsp, message);
}

// One ear of a measured HRIR: inverse real transform, then undo the common delay.
void ear_impulse(const std::vector<std::complex<double>>& spectrum, std::size_t ear,
                 std::size_t bins, std::size_t fft_size, std::vector<double>* out) {
    std::vector<std::complex<double>> buffer(fft_size, std::complex<double>(0.0, 0.0));
    for (std::size_t bin = 0; bin < bins; ++bin) {
        buffer[bin] = spectrum[ear * bins + bin];
    }
    for (std::size_t bin = 1; bin + 1 < bins; ++bin) {
        buffer[fft_size - bin] = std::conj(spectrum[ear * bins + bin]);
    }
    dsp::fft_radix2(&buffer, true);
    out->resize(fft_size);
    for (std::size_t sample = 0; sample < fft_size; ++sample) {
        (*out)[(sample + fft_size / 2u) % fft_size] = buffer[sample].real();
    }
}

// scipy.signal.firwin(129, 150, fs=48000): windowed-sinc, unit DC gain.
std::vector<double> lowpass_150(double sample_rate) {
    constexpr int kTaps = 129;
    const double cutoff = 150.0 / (sample_rate * 0.5);
    std::vector<double> taps(kTaps, 0.0);
    double total = 0.0;
    for (int index = 0; index < kTaps; ++index) {
        const double position = static_cast<double>(index) - 0.5 * (kTaps - 1);
        double value = 0.0;
        if (std::fabs(position) < 1e-12) {
            value = cutoff;
        } else {
            value = cutoff * std::sin(kPi * cutoff * position) / (kPi * position);
        }
        const double window = 0.54 - 0.46 * std::cos(2.0 * kPi * index / (kTaps - 1));
        taps[static_cast<std::size_t>(index)] = value * window;
        total += value * window;
    }
    for (double& value : taps) {
        value /= total;
    }
    return taps;
}

}  // namespace

DirectSofaRuntime::DirectSofaRuntime() = default;
DirectSofaRuntime::~DirectSofaRuntime() = default;

Status DirectSofaRuntime::open(const std::string& sofa_path, const DirectSofaOptions& options) {
    output_gain_ = options.output_gain;
    Status status = reference_.configure(options.mode, options.has_reference_distance,
                                        options.reference_distance_m);
    if (!status.ok()) {
        return status;
    }
    hrtf::SofaHrir sofa;
    status = hrtf::load_sofa(sofa_path, &sofa);
    if (!status.ok()) {
        return status;
    }
    hrtf::CanonicalHrtf canonical;
    status = hrtf::canonicalize_sofa(sofa, &canonical);
    if (!status.ok()) {
        return status;
    }
    hrtf::FullHrirOptions field_options;
    field_options.method = options.interpolation;
    field_options.has_shell_radius = options.has_shell_radius;
    field_options.shell_radius_m = options.shell_radius_m;
    status = field_.build(canonical, field_options);
    if (!status.ok()) {
        return status;
    }
    fft_size_ = field_.fft_size();
    bins_ = field_.frequency_bins();
    origin_samples_ = field_.recording_origin_samples();
    if (origin_samples_ < 0) {
        return runtime_error("negative recording origin requires an explicit time base");
    }
    latency_samples_ = field_.common_latency_samples();
    filter_taps_ = fft_size_;

    if (options.sofa_room) {
        const std::string preset = options.room_preset.empty()
                                       ? (reference_.mode() == "far" ? "legacy" : "compact")
                                       : options.room_preset;
        if (preset != "compact" && preset != "legacy") {
            return runtime_error("room preset must be compact or legacy");
        }
        status = room_.configure(static_cast<std::size_t>(kDirectSofaSources), reference_,
                                 latency_samples_, origin_samples_, preset);
        if (!status.ok()) {
            return status;
        }
        filter_taps_ = room_.filter_budget(field_);
    }
    convolver_.configure(static_cast<std::size_t>(kDirectSofaSources), kDirectSofaBlock,
                         filter_taps_, 2u);
    status = build_lfe_filter();
    if (!status.ok()) {
        return status;
    }
    hop_block_.assign(kDirectSofaBlock * kDirectSofaSources, 0.0);
    staging_.assign(static_cast<std::size_t>(kDirectSofaFrame) * kDirectSofaSources, 0.0);
    direct_.assign(kDirectSofaBlock * 2u, 0.0);
    distance_states_.assign(kDirectSofaSources, PointSourcePosition{});
    has_state_.assign(kDirectSofaSources, 0);
    std::memset(last_key_, 0, sizeof(last_key_));
    pending_filters_.clear();
    return reset();
}

Status DirectSofaRuntime::build_lfe_filter() {
    const std::vector<double> lowpass = lowpass_150(48000.0);
    lfe_filter_.assign(2u * fft_size_, 0.0);
    const int start = latency_samples_ + std::max(origin_samples_, 0) - 64;
    if (start < 0 || static_cast<std::size_t>(start) + lowpass.size() > fft_size_) {
        return render_fail("LFE latency exceeds the filter budget");
    }
    const double scale = 1.0 / std::sqrt(2.0);
    for (std::size_t ear = 0; ear < 2u; ++ear) {
        for (std::size_t tap = 0; tap < lowpass.size(); ++tap) {
            lfe_filter_[ear * fft_size_ + static_cast<std::size_t>(start) + tap] =
                lowpass[tap] * scale;
        }
    }
    return Status::success();
}

Status DirectSofaRuntime::reset() {
    convolver_.reset();
    room_.reset();
    timeline_.reset();
    output_.clear();
    pending_filters_.clear();
    std::fill(has_state_.begin(), has_state_.end(), 0);
    std::memset(last_key_, 0, sizeof(last_key_));
    input_samples_ = 0;
    processed_input_samples_ = 0;
    parameter_updates_ = 0;
    output_samples_ = 0;
    latency_remaining_ = latency_samples_;
    return Status::success();
}

void DirectSofaRuntime::schedule_filter(std::size_t source,
                                        const std::vector<double>& filters, std::size_t taps,
                                        bool first, bool fade) {
    if (first) {
        convolver_.set_filter(source, filters.data(), taps, 0u);
        return;
    }
    const std::int64_t due =
        processed_input_samples_ + static_cast<std::int64_t>(field_.common_latency_samples());
    for (std::size_t index = pending_filters_.size(); index-- > 0;) {
        if (pending_filters_[index].due != due) {
            break;
        }
        if (pending_filters_[index].source == source) {
            pending_filters_[index].filters = filters;
            pending_filters_[index].taps = taps;
            pending_filters_[index].fade_blocks = fade ? 1u : 0u;
            return;
        }
    }
    PendingFilter pending;
    pending.due = due;
    pending.source = source;
    pending.filters = filters;
    pending.taps = taps;
    pending.fade_blocks = fade ? 1u : 0u;
    pending_filters_.push_back(std::move(pending));
}

void DirectSofaRuntime::apply_due_filters() {
    while (!pending_filters_.empty() &&
           pending_filters_.front().due <= processed_input_samples_) {
        PendingFilter pending = std::move(pending_filters_.front());
        pending_filters_.pop_front();
        convolver_.set_filter(pending.source, pending.filters.data(), pending.taps,
                              pending.fade_blocks);
    }
}

Status DirectSofaRuntime::set_source(int source, const double position[3], double gain,
                                     bool special_lfe, bool fade) {
    if (source < 0 || source >= kDirectSofaSources) {
        return runtime_error("source index is out of range");
    }
    if (!std::isfinite(gain)) {
        return runtime_error("source gain must be finite");
    }
    const std::size_t slot = static_cast<std::size_t>(source);
    double key[5] = {position[0], position[1], position[2], gain, special_lfe ? 1.0 : 0.0};
    bool unchanged = has_state_[slot] != 0;
    for (int axis = 0; axis < 5 && unchanged; ++axis) {
        unchanged = last_key_[slot][axis] == key[axis];
    }
    if (unchanged) {
        return Status::success();
    }
    for (int axis = 0; axis < 5; ++axis) {
        last_key_[slot][axis] = key[axis];
    }
    const bool first = processed_input_samples_ == 0;
    if (special_lfe) {
        if (room_.active()) {
            const Status room_status = room_.set_source(slot, nullptr, gain, first, true);
            if (!room_status.ok()) {
                return room_status;
            }
        }
        std::vector<double> filters = lfe_filter_;
        for (double& value : filters) {
            value *= gain;
        }
        schedule_filter(slot, filters, fft_size_, first, fade);
    } else {
        const std::array<double, 3> xyz = {position[0], position[1], position[2]};
        const std::array<double, 3>* previous =
            has_state_[slot] != 0 ? &distance_states_[slot].direction : nullptr;
        PointSourcePosition state;
        Status status = reference_.map_position(xyz, previous, &state);
        if (!status.ok()) {
            return status;
        }
        std::vector<double> filters;
        std::size_t taps = 0;
        if (room_.active()) {
            status = room_.full_hrir(field_, state, filter_taps_, &filters);
            if (!status.ok()) {
                return status;
            }
            taps = filter_taps_;
        } else {
            std::vector<std::complex<double>> spectrum;
            status = field_.spectrum(state.direction.data(), &spectrum);
            if (!status.ok()) {
                return status;
            }
            filters.assign(2u * fft_size_, 0.0);
            std::vector<double> impulse;
            for (std::size_t ear = 0; ear < 2u; ++ear) {
                ear_impulse(spectrum, ear, bins_, fft_size_, &impulse);
                std::copy(impulse.begin(), impulse.end(),
                          filters.begin() + static_cast<std::ptrdiff_t>(ear * fft_size_));
            }
            taps = fft_size_;
        }
        for (double& value : filters) {
            value *= gain;
        }
        schedule_filter(slot, filters, taps, first, fade);
        distance_states_[slot] = state;
        if (room_.active()) {
            status = room_.set_source(slot, &state, gain, first, false);
            if (!status.ok()) {
                return status;
            }
        }
    }
    has_state_[slot] = 1;
    ++parameter_updates_;
    return Status::success();
}

Status DirectSofaRuntime::process_block() {
    apply_due_filters();
    const Status status = convolver_.process_block(hop_block_.data(), direct_.data());
    if (!status.ok()) {
        return status;
    }
    processed_input_samples_ += static_cast<std::int64_t>(kDirectSofaBlock);
    if (room_.active()) {
        bool produced = false;
        const Status room_status = room_.process(hop_block_.data(), kDirectSofaSources,
                                                 direct_.data(), &room_out_, &produced);
        if (!room_status.ok()) {
            return room_status;
        }
        if (produced) {
            emit(room_out_.data(), room_out_.size() / 2u);
        }
        return Status::success();
    }
    emit(direct_.data(), kDirectSofaBlock);
    return Status::success();
}

void DirectSofaRuntime::emit(const double* interleaved, std::size_t samples) {
    const std::size_t skip = static_cast<std::size_t>(
        std::min<std::int64_t>(static_cast<std::int64_t>(samples), latency_remaining_));
    latency_remaining_ -= static_cast<std::int64_t>(skip);
    for (std::size_t sample = skip; sample < samples; ++sample) {
        output_.push_back(interleaved[sample * 2u] * output_gain_);
        output_.push_back(interleaved[sample * 2u + 1u] * output_gain_);
    }
    output_samples_ += samples - skip;
}

Status DirectSofaRuntime::submit_frame(const float* objects16_planar,
                                       const oamd::OamdUpdate* update,
                                       std::int64_t /*frame_index*/,
                                       std::int64_t outer_sample_offset,
                                       std::int64_t object_delay_samples) {
    if (objects16_planar == nullptr) {
        return runtime_error("frame buffer is null");
    }
    for (std::size_t channel = 0; channel < kDirectSofaSources; ++channel) {
        for (std::size_t sample = 0; sample < static_cast<std::size_t>(kDirectSofaFrame);
             ++sample) {
            staging_[sample * kDirectSofaSources + channel] =
                static_cast<double>(objects16_planar[channel * kDirectSofaFrame + sample]);
        }
    }
    if (update != nullptr) {
        const Status status = timeline_.submit_update(*update, input_samples_,
                                                      outer_sample_offset,
                                                      object_delay_samples,
                                                      processed_input_samples_);
        if (!status.ok()) {
            return status;
        }
    }
    input_samples_ += kDirectSofaFrame;
    double positions[timeline::kTimelineObjects][3] = {};
    for (std::size_t block = 0; block < 12u; ++block) {
        Status status = timeline_.positions_at(processed_input_samples_, positions);
        if (!status.ok()) {
            return status;
        }
        status = set_source(0, kDirectSofaLfePosition, 1.0, true, true);
        if (!status.ok()) {
            return status;
        }
        for (int object = 0; object < timeline::kTimelineObjects; ++object) {
            status = set_source(object + 1, positions[object], 1.0, false, true);
            if (!status.ok()) {
                return status;
            }
        }
        for (std::size_t sample = 0; sample < kDirectSofaBlock; ++sample) {
            for (std::size_t channel = 0; channel < kDirectSofaSources; ++channel) {
                hop_block_[sample * kDirectSofaSources + channel] =
                    staging_[(block * kDirectSofaBlock + sample) * kDirectSofaSources + channel];
            }
        }
        status = process_block();
        if (!status.ok()) {
            return status;
        }
    }
    return Status::success();
}

std::uint32_t DirectSofaRuntime::finish_capacity(double tail_seconds) const {
    const double samples = tail_seconds * 48000.0 + static_cast<double>(latency_samples_) +
                           static_cast<double>(filter_taps_);
    const double blocks = std::ceil(samples / 128.0);
    return static_cast<std::uint32_t>(std::max(0.0, blocks * 128.0));
}

Status DirectSofaRuntime::finish(std::uint32_t flush_samples, std::vector<double>* out) {
    if (out == nullptr) {
        return runtime_error("finish output is null");
    }
    const std::size_t before = output_.size();
    std::size_t count = flush_samples;
    const std::int64_t expected = std::max<std::int64_t>(
        0, processed_input_samples_ + static_cast<std::int64_t>(count) - latency_samples_ -
               static_cast<std::int64_t>(output_samples_));
    if (room_.active()) {
        const std::size_t pending = room_.pending_samples();
        const double batches = std::ceil(static_cast<double>(count + pending) /
                                         static_cast<double>(kSofaRoomBatchSamples));
        count = static_cast<std::size_t>(
            std::max(0.0, batches * static_cast<double>(kSofaRoomBatchSamples) -
                              static_cast<double>(pending)));
    }
    std::fill(hop_block_.begin(), hop_block_.end(), 0.0);
    double positions[timeline::kTimelineObjects][3] = {};
    for (std::size_t block = 0; block < count / kDirectSofaBlock; ++block) {
        Status status = timeline_.positions_at(processed_input_samples_, positions);
        if (!status.ok()) {
            return status;
        }
        status = set_source(0, kDirectSofaLfePosition, 1.0, true, true);
        if (!status.ok()) {
            return status;
        }
        for (int object = 0; object < timeline::kTimelineObjects; ++object) {
            status = set_source(object + 1, positions[object], 1.0, false, true);
            if (!status.ok()) {
                return status;
            }
        }
        status = process_block();
        if (!status.ok()) {
            return status;
        }
    }
    std::vector<double> tail(output_.begin() + static_cast<std::ptrdiff_t>(before), output_.end());
    if (static_cast<std::int64_t>(tail.size() / 2u) > expected) {
        tail.resize(static_cast<std::size_t>(expected) * 2u);
    }
    output_.resize(before + tail.size());
    out->swap(tail);
    return Status::success();
}

}  // namespace joc::binaural
