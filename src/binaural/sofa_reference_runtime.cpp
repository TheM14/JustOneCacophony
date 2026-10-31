
#include "binaural/sofa_reference_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

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

// Inverse real transform: Hermitian half -> n real samples (numpy irfft).
void irfft_to_real(const std::vector<std::complex<double>>& half, std::size_t count,
                   std::vector<double>* out) {
    std::vector<std::complex<double>> buffer(count);
    const std::size_t bins = count / 2 + 1;
    for (std::size_t bin = 0; bin < bins; ++bin) {
        buffer[bin] = half[bin];
    }
    for (std::size_t bin = 1; bin + 1 < bins; ++bin) {
        buffer[count - bin] = std::conj(half[bin]);
    }
    dsp::fft_radix2(&buffer, true);
    out->resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        (*out)[index] = buffer[index].real();
    }
}

// numpy.roll along the sample axis.
void roll_forward(const std::vector<double>& source, std::size_t shift,
                  std::vector<double>* out) {
    const std::size_t count = source.size();
    std::vector<double> copy(source);
    out->resize(count);
    if (count == 0) {
        return;
    }
    const std::size_t offset = shift % count;
    for (std::size_t index = 0; index < count; ++index) {
        (*out)[index] = copy[(index + count - offset) % count];
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

void stereo_delay(const std::vector<double>& input, std::size_t delay, std::vector<double>* state,
                  std::vector<double>* out) {
    const std::size_t samples = input.size() / 2;
    out->assign(input.size(), 0.0);
    const std::size_t capacity = delay * 2;
    std::vector<double> joined(capacity + input.size(), 0.0);
    for (std::size_t index = 0; index < capacity; ++index) {
        joined[index] = (*state)[index];
    }
    std::copy(input.begin(), input.end(), joined.begin() + static_cast<std::ptrdiff_t>(capacity));
    for (std::size_t index = 0; index < input.size(); ++index) {
        (*out)[index] = joined[index];
    }
    state->assign(capacity, 0.0);
    for (std::size_t index = 0; index < capacity; ++index) {
        (*state)[index] = joined[input.size() + index];
    }
}

}  // namespace

SofaReferenceRuntime::SofaReferenceRuntime() = default;
SofaReferenceRuntime::~SofaReferenceRuntime() = default;

void SofaReferenceRuntime::rebuild_latency() {
    latency_samples_ = std::max(implementation_latency_, bank_latency_) +
                       std::max(origin_samples_, 0);
    room_delay_samples_ = latency_samples_ - bank_latency_;
    direct_delay_samples_ = std::max(
        0, latency_samples_ - implementation_latency_ - std::max(origin_samples_, 0));
    latency_remaining_ = latency_samples_;
    room_delay_state_.assign(static_cast<std::size_t>(room_delay_samples_) * 2, 0.0);
    direct_delay_state_.assign(static_cast<std::size_t>(direct_delay_samples_) * 2, 0.0);
}

Status SofaReferenceRuntime::open(const std::string& sofa_path,
                                  const SofaReferenceOptions& options) {
    output_gain_ = options.output_gain;
    Status status = environment_.load(options.reference_config, options.mode_index,
                                      options.mode_specified);
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
    frequencies_ = field_.frequencies();
    implementation_latency_ = field_.common_latency_samples();
    origin_samples_ = field_.recording_origin_samples();
    if (origin_samples_ < 0) {
        return runtime_error("negative recording origin requires an explicit time base");
    }
    bank_latency_ = static_cast<int>(hrtf::rosella_bank_latency(hrtf::RosellaBankKind::Public));
    if (environment_.minimum_distance_m() <= field_.head_radius_m()) {
        return runtime_error("reference near-centre regularization lies inside the head");
    }
    rebuild_latency();

    const auto callback = [this](const ReferencePosition& state,
                                 std::vector<std::complex<double>>* out) {
        return spectrum_at(state, out);
    };
    status = environment_.calibrate(fft_size_, frequencies_, callback, &level_calibration_);
    if (!status.ok()) {
        return status;
    }
    analysis_bank_ = environment_.create_analysis_bank(kSofaReferenceSources);
    synthesis_bank_ = environment_.create_synthesis_bank();
    status = environment_.create_room(&room_);
    if (!status.ok()) {
        return status;
    }
    convolver_.configure(kSofaReferenceSources, kSofaReferenceBlock, fft_size_, 2);
    hop_block_.assign(kSofaReferenceBlock * kSofaReferenceSources, 0.0);
    staging_.assign(kSofaReferenceBatch * kSofaReferenceSources, 0.0);
    direct_.assign(kSofaReferenceBlock * 2, 0.0);
    qmf_.assign(2 * kSofaReferenceSources * 64, std::complex<double>(0.0, 0.0));
    hybrid_.assign(2 * kSofaReferenceSources * 77, std::complex<double>(0.0, 0.0));
    room_in_.assign(2 * 77, std::complex<double>(0.0, 0.0));
    room_out_.assign(2 * 2 * 77, std::complex<double>(0.0, 0.0));
    synth_.assign(2 * 2 * 64, std::complex<double>(0.0, 0.0));
    synth_time_.assign(2 * 2 * 64, 0.0);
    wet_.assign(kSofaReferenceBlock * 2, 0.0);
    sends_.assign(kSofaReferenceSources, 0.0);
    send_targets_.assign(kSofaReferenceSources, 0.0);
    distance_states_.assign(kSofaReferenceSources, ReferencePosition{});
    has_state_.assign(kSofaReferenceSources, 0);
    last_position_.assign(kSofaReferenceSources * 3, 0.0);
    std::memset(last_key_, 0, sizeof(last_key_));
    for (double& value : last_gain_) {
        value = std::numeric_limits<double>::quiet_NaN();
    }
    status = build_lfe_filter();
    if (!status.ok()) {
        return status;
    }
    return reset();
}

Status SofaReferenceRuntime::build_lfe_filter() {
    const std::vector<double> lowpass = lowpass_150(48000.0);
    lfe_filter_.assign(2 * fft_size_, 0.0);
    const int start = latency_samples_ - 64;
    if (start < 0 || static_cast<std::size_t>(start) + lowpass.size() > fft_size_) {
        return runtime_error("LFE latency exceeds the filter budget");
    }
    const double scale = 1.0 / std::sqrt(2.0);
    for (int ear = 0; ear < 2; ++ear) {
        for (std::size_t tap = 0; tap < lowpass.size(); ++tap) {
            lfe_filter_[static_cast<std::size_t>(ear) * fft_size_ +
                        static_cast<std::size_t>(start) + tap] = lowpass[tap] * scale;
        }
    }
    return Status::success();
}

Status SofaReferenceRuntime::reset() {
    for (hrtf::RosellaBank* bank : {analysis_bank_.get(), synthesis_bank_.get()}) {
        if (bank != nullptr) {
            bank->reset();
        }
    }
    if (room_) {
        room_->reset();
    }
    convolver_.reset();
    timeline_.reset();
    output_.clear();
    std::fill(sends_.begin(), sends_.end(), 0.0);
    std::fill(send_targets_.begin(), send_targets_.end(), 0.0);
    std::fill(has_state_.begin(), has_state_.end(), 0);
    for (double& value : last_gain_) {
        value = std::numeric_limits<double>::quiet_NaN();
    }
    spectrum_cache_.clear();
    pending_room_.clear();
    pending_direct_.clear();
    pending_slots_ = 0;
    pending_direct_samples_ = 0;
    input_samples_ = 0;
    processed_input_samples_ = 0;
    parameter_updates_ = 0;
    latency_remaining_ = latency_samples_;
    std::fill(room_delay_state_.begin(), room_delay_state_.end(), 0.0);
    std::fill(direct_delay_state_.begin(), direct_delay_state_.end(), 0.0);
    return Status::success();
}

Status SofaReferenceRuntime::spectrum_at(const ReferencePosition& state,
                                         std::vector<std::complex<double>>* out) {
    double key[5] = {state.direction[0], state.direction[1], state.direction[2],
                     state.ear_distance_m, state.centre_weight};
    for (const SpectrumEntry& entry : spectrum_cache_) {
        bool match = true;
        for (int axis = 0; axis < 5; ++axis) {
            if (entry.key[axis] != key[axis]) {
                match = false;
                break;
            }
        }
        if (match) {
            *out = entry.spectrum;
            return Status::success();
        }
    }
    std::vector<std::complex<double>> interpolated;
    Status status = field_.spectrum(state.direction, &interpolated);
    if (!status.ok()) {
        return status;
    }
    std::vector<std::complex<double>> ratio;
    status = field_.radial_ratio(state.direction, state.ear_distance_m, &ratio);
    if (!status.ok()) {
        return status;
    }
    out->assign(2 * bins_, std::complex<double>(0.0, 0.0));
    for (int ear = 0; ear < 2; ++ear) {
        for (std::size_t bin = 0; bin < bins_; ++bin) {
            const std::size_t offset = static_cast<std::size_t>(ear) * bins_ + bin;
            const std::complex<double> radial =
                interpolated[offset] * ratio[offset] * state.centre_weight;
            (*out)[offset] =
                radial + (1.0 - state.centre_weight) * field_.centre_spectrum()[bin];
        }
    }
    SpectrumEntry entry;
    for (int axis = 0; axis < 5; ++axis) {
        entry.key[axis] = key[axis];
    }
    entry.spectrum = *out;
    if (spectrum_cache_.size() >= 128) {
        spectrum_cache_.erase(spectrum_cache_.begin());
    }
    spectrum_cache_.push_back(std::move(entry));
    return Status::success();
}

Status SofaReferenceRuntime::set_source(int source, const double position[3], double gain,
                                        bool special_lfe, bool fade) {
    if (source < 0 || source >= kSofaReferenceSources) {
        return runtime_error("source index is out of range");
    }
    if (!std::isfinite(gain)) {
        return runtime_error("source gain must be finite");
    }
    const std::size_t slot = static_cast<std::size_t>(source);
    double key[5] = {position[0], position[1], position[2], gain,
                     special_lfe ? 1.0 : 0.0};
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
    std::vector<double> filters(2 * fft_size_, 0.0);
    if (special_lfe) {
        send_targets_[slot] = 0.0;
        std::copy(lfe_filter_.begin(), lfe_filter_.end(), filters.begin());
        for (double& value : filters) {
            value *= gain;
        }
    } else {
        ReferencePosition state;
        Status status = environment_.map_position(position, &state);
        if (!status.ok()) {
            return status;
        }
        distance_states_[slot] = state;
        send_targets_[slot] = state.room_send * gain;
        std::vector<std::complex<double>> spectrum;
        status = spectrum_at(state, &spectrum);
        if (!status.ok()) {
            return status;
        }
        const double scale = state.direct_gain * level_calibration_ * gain;
        for (std::complex<double>& value : spectrum) {
            value *= scale;
        }
        // One ear at a time: the field returns both ears side by side.
        std::vector<std::complex<double>> ear_spectrum(bins_);
        for (int ear = 0; ear < 2; ++ear) {
            std::copy(spectrum.begin() + static_cast<std::ptrdiff_t>(ear * bins_),
                      spectrum.begin() + static_cast<std::ptrdiff_t>((ear + 1) * bins_),
                      ear_spectrum.begin());
            std::vector<double> impulse;
            irfft_to_real(ear_spectrum, fft_size_, &impulse);
            roll_forward(impulse, fft_size_ / 2, &impulse);
            std::copy(impulse.begin(), impulse.end(),
                      filters.begin() + static_cast<std::ptrdiff_t>(ear * fft_size_));
        }
    }
    const std::size_t fade_blocks = (first || !fade) ? 0u : 1u;
    const Status status = convolver_.set_filter(static_cast<std::size_t>(source),
                                                filters.data(), fft_size_, fade_blocks);
    if (!status.ok()) {
        return status;
    }
    has_state_[slot] = 1;
    if (first) {
        sends_[slot] = send_targets_[slot];
    }
    ++parameter_updates_;
    return Status::success();
}

Status SofaReferenceRuntime::process_block() {
    // One 128-sample block is two 64-sample hops of the 16 source channels.
    std::vector<double> hops(2 * kSofaReferenceSources * 64, 0.0);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        for (std::size_t channel = 0; channel < kSofaReferenceSources; ++channel) {
            for (std::size_t tap = 0; tap < 64; ++tap) {
                hops[(slot * kSofaReferenceSources + channel) * 64 + tap] =
                    hop_block_[(slot * 64 + tap) * kSofaReferenceSources + channel];
            }
        }
    }
    analysis_bank_->analyze_qmf(hops, 2, &qmf_);
    analysis_bank_->analyze_hybrid(qmf_, 2, &hybrid_);

    const double amount[2] = {0.5, 1.0};
    std::fill(room_in_.begin(), room_in_.end(), std::complex<double>(0.0, 0.0));
    for (std::size_t slot = 0; slot < 2; ++slot) {
        for (std::size_t source = 0; source < kSofaReferenceSources; ++source) {
            const double ramp = sends_[source] +
                                (send_targets_[source] - sends_[source]) * amount[slot];
            const std::complex<double>* hybrid =
                hybrid_.data() + (slot * kSofaReferenceSources + source) * 77;
            for (std::size_t band = 0; band < 77; ++band) {
                room_in_[slot * 77 + band] += hybrid[band] * ramp;
            }
        }
    }
    sends_ = send_targets_;

    const Status status = convolver_.process_block(hop_block_.data(), direct_.data());
    if (!status.ok()) {
        return status;
    }
    pending_room_.insert(pending_room_.end(), room_in_.begin(), room_in_.end());
    pending_direct_.insert(pending_direct_.end(), direct_.begin(), direct_.end());
    pending_slots_ += 2;
    pending_direct_samples_ += kSofaReferenceBlock;
    processed_input_samples_ += kSofaReferenceBlock;
    if (pending_slots_ * 64 < static_cast<std::size_t>(kSofaReferenceBatch)) {
        return Status::success();
    }
    return flush_batch(nullptr);
}

Status SofaReferenceRuntime::flush_batch(std::vector<double>* /*out*/) {
    const std::size_t slots = pending_slots_;
    Status status = room_->process_chunk(pending_room_, slots, &room_out_);
    if (!status.ok()) {
        return status;
    }
    synthesis_bank_->synthesize_hybrid(room_out_, slots, &synth_);
    synth_time_.assign(slots * 2 * 64, 0.0);
    synthesis_bank_->synthesize_qmf(synth_, slots, &synth_time_);
    const std::size_t samples = slots * 64;
    wet_.assign(samples * 2, 0.0);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const std::size_t slot = sample / 64;
        const std::size_t tap = sample % 64;
        for (std::size_t ear = 0; ear < 2; ++ear) {
            wet_[sample * 2 + ear] = synth_time_[(slot * 2 + ear) * 64 + tap];
        }
    }
    std::vector<double> wet_delayed;
    std::vector<double> direct_delayed;
    stereo_delay(wet_, static_cast<std::size_t>(room_delay_samples_), &room_delay_state_,
                 &wet_delayed);
    stereo_delay(pending_direct_, static_cast<std::size_t>(direct_delay_samples_),
                 &direct_delay_state_, &direct_delayed);
    std::vector<double> combined(samples * 2, 0.0);
    for (std::size_t index = 0; index < combined.size(); ++index) {
        combined[index] = wet_delayed[index] + direct_delayed[index];
    }
    const std::size_t skip = static_cast<std::size_t>(
        std::min<std::int64_t>(static_cast<std::int64_t>(samples), latency_remaining_));
    latency_remaining_ -= static_cast<std::int64_t>(skip);
    for (std::size_t sample = skip; sample < samples; ++sample) {
        for (std::size_t ear = 0; ear < 2; ++ear) {
            output_.push_back(combined[sample * 2 + ear] * output_gain_);
        }
    }
    pending_room_.clear();
    pending_direct_.clear();
    pending_slots_ = 0;
    pending_direct_samples_ = 0;
    return Status::success();
}

Status SofaReferenceRuntime::submit_frame(const float* objects16_planar,
                                          const oamd::OamdUpdate* update,
                                          std::int64_t /*frame_index*/,
                                          std::int64_t outer_sample_offset,
                                          std::int64_t object_delay_samples) {
    if (objects16_planar == nullptr) {
        return runtime_error("frame buffer is null");
    }
    for (std::size_t channel = 0; channel < kSofaReferenceSources; ++channel) {
        for (std::size_t sample = 0; sample < static_cast<std::size_t>(kSofaReferenceBatch);
             ++sample) {
            staging_[sample * kSofaReferenceSources + channel] =
                static_cast<double>(objects16_planar[channel * kSofaReferenceBatch + sample]);
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
    input_samples_ += kSofaReferenceBatch;
    for (std::size_t block = 0; block < 12; ++block) {
        double positions[timeline::kTimelineObjects][3] = {};
        Status status = timeline_.positions_at(processed_input_samples_, positions);
        if (!status.ok()) {
            return status;
        }
        status = set_source(0, kSofaReferenceLfePosition, 1.0, true, true);
        if (!status.ok()) {
            return status;
        }
        for (int object = 0; object < timeline::kTimelineObjects; ++object) {
            status = set_source(object + 1, positions[object], 1.0, false, true);
            if (!status.ok()) {
                return status;
            }
        }
        for (std::size_t sample = 0; sample < static_cast<std::size_t>(kSofaReferenceBlock);
             ++sample) {
            for (std::size_t channel = 0; channel < kSofaReferenceSources; ++channel) {
                hop_block_[sample * kSofaReferenceSources + channel] =
                    staging_[static_cast<std::size_t>(block * kSofaReferenceBlock + sample) *
                                 kSofaReferenceSources + channel];
            }
        }
        status = process_block();
        if (!status.ok()) {
            return status;
        }
    }
    return Status::success();
}

std::uint32_t SofaReferenceRuntime::finish_capacity(double tail_seconds) const {
    const double tail = tail_seconds;
    const double samples = tail * 48000.0 + static_cast<double>(latency_samples_) +
                           static_cast<double>(fft_size_);
    const double blocks = std::ceil(samples / 128.0);
    return static_cast<std::uint32_t>(std::max(0.0, blocks * 128.0));
}

Status SofaReferenceRuntime::finish(std::uint32_t flush_samples, std::vector<double>* out) {
    if (out == nullptr) {
        return runtime_error("finish output is null");
    }
    std::vector<double> tail;
    const std::size_t before = output_.size();
    const std::size_t output_samples_before = before;
    const std::size_t pending = pending_slots_ * 64;
    const std::size_t count = flush_samples;
    const double padded_blocks =
        std::ceil((static_cast<double>(count + pending)) /
                  static_cast<double>(kSofaReferenceBatch));
    const std::size_t padded = static_cast<std::size_t>(
        std::max(0.0, padded_blocks * kSofaReferenceBatch - static_cast<double>(pending)));
    const std::int64_t expected =
        std::max<std::int64_t>(0, processed_input_samples_ + static_cast<std::int64_t>(count) -
                                      latency_samples_ -
                                      static_cast<std::int64_t>(output_samples_before));
    std::fill(hop_block_.begin(), hop_block_.end(), 0.0);
    for (std::size_t block = 0; block < padded / kSofaReferenceBlock; ++block) {
        double positions[timeline::kTimelineObjects][3] = {};
        Status status = timeline_.positions_at(processed_input_samples_, positions);
        if (!status.ok()) {
            return status;
        }
        status = set_source(0, kSofaReferenceLfePosition, 1.0, true, true);
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
    tail.assign(output_.begin() + static_cast<std::ptrdiff_t>(before), output_.end());
    if (static_cast<std::int64_t>(tail.size()) > expected) {
        tail.resize(static_cast<std::size_t>(expected));
    }
    output_.resize(before + tail.size() / 2 * 2);
    out->swap(tail);
    return Status::success();
}

}  // namespace joc::binaural
