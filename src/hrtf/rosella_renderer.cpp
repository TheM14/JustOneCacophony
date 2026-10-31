#include "hrtf/rosella_renderer.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

#include "hrtf/rosella_bank.h"
#include "hrtf/rosella_reference.h"

namespace joc::hrtf {

namespace {

constexpr int kSourceChannels = 16;
constexpr int kOutputChannels = 2;
constexpr int kBlockSamples = 512;
constexpr int kFrameSamples = 1536;
constexpr double kPi = 3.14159265358979323846;

Status render_fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kRender, message);
}

// rosella_core.RosellaRenderer: per-source direct gains plus the room send.
class Core {
public:
    Status configure(const RosellaModel& model, const std::vector<double>& field_left,
                     const std::vector<double>& field_right, int room_impulse_slots) {
        model_ = &model;
        field_left_ = &field_left;
        field_right_ = &field_right;
        room_ = std::make_unique<RosellaRoomFir>();
        const Status room_status = room_->configure(model, room_impulse_slots);
        if (!room_status.ok()) {
            return room_status;
        }
        gains_.assign(static_cast<std::size_t>(kSourceChannels) * 2u * kHybridBands,
                      Complex(0.0, 0.0));
        room_sends_.assign(kSourceChannels, 0.0);
        keys_.assign(kSourceChannels, Key{});
        for (int source = 0; source < kSourceChannels; ++source) {
            const double position[3] = {0.0, 1.0, 0.0};
            Status status = set_source(source, position,
                                       static_cast<int>(RosellaProfile::Mid), false);
            if (!status.ok()) {
                return status;
            }
        }
        return Status::success();
    }

    void reset() { room_->reset(); }

    Status set_source(int source, const double position[3], int profile, bool special_lfe) {
        const int effective_profile = special_lfe ? 0 : profile;
        Key key;
        key.special_lfe = special_lfe;
        key.profile = effective_profile;
        key.position[0] = position[0];
        key.position[1] = position[1];
        key.position[2] = position[2];
        if (keys_[static_cast<std::size_t>(source)].equals(key)) {
            return Status::success();
        }
        DirectResult result;
        if (special_lfe) {
            result = special_lfe_direct();
        } else {
            const Status status = direct_and_room_send(*model_, *field_left_, *field_right_,
                                                       position, effective_profile, &result);
            if (!status.ok()) {
                return status;
            }
        }
        for (int ear = 0; ear < 2; ++ear) {
            for (int band = 0; band < kHybridBands; ++band) {
                gains_[(static_cast<std::size_t>(source) * 2u + static_cast<std::size_t>(ear)) *
                           kHybridBands +
                       static_cast<std::size_t>(band)] = result.gains[ear][band];
            }
        }
        room_sends_[static_cast<std::size_t>(source)] = result.room_send;
        keys_[static_cast<std::size_t>(source)] = key;
        return Status::success();
    }

    // rosella_core.RosellaRenderer.direct_and_send_static: the sources are summed in
    // reverse index order, which is part of the reference's arithmetic.
    void direct_and_send_static(const std::vector<Complex>& hybrid, std::size_t slots,
                                std::vector<Complex>* direct, std::vector<Complex>* room_send) {
        direct->assign(slots * 2u * kHybridBands, Complex(0.0, 0.0));
        room_send->assign(slots * kHybridBands, Complex(0.0, 0.0));
        for (int source = kSourceChannels - 1; source >= 0; --source) {
            const std::size_t base = static_cast<std::size_t>(source) * kHybridBands;
            for (std::size_t slot = 0u; slot < slots; ++slot) {
                const std::size_t input = slot * kSourceChannels * kHybridBands + base;
                const std::size_t room_target = slot * kHybridBands;
                for (int band = 0; band < kHybridBands; ++band) {
                    const Complex value = hybrid[input + static_cast<std::size_t>(band)];
                    for (int ear = 0; ear < 2; ++ear) {
                        (*direct)[slot * 2u * kHybridBands +
                                  static_cast<std::size_t>(ear) * kHybridBands +
                                  static_cast<std::size_t>(band)] +=
                            value * gains_[(static_cast<std::size_t>(source) * 2u +
                                            static_cast<std::size_t>(ear)) *
                                               kHybridBands +
                                           static_cast<std::size_t>(band)];
                    }
                    (*room_send)[room_target + static_cast<std::size_t>(band)] +=
                        value * room_sends_[static_cast<std::size_t>(source)];
                }
            }
        }
    }

    Status process_chunk(const std::vector<Complex>& hybrid, std::size_t slots,
                         std::vector<Complex>* output) {
        std::vector<Complex> direct;
        std::vector<Complex> room_send;
        direct_and_send_static(hybrid, slots, &direct, &room_send);
        std::vector<Complex> room;
        const Status status = room_->process_chunk(room_send, slots, &room);
        if (!status.ok()) {
            return status;
        }
        for (std::size_t index = 0u; index < direct.size(); ++index) {
            direct[index] += room[index];
        }
        *output = std::move(direct);
        return Status::success();
    }

    Status room_process(const std::vector<Complex>& room_send, std::size_t slots,
                        std::vector<Complex>* output) {
        return room_->process_chunk(room_send, slots, output);
    }

private:
    struct Key {
        bool special_lfe = false;
        int profile = 0;
        double position[3] = {0.0, 1.0, 0.0};
        bool equals(const Key& other) const {
            return special_lfe == other.special_lfe && profile == other.profile &&
                   position[0] == other.position[0] && position[1] == other.position[1] &&
                   position[2] == other.position[2];
        }
    };

    const RosellaModel* model_ = nullptr;
    const std::vector<double>* field_left_ = nullptr;
    const std::vector<double>* field_right_ = nullptr;
    std::unique_ptr<RosellaRoomFir> room_;
    std::vector<Complex> gains_;
    std::vector<double> room_sends_;
    std::vector<Key> keys_;
};

}  // namespace

struct RosellaRuntime::Impl {
    RosellaModel model;
    RosellaRenderOptions options;
    std::vector<double> field_left;
    std::vector<double> field_right;
    Core core;
    std::unique_ptr<RosellaBank> analysis_bank;
    std::unique_ptr<RosellaBank> synthesis_bank;
    timeline::OamdPositionTimeline timeline{timeline::kTimelineObjects};

    std::vector<double> buffer;  // [chunk_samples][16]
    std::size_t buffer_used = 0;
    std::vector<double> output;
    std::uint64_t input_samples = 0;
    std::uint64_t processed_input_samples = 0;
    std::uint64_t raw_output_samples = 0;
    std::uint64_t metadata_block_updates = 0;
    bool open = false;
    bool finished = false;

    std::size_t chunk_samples() const {
        return static_cast<std::size_t>(options.chunk_frames) * kFrameSamples;
    }

    // rosella_binaural_renderer._process_samples.
    Status process_samples(const double* values, std::size_t count) {
        const std::size_t blocks = count / kBlockSamples;
        const std::size_t slots = blocks * (kBlockSamples / kQmfHop);
        // Every plane below is written in full before it is read, so they live as
        // reusable buffers rather than as per-chunk allocations: the chunk geometry is
        // constant, which turns each one into a no-op size check after the first chunk.
        const std::size_t hops_size = slots * kSourceChannels * kQmfHop;
        if (hops_.size() < hops_size) {
            hops_.resize(hops_size);
        }
        for (std::size_t block = 0u; block < blocks; ++block) {
            for (std::size_t hop = 0u; hop < kBlockSamples / kQmfHop; ++hop) {
                for (std::size_t channel = 0u; channel < kSourceChannels; ++channel) {
                    for (int index = 0; index < kQmfHop; ++index) {
                        hops_[((block * (kBlockSamples / kQmfHop) + hop) * kSourceChannels +
                               channel) *
                                  kQmfHop +
                              static_cast<std::size_t>(index)] =
                            values[(block * kBlockSamples + hop * kQmfHop +
                                    static_cast<std::size_t>(index)) *
                                       kSourceChannels +
                                   channel];
                    }
                }
            }
        }
        analysis_bank->analyze_qmf(hops_, slots, &qmf_);
        analysis_bank->analyze_hybrid(qmf_, slots, &hybrid_);

        rendered_.resize(slots * 2u * kHybridBands);
        if (room_sends_buffer_.size() != slots * kHybridBands) {
            room_sends_buffer_.resize(slots * kHybridBands);
        }
        for (std::size_t block = 0u; block < blocks; ++block) {
            const std::size_t first_slot = block * (kBlockSamples / kQmfHop);
            const std::size_t block_slots = kBlockSamples / kQmfHop;
            const std::int64_t sample =
                static_cast<std::int64_t>(processed_input_samples) +
                static_cast<std::int64_t>(block * kBlockSamples);
            Status status = set_block_parameters(sample);
            if (!status.ok()) {
                return status;
            }
            std::vector<Complex> block_hybrid(
                hybrid_.begin() + static_cast<std::ptrdiff_t>(first_slot * kSourceChannels *
                                                              kHybridBands),
                hybrid_.begin() + static_cast<std::ptrdiff_t>((first_slot + block_slots) *
                                                              kSourceChannels * kHybridBands));
            std::vector<Complex> direct;
            std::vector<Complex> room_send;
            core.direct_and_send_static(block_hybrid, block_slots, &direct, &room_send);
            std::copy(direct.begin(), direct.end(),
                      rendered_.begin() + static_cast<std::ptrdiff_t>(first_slot * 2u * kHybridBands));
            std::copy(room_send.begin(), room_send.end(),
                      room_sends_buffer_.begin() +
                          static_cast<std::ptrdiff_t>(first_slot * kHybridBands));
        }
        // The room runs once for the whole chunk, as the reference does.  Its output
        // keeps a zero fill of its own: only the first 64 of the 77 bands of each ear
        // are written, and the caller adds the whole 77-band slot back into `rendered`.
        Status status = core.room_process(room_sends_buffer_, slots, &room_out_);
        if (!status.ok()) {
            return status;
        }
        for (std::size_t index = 0u; index < rendered_.size(); ++index) {
            rendered_[index] += room_out_[index];
        }
        synthesis_bank->synthesize_hybrid(rendered_, slots, &qmf_out_);
        // The synthesis bank returns [slots, ears, 64]; the reference transposes it to
        // sample-major stereo before the latency skip.
        synthesis_bank->synthesize_qmf(qmf_out_, slots, &bands_);
        const std::size_t produced = blocks * kBlockSamples;
        if (time_.size() != produced * kOutputChannels) {
            time_.resize(produced * kOutputChannels);
        }
        for (std::size_t slot = 0u; slot < slots; ++slot) {
            for (std::size_t channel = 0u; channel < kOutputChannels; ++channel) {
                for (int band = 0; band < kQmfHop; ++band) {
                    time_[(slot * kQmfHop + static_cast<std::size_t>(band)) * kOutputChannels +
                          channel] =
                        bands_[(slot * kOutputChannels + channel) * kQmfHop +
                               static_cast<std::size_t>(band)];
                }
            }
        }
        for (std::size_t sample = 0u; sample < produced; ++sample) {
            for (std::size_t channel = 0u; channel < kOutputChannels; ++channel) {
                time_[sample * kOutputChannels + channel] *= options.output_gain;
            }
        }
        append_output(time_.data(), produced);
        processed_input_samples += count;
        return Status::success();
    }

    // rosella_binaural_renderer._set_block_parameters.
    Status set_block_parameters(std::int64_t sample) {
        double positions[timeline::kTimelineObjects][3] = {};
        Status status = timeline.positions_at(sample, positions);
        if (!status.ok()) {
            return status;
        }
        const double lfe[3] = {0.0, 1.0, 0.0};
        status = core.set_source(0, lfe, static_cast<int>(RosellaProfile::Mid), true);
        if (!status.ok()) {
            return status;
        }
        for (int object = 0; object < timeline::kTimelineObjects; ++object) {
            status = core.set_source(object + 1, positions[object],
                                     static_cast<int>(options.profile), false);
            if (!status.ok()) {
                return status;
            }
        }
        return Status::success();
    }

    // The reference drops the first `latency` samples of the raw stream (the
    // filterbank's own delay) and returns everything after that; the latency is
    // the selected bank's, so the two banks start at different sample offsets.
    void append_output(const double* stereo, std::size_t count) {
        const std::int64_t remaining =
            static_cast<std::int64_t>(rosella_bank_latency(options.bank)) -
            static_cast<std::int64_t>(raw_output_samples);
        const std::size_t skip = static_cast<std::size_t>(
            std::max<std::int64_t>(0, std::min<std::int64_t>(static_cast<std::int64_t>(count),
                                                             remaining)));
        raw_output_samples += count;
        for (std::size_t sample = skip; sample < count; ++sample) {
            output.push_back(stereo[sample * kOutputChannels]);
            output.push_back(stereo[sample * kOutputChannels + 1u]);
        }
    }

    // Per-chunk scratch.  Every one of these planes is written in full before it is
    // read, so they are sized once and reused instead of being allocated per chunk.
    std::vector<double> hops_;          // [slots][kSourceChannels][64]
    std::vector<Complex> qmf_;          // [slots][kSourceChannels][64]
    std::vector<Complex> hybrid_;       // [slots][kSourceChannels][77]
    std::vector<Complex> rendered_;     // [slots][2][77]
    std::vector<Complex> room_out_;     // [slots][2][77], partially written (see above)
    std::vector<Complex> qmf_out_;      // [slots][2][64]
    std::vector<double> bands_;         // [slots][2][64]
    std::vector<double> time_;          // [produced][2]
    std::vector<Complex> room_sends_buffer_;
};

RosellaRuntime::RosellaRuntime() : impl_(std::make_unique<Impl>()) {}
RosellaRuntime::~RosellaRuntime() = default;

Status RosellaRuntime::open(const RosellaModel& model, const RosellaRenderOptions& options) {
    if (impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is already open");
    }
    if (options.chunk_frames <= 0) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "chunk_frames must be positive");
    }
    if (options.room_impulse_slots <= 0) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "room_impulse_slots must be positive");
    }
    if (!(options.tail_seconds >= 0.0)) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "tail_seconds must be non-negative");
    }
    if (!std::isfinite(options.output_gain)) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "output_gain must be finite");
    }
    if (options.object_delay_samples < 0) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "object_delay_samples must be non-negative");
    }
    if (options.bank != RosellaBankKind::Public && options.bank != RosellaBankKind::Captured) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "unknown Rosella filterbank");
    }
    impl_->model = model;
    impl_->options = options;
    impl_->field_left = rosella_logical_field(impl_->model.field_left_padded);
    impl_->field_right = rosella_logical_field(impl_->model.field_right_padded);
    // The core and the room keep references, so they must point at this runtime's
    // own copy of the model rather than at the caller's.
    Status status = impl_->core.configure(impl_->model, impl_->field_left, impl_->field_right,
                                          options.room_impulse_slots);
    if (!status.ok()) {
        return status;
    }
    // The analysis bank runs sixteen sources, the synthesis bank two output ears.
    impl_->analysis_bank = RosellaBank::create(options.bank, kSourceChannels);
    impl_->synthesis_bank = RosellaBank::create(options.bank, kOutputChannels);
    impl_->buffer.assign(impl_->chunk_samples() * kSourceChannels, 0.0);
    impl_->room_sends_buffer_.assign(
        (impl_->chunk_samples() / kQmfHop) * kHybridBands, Complex(0.0, 0.0));
    impl_->buffer_used = 0;
    impl_->output.clear();
    impl_->input_samples = 0u;
    impl_->processed_input_samples = 0u;
    impl_->raw_output_samples = 0u;
    impl_->metadata_block_updates = 0u;
    impl_->finished = false;
    impl_->open = true;
    return Status::success();
}

Status RosellaRuntime::submit_frame(const float* objects16_planar, const oamd::OamdUpdate* update,
                                    std::int64_t frame_index, std::int64_t outer_sample_offset,
                                    std::int64_t object_delay_samples) {
    // The timeline is addressed by the number of samples submitted so far, exactly as
    // the reference addresses it, rather than by the caller's frame index.
    (void)frame_index;
    if (!impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is not open");
    }
    if (impl_->finished) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is already finished");
    }
    if (objects16_planar == nullptr) {
        return render_fail(JOC_ERR_INVALID_ARGUMENT, "null frame");
    }
    if (update != nullptr) {
        const Status submitted = impl_->timeline.submit_update(
            *update, static_cast<std::int64_t>(impl_->input_samples), outer_sample_offset,
            object_delay_samples, static_cast<std::int64_t>(impl_->processed_input_samples));
        if (!submitted.ok()) {
            return submitted;
        }
        ++impl_->metadata_block_updates;
    }
    const std::size_t chunk = impl_->chunk_samples();
    std::size_t offset = 0u;
    while (offset < kFrameSamples) {
        const std::size_t room = chunk - impl_->buffer_used;
        const std::size_t count = std::min<std::size_t>(room, kFrameSamples - offset);
        for (std::size_t sample = 0u; sample < count; ++sample) {
            double* row = impl_->buffer.data() + (impl_->buffer_used + sample) * kSourceChannels;
            for (std::size_t channel = 0u; channel < kSourceChannels; ++channel) {
                row[channel] = static_cast<double>(
                    objects16_planar[channel * kFrameSamples + offset + sample]);
            }
        }
        impl_->buffer_used += count;
        offset += count;
        if (impl_->buffer_used == chunk) {
            const Status status = impl_->process_samples(impl_->buffer.data(), chunk);
            if (!status.ok()) {
                return status;
            }
            impl_->buffer_used = 0u;
        }
    }
    impl_->input_samples += kFrameSamples;
    return Status::success();
}

Status RosellaRuntime::finish(std::uint32_t flush_samples, std::vector<double>* out) {
    if (!impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is not open");
    }
    if (out != nullptr) {
        out->clear();
    }
    if (impl_->finished) {
        return Status::success();
    }
    // Only the flush tail is returned: the caller has already taken the program.
    const std::size_t program_end = impl_->output.size();
    if (impl_->buffer_used != 0u) {
        const Status status = impl_->process_samples(impl_->buffer.data(), impl_->buffer_used);
        if (!status.ok()) {
            return status;
        }
        impl_->buffer_used = 0u;
    }
    const std::size_t chunk = impl_->chunk_samples();
    std::vector<double> silence;
    std::uint32_t remaining = flush_samples;
    while (remaining > 0u) {
        const std::size_t count = std::min<std::size_t>(remaining, chunk);
        silence.assign(count * kSourceChannels, 0.0);
        const Status status = impl_->process_samples(silence.data(), count);
        if (!status.ok()) {
            return status;
        }
        remaining -= static_cast<std::uint32_t>(count);
    }
    impl_->finished = true;
    if (out != nullptr) {
        out->assign(impl_->output.begin() + static_cast<std::ptrdiff_t>(program_end),
                    impl_->output.end());
    }
    return Status::success();
}

std::uint32_t RosellaRuntime::finish_capacity(double tail_seconds) const {
    const double samples = tail_seconds * 48000.0 +
                           static_cast<double>(rosella_bank_latency(impl_->options.bank)) +
                           static_cast<double>(kBlockSamples);
    const double blocks = std::ceil(samples / static_cast<double>(kBlockSamples));
    return static_cast<std::uint32_t>(blocks) * kBlockSamples;
}

Status RosellaRuntime::reset() {
    if (!impl_->open) {
        return render_fail(JOC_ERR_STATE, "Rosella runtime is not open");
    }
    impl_->core.reset();
    impl_->analysis_bank->reset();
    impl_->synthesis_bank->reset();
    impl_->buffer_used = 0u;
    impl_->output.clear();
    impl_->input_samples = 0u;
    impl_->processed_input_samples = 0u;
    impl_->raw_output_samples = 0u;
    impl_->metadata_block_updates = 0u;
    impl_->finished = false;
    impl_->timeline.reset();
    return Status::success();
}

const std::vector<double>& RosellaRuntime::output() const { return impl_->output; }

void RosellaRuntime::take_output(std::vector<double>* out) {
    if (out == nullptr) {
        return;
    }
    // Replace semantics, exactly like SofaBinauralRuntime::take_output: the caller's
    // previous contents are discarded, and the internal buffer is emptied afterwards
    // because the swap leaves it holding whatever the caller had.
    out->swap(impl_->output);
    impl_->output.clear();
}

std::uint64_t RosellaRuntime::input_samples() const { return impl_->input_samples; }
std::uint64_t RosellaRuntime::processed_input_samples() const {
    return impl_->processed_input_samples;
}
std::uint64_t RosellaRuntime::metadata_block_updates() const {
    return impl_->metadata_block_updates;
}
const timeline::OamdPositionTimeline& RosellaRuntime::timeline() const { return impl_->timeline; }

void RosellaRuntime::set_fixed_position(int object_index, const double position[3]) {
    impl_->timeline.set_fixed_position(object_index, position);
}

}  // namespace joc::hrtf
