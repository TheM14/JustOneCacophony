#include "binaural/binaural_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace joc::binaural {

namespace {

double max_delay_bound(const hrtf::Field& field) {
    double maximum = 0.0;
    for (const double value : field.delay_bounds) {
        maximum = std::max(maximum, value);
    }
    return maximum;
}

}  // namespace

bool profile_from_name(const char* name, Profile* out) {
    if (name == nullptr || out == nullptr) {
        return false;
    }
    if (std::strcmp(name, "near") == 0) { *out = Profile::Near; return true; }
    if (std::strcmp(name, "mid") == 0) { *out = Profile::Mid; return true; }
    if (std::strcmp(name, "far") == 0) { *out = Profile::Far; return true; }
    return false;
}

SofaBinauralRuntime::~SofaBinauralRuntime() {
    if (handle_ != nullptr) {
        ejoc_sofa_binaural_destroy(handle_);
        handle_ = nullptr;
    }
}

Status SofaBinauralRuntime::open(const hrtf::Field& field, const hrtf::Kernels& kernels,
                                Profile profile, const RoomConstants& room) {
    if (handle_ != nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "binaural runtime already open");
    }
    if (field.coefficients.size() != static_cast<std::size_t>(hrtf::kShTerms * hrtf::kEars *
                                                             hrtf::kHybridBands * 2) ||
        field.band_centers_hz.size() != static_cast<std::size_t>(hrtf::kHybridBands)) {
        return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                            "compiled HRTF field has unexpected array sizes");
    }
    handle_ = ejoc_sofa_binaural_create();
    if (handle_ == nullptr) {
        return Status::fail(JOC_ERR_OUT_OF_MEMORY, stage::kRender,
                            "ejoc_sofa_binaural_create failed");
    }
    profile_ = profile;

    if (ejoc_sofa_binaural_configure_kernels(
            handle_, kernels.qmf_analysis.data(), kernels.hybrid_low.data(),
            kernels.hybrid_indices.data(), kernels.hybrid_values.data(), kernels.hybrid_count,
            kernels.qmf_basis.data(), kernels.qmf_taps.data()) != 0) {
        const char* message = ejoc_sofa_binaural_last_error(handle_);
        return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                            std::string("configure_kernels failed: ") +
                                (message != nullptr ? message : "unknown"));
    }
    if (ejoc_sofa_binaural_configure_field(handle_, field.coefficients.data(),
                                          field.delay_coefficients.data(),
                                          field.delay_bounds.data(), field.band_centers_hz.data(),
                                          field.measurement_radius_m) != 0) {
        const char* message = ejoc_sofa_binaural_last_error(handle_);
        return Status::fail(JOC_ERR_HRTF_FORMAT, stage::kRender,
                            std::string("configure_field failed: ") +
                                (message != nullptr ? message : "unknown"));
    }
    if (ejoc_sofa_binaural_configure_room(
            handle_, room.dims, room.listener, room.walls, room.speed_of_sound, room.fdn_delays,
            room.fdn_feedback, room.damping, room.fdn_output_gain, room.allpass_delays,
            room.allpass_gains, room.enable_early_reflections, room.enable_late_room) != 0) {
        const char* message = ejoc_sofa_binaural_last_error(handle_);
        return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender,
                            std::string("configure_room failed: ") +
                                (message != nullptr ? message : "unknown"));
    }

    maximum_hrtf_delay_ =
        static_cast<std::int64_t>(std::ceil(max_delay_bound(field) - 1e-9));
    hrtf_history_slots_ = static_cast<std::uint32_t>(std::max<std::int64_t>(1, (maximum_hrtf_delay_ + 63) / 64));
    staging_.assign(kBlockSamples * kSourceCount, 0.0);
    block_output_.assign(kBlockSamples * 2u, 0.0);
    output_.clear();
    staged_ = 0;
    input_samples_ = 0;
    processed_samples_ = 0;
    blocks_processed_ = 0;
    return Status::success();
}

Status SofaBinauralRuntime::process_block() {
    double positions[timeline::kTimelineObjects][3] = {};
    const Status queried = timeline_.positions_at(static_cast<std::int64_t>(processed_samples_),
                                                  positions);
    if (!queried.ok()) {
        return queried;
    }
    // The reference adapter calls set_source without a `fade` argument, so the
    // backend default (fade enabled) applies - the per-object path crossfade is
    // part of the reference behaviour, not an optional extra.
    constexpr std::uint32_t kFade = 1u;
    if (ejoc_sofa_binaural_set_source(handle_, 0u, kLfePosition,
                                      static_cast<std::uint32_t>(profile_), 1.0, 1u, 1u,
                                      kFade) != 0) {
        const char* message = ejoc_sofa_binaural_last_error(handle_);
        return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender,
                            std::string("set_source(LFE) failed: ") +
                                (message != nullptr ? message : "unknown"));
    }
    for (std::uint32_t source = 0; source < timeline::kTimelineObjects; ++source) {
        if (ejoc_sofa_binaural_set_source(handle_, source + 1u, positions[source],
                                          static_cast<std::uint32_t>(profile_), 1.0, 1u, 0u,
                                          kFade) != 0) {
            const char* message = ejoc_sofa_binaural_last_error(handle_);
            return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender,
                                std::string("set_source(object ") + std::to_string(source + 1u) +
                                    ") failed: " + (message != nullptr ? message : "unknown"));
        }
    }
    const int trimmed = ejoc_sofa_binaural_process(handle_, staging_.data(), kBlockSamples, 1.0,
                                                   block_output_.data());
    if (trimmed < 0) {
        const char* message = ejoc_sofa_binaural_last_error(handle_);
        return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender,
                            std::string("sofa process failed: ") +
                                (message != nullptr ? message : "unknown"));
    }
    if (trimmed > 0) {
        output_.insert(output_.end(), block_output_.begin(),
                       block_output_.begin() + static_cast<std::ptrdiff_t>(trimmed) * 2);
    }
    staged_ = 0;
    processed_samples_ += kBlockSamples;
    ++blocks_processed_;
    return Status::success();
}

Status SofaBinauralRuntime::submit_frame(const float* objects16_planar,
                                        const oamd::OamdUpdate* update, std::int64_t frame_index,
                                        std::int64_t outer_sample_offset,
                                        std::int64_t object_delay_samples) {
    if (handle_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "binaural runtime is not open");
    }
    if (objects16_planar == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender, "null frame");
    }
    // A frame without an ID11 payload submits nothing at all (the reference only
    if (update != nullptr) {
        const Status submitted = timeline_.submit_update(
            *update, frame_index * JOC_FRAME_SAMPLES, outer_sample_offset, object_delay_samples,
            static_cast<std::int64_t>(input_samples_));
        if (!submitted.ok()) {
            return submitted;
        }
    }

    std::size_t offset = 0;
    while (offset < JOC_FRAME_SAMPLES) {
        const std::size_t room = kBlockSamples - staged_;
        const std::size_t count = std::min<std::size_t>(room, JOC_FRAME_SAMPLES - offset);
        for (std::size_t sample = 0; sample < count; ++sample) {
            double* row = staging_.data() + (staged_ + sample) * kSourceCount;
            for (std::size_t channel = 0; channel < kSourceCount; ++channel) {
                row[channel] = static_cast<double>(
                    objects16_planar[channel * JOC_FRAME_SAMPLES + offset + sample]);
            }
        }
        staged_ += count;
        offset += count;
        if (staged_ == kBlockSamples) {
            const Status status = process_block();
            if (!status.ok()) {
                return status;
            }
        }
    }
    input_samples_ += JOC_FRAME_SAMPLES;
    return Status::success();
}

std::uint32_t SofaBinauralRuntime::finish_capacity(double tail_seconds) const {
    std::int64_t requested = tail_samples_;
    if (tail_seconds >= 0.0) {
        requested = static_cast<std::int64_t>(std::ceil(tail_seconds * 48000.0 - 1e-9));
    }
    const std::int64_t hrtf_bound = static_cast<std::int64_t>(hrtf_history_slots_) * 64;
    const std::int64_t early_bound = hrtf_bound + 2048 + 256 * 64;
    std::int64_t drain = std::max(requested, early_bound) + 961;
    drain = ((drain + 63) / 64) * 64;
    return static_cast<std::uint32_t>(drain);
}

Status SofaBinauralRuntime::reset() {
    if (handle_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "binaural runtime is not open");
    }
    if (ejoc_sofa_binaural_reset(handle_) != 0) {
        return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender, "sofa reset failed");
    }
    timeline_ = timeline::OamdPositionTimeline();
    output_.clear();
    staged_ = 0;
    input_samples_ = 0;
    processed_samples_ = 0;
    blocks_processed_ = 0;
    return Status::success();
}

Status SofaBinauralRuntime::finish(std::uint32_t flush_samples, std::vector<double>* out) {
    if (handle_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "binaural runtime is not open");
    }
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender, "null output");
    }
    out->clear();
    if (flush_samples == 0u) {
        return Status::success();
    }
    std::vector<double> chunk(static_cast<std::size_t>(kBlockSamples) * 2u, 0.0);
    std::uint32_t produced_total = 0;
    std::uint32_t remaining = flush_samples;
    while (remaining > 0) {
        const std::uint32_t request = std::min<std::uint32_t>(remaining, kBlockSamples);
        const int produced = ejoc_sofa_binaural_finish(handle_, request, chunk.data(), request);
        if (produced < 0) {
            const char* message = ejoc_sofa_binaural_last_error(handle_);
            return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender,
                                std::string("sofa finish failed: ") +
                                    (message != nullptr ? message : "unknown"));
        }
        if (produced == 0) {
            break;
        }
        out->insert(out->end(), chunk.begin(),
                    chunk.begin() + static_cast<std::ptrdiff_t>(produced) * 2);
        produced_total += static_cast<std::uint32_t>(produced);
        remaining -= std::min<std::uint32_t>(remaining, static_cast<std::uint32_t>(produced));
    }
    return Status::success();
}

}  // namespace joc::binaural
