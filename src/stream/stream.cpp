#include "stream/stream.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "adm/adm_metadata.h"
#include "foundation/status.h"
#include "hrtf/jochrtf.h"
#include "hrtf/rosella_model.h"
#include "hrtf/sofa_cache.h"
#include "joc_bitstream/joc_parser.h"
#include "joc_core/objects16.h"

namespace joc::stream {

namespace {

constexpr std::size_t kFrameSamples = JOC_FRAME_SAMPLES;
constexpr std::size_t kBedChannels = 6;
constexpr int kCoreChannels[5] = {0, 1, 2, 4, 5};
constexpr int kLfeChannel = 3;

}  // namespace

Stream::~Stream() {
    if (rebuilder_ != nullptr) {
        ejoc_renderer_destroy(rebuilder_);
        rebuilder_ = nullptr;
    }
    if (speaker_.handle != nullptr) {
        ejoc_speaker_renderer_destroy(speaker_.handle);
        speaker_.handle = nullptr;
    }
}

void Stream::reset_state() {
    reader_ = eac3::FrameReader();
    metadata_.clear();
    bed_pending_.clear();
    bed_read_offset_ = 0;
    objects16_.clear();
    objects_pending_.clear();
    objects_read_offset_ = 0;
    output_.clear();
    read_offset_ = 0;
    info_ = Info();
    info_.output_channels = output_channels_;
    if (rebuilder_ != nullptr) {
        ejoc_renderer_reset(rebuilder_);
    }
    if (speaker_.handle != nullptr) {
        ejoc_speaker_renderer_reset(speaker_.handle);
        speaker_.state.reset();
        speaker_.output.clear();
        speaker_.last_had_payload = false;
    }
    if (binaural_ready_) {
        binaural_.reset();
    }
    if (rosella_ready_) {
        rosella_.reset();
    }
    rosella_pending_.clear();
    rosella_read_offset_ = 0;
}

Status Stream::create(const Config& config) {
    if (rebuilder_ != nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "stream already created");
    }
    config_ = config;
    if (config_.input > JOC_STREAM_IN_PCM_OBJECTS16 ||
        config_.output > JOC_STREAM_OUT_BINAURAL) {
        return Status::fail(JOC_ERR_INVALID_CONFIG, stage::kRender, "unknown stream kind");
    }
    if (config_.input == JOC_STREAM_IN_PCM_OBJECTS16 &&
        config_.output == JOC_STREAM_OUT_PCM_OBJECTS16) {
        return Status::fail(JOC_ERR_INVALID_CONFIG, stage::kRender,
                            "objects16 input with objects16 output would do nothing");
    }
    gain_ = static_cast<float>(std::pow(10.0, config_.gain_db / 20.0));

    if (config_.input == JOC_STREAM_IN_EAC3) {
        rebuilder_ = ejoc_renderer_create();
        if (rebuilder_ == nullptr) {
            return Status::fail(JOC_ERR_OUT_OF_MEMORY, stage::kDsp, "cannot create the JOC kernel");
        }
        if (config_.native_threads != 0u) {
            ejoc_renderer_set_threads(rebuilder_, config_.native_threads);
        }
    }

    if (config_.output == JOC_STREAM_OUT_SPEAKER) {
        speaker_enabled_ = true;
        if (!speaker::layout_by_name(config_.layout.c_str(), &speaker_.layout)) {
            return Status::fail(JOC_ERR_LAYOUT_UNSUPPORTED, stage::kRender,
                                "unknown speaker layout: " + config_.layout);
        }
        speaker_.metadata_offset = config_.metadata_offset;
        speaker_.handle = ejoc_speaker_renderer_create(speaker_.layout.speaker_bitfield);
        if (speaker_.handle == nullptr) {
            return Status::fail(JOC_ERR_RENDER_FAILED, stage::kRender,
                                "cannot create the speaker renderer");
        }
        output_channels_ = speaker_.layout.channel_count;
    } else if (config_.output == JOC_STREAM_OUT_BINAURAL) {
        binaural_enabled_ = true;
        // The HRTF input precedence is joc_task_config's: a Rosella
        // .personalized_headphone wins over a SOFA that the library compiles, and
        // the compiled .jochrtf (hrtf_path) stays the fallback input.
        if (!config_.personalized_headphone_path.empty()) {
            hrtf::RosellaModel model;
            Status status =
                hrtf::load_personalized_headphone(config_.personalized_headphone_path, &model);
            hrtf::RosellaRenderOptions render_options;
            if (config_.binaural_mode == JOC_BINAURAL_NEAR) {
                render_options.profile = hrtf::RosellaProfile::Near;
            } else if (config_.binaural_mode == JOC_BINAURAL_FAR) {
                render_options.profile = hrtf::RosellaProfile::Far;
            }
            render_options.object_delay_samples = config_.object_delay_samples;
            render_options.tail_seconds = config_.tail_seconds;
            if (status.ok()) {
                status = rosella_.open(model, render_options);
            }
            if (!status.ok()) {
                return Status::fail(status.code(), stage::kRender,
                                    "binaural setup failed: " + status.message());
            }
            rosella_ready_ = true;
            output_channels_ = 2;
        } else {
            hrtf::Field field;
            hrtf::Kernels kernels;
            Status status = Status::success();
            if (!config_.hrtf_sofa_path.empty()) {
                // The .jochrtf is an internal cache: the SOFA is the user-facing input.
                hrtf::SofaFieldRequest request;
                request.sofa_path = config_.hrtf_sofa_path;
                request.options.shell_radius_m = config_.hrtf_radius_m;
                request.cache_dir = config_.hrtf_cache_dir;
                switch (config_.hrtf_cache_policy) {
                    case JOC_HRTF_CACHE_NONE: request.policy = hrtf::CachePolicy::None; break;
                    case JOC_HRTF_CACHE_DISK: request.policy = hrtf::CachePolicy::Disk; break;
                    default: request.policy = hrtf::CachePolicy::Memory; break;
                }
                std::string cache_path;
                status = hrtf::load_or_compile_sofa_field(request, &field, &cache_path);
                if (!status.ok()) {
                    return Status::fail(status.code(), stage::kRender,
                                        "binaural setup failed: " + status.message());
                }
            } else {
                status = hrtf::load_jochrtf(config_.hrtf_path, &field);
            }
            if (status.ok()) {
                status = config_.kernels_path.empty()
                             ? (kernels = hrtf::builtin_kernels(), Status::success())
                             : hrtf::load_kernels(config_.kernels_path, &kernels);
            }
            binaural::Profile profile = binaural::Profile::Mid;
            if (status.ok() && config_.binaural_mode == JOC_BINAURAL_NEAR) {
                profile = binaural::Profile::Near;
            } else if (status.ok() && config_.binaural_mode == JOC_BINAURAL_FAR) {
                profile = binaural::Profile::Far;
            }
            if (status.ok()) {
                status = binaural_.open(field, kernels, profile);
            }
            if (!status.ok()) {
                return status;
            }
            binaural_ready_ = true;
            output_channels_ = 2;
        }
    } else {
        output_channels_ = JOC_OUTPUT_CHANNELS;
    }
    reset_state();
    return Status::success();
}

Status Stream::push_eac3(const std::uint8_t* data, std::size_t size, std::size_t* consumed) {
    if (rebuilder_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kEmdf, "stream is not an E-AC-3 input");
    }
    if (consumed != nullptr) {
        *consumed = size;
    }
    if (data != nullptr && size != 0u) {
        reader_.push(data, size);
        info_.bytes_in += size;
    }
    for (;;) {
        eac3::Frame frame;
        const eac3::FrameReader::Next state = reader_.next(&frame);
        if (state == eac3::FrameReader::Next::End) {
            break;
        }
        if (state == eac3::FrameReader::Next::Fail) {
            return Status::fail(reader_.error(), stage::kEac3, reader_.error_message());
        }
        FrameMetadata entry;
        emdf::Container container;
        const Status parsed =
            joc::parse_eac3_frame(frame.data, frame.size, &entry.params, &container, nullptr);
        if (!parsed.ok()) {
            return parsed;
        }
        if (const emdf::Payload* payload = container.find(emdf::kIdOamd)) {
            std::vector<std::uint8_t> bytes;
            const Status extracted =
                emdf::extract_payload_bytes(frame.data, frame.size, *payload, &bytes);
            if (!extracted.ok()) {
                return extracted;
            }
            const Status oamd = oamd::parse_id11(bytes.data(), bytes.size(), &entry.update);
            if (!oamd.ok()) {
                return oamd;
            }
            entry.has_update = true;
            entry.outer_offset = static_cast<std::int64_t>(payload->sample_offset);
        }
        metadata_.push_back(entry);
    }
    return process_ready_frames(false);
}

Status Stream::push_bed(const float* interleaved6, std::size_t samples, std::size_t* consumed) {
    if (rebuilder_ == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kDsp, "stream is not an E-AC-3 input");
    }
    if (consumed != nullptr) {
        *consumed = samples;
    }
    if (interleaved6 != nullptr && samples != 0u) {
        bed_pending_.insert(bed_pending_.end(), interleaved6,
                            interleaved6 + samples * kBedChannels);
    }
    return process_ready_frames(false);
}

Status Stream::push_objects16(const float* planar16, std::size_t samples, std::size_t* consumed) {
    if (consumed != nullptr) {
        *consumed = samples;
    }
    if (planar16 == nullptr || samples == 0u) {
        return Status::success();
    }
    // Queued as whole frames, each planar [16][kFrameSamples]: the shape the
    // objects16 output writes, so a host can feed a batch back unchanged.
    constexpr std::size_t kFrameValues =
        static_cast<std::size_t>(JOC_OUTPUT_CHANNELS) * kFrameSamples;
    for (std::size_t offset = 0; offset < samples; offset += kFrameSamples) {
        const std::size_t count = std::min(kFrameSamples, samples - offset);
        const std::size_t base = objects_pending_.size();
        const float* frame = planar16 + (offset / kFrameSamples) * kFrameValues;
        objects_pending_.resize(base + kFrameValues, 0.0f);
        for (std::size_t channel = 0; channel < JOC_OUTPUT_CHANNELS; ++channel) {
            std::memcpy(objects_pending_.data() + base + channel * kFrameSamples,
                        frame + channel * kFrameSamples, count * sizeof(float));
        }
    }
    return process_objects16_frames(false);
}

Status Stream::process_objects16_frames(bool drain_all) {
    constexpr std::size_t kFrameValues =
        static_cast<std::size_t>(JOC_OUTPUT_CHANNELS) * kFrameSamples;
    while (objects_pending_.size() - objects_read_offset_ >= kFrameValues) {
        // Leave the rest queued, in order, for a later push or for flush().
        if (!drain_all && buffered_samples() >= kMaxRenderAheadSamples) {
            break;
        }
        const auto first = objects_pending_.begin() +
                           static_cast<std::ptrdiff_t>(objects_read_offset_);
        objects_frame_.assign(first, first + static_cast<std::ptrdiff_t>(kFrameValues));
        objects_read_offset_ += kFrameValues;
        const Status rendered = render_objects16(objects_frame_);
        if (!rendered.ok()) {
            return rendered;
        }
        ++info_.frames_in;
        info_.samples_in += kFrameSamples;
    }
    if (objects_read_offset_ == objects_pending_.size()) {
        objects_pending_.clear();
        objects_read_offset_ = 0;
    } else if (objects_read_offset_ >= (1u << 20)) {
        // Erasing from the front moves the remainder, so it is only worth doing
        // once the consumed prefix is large enough to pay for the move.
        objects_pending_.erase(
            objects_pending_.begin(),
            objects_pending_.begin() + static_cast<std::ptrdiff_t>(objects_read_offset_));
        objects_read_offset_ = 0;
    }
    return Status::success();
}

Status Stream::process_ready_frames(bool drain_all) {
    while (bed_pending_.size() - bed_read_offset_ >= kFrameSamples * kBedChannels &&
           !metadata_.empty()) {
        // Leave the rest queued, in order, for a later push or for flush().
        if (!drain_all && buffered_samples() >= kMaxRenderAheadSamples) {
            break;
        }
        const FrameMetadata entry = metadata_.front();
        metadata_.pop_front();

        const float* bed = bed_pending_.data() + bed_read_offset_;
        bed5_.resize(static_cast<std::size_t>(JOC_CORE_CHANNELS) * kFrameSamples);
        lfe_.resize(kFrameSamples);
        for (std::size_t sample = 0; sample < kFrameSamples; ++sample) {
            for (std::size_t channel = 0; channel < JOC_CORE_CHANNELS; ++channel) {
                bed5_[channel * kFrameSamples + sample] =
                    bed[sample * kBedChannels + kCoreChannels[channel]];
            }
            lfe_[sample] = bed[sample * kBedChannels + kLfeChannel];
        }
        bed_read_offset_ += kFrameSamples * kBedChannels;
        compact_bed_pending();

        std::string error;
        const Status rebuilt = joc::rebuild_objects16(rebuilder_, entry.params, bed5_.data(),
                                                      lfe_.data(), gain_, &objects16_, &error);
        if (!rebuilt.ok()) {
            return Status::fail(rebuilt.code(), stage::kDsp, error);
        }
        pending_metadata_ = entry;
        const Status rendered = render_objects16(objects16_);
        if (!rendered.ok()) {
            return rendered;
        }
        ++info_.frames_in;
        info_.samples_in += kFrameSamples;
    }
    return Status::success();
}

void Stream::compact_bed_pending() {
    if (bed_read_offset_ == 0) {
        return;
    }
    if (bed_read_offset_ == bed_pending_.size()) {
        bed_pending_.clear();
        bed_read_offset_ = 0;
    } else if (bed_read_offset_ >= (1u << 20)) {
        // Erasing from the front moves the remainder, so it is only worth doing
        // once the consumed prefix is large enough to pay for the move.
        bed_pending_.erase(bed_pending_.begin(),
                           bed_pending_.begin() + static_cast<std::ptrdiff_t>(bed_read_offset_));
        bed_read_offset_ = 0;
    }
}

Status Stream::render_objects16(const std::vector<float>& objects16) {
    if (config_.output == JOC_STREAM_OUT_PCM_OBJECTS16) {
        output_.insert(output_.end(), objects16.begin(), objects16.end());
        info_.frames_out++;
        info_.samples_out += kFrameSamples;
        return Status::success();
    }
    if (speaker_enabled_) {
        std::string error;
        const Status stepped =
            speaker::step(&speaker_, objects16, pending_metadata_.has_update
                                                   ? &pending_metadata_.update
                                                   : nullptr,
                          &error);
        if (!stepped.ok()) {
            return Status::fail(stepped.code(), stage::kRender, error);
        }
        output_.reserve(output_.size() + speaker_.output.size());
        for (const double value : speaker_.output) {
            output_.push_back(static_cast<float>(value));
        }
        info_.frames_out++;
        info_.samples_out += kFrameSamples;
        return Status::success();
    }
    if (rosella_ready_) {
        return render_rosella_objects16(objects16);
    }

    const Status submitted =
        binaural_.submit_frame(objects16.data(),
                               pending_metadata_.has_update ? &pending_metadata_.update : nullptr,
                               static_cast<std::int64_t>(info_.frames_out),
                               pending_metadata_.outer_offset,
                               static_cast<std::int64_t>(config_.object_delay_samples));
    if (!submitted.ok()) {
        return submitted;
    }
    binaural_.take_output(&produced_);
    output_.reserve(output_.size() + produced_.size());
    for (const double value : produced_) {
        output_.push_back(static_cast<float>(value));
    }
    info_.frames_out++;
    info_.samples_out += produced_.size() / 2u;
    return Status::success();
}

// The Rosella runtime is driven exactly like the SOFA runtime (the same frame,
// update, frame index, outer offset and object delay), but it renders in chunks
// of its own size (64 frames by default), so a frame usually yields either
// nothing or a whole chunk.  Its output therefore waits in a FIFO and is released
// one frame's worth at a time, which keeps the stream's contract intact: pushing
// one syncframe leaves exactly JOC_FRAME_SAMPLES samples for the caller to pull,
// and nothing is dropped or counted twice.  pull() and flush() release the rest.
Status Stream::render_rosella_objects16(const std::vector<float>& objects16) {
    const Status submitted =
        rosella_.submit_frame(objects16.data(),
                              pending_metadata_.has_update ? &pending_metadata_.update : nullptr,
                              static_cast<std::int64_t>(info_.frames_out),
                              pending_metadata_.outer_offset,
                              static_cast<std::int64_t>(config_.object_delay_samples));
    if (!submitted.ok()) {
        return submitted;
    }
    rosella_.take_output(&produced_);
    if (!produced_.empty()) {
        rosella_pending_.insert(rosella_pending_.end(), produced_.begin(), produced_.end());
    }
    release_rosella_output(kFrameSamples);
    info_.frames_out++;
    // Counted as the runtime produces it, which is also how the SOFA path counts:
    // the totals are identical, only the frame they appear on differs.
    info_.samples_out += produced_.size() / 2u;
    return Status::success();
}

void Stream::release_rosella_output(std::size_t limit) {
    if (!rosella_ready_ || limit == 0u) {
        return;
    }
    const std::size_t count = std::min(limit, rosella_pending_samples());
    if (count == 0u) {
        return;
    }
    const std::size_t values = count * 2u;
    output_.reserve(output_.size() + values);
    for (std::size_t index = 0; index < values; ++index) {
        output_.push_back(static_cast<float>(rosella_pending_[rosella_read_offset_ + index]));
    }
    rosella_read_offset_ += values;
    if (rosella_read_offset_ == rosella_pending_.size()) {
        rosella_pending_.clear();
        rosella_read_offset_ = 0;
    }
}

Status Stream::pull(float* destination, std::size_t capacity_samples, std::size_t* produced) {
    if (produced != nullptr) {
        *produced = 0;
    }
    if (destination == nullptr || produced == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput, "null pull buffer");
    }
    // Rendered Rosella samples that the frame-at-a-time release above has not
    // handed over yet are still the caller's to take; releasing them here keeps
    // buffered_samples() and the amount pull() can deliver the same number.
    release_rosella_output(capacity_samples);
    const std::size_t available = buffered_samples();
    const std::size_t count = std::min(capacity_samples, available);
    if (count != 0u) {
        std::memcpy(destination, output_.data() + read_offset_,
                    count * output_channels_ * sizeof(float));
        read_offset_ += count * output_channels_;
        if (read_offset_ == output_.size()) {
            output_.clear();
            read_offset_ = 0;
        } else if (read_offset_ > (1u << 20)) {
            output_.erase(output_.begin(),
                          output_.begin() + static_cast<std::ptrdiff_t>(read_offset_));
            read_offset_ = 0;
        }
    }
    *produced = static_cast<std::uint32_t>(count);
    return Status::success();
}

Status Stream::flush() {
    // Input has ended, so drain what the cap held back: nothing else will
    // trigger rendering.
    const Status remaining = process_ready_frames(true);
    if (!remaining.ok()) {
        return remaining;
    }
    const Status objects = process_objects16_frames(true);
    if (!objects.ok()) {
        return objects;
    }
    if (binaural_ready_) {
        std::vector<double> tail;
        const Status drained =
            binaural_.finish(binaural_.finish_capacity(config_.tail_seconds), &tail);
        if (!drained.ok()) {
            return drained;
        }
        output_.reserve(output_.size() + tail.size());
        for (const double value : tail) {
            output_.push_back(static_cast<float>(value));
        }
        info_.samples_out += tail.size() / 2u;
    }
    if (rosella_ready_) {
        std::vector<double> tail;
        const Status drained =
            rosella_.finish(rosella_.finish_capacity(config_.tail_seconds), &tail);
        if (!drained.ok()) {
            return drained;
        }
        // Everything the runtime produced as the program is released first: the
        // tail only sounds after it.  The program samples were already counted by
        // render_rosella_objects16, so only the tail is added here.
        release_rosella_output(rosella_pending_samples());
        output_.reserve(output_.size() + tail.size());
        for (const double value : tail) {
            output_.push_back(static_cast<float>(value));
        }
        info_.samples_out += tail.size() / 2u;
    }
    info_.ended = 1;
    return Status::success();
}

Status Stream::reset() {
    if (rebuilder_ == nullptr && !speaker_enabled_ && !binaural_ready_ && !rosella_ready_ &&
        config_.input != JOC_STREAM_IN_PCM_OBJECTS16) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "stream is not created");
    }
    reset_state();
    return Status::success();
}

}  // namespace joc::stream
