#include "stream/stream.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "adm/adm_metadata.h"
#include "foundation/status.h"
#include "hrtf/jochrtf.h"
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
    objects16_.clear();
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
        hrtf::Field field;
        hrtf::Kernels kernels;
        Status status = hrtf::load_jochrtf(config_.hrtf_path, &field);
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
    return process_ready_frames();
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
    return process_ready_frames();
}

Status Stream::push_objects16(const float* planar16, std::size_t samples, std::size_t* consumed) {
    if (consumed != nullptr) {
        *consumed = samples;
    }
    if (planar16 == nullptr || samples == 0u) {
        return Status::success();
    }
    // Rendered immediately: the host has already done the JOC rebuild.
    for (std::size_t offset = 0; offset < samples; offset += kFrameSamples) {
        const std::size_t count = std::min(kFrameSamples, samples - offset);
        std::vector<float> frame(static_cast<std::size_t>(JOC_OUTPUT_CHANNELS) * kFrameSamples,
                                 0.0f);
        for (std::size_t channel = 0; channel < JOC_OUTPUT_CHANNELS; ++channel) {
            std::memcpy(frame.data() + channel * kFrameSamples,
                        planar16 + channel * samples + offset, count * sizeof(float));
        }
        ++info_.frames_in;
        info_.samples_in += count;
        const Status rendered = render_objects16(frame);
        if (!rendered.ok()) {
            return rendered;
        }
        if (count != kFrameSamples) {
            break;  // a partial frame is dropped; the host should push whole frames
        }
    }
    return Status::success();
}

Status Stream::process_ready_frames() {
    while (bed_pending_.size() / kBedChannels >= kFrameSamples && !metadata_.empty()) {
        const FrameMetadata entry = metadata_.front();
        metadata_.pop_front();

        std::vector<float> bed5(static_cast<std::size_t>(JOC_CORE_CHANNELS) * kFrameSamples, 0.0f);
        std::vector<float> lfe(kFrameSamples, 0.0f);
        for (std::size_t sample = 0; sample < kFrameSamples; ++sample) {
            for (std::size_t channel = 0; channel < JOC_CORE_CHANNELS; ++channel) {
                bed5[channel * kFrameSamples + sample] =
                    bed_pending_[sample * kBedChannels + kCoreChannels[channel]];
            }
            lfe[sample] = bed_pending_[sample * kBedChannels + kLfeChannel];
        }
        bed_pending_.erase(bed_pending_.begin(),
                           bed_pending_.begin() + static_cast<std::ptrdiff_t>(kFrameSamples *
                                                                             kBedChannels));

        std::string error;
        const Status rebuilt = joc::rebuild_objects16(rebuilder_, entry.params, bed5.data(),
                                                      lfe.data(), gain_, &objects16_, &error);
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
        for (const double value : speaker_.output) {
            output_.push_back(static_cast<float>(value));
        }
        info_.frames_out++;
        info_.samples_out += kFrameSamples;
        return Status::success();
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
    std::vector<double> produced;
    binaural_.take_output(&produced);
    for (const double value : produced) {
        output_.push_back(static_cast<float>(value));
    }
    info_.frames_out++;
    info_.samples_out += produced.size() / 2u;
    return Status::success();
}

Status Stream::pull(float* destination, std::size_t capacity_samples, std::size_t* produced) {
    if (produced != nullptr) {
        *produced = 0;
    }
    if (destination == nullptr || produced == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput, "null pull buffer");
    }
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
    if (binaural_ready_) {
        std::vector<double> tail;
        const Status drained =
            binaural_.finish(binaural_.finish_capacity(config_.tail_seconds), &tail);
        if (!drained.ok()) {
            return drained;
        }
        for (const double value : tail) {
            output_.push_back(static_cast<float>(value));
        }
        info_.samples_out += tail.size() / 2u;
    }
    info_.ended = 1;
    return Status::success();
}

Status Stream::reset() {
    if (rebuilder_ == nullptr && !speaker_enabled_ && !binaural_ready_ &&
        config_.input != JOC_STREAM_IN_PCM_OBJECTS16) {
        return Status::fail(JOC_ERR_STATE, stage::kRender, "stream is not created");
    }
    reset_state();
    return Status::success();
}

}  // namespace joc::stream
