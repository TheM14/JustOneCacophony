#include "virtual_speaker/virtual_speaker.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "joc_core.h"

namespace joc::virtual_speaker {

namespace {

constexpr std::size_t kFrameSamples = JOC_FRAME_SAMPLES;
constexpr std::size_t kSourceChannels = JOC_OUTPUT_CHANNELS;

// WAV order follows the ffmpeg layouts: 5.1 = L R C LFE BL BR,
// 7.1 = L R C LFE BL BR SL SR.
constexpr Layout kLayouts[] = {
    {"2.0",
     2,
     {{false, -1.0, 1.0, 0.0},
      {false, 1.0, 1.0, 0.0}}},
    {"5.1",
     6,
     {{false, -1.0, 1.0, 0.0},
      {false, 1.0, 1.0, 0.0},
      {false, 0.0, 1.0, 0.0},
      {true, 0.0, 1.0, 0.0},
      {false, -1.0, -1.0, 0.0},
      {false, 1.0, -1.0, 0.0}}},
    {"7.1",
     8,
     {{false, -1.0, 1.0, 0.0},
      {false, 1.0, 1.0, 0.0},
      {false, 0.0, 1.0, 0.0},
      {true, 0.0, 1.0, 0.0},
      {false, -1.0, -1.0, 0.0},
      {false, 1.0, -1.0, 0.0},
      {false, -1.0, 0.0, 0.0},
      {false, 1.0, 0.0, 0.0}}},
};

std::int16_t q_for_axis(double value) {
    if (value <= -0.999) {
        return 0;
    }
    if (value >= 0.999) {
        return 32767;
    }
    return 16384;
}

void fill_fixed_positions(const Layout& layout, oamd::OamdUpdate* update) {
    update->object_count = static_cast<std::uint32_t>(kSourceChannels);
    update->block_offset_samples = 0;
    update->ramp_duration_samples = 0;
    int slot = 1;
    for (int channel = 0; channel < layout.channels; ++channel) {
        const ChannelSpec& spec = layout.channel_specs[channel];
        const int destination = spec.lfe ? 0 : slot++;
        update->slots[destination].q1 = q_for_axis(spec.x);
        update->slots[destination].q2 = q_for_axis(-spec.y);
        update->slots[destination].q3 = q_for_axis(spec.z);
    }
}

}  // namespace

const Layout* find_layout(const std::string& name) {
    for (const Layout& layout : kLayouts) {
        if (name == layout.name) {
            return &layout;
        }
    }
    return nullptr;
}

const Layout* detect_layout(int channels) {
    for (const Layout& layout : kLayouts) {
        if (layout.channels == channels) {
            return &layout;
        }
    }
    return nullptr;
}

Status Input::configure(const std::string& layout_name, int channels, double gain_db) {
    const Layout* layout =
        (layout_name.empty() || layout_name == "auto")
            ? detect_layout(channels)
            : find_layout(layout_name);
    if (layout == nullptr || layout->channels != channels) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "PCM layout does not match the channel count");
    }
    layout_ = layout;
    channels_ = channels;
    gain_ = static_cast<double>(std::pow(10.0, gain_db / 20.0));
    pending_.clear();
    read_offset_ = 0;
    fill_fixed_positions(*layout_, &fixed_update_);
    return Status::success();
}

void Input::reset() {
    pending_.clear();
    read_offset_ = 0;
}

Status Input::append(const float* interleaved, std::size_t frames) {
    if (interleaved == nullptr || frames == 0u) {
        return Status::success();
    }
    if (layout_ == nullptr || channels_ <= 0) {
        return Status::fail(JOC_ERR_STATE, stage::kRender,
                            "virtual speaker input is not configured");
    }
    pending_.insert(pending_.end(), interleaved,
                    interleaved + frames * static_cast<std::size_t>(channels_));
    return Status::success();
}

bool Input::frame_ready() const {
    return layout_ != nullptr &&
           pending_.size() - read_offset_ >=
               kFrameSamples * static_cast<std::size_t>(channels_);
}

Status Input::take_frame(std::vector<float>* objects16) {
    if (objects16 == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "objects16 output is null");
    }
    if (!frame_ready()) {
        return Status::fail(JOC_ERR_STATE, stage::kRender,
                            "no complete PCM frame is buffered");
    }
    objects16->assign(kSourceChannels * kFrameSamples, 0.0f);
    const float* source = pending_.data() + read_offset_;
    int slot = 1;
    for (int channel = 0; channel < layout_->channels; ++channel) {
        const int destination = layout_->channel_specs[channel].lfe ? 0 : slot++;
        for (std::size_t sample = 0; sample < kFrameSamples; ++sample) {
            (*objects16)[static_cast<std::size_t>(destination) * kFrameSamples + sample] =
                source[sample * static_cast<std::size_t>(channels_) + channel] *
                static_cast<float>(gain_);
        }
    }
    read_offset_ += kFrameSamples * static_cast<std::size_t>(channels_);
    if (read_offset_ == pending_.size()) {
        pending_.clear();
        read_offset_ = 0;
    } else if (read_offset_ >= (1u << 20)) {
        pending_.erase(pending_.begin(),
                       pending_.begin() + static_cast<std::ptrdiff_t>(read_offset_));
        read_offset_ = 0;
    }
    return Status::success();
}

}  // namespace joc::virtual_speaker
