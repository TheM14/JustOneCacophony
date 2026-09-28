#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "binaural/binaural_runtime.h"
#include "eac3_transport/eac3_reader.h"
#include "foundation/status.h"
#include "hrtf/rosella_renderer.h"
#include "joc_core.h"
#include "joc_stream.h"
#include "oamd/oamd_parser.h"
#include "speaker/speaker_step.h"

namespace joc::stream {

struct Config {
    std::uint32_t input = JOC_STREAM_IN_EAC3;
    std::uint32_t output = JOC_STREAM_OUT_PCM_OBJECTS16;
    std::string layout;
    std::uint32_t metadata_offset = 1473;
    std::uint32_t binaural_mode = JOC_BINAURAL_MID;
    std::string hrtf_path;
    std::string kernels_path;
    double tail_seconds = 5.0;
    std::uint32_t object_delay_samples = 1473;
    double gain_db = 0.0;
    std::uint32_t native_threads = 0;
    // The three HRTF shapes of joc_task_config, in its precedence order: a Rosella
    // model wins over a SOFA, and hrtf_path (.jochrtf) is the fallback.
    std::string hrtf_sofa_path;
    std::string personalized_headphone_path;
    std::string hrtf_cache_dir;
    std::uint32_t hrtf_cache_policy = JOC_HRTF_CACHE_MEMORY;
    double hrtf_radius_m = 1.0;
};

struct Info {
    std::uint64_t frames_in = 0;
    std::uint64_t frames_out = 0;
    std::uint64_t samples_in = 0;
    std::uint64_t samples_out = 0;
    std::uint64_t bytes_in = 0;
    std::uint64_t oamd_payloads = 0;
    std::uint64_t oamd_transitions = 0;
    std::uint32_t output_channels = 0;
    std::uint32_t ended = 0;
};

// One frame's metadata, queued while the matching core PCM arrives.
struct FrameMetadata {
    joc_frame_params params{};
    oamd::OamdUpdate update{};
    bool has_update = false;
    std::int64_t outer_offset = 0;
};

class Stream {
public:
    Stream() = default;
    ~Stream();

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Status create(const Config& config);
    Status push_eac3(const std::uint8_t* data, std::size_t size, std::size_t* consumed);
    Status push_bed(const float* interleaved6, std::size_t samples, std::size_t* consumed);
    Status push_objects16(const float* planar16, std::size_t samples, std::size_t* consumed);
    Status pull(float* destination, std::size_t capacity_samples, std::size_t* produced);
    Status flush();
    Status reset();

    const Info& info() const { return info_; }
    // Per-channel sample count, not the interleaved float count.  The Rosella
    // runtime renders in its own chunk size, so its output waits in a FIFO before
    // it is released one frame at a time; those samples are rendered and unpulled
    // as well, so the reported backlog has to include them.
    std::size_t buffered_samples() const {
        return output_channels_ != 0u
                   ? (output_.size() - read_offset_) / output_channels_ + rosella_pending_samples()
                   : 0u;
    }

private:
    Status process_ready_frames();
    Status render_objects16(const std::vector<float>& objects16);
    Status render_rosella_objects16(const std::vector<float>& objects16);
    // Moves at most `limit` rendered stereo samples per channel out of the FIFO
    // into output_, oldest sample first.
    void release_rosella_output(std::size_t limit);
    std::size_t rosella_pending_samples() const {
        return rosella_ready_ ? (rosella_pending_.size() - rosella_read_offset_) / 2u : 0u;
    }
    void reset_state();

    Config config_;
    Info info_;
    eac3::FrameReader reader_;
    std::deque<FrameMetadata> metadata_;
    FrameMetadata pending_metadata_;
    std::vector<float> bed_pending_;
    std::vector<std::uint8_t> frame_copy_;
    std::vector<float> objects16_;
    std::vector<float> output_;
    std::size_t read_offset_ = 0;
    std::uint32_t output_channels_ = 0;

    ejoc_renderer_handle rebuilder_ = nullptr;
    speaker::SpeakerStep speaker_;
    binaural::SofaBinauralRuntime binaural_;
    hrtf::RosellaRuntime rosella_;
    std::vector<double> rosella_pending_;
    std::size_t rosella_read_offset_ = 0;
    bool speaker_enabled_ = false;
    bool binaural_enabled_ = false;
    bool binaural_ready_ = false;
    bool rosella_ready_ = false;
    float gain_ = 1.0f;
};

}  // namespace joc::stream
