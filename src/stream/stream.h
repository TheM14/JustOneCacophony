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
#include "joc_bitstream/joc_parser.h"
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
    std::uint64_t missing_joc_frames = 0;
    std::uint64_t objects_clamped_frames = 0;
    // Audio frames whose syncframe held more than one top-level JOC container and
    // were decoded from the single one that carries a complete joc() payload.
    std::uint64_t extra_container_frames = 0;
    // Audio frames whose extra-pair identity disagreed across layers (spec 11.2
    // checks 1 and 2).  The identity is metadata only, so these still render.
    std::uint64_t chanmap_pair_mismatch_frames = 0;
    std::uint64_t bed_mask_mismatch_frames = 0;
    std::uint32_t output_channels = 0;
    std::uint32_t ended = 0;
};

// One audio frame's metadata, queued while the matching core PCM arrives.
//
// One audio frame is the group of syncframes up to (and excluding) the first
// repeated (codec, strmtyp, substreamid) combination; the JOC payload usually
// lives in the dependent substream while the independent substream carries only
// the 5.1 core.  The L2 fields below feed the spec section 11.2 cross-layer
// checks.
struct FrameMetadata {
    joc_frame_params params{};
    oamd::OamdUpdate update{};
    bool has_update = false;
    std::int64_t outer_offset = 0;
    bool has_joc = false;
    eac3::Frame core_frame{};   // independent substream header (codec/strmtyp=0)
    bool has_core = false;
    bool has_dependent = false;
    bool has_chanmap = false;
    std::uint16_t chanmap = 0;
    std::size_t syncframe_offset = 0;
    std::size_t syncframe_size = 0;
    std::vector<eac3::Frame> streams;
    std::vector<oamd::BedAssignment> bed_assignments;
};

class Stream {
public:
    Stream() = default;
    ~Stream();

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Status create(const Config& config);
    Status push_eac3(const std::uint8_t* data, std::size_t size, std::size_t* consumed);
    Status push_bed(const float* interleaved, std::size_t samples, std::uint32_t channels,
                    std::size_t* consumed);
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

    // Cap on rendered samples that have not been pulled.  A push renders what it
    // makes ready, so a caller that feeds faster than it pulls renders ahead.
    static constexpr std::size_t kMaxRenderAheadSamples = 16384;

private:
    // `drain_all` ignores kMaxRenderAheadSamples and renders every ready frame.
    Status process_ready_frames(bool drain_all);
    Status render_objects16(const std::vector<float>& objects16);
    Status render_rosella_objects16(const std::vector<float>& objects16);
    // Moves at most `limit` rendered stereo samples per channel out of the FIFO
    // into output_, oldest sample first.
    void release_rosella_output(std::size_t limit);
    std::size_t rosella_pending_samples() const {
        return rosella_ready_ ? (rosella_pending_.size() - rosella_read_offset_) / 2u : 0u;
    }
    void reset_state();
    // 声明的 bed 布局只在首帧解析一次：bed 声道在 OAMD 里没有位置字段，方位由布局
    // 给出；流式路径和文件任务走同一套 SpeakerStep/双耳运行时，所以这里也要下发。
    void apply_bed_positions(const FrameMetadata& entry);
    // Drops the bed samples that have already been rendered, keeping the rest.
    void compact_bed_pending();
    // Closes the audio frame currently being assembled and queues it.
    Status flush_pending_frame();
    // Resolves the bed layout from the first queued frame that carries JOC;
    // returns false while it is still unknown and more input may arrive.
    bool resolve_bed_plan();
    // Renders the queued objects16 frames, bounded by kMaxRenderAheadSamples.
    Status process_objects16_frames(bool drain_all);

    Config config_;
    Info info_;
    eac3::FrameReader reader_;
    std::deque<FrameMetadata> metadata_;
    FrameMetadata pending_metadata_;
    FrameMetadata pending_frame_;
    joc::BedPlan bed_plan_;
    std::size_t bed_channels_ = 0;
    int config_idx_ = -1;
    int core_channels_ = 0;
    bool bed_plan_failed_ = false;
    Status bed_plan_error_;
    std::uint32_t bed_input_channels_ = 0;
    std::vector<float> bed_pending_;
    std::size_t bed_read_offset_ = 0;
    std::vector<float> bed5_;
    std::vector<float> lfe_;
    std::vector<float> objects_pending_;
    std::size_t objects_read_offset_ = 0;
    std::vector<float> objects_frame_;
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
    std::vector<double> produced_;
    bool speaker_enabled_ = false;
    bool binaural_enabled_ = false;
    bool binaural_ready_ = false;
    bool bed_positions_ready_ = false;
    bool rosella_ready_ = false;
    float gain_ = 1.0f;
};

}  // namespace joc::stream
