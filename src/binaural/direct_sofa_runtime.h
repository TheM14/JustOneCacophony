#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "binaural/point_source.h"
#include "binaural/sofa_room.h"
#include "foundation/status.h"
#include "hrtf/full_hrir_field.h"
#include "hrtf/partitioned_convolution.h"
#include "oamd/oamd_parser.h"
#include "timeline/position_timeline.h"

// Direct SOFA renderer: full HRIR per source, point-source coordinates, and the
// optional project-owned room.  Ported from the reference src/sofa_direct_backend.py;
// there is no distance-dependent HRTF adaptation and no external reference model.
namespace joc::binaural {

inline constexpr int kDirectSofaSources = 16;
inline constexpr std::size_t kDirectSofaBlock = 128;
inline constexpr int kDirectSofaFrame = 1536;
inline constexpr double kDirectSofaLfePosition[3] = {0.0, 1.0, 0.0};

struct DirectSofaOptions {
    std::string mode;  // near/mid/far; empty selects mid
    bool has_reference_distance = false;
    double reference_distance_m = 0.0;
    hrtf::HrirInterpolation interpolation = hrtf::HrirInterpolation::kLogMagnitudePhase;
    bool has_shell_radius = false;
    double shell_radius_m = 0.0;
    double output_gain = 1.0;
    // Optional project room; `room_preset` is compact/legacy (empty selects
    // legacy for far and compact otherwise).
    bool sofa_room = false;
    std::string room_preset;
    std::int64_t object_delay_samples = 1473;
    double tail_seconds = 5.0;
};

class DirectSofaRuntime {
public:
    DirectSofaRuntime();
    ~DirectSofaRuntime();
    DirectSofaRuntime(const DirectSofaRuntime&) = delete;
    DirectSofaRuntime& operator=(const DirectSofaRuntime&) = delete;

    Status open(const std::string& sofa_path, const DirectSofaOptions& options);

    // objects16_planar is channel-major: channel * 1536 + sample.
    Status submit_frame(const float* objects16_planar, const oamd::OamdUpdate* update,
                        std::int64_t frame_index, std::int64_t outer_sample_offset,
                        std::int64_t object_delay_samples);

    Status finish(std::uint32_t flush_samples, std::vector<double>* out);
    std::uint32_t finish_capacity(double tail_seconds) const;
    Status reset();

    const std::vector<double>& output() const { return output_; }
    void take_output(std::vector<double>* out) {
        out->swap(output_);
        output_.clear();
    }

    void set_fixed_position(int object_index, const double position[3]) {
        timeline_.set_fixed_position(object_index, position);
    }

    const timeline::OamdPositionTimeline& timeline() const { return timeline_; }
    std::uint64_t input_samples() const { return input_samples_; }
    std::uint64_t processed_input_samples() const { return processed_input_samples_; }
    std::uint64_t parameter_updates() const { return parameter_updates_; }
    int latency_samples() const { return latency_samples_; }
    int recording_origin_samples() const { return origin_samples_; }
    std::size_t filter_taps() const { return filter_taps_; }
    bool room_active() const { return room_.active(); }
    const std::string& room_preset() const { return room_.preset(); }
    const hrtf::FullHrirField& field() const { return field_; }
    const PointSourceReference& reference() const { return reference_; }

private:
    struct PendingFilter {
        std::int64_t due = 0;
        std::size_t source = 0;
        std::vector<double> filters;
        std::size_t taps = 0;
        std::uint32_t fade_blocks = 0;
    };

    Status set_source(int source, const double position[3], double gain, bool special_lfe,
                      bool fade);
    void schedule_filter(std::size_t source, const std::vector<double>& filters,
                         std::size_t taps, bool first, bool fade);
    void apply_due_filters();
    Status process_block();
    void emit(const double* interleaved, std::size_t samples);
    Status build_lfe_filter();

    PointSourceReference reference_;
    hrtf::FullHrirField field_;
    SofaRoom room_;
    hrtf::MultiSourcePartitionedConvolver convolver_;

    std::vector<double> lfe_filter_;  // [2][fft_size]
    std::deque<PendingFilter> pending_filters_;

    std::vector<PointSourcePosition> distance_states_;
    std::vector<std::uint8_t> has_state_;
    double last_key_[kDirectSofaSources][5] = {};

    std::vector<double> hop_block_;  // [128][sources]
    std::vector<double> staging_;    // [1536][sources]
    std::vector<double> direct_;     // [128][2]
    std::vector<double> room_out_;   // [1536][2] when the room produced a batch
    std::vector<double> output_;

    timeline::OamdPositionTimeline timeline_;
    std::int64_t input_samples_ = 0;
    std::int64_t processed_input_samples_ = 0;
    std::uint64_t parameter_updates_ = 0;
    std::uint64_t output_samples_ = 0;
    int origin_samples_ = 0;
    int latency_samples_ = 0;
    std::size_t filter_taps_ = 0;
    std::size_t fft_size_ = 0;
    std::size_t bins_ = 0;
    std::int64_t latency_remaining_ = 0;
    double output_gain_ = 1.0;
};

}  // namespace joc::binaural
