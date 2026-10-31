
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "binaural/binaural_reference.h"
#include "foundation/status.h"
#include "hrtf/full_hrir_field.h"
#include "hrtf/partitioned_convolution.h"
#include "hrtf/rosella_bank.h"
#include "hrtf/rosella_reference.h"
#include "oamd/oamd_parser.h"
#include "timeline/position_timeline.h"

// SOFA binaural renderer with a configured reference environment, ported from
// the reference sofa_reference_backend.py: the direct sound is the interpolated
// full HRIR (delay-aligned, partitioned convolution), the room send goes through
// the reference model's room, and the two are summed after their own latency
// compensation.
namespace joc::binaural {

inline constexpr int kSofaReferenceSources = 16;
inline constexpr int kSofaReferenceBlock = 128;
inline constexpr int kSofaReferenceBatch = 1536;
inline constexpr double kSofaReferenceLfePosition[3] = {0.0, 1.0, 0.0};

struct SofaReferenceOptions {
    int mode_index = kProfileMid;
    bool mode_specified = false;
    std::string reference_config;  // empty selects the default configuration
    hrtf::HrirInterpolation interpolation = hrtf::HrirInterpolation::kLogMagnitudePhase;
    bool has_shell_radius = false;
    double shell_radius_m = 0.0;
    double output_gain = 1.0;
    std::int64_t object_delay_samples = 1473;
    double tail_seconds = 5.0;
};

class SofaReferenceRuntime {
public:
    SofaReferenceRuntime();
    ~SofaReferenceRuntime();
    SofaReferenceRuntime(const SofaReferenceRuntime&) = delete;
    SofaReferenceRuntime& operator=(const SofaReferenceRuntime&) = delete;

    Status open(const std::string& sofa_path, const SofaReferenceOptions& options);

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

    std::uint64_t input_samples() const { return input_samples_; }
    std::uint64_t processed_input_samples() const { return processed_input_samples_; }
    std::uint64_t parameter_updates() const { return parameter_updates_; }
    const timeline::OamdPositionTimeline& timeline() const { return timeline_; }

    void set_fixed_position(int object_index, const double position[3]) {
        timeline_.set_fixed_position(object_index, position);
    }

    // Reporting surface (mirrors the reference backend's info()).
    int latency_samples() const { return latency_samples_; }
    int recording_origin_samples() const { return origin_samples_; }
    double level_calibration() const { return level_calibration_; }
    const ReferenceEnvironment& environment() const { return environment_; }
    const hrtf::FullHrirField& field() const { return field_; }
    bool fixed_reference() const { return true; }

private:
    Status process_block();
    Status flush_batch(std::vector<double>* out);
    Status set_source(int source, const double position[3], double gain, bool special_lfe,
                      bool fade);
    Status spectrum_at(const ReferencePosition& state,
                       std::vector<std::complex<double>>* out);
    Status build_lfe_filter();
    void rebuild_latency();

    ReferenceEnvironment environment_;
    hrtf::FullHrirField field_;
    std::unique_ptr<hrtf::RosellaBank> analysis_bank_;
    std::unique_ptr<hrtf::RosellaBank> synthesis_bank_;
    std::unique_ptr<hrtf::RosellaRoomFir> room_;
    hrtf::MultiSourcePartitionedConvolver convolver_;

    std::vector<double> lfe_filter_;   // [2][fft_size]
    std::vector<double> sends_;        // [sources]
    std::vector<double> send_targets_;
    std::vector<ReferencePosition> distance_states_;
    std::vector<std::uint8_t> has_state_;
    double last_gain_[kSofaReferenceSources] = {};
    std::vector<double> last_position_;  // [sources][3]
    double last_key_[kSofaReferenceSources][5] = {};

    std::vector<std::complex<double>> pending_room_;   // [slots][77]
    std::vector<double> pending_direct_;               // [samples][2]
    std::size_t pending_slots_ = 0;
    std::size_t pending_direct_samples_ = 0;

    std::vector<double> hop_block_;     // [block][sources]
    std::vector<double> staging_;      // [frame][sources]
    std::vector<double> output_;
    std::vector<std::complex<double>> qmf_;      // [slots][sources][64]
    std::vector<std::complex<double>> hybrid_;   // [slots][sources][77]
    std::vector<std::complex<double>> room_in_;  // [bins][77]
    std::vector<std::complex<double>> room_out_; // [bins][2][77]
    std::vector<std::complex<double>> synth_;    // [bins][2][64]
    std::vector<double> synth_time_;             // [bins][2][64]
    std::vector<double> wet_;                    // [samples][2]
    std::vector<double> direct_;                 // [samples][2]

    // Reference-position spectrum cache (the reference keeps 128 entries).
    struct SpectrumEntry {
        double key[5] = {};
        std::vector<std::complex<double>> spectrum;
    };
    std::vector<SpectrumEntry> spectrum_cache_;

    timeline::OamdPositionTimeline timeline_;
    std::int64_t input_samples_ = 0;
    std::int64_t processed_input_samples_ = 0;
    std::uint64_t parameter_updates_ = 0;
    int origin_samples_ = 0;
    int latency_samples_ = 0;
    int implementation_latency_ = 0;
    int bank_latency_ = 0;
    int room_delay_samples_ = 0;
    int direct_delay_samples_ = 0;
    std::int64_t latency_remaining_ = 0;
    double level_calibration_ = 1.0;
    double output_gain_ = 1.0;
    std::size_t fft_size_ = 0;
    std::size_t bins_ = 0;
    std::vector<double> frequencies_;
    std::vector<double> room_delay_state_;
    std::vector<double> direct_delay_state_;
    bool empty_block_ = true;
};

}  // namespace joc::binaural
