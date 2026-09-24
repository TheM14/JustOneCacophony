
#pragma once

#include <cstdint>
#include <vector>

#include "eac3joc_core.h"
#include "foundation/status.h"
#include "hrtf/jochrtf.h"
#include "oamd/oamd_parser.h"
#include "timeline/position_timeline.h"

namespace joc::binaural {

// ADM direction of the LFE source, as the reference passes it.
inline constexpr double kLfePosition[3] = {0.0, 1.0, 0.0};
inline constexpr int kSourceCount = 16;
inline constexpr std::uint32_t kBlockSamples = 512;

// Room constants the reference's native bridge passes for the default shoebox.
struct RoomConstants {
    double dims[3] = {18.0, 18.0, 14.0};
    double listener[3] = {9.0, 9.0, 7.0};
    double walls[6] = {0.62, 0.60, 0.58, 0.61, 0.52, 0.56};
    double speed_of_sound = 343.3;
    std::uint32_t fdn_delays[4] = {1427u, 1783u, 1973u, 2099u};
    double fdn_feedback[4] = {0.7853685923259284, 0.7394299865898056, 0.7160221718631921,
                              0.7009092068085467};
    double damping = 0.32;
    double fdn_output_gain = 0.22;
    std::uint32_t allpass_delays[2] = {113u, 331u};
    double allpass_gains[2] = {0.63, 0.51};
    std::uint32_t enable_early_reflections = 1;
    std::uint32_t enable_late_room = 1;
};

enum class Profile : std::uint32_t { Near = 0, Mid = 1, Far = 2 };

bool profile_from_name(const char* name, Profile* out);

class SofaBinauralRuntime {
public:
    SofaBinauralRuntime() = default;
    ~SofaBinauralRuntime();

    SofaBinauralRuntime(const SofaBinauralRuntime&) = delete;
    SofaBinauralRuntime& operator=(const SofaBinauralRuntime&) = delete;

    Status open(const hrtf::Field& field, const hrtf::Kernels& kernels, Profile profile,
                const RoomConstants& room = RoomConstants{});

    Status submit_frame(const float* objects16_planar, const oamd::OamdUpdate* update,
                        std::int64_t frame_index, std::int64_t outer_sample_offset,
                        std::int64_t object_delay_samples);

    // Resets the kernel, the timeline and the counters (plan 31.2).
    Status reset();

    // Drains the room tail.  `flush_samples` is the drain length; the reference
    Status finish(std::uint32_t flush_samples, std::vector<double>* out);

    // Program output (input minus the 961-sample kernel latency), interleaved.
    const std::vector<double>& output() const { return output_; }

    void take_output(std::vector<double>* out) {
        out->swap(output_);
        output_.clear();
    }

    std::uint64_t input_samples() const { return input_samples_; }
    std::uint64_t blocks_processed() const { return blocks_processed_; }
    std::size_t staged_samples() const { return staged_; }
    const timeline::OamdPositionTimeline& timeline() const { return timeline_; }

    // finish_output_capacity as the reference computes it (plan 21.4).
    std::uint32_t finish_capacity(double tail_seconds) const;

private:
    Status process_block();

    ejoc_sofa_binaural_handle handle_ = nullptr;
    Profile profile_ = Profile::Mid;
    timeline::OamdPositionTimeline timeline_;
    std::vector<double> staging_;
    std::size_t staged_ = 0;
    std::vector<double> block_output_;
    std::vector<double> output_;
    std::uint64_t input_samples_ = 0;
    std::uint64_t processed_samples_ = 0;
    std::uint64_t blocks_processed_ = 0;
    std::int64_t maximum_hrtf_delay_ = 0;
    std::uint32_t hrtf_history_slots_ = 1;
    std::uint32_t tail_samples_ = 61200;
};

}  // namespace joc::binaural
