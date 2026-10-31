
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "foundation/status.h"
#include "hrtf/full_hrir_field.h"
#include "hrtf/rosella_bank.h"
#include "hrtf/rosella_model.h"
#include "hrtf/rosella_reference.h"

// Configured reference environment for the SOFA renderer, ported from the
// reference binaural_reference.py: the JSON configuration names the personalized
// model that defines the reference reproduction (room, bank and level anchor),
// and every SOFA direction is mapped through that model's distance profile.
namespace joc::binaural {

// Profile indices, as in the reference rosella_direct module.
inline constexpr int kProfileNear = 1;
inline constexpr int kProfileFar = 2;
inline constexpr int kProfileMid = 3;

struct ReferencePosition {
    double direction[3] = {1.0, 0.0, 0.0};
    double distance_m = 0.0;        // physical distance implied by the metadata
    double ear_distance_m = 0.0;    // distance floored at the near-centre minimum
    double centre_weight = 1.0;     // direction vs centre blend
    double direct_gain = 1.0;
    double room_send = 0.0;
};

class ReferenceEnvironment {
public:
    // Loads `config_path` (JSON) and the model it names, verifying its SHA-256.
    // An empty path selects `default_reference_config_path()`.
    Status load(const std::string& config_path, int mode_index, bool mode_specified);

    int profile_index() const { return profile_index_; }
    int calibration_profile_index() const { return calibration_profile_index_; }
    int minimum_profile_index() const { return calibration_profile_index_; }
    const std::string& mode_name() const { return mode_name_; }
    const hrtf::RosellaModel& model() const { return model_; }
    const std::string& profile_id() const { return profile_id_; }
    const std::string& model_sha256() const { return model_sha256_; }
    const std::string& configuration_sha256() const { return configuration_sha256_; }
    double reference_distance_m() const { return reference_distance_m_; }
    double minimum_distance_m() const { return minimum_distance_m_; }
    int room_impulse_slots() const { return room_impulse_slots_; }
    int calibration_directions() const { return calibration_directions_; }
    const std::vector<double>& compiled_receivers_m() const { return compiled_receivers_m_; }

    Status map_position(const double position[3], ReferencePosition* out) const;
    // `profile_index` selects the distance profile; negative keeps the mode's.
    Status map_position(const double position[3], int profile_index,
                        ReferencePosition* out) const;

    Status create_room(std::unique_ptr<hrtf::RosellaRoomFir>* out) const;

    // The public bank is the reference bank; the analysis side carries the source
    // channels, the synthesis side the two ears.
    std::unique_ptr<hrtf::RosellaBank> create_analysis_bank(std::size_t channels) const;
    std::unique_ptr<hrtf::RosellaBank> create_synthesis_bank() const;

    // Raises the SOFA field to the reference level: one scalar over the equal-area
    // calibration directions, shared by all three modes.
    using SpectrumAt = std::function<Status(const ReferencePosition&,
                                            std::vector<std::complex<double>>*)>;
    Status calibrate(std::size_t fft_size, const std::vector<double>& frequencies,
                     const SpectrumAt& spectrum_at, double* level) const;

    static std::string default_reference_config_path();

private:
    hrtf::RosellaModel model_;
    std::string config_path_;
    std::string profile_id_;
    std::string model_sha256_;
    std::string configuration_sha256_;
    std::string mode_name_;
    int calibration_profile_index_ = kProfileMid;
    int profile_index_ = kProfileMid;
    int room_impulse_slots_ = 4096;
    int calibration_directions_ = 128;
    double header_constant_ = 1.0;   // header_float_scalars[0]
    double coupling_ = 0.0;          // header_float_scalars[1]
    double minimum_distance_m_ = 0.0;
    double reference_distance_m_ = 0.0;
    std::vector<double> compiled_receivers_m_;  // [2][3], when the config supplies them
};

}  // namespace joc::binaural
