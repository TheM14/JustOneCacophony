#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "foundation/status.h"
#include "hrtf/rosella_model.h"

// One of the room's transforms is a radix-2 FFT plan, so the type is named here and
// kept out of the interface.
namespace joc::dsp {
class FftPlan;
}  // namespace joc::dsp

// The reference arithmetic of the Rosella ".personalized_headphone" renderer
// (rosella_model.py, rosella_direct.py and rosella_room.py): the direction basis and
// the padded lane grid, the per-source direct gains and room send, and the recursive
// table-A room model.  rosella_renderer.cpp drives these through RosellaRuntime; a
// renderer built on another field format can call them directly and get the same
// samples, so the operation order and the literals in the definitions are part of
// the interface.
namespace joc::hrtf {

// rosella_direct._profile_geometry: a virtual position resolved on one distance
// profile's grid, as the direction of the direct path plus the radius factors the
// gain weights are scaled by.
struct ProfileGeometry {
    double direction[3] = {1.0, 0.0, 0.0};
    double radius = 0.0;
    double clamped = 0.0;
    double alpha = 0.0;
};

// Quantises `position` onto the profile's metadata grid, maps it through the
// profile's axis scales and clamps it to the profile bounds; `out` takes the result.
Status profile_geometry(const RosellaModel& model, const double position[3], int profile_index,
                        ProfileGeometry* out);

// rosella_model.direction_basis: the observed 36-term basis of one unit vector.
void direction_basis(double x, double y, double z, double out[36]);

// rosella_direct._logical_field: the padded lane grid as [77][36][2] float64.
std::vector<double> rosella_logical_field(const std::vector<float>& padded);

// One source's direct-path gains and room send, plus the radii and the interaural
// delay the caller reports.  `delayed_ear` is -1 when neither ear is delayed,
// otherwise it is the ear (0 = left, 1 = right) whose phase carries
// `delay_samples`.
struct DirectResult {
    std::complex<double> gains[2][77] = {};
    double room_send = 0.0;
    double physical_radius_m = 0.0;
    double normalized_radius = 0.0;
    double clamped_radius = 0.0;
    double delay_samples = 0.0;
    int delayed_ear = -1;
};

// rosella_direct.special_lfe_direct: the response a special (LFE) source is
// rendered with, built once.
const DirectResult& special_lfe_direct();

// rosella_direct.direct_and_room_send: the direct gains and the room send of the
// position `position` on profile `profile_index`, read through the logical field
// grids of `field_left`/`field_right`.
Status direct_and_room_send(const RosellaModel& model, const std::vector<double>& field_left,
                            const std::vector<double>& field_right, const double position[3],
                            int profile_index, DirectResult* out);

// rosella_room.RosellaRoomFir: the recursive table-A room model, its room FIR and the
// overlap-add convolution that applies the FIR in the hybrid band domain.  State is
// per instance and not thread-safe.
class RosellaRoomFir {
public:
    RosellaRoomFir();
    ~RosellaRoomFir();
    RosellaRoomFir(const RosellaRoomFir&) = delete;
    RosellaRoomFir& operator=(const RosellaRoomFir&) = delete;

    // Builds the room impulse response from `model` over `impulse_slots` slots.  The
    // room can only be processed once this succeeds; the alternate table-A mode is
    // rejected here, as the reference rejects it.
    Status configure(const RosellaModel& model, int impulse_slots);
    // `values` is [slots][2][77], `output` is [slots][2][77] and stays zero outside
    // the room bands.
    Status process_chunk(const std::vector<std::complex<double>>& values, std::size_t slots,
                         std::vector<std::complex<double>>* output);
    void reset();
    bool valid() const;

private:
    int length_ = 0;
    bool valid_ = false;
    std::vector<std::complex<double>> kernel_;
    std::vector<std::complex<double>> tail_;
    std::vector<std::complex<double>> kernel_fft_;
    std::size_t fft_cache_size_ = 0u;
    std::vector<std::complex<double>> line_;       // scratch, exactly fft_size entries
    std::vector<std::complex<double>> input_fft_;  // scratch, [fft_size][64], fully written
    std::vector<std::complex<double>> block_;      // scratch, [needed][2][64], fully written
    std::unique_ptr<dsp::FftPlan> forward_plan_;
    std::unique_ptr<dsp::FftPlan> inverse_plan_;
};

}  // namespace joc::hrtf
