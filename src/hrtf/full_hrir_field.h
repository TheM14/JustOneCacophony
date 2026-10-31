
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "foundation/status.h"
#include "hrtf/spherical_mesh.h"

// Full-HRIR directional field, ported from the reference sofa_hrir.py: the
// selected measurement shell keeps its complete FIR support, the interpolation
// works on delay-aligned complex spectra inside the containing spherical
// triangle, and the radial model is the analytic rigid-sphere series.  This
// replaces the compiled 77-band/SH field for SOFA inputs.
namespace joc::hrtf {

struct CanonicalHrtf;

// Point-source pressure at `radius` over the free-field pressure at the sphere
// centre (rigid sphere, FFT sign convention) for the cosines in `cosines`.
// Output is [cosines][frequencies] complex, row-major.
Status sphere_transfer(const std::vector<double>& frequencies, double radius, double head_radius,
                       const std::vector<double>& cosines,
                       std::vector<std::complex<double>>* out);

enum class HrirInterpolation : std::uint32_t {
    kLogMagnitudePhase = 0,  // lf_logmag_phase: geometric mean magnitude, interpolated phase
    kMcaZero = 1,            // lf_mca_zero: auditory-band magnitude correction
};

struct FullHrirOptions {
    HrirInterpolation method = HrirInterpolation::kLogMagnitudePhase;
    bool has_shell_radius = false;
    double shell_radius_m = 0.0;
    std::size_t fft_size = 0;  // 0 selects the automatic size
};

class FullHrirField {
public:
    // Builds the field from a canonical SimpleFreeFieldHRIR set (48 kHz only).
    Status build(const CanonicalHrtf& canonical, const FullHrirOptions& options);

    std::size_t fft_size() const { return fft_size_; }
    std::size_t frequency_bins() const { return bins_; }
    const std::vector<double>& frequencies() const { return frequencies_; }
    const std::vector<double>& omega() const { return omega_; }
    double measurement_radius_m() const { return measurement_radius_m_; }
    double head_radius_m() const { return head_radius_m_; }
    const std::vector<double>& ear_directions() const { return ear_directions_; }  // [2][3]
    std::size_t direction_count() const { return mesh_.vertex_count(); }
    std::size_t triangle_count() const { return mesh_.face_count(); }
    int recording_origin_samples() const { return recording_origin_samples_; }
    int common_latency_samples() const { return static_cast<int>(fft_size_ / 2); }
    double max_triangle_edge_degrees() const { return mesh_.max_triangle_edge_degrees(); }
    double phase_fit_rms_rad_p95() const { return phase_fit_rms_rad_p95_; }
    double floor() const { return floor_; }
    HrirInterpolation method() const { return method_; }

    // Interpolated two-ear spectrum at a direction (`out` is [2][bins] complex).
    Status spectrum(const double direction[3], std::vector<std::complex<double>>* out);

    // Radial (distance) correction for a source at `radius` metres.
    Status radial_ratio(const double direction[3], double radius,
                        std::vector<std::complex<double>>* out) const;

    // Centre (omnidirectional) spectrum used when a position is inside the head.
    const std::vector<std::complex<double>>& centre_spectrum() const { return centre_spectrum_; }

private:
    Status build_interpolation(const CanonicalHrtf& canonical, const FullHrirOptions& options);

    std::size_t fft_size_ = 0;
    std::size_t bins_ = 0;
    std::size_t taps_ = 0;
    HrirInterpolation method_ = HrirInterpolation::kLogMagnitudePhase;
    SphericalMesh mesh_;
    std::vector<double> frequencies_;  // [bins]
    std::vector<double> omega_;        // [bins]
    std::vector<double> delays_;       // [M][2] samples (estimated + Data.Delay)
    std::vector<std::complex<double>> aligned_;       // [M][2][bins]
    std::vector<double> log_magnitude_;               // [M][2][bins]
    std::vector<double> auditory_;                    // [M][2][41] (MCA method only)
    std::vector<double> centres_hz_;                  // [41] (MCA method only)
    std::vector<double> bank_;                        // [41][bins] (MCA method only)
    std::vector<std::complex<double>> centre_spectrum_;  // [bins]
    double measurement_radius_m_ = 0.0;
    double head_radius_m_ = 0.0;
    std::vector<double> ear_directions_;              // [2][3]
    double floor_ = 1.0e-15;
    int recording_origin_samples_ = 0;
    double phase_fit_rms_rad_p95_ = 0.0;
};

}  // namespace joc::hrtf
