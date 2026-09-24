#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"
#include "hrtf/jochrtf.h"
#include "hrtf/sofa.h"

// SOFA SimpleFreeFieldHRIR -> compiled directional field, ported from the
// reference chain (sofa_canonical.py + sofa_hrtf_field.py + the public
// filterbank): the measurement shell is selected, one delay representation is
// separated, the FIRs are projected onto the 64-QMF/77-hybrid filterbank and the
// result is fitted with fifth-order ACN/N3D real spherical harmonics.
namespace joc::hrtf {

inline constexpr int kFieldOrder = 5;
inline constexpr int kFieldTerms = 36;
inline constexpr double kFieldSampleRateHz = 48000.0;
inline constexpr double kDefaultShellRadiusM = 1.0;
inline constexpr double kDefaultProjectionRidge = 1.0e-3;
inline constexpr double kDefaultSphericalHarmonicRidge = 1.0e-5;
inline constexpr const char* kCompilerVersion = "joc-sofa-compiler-v1";
inline constexpr const char* kPhasePolicyVersion = "sofa-delay-exactly-once-v1";
inline constexpr const char* kShConvention = "ACN/N3D real";
inline constexpr const char* kFilterbankTableVersion = "joc-public-64qmf-77hybrid-v1";
// SHA-256 of the standard filterbank archive the embedded tables came from.  It
// participates in the cache key, so it is part of the file-format contract.
inline constexpr const char* kFilterbankArchiveSha256 =
    "C05BEF4D26E96ECBD4694E2572F05DA400255C777BA5047300B9D3B1F81081CD";

struct CompileOptions {
    double shell_radius_m = kDefaultShellRadiusM;
    int order = kFieldOrder;
    double projection_ridge = kDefaultProjectionRidge;
    double sh_ridge = kDefaultSphericalHarmonicRidge;
};

// Canonical HRIR set: Data.IR and Data.Delay stay separate, the listener frame is
// applied to the source positions and the ears are ordered left/right.
struct CanonicalHrtf {
    std::string source_path;
    std::string source_sha256;
    std::string convention;
    std::string convention_version;
    std::string processing_label;
    double sample_rate_hz = 0.0;
    std::uint32_t measurements = 0;
    std::uint32_t taps = 0;
    int left_receiver_index = 0;
    int right_receiver_index = 1;
    std::vector<double> source_position_cartesian_m;  // [M,3] listener-local
    std::vector<double> unit_directions;              // [M,3]
    std::vector<double> measurement_radius_m;         // [M]
    std::vector<double> hrir;                         // [M,2,N] canonical L/R
    std::vector<double> delay_samples;                // [M,2], not applied
};

// Port of load_simple_free_field_hrir(): strict SimpleFreeFieldHRIR import.
Status canonicalize_sofa(const SofaHrir& sofa, CanonicalHrtf* out);

// Compiles the canonical set into the runtime field (port of SofaHrtfField.fit).
Status compile_sofa_field(const SofaHrir& sofa, const CompileOptions& options, Field* out);

// The measurements on the shell nearest to radius_m; actual_radius_m receives the
// mean radius of that shell (upstream CanonicalHrtf.shell_indices).
std::vector<std::size_t> canonical_shell_indices(const CanonicalHrtf& canonical, double radius_m,
                                                 double* actual_radius_m);

// Compiles an already canonicalized set (used by tests and the cache layer).
Status compile_canonical_field(const CanonicalHrtf& canonical, const CompileOptions& options,
                               Field* out);

// Configuration hash that names the cache file (upstream compiled_hrtf_cache_key).
std::string compiled_hrtf_cache_key(const std::string& source_sha256, double sample_rate_hz,
                                    double shell_radius_m, int order, double projection_ridge,
                                    double sh_ridge);

// Payload hash over the four arrays (upstream _payload_sha256).
std::string field_payload_sha256(const Field& field);

// "<stem>.<first 20 key digits>.jochrtf", the upstream cache file name.
std::string cache_file_name(const std::string& display_name, const std::string& cache_key);

// Serializes the field as a .jochrtf cache the upstream loader also accepts.
Status write_jochrtf(const Field& field, const std::string& path);

// The analysis/gain/synthesis dictionary the projection solves against (dev check).
std::vector<double> hybrid_gain_synthesis_dictionary_for_check(std::size_t sample_count);

// Shell directions and their spherical Voronoi weights (dev check).
void shell_directions_and_weights_for_check(const SofaHrir& sofa, double radius_m,
                                            std::vector<double>* directions,
                                            std::vector<double>* weights);

// PublicAnalysis77 on a unit impulse, interleaved complex (dev check).
std::vector<double> analysis_impulse_for_check(std::size_t total_samples);

// The 77 hybrid-band centre frequencies at 48 kHz.
const std::vector<double>& hybrid_band_center_frequencies_hz();

}  // namespace joc::hrtf
