
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "foundation/status.h"

namespace joc::hrtf {

inline constexpr int kShTerms = 36;
inline constexpr int kEars = 2;
inline constexpr int kHybridBands = 77;
inline constexpr int kFormatVersion = 1;
inline constexpr const char* kMagic = "JOC-HRTF-CACHE";
inline constexpr const char* kCacheSchema = "joc-compiled-hrtf-v1";

struct Field {
    std::vector<double> coefficients;
    std::vector<double> delay_coefficients;
    std::vector<double> delay_bounds;
    std::vector<double> band_centers_hz;
    double measurement_radius_m = 1.0;
    long long order = 5;
    std::string source_sha256;
    std::string cache_key;
    std::string payload_sha256;
    std::string delay_source;
    std::string compiler_version;
    std::string phase_policy_version;
    std::string sh_convention;
    std::string filterbank_json;
    std::string metadata_json;
    // Compile-side metadata, needed to write the cache back out unchanged.
    std::string source_display_name;
    std::string fit_report_json;
    double projection_ridge = 0.0;
    double spherical_harmonic_ridge = 0.0;
};

Status load_jochrtf(const std::string& path, Field* out);

// Binaural filterbank kernels, as the reused kernel expects them (C order, the
// exact dtypes of the ABI parameters).
struct Kernels {
    std::vector<double> qmf_analysis;
    std::vector<double> hybrid_low;
    std::vector<std::int16_t> hybrid_indices;
    std::vector<double> hybrid_values;
    std::vector<double> qmf_basis;
    std::vector<double> qmf_taps;
    std::uint32_t hybrid_count = 0;
};

// Loads a kernel-table archive.  The file path is an override for verification;
// the shipped tables are embedded (see builtin_kernels) so no data file is needed.
// The Fortran-order index member is transposed into C order on purpose: the reused
// kernel indexes the hybrid synthesis table row-major.
Status load_kernels(const std::string& npz_path, Kernels* out);

// The public filterbank tables compiled into the library (identical values to the
// archive the file loader accepts; the unit test checks their hashes).
const Kernels& builtin_kernels();

}  // namespace joc::hrtf
