// Port of src/oamd_math.py: float32 equations for the metadata application path.
//
// The routines implement the fixed float32 equations for the size tables, ISF
// basis assembly, bed channel routing and the position-dependent compensation.
// The size tables are generated from the region geometry with grouped equal-power
// panning, an S448 weight and a power sum; every operation is evaluated in
// float32 in a fixed order so the coefficients are reproducible.
//
// The caller passes the parameter mappings from oamd_params; this module does
// not read files itself.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "oamd/oamd_params.h"
#include "speaker/speaker_layout_lookup.h"
#include "speaker/speaker_layouts.h"

namespace joc::oamd {

// The specified 2**x approximation used by the compensation kernels.
float exp2_approx(float value);

// The specified x**power approximation; power is a float32.
float pow_approx(float value, float power);

// Stable float32 normalisation with the tiny-norm floor.
std::vector<float> norm(const std::vector<float>& vector);

// One boundary weight of the size kernel.
float boundary(int axis, int edge, float position, float t);

// Maps a scalar size in [0, 1] to t on the size curve.
float size_curve_value(double size);

// Size kernel response in region-speaker order (the region point order).
std::vector<float> size_kernel(const ::ejoc::speaker_tables::RegionGeometry& region,
                               const float position[3], float t, bool height);

// Size-kernel target vector in layout channel order.
std::vector<float> size_targets(const MetadataParams& meta,
                                const ::ejoc::speaker_tables::LayoutGeometry& layout,
                                int region_index, const double position[3],
                                double size, bool height);

// One assembled ISF basis matrix (row_count x layout output_count, row major).
struct IsfBasis {
    int rows = 0;
    int columns = 0;
    std::vector<float> values;

    float at(int row, int column) const {
        return values[static_cast<std::size_t>(row) * static_cast<std::size_t>(columns) +
                      static_cast<std::size_t>(column)];
    }
};

// Assembles one ISF basis matrix from the compressed base tables.
IsfBasis assemble_isf(const MetadataParams& meta, const char* layout_key, int isf_idx);

// Returns the per-basis ISF compensation gain.
float isf_compensation(const MetadataParams& meta, const char* layout_key, int isf_idx,
                       int row, bool disabled);

// Per-label bed redistribution weight, in the reference-position index space
// used by the layout descriptors.
float bed_weight(int label, std::uint32_t mask, int target_label);

// The five height-candidate flags for one input bed label.
std::array<int, 5> bed_candidates(int label, std::uint32_t mask,
                                  const std::vector<int>& output_labels);

// Route and weights for one input bed label.  route == -1 means the returned
// weights are the target; otherwise route is the reference position used to
// synthesise a point source.
struct BedDistribution {
    int route = -1;
    std::vector<float> weights;
};

// Derives the bed routing of one input bed label from the target layout.
BedDistribution bed_distribution(const ::joc::speaker::LayoutInfo& layout, int label,
                                 std::uint32_t mask, bool distribute);

// Returns the Q15 reference position used by the bed fallback path.
void bed_reference_position(int index, int q15[3]);

// Returns the bed path compensation gain for one input bed label.  ``mode`` is
// the trim mode (0 automatic, 1 disabled, 2 explicit) and ``levels`` holds the
// surround, height and centre levels in dB.
float bed_compensation(const ::joc::speaker::LayoutInfo& layout, int label, int mode,
                       bool bypass, const double levels[3]);

// Automatic attenuation of a target layout in dB.
float bed_attenuation_db(const ::joc::speaker::LayoutInfo& layout);

}  // namespace joc::oamd
