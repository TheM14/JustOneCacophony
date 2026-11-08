// Port of src/oamd_params.py: the static ISF lookup data of the metadata path.
//
// The numbers are compiled in as the generated header `oamd/isf_tables.h`:
// ISF base coefficients and layout descriptors, applied as lookup tables.
// The remaining values (gain, size, priority, position, delta and refinement
// tables) are computed in code; the size tables are generated from the region
// geometry at runtime, and bed routing is derived from the speaker layouts.
// The arrays are adapted once per process into the structures below; nothing
// reads a data file.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "foundation/status.h"

namespace joc::oamd {

// One ISF configuration: the compressed basis matrix and the two row maps
// that expand it to the layout output channels.
struct IsfConfig {
    // mid, upper, lower, zenith, row_count (the reference's counts tuple).
    std::array<int, 5> counts{};
    std::vector<int> primary_row_map;
    std::vector<int> secondary_row_map;
    // row_count * 73 float32 coefficients, row major.
    std::vector<float> matrix;

    int row_count() const { return counts[4]; }
};

// One layout descriptor: how many output channels the basis has,
// the layout-group selectors and the automatic compensation value.
struct IsfLayout {
    int output_count = 0;
    std::array<int, 4> selectors{};  // floor_a, floor_b, height, additional
    double compensation_value = 0.0;
    // ISF channel id of each matrix output column, padded with zeros to 37.
    std::vector<std::uint8_t> output_ids;
};

// One record of a layout group: selector id and the half-open column range
// [begin, end).
struct IsfGroupRecord {
    int selector = 0;
    int begin = 0;
    int end = 0;
};

// The four layout groups of the ISF tables.
struct IsfLayoutGroups {
    std::vector<IsfGroupRecord> floor_a;
    std::vector<IsfGroupRecord> floor_b;
    std::vector<IsfGroupRecord> height;
    std::vector<IsfGroupRecord> additional;

    const std::vector<IsfGroupRecord>* find(const char* name) const;
};

// The ISF lookup data in the shape the metadata renderer consumes.
struct MetadataParams {
    std::vector<IsfConfig> configs;
    std::unordered_map<std::string, IsfLayout> layouts;
    IsfLayoutGroups layout_groups;
    // One flag per matrix column: whether the column also contributes through
    // the secondary row map.
    std::vector<bool> paired_columns;

    const IsfLayout* layout(const char* key) const;
};

// Returns the process-wide tables.  They are compiled in, so the pointer is
// never null; ``metadata_params_error`` is kept for callers that log a load
// failure and always returns an empty string.
const MetadataParams* metadata_params();
const char* metadata_params_error();

}  // namespace joc::oamd
