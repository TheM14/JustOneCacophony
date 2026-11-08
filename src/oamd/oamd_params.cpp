#include "oamd/oamd_params.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "oamd/isf_tables.h"

namespace joc::oamd {

namespace {

namespace tables = isf_tables;

// Builds the renderer-facing view from the checked-in table header.  There is no
// file I/O and no runtime parsing: the numbers are compiled in as static data.
MetadataParams build_params() {
    MetadataParams params;
    params.configs.reserve(tables::kConfigCount);
    for (const tables::Config& source : tables::kConfigs) {
        IsfConfig config;
        for (std::size_t index = 0; index < config.counts.size(); ++index) {
            config.counts[index] = source.counts[index];
        }
        const std::size_t rows = static_cast<std::size_t>(config.row_count());
        config.primary_row_map.assign(source.primary_row_map, source.primary_row_map + rows);
        config.secondary_row_map.assign(source.secondary_row_map,
                                        source.secondary_row_map + rows);
        const std::size_t coefficients = rows * static_cast<std::size_t>(tables::kMatrixColumns);
        config.matrix.assign(source.matrix, source.matrix + coefficients);
        params.configs.push_back(std::move(config));
    }

    params.layouts.reserve(tables::kLayoutCount);
    for (const tables::Layout& source : tables::kLayouts) {
        IsfLayout layout;
        layout.output_count = source.output_count;
        for (std::size_t index = 0; index < layout.selectors.size(); ++index) {
            layout.selectors[index] = source.selectors[index];
        }
        layout.compensation_value = source.compensation_value;
        layout.output_ids.assign(source.output_ids,
                                 source.output_ids + tables::kOutputIdSlots);
        params.layouts.emplace(source.key, std::move(layout));
    }

    const auto copy_groups = [](const tables::GroupRecord* records, int count,
                                std::vector<IsfGroupRecord>* out) {
        out->reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            out->push_back(IsfGroupRecord{records[index].selector, records[index].begin,
                                          records[index].end});
        }
    };
    copy_groups(tables::kLayoutGroups.floor_a, tables::kLayoutGroups.floor_a_count,
                &params.layout_groups.floor_a);
    copy_groups(tables::kLayoutGroups.floor_b, tables::kLayoutGroups.floor_b_count,
                &params.layout_groups.floor_b);
    copy_groups(tables::kLayoutGroups.height, tables::kLayoutGroups.height_count,
                &params.layout_groups.height);
    copy_groups(tables::kLayoutGroups.additional, tables::kLayoutGroups.additional_count,
                &params.layout_groups.additional);

    params.paired_columns.assign(tables::kPairedColumns,
                                 tables::kPairedColumns + tables::kPairedColumnCount);
    return params;
}

const MetadataParams& loaded_params() {
    static const MetadataParams params = build_params();
    return params;
}

}  // namespace

const std::vector<IsfGroupRecord>* IsfLayoutGroups::find(const char* name) const {
    if (name == nullptr) {
        return nullptr;
    }
    const std::string key(name);
    if (key == "floor_a") {
        return &floor_a;
    }
    if (key == "floor_b") {
        return &floor_b;
    }
    if (key == "height") {
        return &height;
    }
    if (key == "additional") {
        return &additional;
    }
    return nullptr;
}

const IsfLayout* MetadataParams::layout(const char* key) const {
    if (key == nullptr) {
        return nullptr;
    }
    const auto found = layouts.find(key);
    return found != layouts.end() ? &found->second : nullptr;
}

const MetadataParams* metadata_params() { return &loaded_params(); }

const char* metadata_params_error() { return ""; }

}  // namespace joc::oamd
