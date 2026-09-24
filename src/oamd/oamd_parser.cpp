#include "oamd/oamd_parser.h"

#include <cmath>
#include <string>

#include "foundation/bit_reader.h"

namespace joc::oamd {

namespace {

constexpr int kSampleOffsetIndex[4] = {8, 16, 18, 24};
constexpr int kRampDurations[3] = {0, 512, 1536};
constexpr int kRampDurationIndex[16] = {32,  64,   128,  256,  320,  480,
                                        1000, 1001, 1024, 1600, 1601, 1602,
                                        1920, 2000, 2002, 2048};
constexpr std::uint32_t kObjectElementId = 1;
constexpr int kIsfObjectCounts[6] = {4, 8, 10, 14, 15, 30};
constexpr int kStandardBedChannelCount[10] = {2, 1, 1, 2, 2, 2, 2, 2, 2, 1};

Status variant(const std::string& name, const std::string& message) {
    return Status::fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT, stage::kOamd,
                        name + ": " + message);
}

Status syntax(const std::string& message) {
    return Status::fail(JOC_ERR_OAMD_SYNTAX, stage::kOamd, message);
}

bool variable_bits_max(bits::BitReader& reader, unsigned width, unsigned max_groups,
                       std::uint32_t* out_value) {
    std::uint32_t value = reader.read(width);
    if (reader.failed()) {
        return false;
    }
    std::uint32_t more = reader.read(1);
    if (reader.failed()) {
        return false;
    }
    unsigned num_group = 1;
    if (max_groups > num_group) {
        if (more != 0u) {
            value = (value + 1u) << width;
        }
        while (more != 0u) {
            value += reader.read(width);
            more = reader.read(1);
            if (reader.failed()) {
                return false;
            }
            if (num_group >= max_groups) {
                break;
            }
            if (more != 0u) {
                value = (value + 1u) << width;
                num_group += 1;
            }
        }
    }
    *out_value = value;
    return true;
}

struct ProgramInfo {
    bool dynamic_object_only = false;
    bool lfe_present = false;
    std::uint32_t num_bed_objects = 0;
    std::uint32_t num_isf_objects = 0;
    std::int32_t num_dynamic_objects = -1;
    std::vector<BedAssignment> bed_assignments;
};

Status parse_program_assignment(bits::BitReader& reader, ProgramInfo* program) {
    program->dynamic_object_only = reader.read(1) != 0u;
    if (reader.failed()) {
        return syntax("program_assignment truncated");
    }
    if (program->dynamic_object_only) {
        program->lfe_present = reader.read(1) != 0u;
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        program->num_bed_objects = program->lfe_present ? 1u : 0u;
        return Status::success();
    }

    const std::uint32_t mask = reader.read(4);
    if (reader.failed()) {
        return syntax("program_assignment truncated");
    }
    if ((mask & 0x1u) != 0u) {
        reader.read(1);
        const std::uint32_t multi = reader.read(1);
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        std::uint32_t instances = 1;
        if (multi != 0u) {
            instances = reader.read(3) + 2u;
        }
        for (std::uint32_t instance = 0; instance < instances; ++instance) {
            BedAssignment assignment;
            if (reader.read(1) != 0u) {
                assignment.lfe_only = true;
                assignment.mask = 0;
                program->bed_assignments.push_back(assignment);
                program->num_bed_objects += 1;
                continue;
            }
            if (reader.read(1) != 0u) {
                assignment.standard = true;
                assignment.mask = reader.read(10);
                for (int bit = 0; bit < 10; ++bit) {
                    if (((assignment.mask >> bit) & 1u) != 0u) {
                        program->num_bed_objects +=
                            static_cast<std::uint32_t>(kStandardBedChannelCount[bit]);
                    }
                }
            } else {
                assignment.standard = false;
                assignment.mask = reader.read(17);
                for (int bit = 0; bit < 17; ++bit) {
                    if (((assignment.mask >> bit) & 1u) != 0u) {
                        program->num_bed_objects += 1;
                    }
                }
            }
            program->bed_assignments.push_back(assignment);
            if (reader.failed()) {
                return syntax("program_assignment truncated");
            }
        }
    }
    if ((mask & 0x2u) != 0u) {
        const std::uint32_t isf_idx = reader.read(3);
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        program->num_isf_objects =
            (isf_idx < 6u) ? static_cast<std::uint32_t>(kIsfObjectCounts[isf_idx]) : 0u;
    }
    if ((mask & 0x4u) != 0u) {
        std::uint32_t count = reader.read(5);
        if (count == 0x1Fu) {
            count += reader.read(7);
        }
        if (reader.failed()) {
            return syntax("program_assignment truncated");
        }
        program->num_dynamic_objects = static_cast<std::int32_t>(count + 1u);
    }
    if ((mask & 0x8u) != 0u) {
        const std::uint32_t reserved_bytes = reader.read(4) + 1u;
        if (!reader.skip(static_cast<std::size_t>(reserved_bytes) * 8u)) {
            return syntax("program_assignment reserved data truncated");
        }
    }
    return Status::success();
}

Status parse_object_info_block(bits::BitReader& reader, int object_index, bool in_bed_or_isf,
                               int* position_x, int* position_y, int* position_z, bool* has_position) {
    *has_position = false;
    const bool not_active = reader.read(1) != 0u;
    if (reader.failed()) {
        return syntax("object_info_block truncated");
    }
    const std::uint32_t basic_status = not_active ? 0u : 1u;
    if (basic_status == 1u) {
        const std::uint32_t gain_idx = reader.read(2);
        if (gain_idx == 2u) {
            reader.read(6);
        }
        const bool default_priority = reader.read(1) != 0u;
        if (!default_priority) {
            reader.read(5);
        }
        if (reader.failed()) {
            return syntax("object_info_block truncated");
        }
    }
    const std::uint32_t render_status = (not_active || in_bed_or_isf) ? 0u : 1u;
    if (render_status == 1u) {
        const int x = static_cast<int>(reader.read(6));
        const int y = static_cast<int>(reader.read(6));
        const int z_sign = static_cast<int>(reader.read(1));
        const int z = static_cast<int>(reader.read(4));
        if (reader.failed()) {
            return syntax("object_info_block truncated");
        }
        *position_x = x;
        *position_y = y;
        *position_z = z_sign != 0 ? z : -z;
        *has_position = true;
        if (reader.read(1) != 0u) {
            if (reader.read(1) == 0u) {
                reader.read(4);
            }
        }
        reader.read(3);
        reader.read(1);
        const std::uint32_t size_idx = reader.read(2);
        if (size_idx == 1u) {
            reader.read(5);
        } else if (size_idx == 2u) {
            reader.read(15);
        }
        if (reader.read(1) != 0u) {
            reader.read(3);
            reader.read(2);
        }
        reader.read(1);
        if (reader.failed()) {
            return syntax("object_info_block truncated");
        }
    }
    if (reader.read(1) != 0u) {
        const std::uint32_t additional_bytes = reader.read(4) + 1u;
        if (!reader.skip(static_cast<std::size_t>(additional_bytes) * 8u)) {
            return syntax("object_info_block additional table truncated");
        }
    }
    (void)object_index;
    return Status::success();
}

struct ObjectElementInfo {
    std::uint32_t sample_offset_code = 0;
    std::uint32_t sample_offset = 0;
    std::uint32_t block_count = 0;
    std::uint32_t block_offset_samples = 0;
    std::uint32_t ramp_duration_samples = 0;
    bool reserved_data_not_present = false;
};

Status parse_object_element(bits::BitReader& reader, std::uint32_t object_count,
                            std::uint32_t bed_isf_objects, ObjectElementInfo* out, OamdUpdate* update) {
    out->sample_offset_code = reader.read(2);
    if (reader.failed()) {
        return syntax("object_element truncated");
    }
    switch (out->sample_offset_code) {
        case 0: out->sample_offset = 0; break;
        case 1: {
            const std::uint32_t index = reader.read(2);
            if (reader.failed()) {
                return syntax("object_element truncated");
            }
            out->sample_offset = static_cast<std::uint32_t>(kSampleOffsetIndex[index & 3u]);
            break;
        }
        case 2: out->sample_offset = reader.read(5); break;
        default:
            return variant("md_sample_offset_mode",
                           "MD sample-offset mode " +
                               std::to_string(out->sample_offset_code) +
                               " is not covered by the 16-slot model");
    }
    out->block_count = reader.read(3) + 1u;
    if (reader.failed()) {
        return syntax("object_element truncated");
    }
    for (std::uint32_t block = 0; block < out->block_count; ++block) {
        const std::uint32_t block_offset_factor = reader.read(6);
        const std::uint32_t ramp_code = reader.read(2);
        std::uint32_t ramp_duration = 0;
        if (ramp_code == 3u) {
            if (reader.read(1) != 0u) {
                const std::uint32_t index = reader.read(4);
                if (reader.failed()) {
                    return syntax("object_element truncated");
                }
                ramp_duration = static_cast<std::uint32_t>(kRampDurationIndex[index & 15u]);
            } else {
                ramp_duration = reader.read(11);
            }
        } else {
            ramp_duration = static_cast<std::uint32_t>(kRampDurations[ramp_code]);
        }
        if (reader.failed()) {
            return syntax("object_element truncated");
        }
        if (block == 0) {
            out->block_offset_samples = out->sample_offset + block_offset_factor * 32u;
            out->ramp_duration_samples = ramp_duration;
        }
    }
    out->reserved_data_not_present = reader.read(1) != 0u;
    if (!out->reserved_data_not_present) {
        reader.read(5);
    }
    if (reader.failed()) {
        return syntax("object_element truncated");
    }

    for (std::uint32_t index = 0; index < object_count; ++index) {
        int x = 0;
        int y = 0;
        int z = 0;
        bool has_position = false;
        const bool in_bed_or_isf = index < bed_isf_objects;
        const Status status =
            parse_object_info_block(reader, static_cast<int>(index), in_bed_or_isf, &x, &y, &z,
                                    &has_position);
        if (!status.ok()) {
            return status;
        }
        if (!has_position || index >= static_cast<std::uint32_t>(kMaxSlots)) {
            continue;
        }
        update->slots[index].q1 =
            (x >= 0 && x <= kNQ12) ? static_cast<std::int16_t>(q_of(x, kNQ12)) : static_cast<std::int16_t>(-1);
        update->slots[index].q2 =
            (y >= 0 && y <= kNQ12) ? static_cast<std::int16_t>(q_of(y, kNQ12)) : static_cast<std::int16_t>(-1);
        if (z >= 0 && z <= kNQ3) {
            update->slots[index].q3 = static_cast<std::int16_t>(q_of(z, kNQ3));
        } else if (z < 0) {
            update->slots[index].q3 = 0;
        } else {
            update->slots[index].q3 = static_cast<std::int16_t>(-1);
        }
    }
    return Status::success();
}

}  // namespace

int q_of(int k, int n) {
    const double value = std::floor(32768.0 * static_cast<double>(k) / static_cast<double>(n) + 0.5);
    const int quantised = static_cast<int>(value);
    return quantised > 32767 ? 32767 : quantised;
}

Status parse_id11(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out) {
    return parse_id11_verbose(payload, payload_size, out, nullptr);
}

Status parse_id11_verbose(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out,
                          ParseTrace* trace) {
    if (payload == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOamd, "null payload or output");
    }
    if (payload_size == 0) {
        return variant("header_truncated", "OAMD payload is empty");
    }
    *out = OamdUpdate{};

    bits::BitReader reader(payload, payload_size);
    std::uint32_t version = reader.read(2);
    if (reader.failed()) {
        return variant("header_truncated", "OAMD payload cannot hold the 2-bit version");
    }
    if (version == 3u) {
        version += reader.read(3);
        if (reader.failed()) {
            return syntax("OAMD version extension truncated");
        }
    }
    std::uint32_t object_count_bits = reader.read(5);
    if (object_count_bits == 0x1Fu) {
        object_count_bits += reader.read(7);
    }
    if (reader.failed()) {
        return syntax("OAMD object count truncated");
    }
    const std::uint32_t object_count = object_count_bits + 1u;

    ProgramInfo program;
    Status status = parse_program_assignment(reader, &program);
    if (!status.ok()) {
        return status;
    }
    const std::uint32_t alternate_present = reader.read(1);
    if (reader.failed()) {
        return syntax("OAMD alternate data flag truncated");
    }

    std::uint32_t element_count = reader.read(4);
    if (element_count == 0xFu) {
        element_count += reader.read(5);
    }
    if (reader.failed()) {
        return syntax("OAMD element count truncated");
    }
    if (element_count == 0u) {
        return variant("missing_object_element", "OAMD declares no element");
    }
    const std::uint32_t bed_isf_objects = program.num_bed_objects + program.num_isf_objects;

    bool have_object_element = false;
    ObjectElementInfo object_element;
    for (std::uint32_t ordinal = 0; ordinal < element_count; ++ordinal) {
        const std::size_t header_start = reader.position();
        const std::uint32_t element_id = reader.read(4);
        std::uint32_t size_field = 0;
        if (!variable_bits_max(reader, 4, 4, &size_field)) {
            return syntax("OAMD element header truncated at ordinal " + std::to_string(ordinal));
        }
        const std::uint32_t size_bytes = size_field + 1u;
        const std::size_t region_start = reader.position();
        const std::size_t region_end = region_start + static_cast<std::size_t>(size_bytes) * 8u;
        if (region_end > reader.limit()) {
            return variant("element_bounds",
                           "element " + std::to_string(ordinal) + " (id " +
                               std::to_string(element_id) + ") declares " +
                               std::to_string(size_bytes) +
                               " bytes, beyond the payload (header at bit " +
                               std::to_string(header_start) + ")");
        }
        if (alternate_present != 0u) {
            reader.read(4);
        }
        reader.read(1);
        if (reader.failed()) {
            return syntax("OAMD element control fields truncated at ordinal " +
                          std::to_string(ordinal));
        }

        if (element_id == kObjectElementId) {
            if (have_object_element) {
                return variant("multiple_object_elements", "OAMD contains several object elements");
            }
            have_object_element = true;
            const Status parsed = parse_object_element(reader, object_count, bed_isf_objects,
                                                       &object_element, out);
            if (!parsed.ok()) {
                return parsed;
            }
            if (trace != nullptr) {
                trace->parsed_end_bit = static_cast<std::uint32_t>(reader.position());
            }
            if (reader.position() < region_end) {
                reader.set_limit_bits(reader.limit());
                reader.skip(region_end - reader.position());
            }
        } else {
            if (!reader.skip(region_end - reader.position())) {
                return syntax("OAMD element skip past payload end");
            }
        }
        if (reader.failed()) {
            return syntax("OAMD element parse failed at ordinal " + std::to_string(ordinal));
        }
    }

    if (!have_object_element) {
        return variant("missing_object_element", "OAMD has no object element");
    }

    // Trailing padding must be zero over the whole remainder.
    {
        bits::BitReader tail(payload, payload_size);
        tail.reset(payload, payload_size, reader.position());
        while (tail.remaining_bits() > 0) {
            const std::size_t chunk =
                tail.remaining_bits() > 32u ? 32u : tail.remaining_bits();
            if (tail.read(static_cast<unsigned>(chunk)) != 0u) {
                return Status::fail(JOC_ERR_BITSTREAM_PADDING, stage::kOamd,
                                    "OAMD payload padding is not zero (from bit " +
                                        std::to_string(reader.position()) + ")");
            }
        }
    }

    if (version != 0u) {
        return variant("oamd_version",
                       "OAMD syntax version " + std::to_string(version) + " is not covered");
    }
    if (object_count > static_cast<std::uint32_t>(kMaxSlots)) {
        return variant("object_count", "OAMD object count " + std::to_string(object_count) +
                                           " exceeds the 16-slot model");
    }
    if (object_element.block_count != 1u) {
        return variant("multiple_position_blocks",
                       "OAMD frame carries " + std::to_string(object_element.block_count) +
                           " position blocks; a single frame_update cannot express that");
    }

    out->block_offset_samples = object_element.block_offset_samples;
    out->ramp_duration_samples = object_element.ramp_duration_samples;
    out->object_count = object_count;
    out->dynamic_object_only = program.dynamic_object_only;
    out->lfe_present = program.lfe_present;
    out->num_bed_objects = program.num_bed_objects;
    out->num_isf_objects = program.num_isf_objects;
    out->num_dynamic_objects = program.num_dynamic_objects;

    if (trace != nullptr) {
        trace->bed_assignments = program.bed_assignments;
        trace->element_count = element_count;
        trace->object_element_count = have_object_element ? 1u : 0u;
        trace->sample_offset_code = object_element.sample_offset_code;
        trace->sample_offset = object_element.sample_offset;
        trace->block_count = object_element.block_count;
        trace->reserved_data_not_present = object_element.reserved_data_not_present;
    }
    return Status::success();
}

}  // namespace joc::oamd
