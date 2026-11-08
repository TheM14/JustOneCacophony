// State-only checks of the OAMD parser, ported from tests/oamd_metadata_test.py.
// The speaker-renderer targets are covered by the speaker tests; this file keeps
// the bitstream expansion, formula tables and programme parsing checks.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "oamd/oamd_params.h"
#include "oamd/oamd_parser.h"

namespace {

int failures = 0;

void check(bool passed, const char* name) {
    if (!passed) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

std::vector<std::uint8_t> from_hex(const char* text) {
    std::vector<std::uint8_t> bytes;
    const std::size_t length = std::strlen(text);
    for (std::size_t index = 0; index + 1 < length; index += 2) {
        const auto high = static_cast<unsigned>(text[index]);
        const auto low = static_cast<unsigned>(text[index + 1]);
        const auto digit = [](unsigned value) -> unsigned {
            if (value >= '0' && value <= '9') {
                return value - '0';
            }
            if (value >= 'a' && value <= 'f') {
                return value - 'a' + 10;
            }
            if (value >= 'A' && value <= 'F') {
                return value - 'A' + 10;
            }
            return 0;
        };
        bytes.push_back(static_cast<std::uint8_t>((digit(high) << 4) | digit(low)));
    }
    return bytes;
}

joc::oamd::OamdUpdate parse_state(const char* hex) {
    const std::vector<std::uint8_t> payload = from_hex(hex);
    joc::oamd::OamdUpdate update;
    const joc::Status status =
        joc::oamd::parse_id11_state(payload.data(), payload.size(), &update);
    check(status.ok(), "parse_id11_state succeeds");
    if (!status.ok()) {
        std::fprintf(stderr, "  %s: %s\n", hex, status.message().c_str());
    }
    return update;
}

const joc::oamd::ObjectRenderState& block0(const joc::oamd::OamdUpdate& update,
                                           int object) {
    return update.frame_state.objects[static_cast<std::size_t>(object)]
        .blocks[0]
        .render;
}

// Two object blocks in one object element: block 0 is a full update, block 1 is
// an inactive block.  Ported from test_multiblock_state_expansion.
void test_multiblock_state_expansion() {
    const joc::oamd::OamdUpdate update = parse_state("0104600800450132831d6a00");
    check(update.has_frame_state, "frame state present");
    check(update.frame_state.object_count == 1, "one object");
    check(update.frame_state.block_count == 2, "two blocks");
    check(update.frame_state.objects.size() == 1, "one object state");
    check(update.frame_state.objects[0].blocks.size() == 2, "two object blocks");
    const joc::oamd::ObjectRenderState& first = block0(update, 0);
    check(first.gain_q == 23033, "block 0 gain_q");
    check(first.priority_q == 32767, "block 0 priority_q");
    check(first.qx == 6342 && first.qy == 21141 && first.qz == -13107, "block 0 position");
    check(first.zone_idx == 3 && first.enable_elevation, "block 0 zone/elevation");
    check(first.size_mode == 1, "block 0 size mode");
    check(first.snap, "block 0 snap");
    const joc::oamd::ObjectRenderState& second =
        update.frame_state.objects[0].blocks[1].render;
    // The second block is active but carries basic_status 0: no gain update.
    check(!second.not_active, "block 1 active");
    check(second.gain_q == 0 && second.priority_q == 0, "block 1 gain/priority");
    check(second.qx == 16384 && second.qy == 16384 && second.qz == 0,
          "block 1 default position");
}

// Two objects with three blocks: the first object's block 1 is inactive, the
// second is a full update.  Ported from test_multiblock_gain_inheritance.
void test_multiblock_gain_inheritance() {
    const joc::oamd::OamdUpdate update =
        parse_state("038474100040850146527514c0407c3e00");
    check(update.frame_state.objects.size() == 2, "two objects");
    check(update.frame_state.block_count == 3, "three blocks");
    const std::int32_t expected_gain[3] = {23033, 0, 2052};
    for (int block = 0; block < 3; ++block) {
        check(update.frame_state.objects[0].blocks[static_cast<std::size_t>(block)]
                      .render.gain_q == expected_gain[block],
              "first object gain inheritance");
        check(update.frame_state.objects[1].blocks[static_cast<std::size_t>(block)]
                      .render.gain_q == expected_gain[block],
              "second object gain inheritance");
    }
    check(update.frame_state.objects[0].blocks[1].render.not_active,
          "first object block 1 inactive");
    check(!update.frame_state.objects[1].blocks[1].render.not_active,
          "second object block 1 active");
    for (int block = 0; block < 3; ++block) {
        check(update.frame_state.objects[0].blocks[static_cast<std::size_t>(block)]
                      .render.qx == 16384,
              "first object holds centre x");
        check(update.frame_state.objects[1].blocks[static_cast<std::size_t>(block)]
                      .render.qx == 10570,
              "second object x");
    }
}

// gain_idx 0/1/2: unity, mute and the 6-bit dB table.
void test_gain_formula() {
    check(block0(parse_state("010458000501514C0400"), 0).gain_q == 23033, "gain bits 0");
    check(block0(parse_state("01045800057F514C0400"), 0).gain_q == 14, "gain bits 63");
    check(block0(parse_state("0104540004D4530100"), 0).gain_q == 0, "gain mute");
}

// size_idx 1/2: isotropic and per-axis Q15 sizes.
void test_size_formula() {
    const joc::oamd::OamdUpdate size16 = parse_state("01045800045453016000");
    check(block0(size16, 0).size_mode == 1, "size mode 1");
    check(block0(size16, 0).size_q[0] == 16913, "size value 16");
    check(block0(size16, 0).size_q[1] == 16913 && block0(size16, 0).size_q[2] == 16913,
          "isotropic size");
    const joc::oamd::OamdUpdate size31 = parse_state("01045800045453017E00");
    check(block0(size31, 0).size_q[0] == 32767, "size value 31");
    const joc::oamd::OamdUpdate size2 = parse_state("01045C00045453018AAA00");
    check(block0(size2, 0).size_mode == 2, "size mode 2");
    check(block0(size2, 0).size_q[0] == 5285 && block0(size2, 0).size_q[1] == 10570 &&
              block0(size2, 0).size_q[2] == 21141,
          "per-axis size");
}

// Position quantisation, zone/snap, elevation flag and signed z.
void test_position_formula() {
    const joc::oamd::OamdUpdate base = parse_state("010454000454530100");
    check(block0(base, 0).qx == 10570 && block0(base, 0).qy == 10570 &&
              block0(base, 0).qz == 17476,
          "coarse position");
    check(block0(base, 0).enable_elevation, "elevation enabled");
    check(block0(base, 0).zone_idx == 0, "zone 0");

    const joc::oamd::OamdUpdate zone = parse_state("0104540004420A0B10");
    check(block0(zone, 0).qx == 1057 && block0(zone, 0).qy == 1057 && block0(zone, 0).qz == 0,
          "zone position");
    check(block0(zone, 0).zone_idx == 5, "zone index");
    check(block0(zone, 0).snap, "snap flag");

    const joc::oamd::OamdUpdate height = parse_state("010454000454530000");
    check(!block0(height, 0).enable_elevation, "elevation disabled");

    const joc::oamd::OamdUpdate negative = parse_state("010454000454510100");
    check(block0(negative, 0).qz == -17476, "signed elevation");
}

// Extended object element position refinement.
void test_refinement() {
    const joc::oamd::OamdUpdate update = parse_state("010854000454530100A23E00");
    check(block0(update, 0).qx == 10676, "refined x");
    check(block0(update, 0).qy == 10676, "refined y");
    check(block0(update, 0).qz == 17913, "refined z");
}

// Extended object element divergence table.
void test_divergence() {
    check(block0(parse_state("010854000454530100A27400"), 0).divergence_q == 11548,
          "divergence table value 32");
    check(block0(parse_state("010854000454530100A277E0"), 0).divergence_q == 16384,
          "divergence table value 63");
}

// Programme assignment: bed labels, ISF index, masks and distribution flag.
void test_program_assignment() {
    const joc::oamd::OamdUpdate lfe = parse_state("018448000440");
    check(lfe.frame_state.program.dynamic_object_only, "dynamic object only");
    check(lfe.frame_state.program.lfe_present, "lfe present");
    check(lfe.frame_state.program.num_bed_objects == 1, "one bed object");
    check(lfe.frame_state.program.bed_labels.size() == 1 &&
              lfe.frame_state.program.bed_labels[0] == joc::joc::BedChannel::kLfe,
          "lfe bed label");
    check(lfe.frame_state.program.bed_mask == 0x8u, "lfe bed mask");
    check(lfe.frame_state.objects[0].kind == joc::oamd::ObjectKind::kBed, "bed object kind");
    check(block0(lfe, 0).qx == 16384 && block0(lfe, 0).qy == 16384 && block0(lfe, 0).qz == 0,
          "bed object holds centre");

    const joc::oamd::OamdUpdate isf = parse_state("062011400011084200");
    check(isf.frame_state.object_count == 4, "isf object count");
    check(isf.frame_state.program.isf_idx == 0, "isf index");
    check(isf.frame_state.program.num_isf_objects == 4, "isf object number");
    for (int index = 0; index < 4; ++index) {
        check(isf.frame_state.objects[static_cast<std::size_t>(index)].kind ==
                  joc::oamd::ObjectKind::kIsf,
              "isf object kind");
    }

    const joc::oamd::OamdUpdate standard = parse_state("021100422600022100");
    check(standard.frame_state.program.num_bed_objects == 2, "standard bed object number");
    check(standard.frame_state.program.bed_mask == 0x3u, "standard bed wide mask");
    check(standard.frame_state.program.bed_labels.size() == 2 &&
              standard.frame_state.program.bed_labels[0] == joc::joc::BedChannel::kLeft &&
              standard.frame_state.program.bed_labels[1] == joc::joc::BedChannel::kRight,
          "standard bed labels");

    const joc::oamd::OamdUpdate nonstandard = parse_state("001000800448000440");
    check(nonstandard.frame_state.program.num_bed_objects == 1, "non-standard bed number");
    check(nonstandard.frame_state.program.bed_mask == 0x100u, "non-standard bed mask");
    check(nonstandard.frame_state.program.bed_labels.size() == 1 &&
              nonstandard.frame_state.program.bed_labels[0] ==
                  joc::joc::BedChannel::kTopFrontLeft,
          "non-standard bed label");
    check(!nonstandard.frame_state.program.b_bed_chan_distribute, "distribution off");

    const joc::oamd::OamdUpdate distributed = parse_state("001800800448000440");
    check(distributed.frame_state.program.b_bed_chan_distribute, "distribution on");
}

// The parsed bed declaration keeps its expanded channels for the ADM check.
void test_bed_assignments() {
    const std::vector<std::uint8_t> payload = from_hex("021100422600022100");
    std::vector<joc::oamd::BedAssignment> assignments;
    check(joc::oamd::parse_id11_bed_assignments(payload.data(), payload.size(),
                                                &assignments)
              .ok(),
          "bed assignment parse");
    check(assignments.size() == 1, "one assignment");
    check(assignments[0].standard && !assignments[0].lfe_only, "standard assignment");
    check(assignments[0].channels.size() == 2 &&
              assignments[0].channels[0] == joc::joc::BedChannel::kLeft &&
              assignments[0].channels[1] == joc::joc::BedChannel::kRight,
          "assignment channels");
}

// Legacy frame_update() view: 16 slots from block 0, >16 objects and multi-block
// frames stay rejected; the full state accepts both.
void test_legacy_view() {
    const std::vector<std::uint8_t> base_payload = from_hex("010454000454530100");
    joc::oamd::OamdUpdate legacy;
    check(joc::oamd::parse_id11(base_payload.data(), base_payload.size(), &legacy).ok(),
          "legacy parse");
    check(legacy.has_frame_state, "legacy carries frame state");
    check(legacy.frame_state.block_count == 1, "legacy block count");
    check(legacy.frame_state.blocks[0].block_offset_samples == 0 &&
              legacy.frame_state.blocks[0].ramp_duration_samples == 0,
          "legacy block timing");
    check(legacy.slots[0].q1 == 10570 && legacy.slots[0].q2 == 10570 &&
              legacy.slots[0].q3 == 17476,
          "legacy slot position");
    check(legacy.slots[1].q1 == -1 && legacy.slots[1].q2 == -1 && legacy.slots[1].q3 == -1,
          "legacy unused slot hold");

    // The legacy slot view never publishes a negative elevation.
    const std::vector<std::uint8_t> negative = from_hex("010454000454510100");
    check(joc::oamd::parse_id11(negative.data(), negative.size(), &legacy).ok(),
          "negative legacy parse");
    check(legacy.slots[0].q3 == 0, "legacy negative z clamps to side");

    const std::vector<std::uint8_t> elevation_off = from_hex("010454000454530000");
    check(joc::oamd::parse_id11(elevation_off.data(), elevation_off.size(), &legacy).ok(),
          "elevation-off legacy parse");
    check(legacy.slots[0].q3 == 0, "legacy elevation flag zeroes z");

    // Bed objects have no position field: the slot keeps the previous position.
    const std::vector<std::uint8_t> bed = from_hex("018448000440");
    check(joc::oamd::parse_id11(bed.data(), bed.size(), &legacy).ok(), "bed legacy parse");
    check(legacy.slots[0].q1 == -1 && legacy.slots[0].q2 == -1 && legacy.slots[0].q3 == -1,
          "legacy bed slot holds");

    const std::vector<std::uint8_t> multiblock = from_hex("0104600800450132831d6a00");
    const joc::Status legacy_status =
        joc::oamd::parse_id11(multiblock.data(), multiblock.size(), &legacy);
    check(!legacy_status.ok() &&
              legacy_status.code() == JOC_ERR_OAMD_UNSUPPORTED_VARIANT,
          "legacy rejects multiple blocks");

    const char* many_objects_hex =
        "21044E600022A29808054530100A8A60201514C0402A29808054530100A8A60201514C040"
        "2A29808054530100A8A60201514C0402A29808054530100A8A60201514C0402A2980800";
    const std::vector<std::uint8_t> many_objects = from_hex(many_objects_hex);
    const joc::Status many_legacy =
        joc::oamd::parse_id11(many_objects.data(), many_objects.size(), &legacy);
    check(!many_legacy.ok() && many_legacy.code() == JOC_ERR_OAMD_UNSUPPORTED_VARIANT,
          "legacy rejects more than 16 objects");
    joc::oamd::OamdUpdate many;
    check(joc::oamd::parse_id11_state(many_objects.data(), many_objects.size(), &many).ok(),
          "full state accepts more than 16 objects");
    check(many.frame_state.object_count == 17, "full state object count");
    check(many.frame_state.objects.size() == 17, "full state expands every object");
}

// Optional position fields must be consumed exactly: distance, screen reference
// and the priority mask.
void test_optional_fields() {
    const joc::oamd::OamdUpdate distance = parse_state("0304680004545313880D45301000");
    check(distance.frame_state.object_count == 2, "distance payload object count");
    check(block0(distance, 0).qx == 10570, "distance payload keeps alignment");
    check(block0(distance, 1).gain_q == 0, "distance payload second object");

    const joc::oamd::OamdUpdate infinity = parse_state("010454000454531880");
    check(block0(infinity, 0).qz == 17476, "infinite distance keeps alignment");

    const joc::oamd::OamdUpdate screen = parse_state("01045800045453013600");
    check(block0(screen, 0).qx == 10570 && block0(screen, 0).snap == false,
          "screen reference keeps alignment");

    const joc::oamd::OamdUpdate priority = parse_state("03046800043EA29808000A298080");
    check(block0(priority, 0).priority_q == 31744, "priority bits");
    check(block0(priority, 1).priority_q == 0, "priority zero");

    const joc::oamd::OamdUpdate inactive = parse_state("010448000600");
    check(block0(inactive, 0).not_active, "inactive object");
    check(block0(inactive, 0).gain_q == 0 && block0(inactive, 0).priority_q == 0,
          "inactive gain/priority");
    check(block0(inactive, 0).qx == 16384 && block0(inactive, 0).qy == 16384 &&
              block0(inactive, 0).qz == 0,
          "inactive default position");
}

// The compiled-in ISF tables must be complete: six configurations, twelve
// layout descriptors, four layout groups, 73 paired column flags, and exact
// matrix coefficients.
void test_metadata_params() {
    const joc::oamd::MetadataParams* params = joc::oamd::metadata_params();
    check(params != nullptr, "metadata params");
    if (params == nullptr) {
        std::fprintf(stderr, "  %s\n", joc::oamd::metadata_params_error());
        return;
    }
    check(params->configs.size() == 6, "six ISF configurations");
    check(params->layouts.size() == 12, "twelve ISF layout descriptors");
    check(!params->layout_groups.floor_a.empty() && !params->layout_groups.floor_b.empty() &&
              !params->layout_groups.height.empty() &&
              !params->layout_groups.additional.empty(),
          "four layout groups");
    check(!params->paired_columns.empty(), "paired column flags");
    for (const joc::oamd::IsfConfig& config : params->configs) {
        check(config.counts[4] == static_cast<int>(config.primary_row_map.size()),
              "isf row count");
        // Every configuration stores its basis with a 73-column row stride.
        check(config.matrix.size() ==
                  static_cast<std::size_t>(config.counts[4]) * 73u,
              "isf matrix size");
    }
    const joc::oamd::IsfLayout* layout20 = params->layout("20");
    const joc::oamd::IsfLayout* layout916 = params->layout("916");
    check(layout20 != nullptr && layout20->output_count == 2 &&
              layout20->compensation_value == -4.5,
          "2.0 layout descriptor");
    check(layout916 != nullptr && layout916->output_count == 16 &&
              layout916->compensation_value == 0.0,
          "9.1.6 layout descriptor");
    check(params->layout("60") != nullptr && params->layout("60")->output_count == 6 &&
              params->layout("60")->compensation_value == -1.5,
          "6.0 layout descriptor");
    check(!params->layout("unknown"), "unknown layout descriptor");

    // Exact table contents: the config counts and the first coefficients pin
    // the reference data, and the row/column structure follows from them.
    const joc::oamd::IsfConfig& first = params->configs[0];
    check(first.counts[0] == 3 && first.counts[1] == 1 && first.counts[2] == 0 &&
              first.counts[3] == 0 && first.counts[4] == 4,
          "first config counts");
    check(first.matrix.size() == 4u * 73u && first.matrix[0] == 0.806554496f &&
              first.matrix[1] == 0.55179733f && first.matrix[15] == -0.280463845f,
          "first config coefficients");
    check(first.primary_row_map.size() == 4 && first.primary_row_map[3] == 3 &&
              first.secondary_row_map.size() == 4 && first.secondary_row_map[3] == 3,
          "first config row maps");
    const joc::oamd::IsfConfig& last = params->configs[5];
    check(last.counts[0] == 15 && last.counts[1] == 9 && last.counts[2] == 5 &&
              last.counts[3] == 1 && last.counts[4] == 30 &&
              last.matrix.size() == 30u * 73u,
          "last config shape");

    check(params->layout("20")->output_ids.size() == 37 &&
              params->layout("20")->output_ids[0] == 0 &&
              params->layout("20")->output_ids[1] == 1 &&
              params->layout("20")->output_ids[2] == 0,
          "layout output ids");
    check(params->layout_groups.floor_a.size() == 6 &&
              params->layout_groups.floor_b.size() == 7 &&
              params->layout_groups.height.size() == 6 &&
              params->layout_groups.additional.size() == 5,
          "layout group sizes");
    check(params->layout_groups.floor_a[1].selector == 1 &&
              params->layout_groups.floor_a[1].begin == 1 &&
              params->layout_groups.floor_a[1].end == 3,
          "layout group record");
    const std::size_t paired = static_cast<std::size_t>(
        std::count(params->paired_columns.begin(), params->paired_columns.end(), true));
    check(params->paired_columns.size() == 73 && paired == 67, "paired column flags");
}

// q_of is the reference's Q15 quantiser.
void test_q_of() {
    check(joc::oamd::q_of(0, 62) == 0, "q_of 0");
    check(joc::oamd::q_of(62, 62) == 32767, "q_of max");
    check(joc::oamd::q_of(31, 62) == 16384, "q_of centre");
    check(joc::oamd::q_of(15, 15) == 32767, "q_of z max");
    check(joc::oamd::q_of(20, 62) == 10570, "q_of 20/62");
    check(joc::oamd::q_of(8, 15) == 17476, "q_of 8/15");
}

}  // namespace

int main() {
    test_multiblock_state_expansion();
    test_multiblock_gain_inheritance();
    test_gain_formula();
    test_size_formula();
    test_position_formula();
    test_refinement();
    test_divergence();
    test_program_assignment();
    test_bed_assignments();
    test_legacy_view();
    test_optional_fields();
    test_metadata_params();
    test_q_of();
    if (failures != 0) {
        std::fprintf(stderr, "%d OAMD metadata check(s) failed\n", failures);
        return 1;
    }
    std::printf("OAMD metadata checks passed\n");
    return 0;
}
