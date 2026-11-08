// Port of src/oamd_bits.py.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "foundation/status.h"
#include "joc_core.h"

namespace joc::joc {

// OAMD Table 12 的标准 bed 声道。LSB 顺序与声明顺序一致：L/R、C、LFE、
// Ls/Rs、Lrs/Rrs、TFL/TFR、TSL/TSR、TBL/TBR、LW/RW、LFE2。
// The enum values are the reference's BED_LABEL_INDEX - 1.
enum class BedChannel : std::uint8_t {
    kLeft = 0,
    kRight,
    kCenter,
    kLfe,
    kLeftSurround,
    kRightSurround,
    kLeftRearSurround,
    kRightRearSurround,
    kTopFrontLeft,
    kTopFrontRight,
    kTopSideLeft,
    kTopSideRight,
    kTopBackLeft,
    kTopBackRight,
    kLeftWide,
    kRightWide,
    kLfe2,
    kCount,
};

}  // namespace joc::joc

namespace joc::oamd {

inline constexpr int kMaxSlots = 16;
inline constexpr int kObjects = 15;
inline constexpr int kAxes = 3;
inline constexpr int kNQ12 = 62;
inline constexpr int kNQ3 = 15;
inline constexpr int kQ15Scale = 32768;
// Largest ISF configuration of Table 11b (5.6.0.11). A frame cannot carry more
// objects than this through any of the standard programmes, but the parser does
// not reject a larger count: it only bounds the legacy 16-slot view.
inline constexpr int kMaxObjects = 30;
inline constexpr int kMaxBlocks = 8;
inline constexpr std::uint32_t kTrimElementId = 2;
inline constexpr std::uint32_t kObjectElementId = 1;
inline constexpr std::uint32_t kExtendedElementId = 5;
inline constexpr int kTrimConfigs = 9;
inline constexpr int kBedLabelCount = 17;

// One of the nine per-configuration trim records of the OAMD trim element.
// ``kAbsent`` means the configuration carries no data (the element only
// carries explicit records when global_trim_mode is 2).
enum class TrimConfigKind : std::uint8_t {
    kAbsent = 0,
    kDefault,
    kDisabled,
    kExplicit,
};

struct TrimConfig {
    TrimConfigKind kind = TrimConfigKind::kAbsent;
    std::uint8_t presence = 0;  // b_*_trim_present bits of an explicit record
    double centre = 0.0;        // dB
    double surround = 0.0;      // dB
    double height = 0.0;        // dB
    double balance_overhead = 0.0;  // bal3d_y_tb, [-1, 1]
    double balance_listener = 0.0;  // bal3d_y_lis, [-1, 1]
};

// Programme calibration carried by the OAMD trim element: warp mode, global
// trim mode, the nine configuration records and the per-object bypass flags.
struct OamdTrim {
    std::uint32_t warp_mode = 0;
    std::uint32_t global_trim_mode = 0;
    TrimConfig configs[kTrimConfigs];
    std::uint32_t object_count = 0;
    bool disable_per_object[kMaxSlots] = {};
};

// OAMD object kind: the programme assigns objects to the bed, to an ISF
// configuration, or to the dynamic (positional) set, in that order.
enum class ObjectKind : std::uint8_t {
    kBed = 0,
    kIsf = 1,
    kDynamic = 2,
};

// Render state of one object in one block (5.5.9 object_info_block).
struct ObjectRenderState {
    bool not_active = false;
    std::int32_t gain_q = 0;      // integer gain, unity = 4096
    std::int32_t priority_q = 0;  // Q10 priority, 32767 = default
    std::int32_t qx = 0;          // Q15 room position, 0..32767
    std::int32_t qy = 0;          // Q15 room position, 0..32767
    std::int32_t qz = 0;          // signed Q15 elevation
    std::uint8_t zone_idx = 0;
    bool enable_elevation = true;
    std::uint8_t size_mode = 0;   // 0 none, 1 isotropic, 2 per-axis
    std::uint16_t size_q[3] = {0, 0, 0};  // Q15 size per axis
    bool snap = false;
    std::uint16_t divergence_q = 0;  // Q15 divergence width
};

struct ObjectBlockState {
    ObjectRenderState render;
};

// One object of the frame: its kind and the state of every block.
struct ObjectElementState {
    ObjectKind kind = ObjectKind::kDynamic;
    std::vector<ObjectBlockState> blocks;
};

// Timing shared by every object of one block. ``block_offset_samples`` already
// includes the element's sample_offset, like the reference block dictionary.
struct BlockTiming {
    std::uint32_t block_offset_samples = 0;
    std::uint32_t ramp_duration_samples = 0;
};

// Program assignment (5.5.3) as consumed by the speaker renderer.
struct ProgramState {
    bool dynamic_object_only = false;
    bool lfe_present = false;
    bool b_bed_chan_distribute = false;
    // program_assignment() content_description flags.
    bool content_reserved = false;
    bool content_bed = false;
    bool content_isf = false;
    bool content_dynamic = false;
    std::uint32_t num_bed_objects = 0;
    std::uint32_t num_isf_objects = 0;
    std::int32_t num_dynamic_objects = -1;  // -1 when the frame declares none
    std::uint32_t bed_mask = 0;             // 17-bit non-standard bed mask
    std::int32_t isf_idx = -1;              // -1 when the frame declares none
    // Declared bed labels in channel order, including LFE/LFE2. Index i is the
    // label of object i for i < num_bed_objects.
    std::vector<::joc::joc::BedChannel> bed_labels;
};

// One element whose parsed content ended past its declared size. The reference
// keeps parsing such elements and only records the difference.
struct ElementSizeMismatch {
    std::uint32_t element_ordinal = 0;
    std::uint32_t element_id = 0;
    std::uint32_t declared_size_bytes = 0;
    std::uint32_t declared_end_bit = 0;
    std::uint32_t parsed_end_bit = 0;
};

// Full frame state (frame_state() in the reference): per-object per-block
// render state plus programme assignment, timing and trim.
struct FrameState {
    std::uint32_t object_count = 0;
    std::uint32_t block_count = 0;
    std::uint32_t sample_offset = 0;            // object_element sample offset
    std::uint32_t ramp_duration_samples = 0;    // block 0
    std::vector<BlockTiming> blocks;
    std::vector<ObjectElementState> objects;
    ProgramState program;
    bool has_trim = false;
    OamdTrim trim;
    std::vector<ElementSizeMismatch> size_mismatch;
};

// (the reference's None), i.e. hold the previous position.
struct SlotUpdate {
    std::int16_t q1 = -1;
    std::int16_t q2 = -1;
    std::int16_t q3 = -1;
};

struct OamdUpdate {
    SlotUpdate slots[kMaxSlots];
    std::uint32_t block_offset_samples = 0;
    std::uint32_t ramp_duration_samples = 0;
    std::uint32_t object_count = 0;
    bool dynamic_object_only = false;
    bool lfe_present = false;
    std::uint32_t num_bed_objects = 0;
    std::uint32_t num_isf_objects = 0;
    std::int32_t num_dynamic_objects = -1;
    // program_assignment() 的 content_description：bed / ISF / 动态对象三类内容。
    bool content_bed = false;
    bool content_isf = false;
    bool content_dynamic = false;
    // Trim element of this payload.  The element describes the programme, so
    // ``has_trim`` only marks the payloads that carry it; a payload without it
    // keeps whatever controls the caller already holds.
    OamdTrim trim;
    bool has_trim = false;
    // Full per-object per-block state of this payload. ``has_frame_state`` is
    // true whenever the parser produced it; the speaker metadata path and the
    // binaural position timeline consume it, while the legacy slot fields above
    // stay authoritative for ADM export and the point-renderer path.
    bool has_frame_state = false;
    FrameState frame_state;
};

class OamdState {
public:
    OamdState() { reset(); }

    void reset() {
        for (int slot = 0; slot < kMaxSlots; ++slot) {
            for (int axis = 0; axis < kAxes; ++axis) {
                q_[slot][axis] = 0;
            }
        }
        q_[0][0] = 16384;
        q_[0][1] = 16384;
    }

    void apply(const OamdUpdate& update) {
        for (int slot = 0; slot < kMaxSlots; ++slot) {
            const SlotUpdate& value = update.slots[slot];
            if (value.q1 >= 0) { q_[slot][0] = value.q1; }
            if (value.q2 >= 0) { q_[slot][1] = value.q2; }
            if (value.q3 >= 0) { q_[slot][2] = value.q3; }
        }
    }

    std::int16_t q(int slot, int axis) const { return q_[slot][axis]; }

    // Resolves one frame-state position block for one object.  The legacy
    // apply() entry above keeps the reference frame_update() view (16 slots,
    // block 0); the binaural position timeline uses this entry so a frame may
    // carry more objects and more blocks than that view can express.  Callers
    // skip inactive and non-dynamic objects themselves, exactly like the
    // reference's frame_update() value construction.
    void apply_render_state(int object_index, const ObjectRenderState& render) {
        if (object_index < 0 || object_index >= kMaxObjects) {
            return;
        }
        q_[object_index][0] = static_cast<std::int16_t>(render.qx);
        q_[object_index][1] = static_cast<std::int16_t>(render.qy);
        const std::int32_t elevation = render.enable_elevation ? render.qz : 0;
        q_[object_index][2] =
            static_cast<std::int16_t>(elevation > 0 ? elevation : 0);
    }

    // Objects 1..15 as q15 triples, the layout the speaker renderer consumes.
    void object_positions_q15(std::uint16_t out[kObjects][kAxes]) const {
        for (int object = 0; object < kObjects; ++object) {
            for (int axis = 0; axis < kAxes; ++axis) {
                out[object][axis] = static_cast<std::uint16_t>(q_[object + 1][axis]);
            }
        }
    }

private:
    std::int16_t q_[kMaxObjects][kAxes] = {};
};

// q_of(k, n) = min(32767, floor(32768 k / n + 0.5)), the reference's quantiser.
int q_of(int k, int n);

struct BedAssignment {
    bool lfe_only = false;
    bool standard = false;
    std::uint32_t mask = 0;  // 10-bit standard or 17-bit non-standard mask
    // Declared channels in channel order; LFE is kept in place.
    std::vector<::joc::joc::BedChannel> channels;
};

struct ParseTrace {
    std::vector<BedAssignment> bed_assignments;
    std::uint32_t element_count = 0;
    std::uint32_t object_element_count = 0;
    std::uint32_t sample_offset_code = 0;
    std::uint32_t sample_offset = 0;
    std::uint32_t block_count = 0;
    std::uint32_t parsed_end_bit = 0;
    bool reserved_data_not_present = false;
};

// Legacy single-frame view (frame_update() in the reference): the 16-slot
// position update plus programme and trim.  A payload carrying more than 16
// objects or more than one position block is rejected, exactly like the
// reference legacy entry point.  The ADM export path keeps this view; the
// speaker and binaural renderers use parse_id11_state().
Status parse_id11(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out);

// Full frame_state() view: per-object per-block state, no 16-slot or
// single-block restriction.  The legacy fields are filled as well whenever the
// frame fits them, so a caller can parse once and hand the same update to the
// renderers and to the ADM export path where its shape allows it.
Status parse_id11_state(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out);

// Parses only the trim element of a payload.  ``out`` receives the first trim
// element found; ``*present`` is false when the payload carries none.  decode
// paths that need the OAMD calibration but no object state can use this
// without requiring an object element.
Status parse_trim(const std::uint8_t* payload, std::size_t payload_size, OamdTrim* out,
                  bool* present);

Status parse_id11_verbose(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out,
                          ParseTrace* trace);

// Parses only the OAMD header and program_assignment(), which is all the
// cross-layer bed-mask check (spec section 11.2 item 2) needs.
Status parse_id11_bed_assignments(const std::uint8_t* payload, std::size_t payload_size,
                                  std::vector<BedAssignment>* out);

// 10-bit standard bed mask -> 17-bit non-standard bed mask (5.6.1.1.4).
std::uint32_t standard_bed_mask(std::uint32_t bed_mask);

}  // namespace joc::oamd
