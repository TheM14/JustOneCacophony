// Port of src/oamd_bits.py.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "foundation/status.h"
#include "joc_core.h"

namespace joc::oamd {

inline constexpr int kMaxSlots = 16;
inline constexpr int kObjects = 15;
inline constexpr int kAxes = 3;
inline constexpr int kNQ12 = 62;
inline constexpr int kNQ3 = 15;
inline constexpr int kQ15Scale = 32768;

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

    // Objects 1..15 as q15 triples, the layout the speaker renderer consumes.
    void object_positions_q15(std::uint16_t out[kObjects][kAxes]) const {
        for (int object = 0; object < kObjects; ++object) {
            for (int axis = 0; axis < kAxes; ++axis) {
                out[object][axis] = static_cast<std::uint16_t>(q_[object + 1][axis]);
            }
        }
    }

private:
    std::int16_t q_[kMaxSlots][kAxes] = {};
};

// q_of(k, n) = min(32767, floor(32768 k / n + 0.5)), the reference's quantiser.
int q_of(int k, int n);

Status parse_id11(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out);

struct BedAssignment {
    bool lfe_only = false;
    bool standard = false;
    std::uint32_t mask = 0;
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

Status parse_id11_verbose(const std::uint8_t* payload, std::size_t payload_size, OamdUpdate* out,
                          ParseTrace* trace);

}  // namespace joc::oamd
