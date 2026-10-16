// Port of src/joc_decode.py.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "eac3_transport/eac3_reader.h"
#include "joc_core.h"
#include "emdf/emdf_parser.h"
#include "foundation/status.h"
#include "oamd/oamd_parser.h"

namespace joc::joc {

struct ObjectSymbols {
    std::uint8_t present = 0;
    std::uint8_t sparse = 0;
    std::uint8_t n_bands = 0;
    std::uint8_t n_dpoints = 0;
    std::uint8_t n_channels = 0;
    std::int64_t q[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS] = {};
    std::int16_t mtx[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS] = {};
    std::uint8_t idx[JOC_MAX_DPOINTS][JOC_MAX_PARAMETER_BANDS] = {};
    std::int16_t vec[JOC_MAX_DPOINTS][JOC_MAX_PARAMETER_BANDS] = {};
};

struct FrameSymbols {
    std::uint8_t n_objects = 0;
    std::uint8_t n_channels = 0;
    ObjectSymbols objects[JOC_MAX_OBJECTS];
};

Status parse_id14(const std::uint8_t* payload, std::size_t payload_size, joc_frame_params* out,
                  FrameSymbols* symbols);

Status parse_eac3_frame(const std::uint8_t* frame, std::size_t frame_size, joc_frame_params* out,
                        emdf::Container* container, FrameSymbols* symbols);

Status check_id14_padding(const std::uint8_t* payload, std::size_t payload_size,
                          std::uint32_t* out_trailing_bits);

// Scores this container as the frame's JOC metadata when the syncframe holds more
// than one (spec section 8.2 and its Table 56 associate the OAMD and JOC payloads
// of one frame by an equal group id).  2 = the two payloads share a group id and
// ID14 parses as a complete joc() payload, 1 = ID14 parses, 0 = neither.  A
// container whose ID14 declares far more objects than its bytes could hold, or
// whose ID11 carries no group id at all, is some other structure.
int score_container_as_joc(const std::uint8_t* data, std::size_t size,
                           const emdf::Container& container);

// Parameters for an audio frame that carries no JOC payload: every object is
// "present" with an all-zero matrix, so the kernel renders silence while its
// analysis and synthesis state keep advancing in step with the stream.
void make_silent_frame_params(joc_frame_params* out);

std::int16_t num_channels_for_config(std::uint32_t dmx_config_idx);

std::int16_t num_bands_for_index(std::uint32_t num_bands_idx);

// ---------------------------------------------------------------------------
// Channel identity / bed layout (spec sections 3, 10.2, 10.4)
//
// This is the position layer only: it decides which decoded bed channel feeds
// which JOC input lane and which ADM bed slot the extra pair occupies.  The DSP
// never reads it.
// ---------------------------------------------------------------------------

enum class Position : std::uint8_t {
    kNone = 0,
    kLeft,
    kRight,
    kCenter,
    kLfe,
    kLeftSurround,
    kRightSurround,
    kLeftRearSurround,
    kRightRearSurround,
    kVerticalHeightLeft,
    kVerticalHeightRight,
    kTopSurroundLeft,
    kTopSurroundRight,
    kCenterSurround,
    kTopSurround,
    kVerticalHeightCenter,
    kLfe2,
};

// Table E2.5 expansion; the field's most significant bit is table bit 0.
void expand_chanmap(std::uint16_t chanmap, std::vector<Position>* out);

// A/52 acmod -> full-band positions in decoding order (LFE excluded).
void acmod_positions(std::uint8_t acmod, std::vector<Position>* out);

// The pair joc_dmx_config_idx declares; empty for configurations 0 and 3.
void expected_pair(std::uint8_t dmx_config_idx, std::vector<Position>* out);

// The extra pair a dependent-substream chanmap adds on top of the core channels.
void chanmap_extra_pair(const std::vector<Position>& core, std::uint16_t chanmap,
                        std::vector<Position>* out);

// Decoded-bed channel count implied by the JOC core channel count.
int bed_channels_for_core(int core_channels);

// bed[lane] = decoded PCM channel index feeding JOC lane 'lane'.
struct BedPlan {
    int channels = 0;                                              // 6 or 8
    int lfe_channel = -1;
    int lane[JOC_MAX_CORE_CHANNELS] = {-1, -1, -1, -1, -1, -1, -1};
    int extra_pair_slot[2] = {-1, -1};                             // ADM skeleton
};

Status compute_bed_plan(const eac3::Frame* core_frame, bool has_core, bool has_dependent,
                        bool has_chanmap, std::uint16_t chanmap, std::uint8_t dmx_config_idx,
                        BedPlan* out);

// Bed plan driven by joc_dmx_config_idx alone, for callers that hold no L2
// header: the extra pair is the one the configuration declares.
Status compute_bed_plan_for_config(std::uint8_t dmx_config_idx, BedPlan* out);

// Identity disagreements that leave the DSP input unchanged (spec section 10.1:
// channel identity never enters the DSP): the extra pair's name is metadata only,
// so a mismatch is reported through CrossLayerNotes instead of refusing the frame.
struct CrossLayerNotes {
    bool chanmap_pair_mismatch = false;  // spec 11.2 check 1: chanmap vs k
    bool bed_mask_mismatch = false;      // spec 11.2 check 2: OAMD bed mask vs k
};

// Spec section 11.2 cross-layer consistency.  A mismatch in the core channel count
// or in joc_ext_config_idx changes the DSP input and is a variant error; the two
// identity checks are recorded in `notes` and the frame still renders.
Status check_cross_layer(const joc_frame_params& params, const eac3::Frame* core_frame,
                         bool has_core, bool has_dependent, bool has_chanmap,
                         std::uint16_t chanmap,
                         const std::vector<oamd::BedAssignment>* bed_assignments,
                         std::string* error, CrossLayerNotes* notes = nullptr);

}  // namespace joc::joc
