#include "joc_bitstream/joc_parser.h"

#include "joc_bitstream/bed_layout_map.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "foundation/bit_reader.h"
#include "joc_huffman_tables.h"

namespace joc::joc {

namespace {

struct NumChannelsEntry {
    std::uint32_t config;
    std::int16_t channels;
};
constexpr NumChannelsEntry kNumChannels[] = {
    {0u, 5}, {1u, 7}, {2u, 7}, {3u, 5}, {4u, 7},
};

struct NumBandsEntry {
    std::uint32_t index;
    std::int16_t bands;
};
constexpr NumBandsEntry kNumBands[] = {
    {0u, 1}, {1u, 3}, {2u, 5}, {3u, 7}, {4u, 9}, {5u, 12}, {6u, 15}, {7u, 23},
};

// Floored modulo: Python's % operator semantics, so that the ported
inline std::int64_t floored_mod(std::int64_t value, std::int64_t modulus) {
    const std::int64_t remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

enum class SymbolKind { Mtx, Idx, Vec };

struct Tree {
    const int (*nodes)[2] = nullptr;
    int count = 0;
};

Tree select_tree(std::uint32_t quant_idx, SymbolKind kind, int n_channels) {
    Tree tree;
    switch (kind) {
        case SymbolKind::Idx:
            if (n_channels == 5) {
                tree.nodes = joc_huff_code_5ch_pos_index_sparse;
                tree.count = static_cast<int>(sizeof(joc_huff_code_5ch_pos_index_sparse) /
                                              sizeof(joc_huff_code_5ch_pos_index_sparse[0]));
            } else {
                tree.nodes = joc_huff_code_7ch_pos_index_sparse;
                tree.count = static_cast<int>(sizeof(joc_huff_code_7ch_pos_index_sparse) /
                                              sizeof(joc_huff_code_7ch_pos_index_sparse[0]));
            }
            break;
        case SymbolKind::Vec:
            if (quant_idx == 0u) {
                tree.nodes = joc_huff_code_coarse_coeff_sparse;
                tree.count = static_cast<int>(sizeof(joc_huff_code_coarse_coeff_sparse) /
                                              sizeof(joc_huff_code_coarse_coeff_sparse[0]));
            } else {
                tree.nodes = joc_huff_code_fine_coeff_sparse;
                tree.count = static_cast<int>(sizeof(joc_huff_code_fine_coeff_sparse) /
                                              sizeof(joc_huff_code_fine_coeff_sparse[0]));
            }
            break;
        case SymbolKind::Mtx:
        default:
            if (quant_idx == 0u) {
                tree.nodes = joc_huff_code_coarse_generic;
                tree.count = static_cast<int>(sizeof(joc_huff_code_coarse_generic) /
                                              sizeof(joc_huff_code_coarse_generic[0]));
            } else {
                tree.nodes = joc_huff_code_fine_generic;
                tree.count = static_cast<int>(sizeof(joc_huff_code_fine_generic) /
                                              sizeof(joc_huff_code_fine_generic[0]));
            }
            break;
    }
    return tree;
}

// infinite loop (plan 40.4 BL-6).
bool huff_decode(const Tree& tree, bits::BitReader& reader, std::int16_t* out_value) {
    int node = 0;
    int steps = 0;
    while (node >= 0) {
        if (node >= tree.count || steps > tree.count) {
            reader.fail(JOC_ERR_JOC_SYNTAX, "Huffman tree walk left the valid node range");
            return false;
        }
        ++steps;
        const std::uint32_t bit = reader.read(1);
        if (reader.failed()) {
            return false;
        }
        node = tree.nodes[node][bit];
    }
    *out_value = static_cast<std::int16_t>(-node - 1);
    return true;
}

Status syntax_fail(const std::string& message) {
    return Status::fail(JOC_ERR_JOC_SYNTAX, stage::kJoc, message);
}

Status truncated_fail(const bits::BitReader& reader) {
    if (reader.error() == JOC_ERR_JOC_SYNTAX) {
        return syntax_fail(reader.error_message());
    }
    return Status::fail(JOC_ERR_BITSTREAM_TRUNCATED, stage::kJoc,
                        std::string("JOC bitstream truncated: ") + reader.error_message());
}

void reconstruct_dense(const ObjectSymbols& symbols, std::uint32_t dp, int n_channels, std::int64_t nquant,
                       std::int64_t offset,
                       std::int64_t q[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS]) {
    for (int ch = 0; ch < n_channels; ++ch) {
        q[dp][ch][0] =
            floored_mod(offset + static_cast<std::int64_t>(symbols.mtx[dp][ch][0]), nquant);
        for (int pb = 1; pb < symbols.n_bands; ++pb) {
            q[dp][ch][pb] = floored_mod(
                q[dp][ch][pb - 1] + static_cast<std::int64_t>(symbols.mtx[dp][ch][pb]), nquant);
        }
    }
}

// across parameter bands and is deliberately NOT reset when the active channel
Status reconstruct_sparse(
    const ObjectSymbols& symbols, std::uint32_t dp, int n_channels, std::int64_t nquant, std::int64_t offset,
    std::int64_t q[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS]) {
    if (n_channels != 5 && n_channels != 7) {
        return Status::fail(JOC_ERR_JOC_UNSUPPORTED_VARIANT, stage::kJoc,
                            "sparse JOC requires 5 or 7 core channels, got " +
                                std::to_string(n_channels));
    }
    const int initial_channel = symbols.idx[dp][0];
    if (initial_channel < 0 || initial_channel >= n_channels) {
        return syntax_fail("sparse JOC initial channel " + std::to_string(initial_channel) +
                           " out of range for " + std::to_string(n_channels) + " channels");
    }
    // Non-active entries take nquant/2, which dequantizes to exactly zero.
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int pb = 0; pb < symbols.n_bands; ++pb) {
            q[dp][ch][pb] = nquant / 2;
        }
    }
    int active = initial_channel;
    std::int64_t coefficient = offset;
    for (int pb = 0; pb < symbols.n_bands; ++pb) {
        if (pb != 0) {
            active = static_cast<int>(
                floored_mod(static_cast<std::int64_t>(active) + symbols.idx[dp][pb], n_channels));
        }
        coefficient = floored_mod(coefficient + static_cast<std::int64_t>(symbols.vec[dp][pb]), nquant);
        q[dp][active][pb] = coefficient;
    }
    return Status::success();
}

}  // namespace

std::int16_t num_channels_for_config(std::uint32_t dmx_config_idx) {
    for (const NumChannelsEntry& entry : kNumChannels) {
        if (entry.config == dmx_config_idx) {
            return entry.channels;
        }
    }
    return -1;
}

std::int16_t num_bands_for_index(std::uint32_t num_bands_idx) {
    for (const NumBandsEntry& entry : kNumBands) {
        if (entry.index == num_bands_idx) {
            return entry.bands;
        }
    }
    return -1;
}

Status parse_id14(const std::uint8_t* payload, std::size_t payload_size, joc_frame_params* out,
                  FrameSymbols* symbols) {
    if (payload == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kJoc, "null payload or output");
    }
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(joc_frame_params);
    out->struct_version = JOC_FRAME_PARAMS_VERSION;
    // capture is purely additive and never changes the parse result.
    FrameSymbols local_symbols{};
    FrameSymbols& capture = (symbols != nullptr) ? *symbols : local_symbols;
    std::memset(&capture, 0, sizeof(capture));

    bits::BitReader reader(payload, payload_size);

    out->dmx_config_idx = static_cast<std::uint8_t>(reader.read(3));
    out->num_objects_bits = static_cast<std::uint8_t>(reader.read(6));
    out->ext_config_idx = static_cast<std::uint8_t>(reader.read(3));

    const std::uint32_t declared_objects = static_cast<std::uint32_t>(out->num_objects_bits) + 1u;
    const std::int16_t n_channels = num_channels_for_config(out->dmx_config_idx);
    if (n_channels < 0) {
        return syntax_fail("unknown JOC downmix configuration " +
                           std::to_string(out->dmx_config_idx));
    }
    out->n_channels = static_cast<std::uint8_t>(n_channels);
    // A declared object count above 15 is clamped to 15: the frame is parsed for
    // 15 objects only, and the clamp is reported separately.
    out->objects_clamped = declared_objects > JOC_MAX_OBJECTS ? 1u : 0u;
    const std::uint32_t n_objects =
        std::min<std::uint32_t>(declared_objects, JOC_MAX_OBJECTS);
    out->n_objects = static_cast<std::uint8_t>(n_objects);

    out->clipgain_x_bits = static_cast<std::uint8_t>(reader.read(3));
    out->clipgain_y_bits = static_cast<std::uint8_t>(reader.read(5));
    out->seq_count = reader.read(10);
    // clipgain = 1 + (y/32) * 2^(x-4); ldexp builds the power of two exactly
    // over the whole legal range.
    out->clipgain = 1.0 + static_cast<double>(out->clipgain_y_bits) / 32.0 *
                              std::ldexp(1.0, static_cast<int>(out->clipgain_x_bits) - 4);

    for (std::uint32_t obj = 0; obj < n_objects; ++obj) {
        joc_object_params& info = out->objects[obj];
        info.present = static_cast<std::uint8_t>(reader.read(1));
        if (info.present == 0u) {
            continue;
        }
        info.num_bands_idx = static_cast<std::uint8_t>(reader.read(3));
        const std::int16_t bands = num_bands_for_index(info.num_bands_idx);
        if (bands < 0) {
            return syntax_fail("unknown JOC num_bands index " +
                               std::to_string(info.num_bands_idx));
        }
        info.n_bands = static_cast<std::uint8_t>(bands);
        info.sparse = static_cast<std::uint8_t>(reader.read(1));
        info.quant_idx = static_cast<std::uint8_t>(reader.read(1));
        info.slope_idx = static_cast<std::uint8_t>(reader.read(1));
        info.num_dpoints_bits = static_cast<std::uint8_t>(reader.read(1));
        info.n_dpoints = static_cast<std::uint8_t>(info.num_dpoints_bits + 1u);
        if (info.slope_idx == 1u) {
            for (std::uint32_t dp = 0; dp < info.n_dpoints; ++dp) {
                info.offset_ts[dp] = static_cast<std::uint8_t>(reader.read(5) + 1u);
            }
        }
    }
    if (reader.failed()) {
        return truncated_fail(reader);
    }

    for (std::uint32_t obj = 0; obj < n_objects; ++obj) {
        const joc_object_params& info = out->objects[obj];
        if (info.present == 0u) {
            continue;
        }
        ObjectSymbols* symbol = &capture.objects[obj];
        symbol->present = 1;
        symbol->sparse = info.sparse;
        symbol->n_bands = info.n_bands;
        symbol->n_dpoints = info.n_dpoints;
        symbol->n_channels = out->n_channels;
        for (std::uint32_t dp = 0; dp < info.n_dpoints; ++dp) {
            if (info.sparse == 1u) {
                const Tree idx_tree = select_tree(info.quant_idx, SymbolKind::Idx, n_channels);
                const std::uint32_t first = reader.read(3);
                if (reader.failed()) {
                    return truncated_fail(reader);
                }
                symbol->idx[dp][0] = static_cast<std::uint8_t>(first);
                for (int pb = 1; pb < info.n_bands; ++pb) {
                    std::int16_t value = 0;
                    if (!huff_decode(idx_tree, reader, &value)) {
                        return truncated_fail(reader);
                    }
                    symbol->idx[dp][pb] = static_cast<std::uint8_t>(value);
                }
                const Tree vec_tree = select_tree(info.quant_idx, SymbolKind::Vec, n_channels);
                for (int pb = 0; pb < info.n_bands; ++pb) {
                    std::int16_t value = 0;
                    if (!huff_decode(vec_tree, reader, &value)) {
                        return truncated_fail(reader);
                    }
                    symbol->vec[dp][pb] = value;
                }
            } else {
                const Tree mtx_tree = select_tree(info.quant_idx, SymbolKind::Mtx, n_channels);
                for (int ch = 0; ch < n_channels; ++ch) {
                    for (int pb = 0; pb < info.n_bands; ++pb) {
                        std::int16_t value = 0;
                        if (!huff_decode(mtx_tree, reader, &value)) {
                            return truncated_fail(reader);
                        }
                        symbol->mtx[dp][ch][pb] = value;
                    }
                }
            }
        }
    }

    out->data_end_bits = static_cast<std::uint32_t>(reader.position());
    out->trailing_bits = static_cast<std::uint32_t>(payload_size * 8u - reader.position());
    const std::size_t tail_offset = reader.position() / 8u;
    if (tail_offset < payload_size) {
        const std::size_t tail_bytes = std::min<std::size_t>(8u, payload_size - tail_offset);
        std::memcpy(out->tail_bytes, payload + tail_offset, tail_bytes);
    }

    for (std::uint32_t obj = 0; obj < n_objects; ++obj) {
        const joc_object_params& info = out->objects[obj];
        if (info.present == 0u) {
            continue;
        }
        std::uint32_t mask_bit = 1u << obj;
        out->present_mask |= mask_bit;

        const std::int64_t nquant = (info.quant_idx == 0u) ? 96 : 192;
        std::int64_t q[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS] = {};
        ObjectSymbols* symbol = &capture.objects[obj];

        for (std::uint32_t dp = 0; dp < info.n_dpoints; ++dp) {
            if (info.sparse == 1u) {
                const std::int64_t offset = (info.quant_idx == 0u) ? 50 : 100;
                const Status status =
                    reconstruct_sparse(*symbol, dp, n_channels, nquant, offset, q);
                if (!status.ok()) {
                    return status;
                }
            } else {
                const std::int64_t offset = (info.quant_idx == 0u) ? 48 : 96;
                reconstruct_dense(*symbol, dp, n_channels, nquant, offset, q);
            }
        }

        // Operand order and types are kept identical to the reference so the
        // result is bit-exact, not merely close.
        const double nquant_half = static_cast<double>(nquant) / 2.0;
        const double denominator = 4096.0 * static_cast<double>(1 + static_cast<int>(info.quant_idx));
        for (std::uint32_t dp = 0; dp < info.n_dpoints; ++dp) {
            for (int ch = 0; ch < n_channels; ++ch) {
                for (int pb = 0; pb < info.n_bands; ++pb) {
                    const double value = static_cast<double>(q[dp][ch][pb]) - nquant_half;
                    out->objects[obj].dq[dp][ch][pb] = value * 820.0 / denominator;
                }
            }
        }
        for (std::uint32_t dp = 0; dp < info.n_dpoints; ++dp) {
            for (int ch = 0; ch < n_channels; ++ch) {
                for (int pb = 0; pb < info.n_bands; ++pb) {
                    symbol->q[dp][ch][pb] = q[dp][ch][pb];
                }
            }
        }
    }

    capture.n_objects = out->n_objects;
    capture.n_channels = out->n_channels;
    return Status::success();
}

int score_container_as_joc(const std::uint8_t* data, std::size_t size,
                           const emdf::Container& container) {
    const emdf::Payload* payload = container.find(emdf::kIdJoc);
    const emdf::Payload* oamd = container.find(emdf::kIdOamd);
    if (payload == nullptr || oamd == nullptr) {
        return 0;
    }
    bool parses = false;
    std::vector<std::uint8_t> bytes;
    if (emdf::extract_payload_bytes(data, size, *payload, &bytes).ok()) {
        joc_frame_params params;
        FrameSymbols symbols;
        parses = parse_id14(bytes.data(), bytes.size(), &params, &symbols).ok();
    }
    const bool associated = oamd->has_group_id != 0u && payload->has_group_id != 0u &&
                            oamd->group_id == payload->group_id;
    if (associated && parses) {
        return 2;
    }
    return parses ? 1 : 0;
}

Status parse_eac3_frame(const std::uint8_t* frame, std::size_t frame_size, joc_frame_params* out,
                        emdf::Container* container, FrameSymbols* symbols) {
    if (frame == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kJoc, "null frame or output");
    }
    emdf::Container local;
    const Status status =
        emdf::find_joc_emdf(frame, frame_size, &local, &score_container_as_joc, nullptr);
    if (!status.ok()) {
        return status;
    }
    if (container != nullptr) {
        *container = local;
    }
    const emdf::Payload* payload = local.find(emdf::kIdJoc);
    if (payload == nullptr) {
        return Status::fail(JOC_ERR_EMDF_TRANSPORT, stage::kEmdf,
                            "EMDF container has no ID14 (JOC) payload");
    }
    std::vector<std::uint8_t> bytes;
    const Status extract = emdf::extract_payload_bytes(frame, frame_size, *payload, &bytes);
    if (!extract.ok()) {
        return extract;
    }
    return parse_id14(bytes.data(), bytes.size(), out, symbols);
}

Status check_id14_padding(const std::uint8_t* payload, std::size_t payload_size,
                          std::uint32_t* out_trailing_bits) {
    joc_frame_params params;
    FrameSymbols symbols;
    const Status status = parse_id14(payload, payload_size, &params, &symbols);
    if (!status.ok()) {
        return status;
    }
    if (out_trailing_bits != nullptr) {
        *out_trailing_bits = params.trailing_bits;
    }
    if (params.trailing_bits > 7u) {
        return Status::fail(JOC_ERR_BITSTREAM_PADDING, stage::kJoc,
                            "more than 7 bits left after joc_data (" +
                                std::to_string(params.trailing_bits) + ")");
    }
    for (std::size_t bit = params.data_end_bits; bit < payload_size * 8u; ++bit) {
        if (((payload[bit >> 3] >> (7u - (bit & 7u))) & 1u) != 0u) {
            return Status::fail(JOC_ERR_BITSTREAM_PADDING, stage::kJoc,
                                "non-zero trailing padding bit at " + std::to_string(bit));
        }
    }
    return Status::success();
}

void make_silent_frame_params(joc_frame_params* out) {
    if (out == nullptr) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(joc_frame_params);
    out->struct_version = JOC_FRAME_PARAMS_VERSION;
    out->dmx_config_idx = 0u;
    out->n_channels = static_cast<std::uint8_t>(JOC_CORE_CHANNELS);
    out->n_objects = static_cast<std::uint8_t>(JOC_MAX_OBJECTS);
    out->present_mask = (1u << JOC_MAX_OBJECTS) - 1u;
    out->clipgain = 1.0;
    for (unsigned object = 0; object < JOC_MAX_OBJECTS; ++object) {
        out->objects[object].present = 1u;
        out->objects[object].num_bands_idx = 0u;  // one parameter band
        out->objects[object].n_bands = 1u;
        out->objects[object].sparse = 0u;
        out->objects[object].quant_idx = 0u;
        out->objects[object].slope_idx = 0u;
        out->objects[object].num_dpoints_bits = 0u;
        out->objects[object].n_dpoints = 1u;
    }
}

// ---------------------------------------------------------------------------
// Channel identity / bed layout (spec sections 3, 10.2, 10.4)
// ---------------------------------------------------------------------------

namespace {

struct PositionBit {
    Position position;
    std::uint32_t bit;
};

// Position -> ffmpeg channel bit.  A decoded bed frame lists its channels in
// ascending bit order, so sorting the positions by bit reproduces the decoder's
// channel order without asking the decoder.
constexpr PositionBit kPositionBits[] = {
    {Position::kLeft, 0x1u},
    {Position::kRight, 0x2u},
    {Position::kCenter, 0x4u},
    {Position::kLfe, 0x8u},
    {Position::kLeftRearSurround, 0x10u},
    {Position::kRightRearSurround, 0x20u},
    {Position::kCenterSurround, 0x100u},
    {Position::kLeftSurround, 0x200u},
    {Position::kRightSurround, 0x400u},
    {Position::kTopSurround, 0x800u},
    {Position::kVerticalHeightLeft, 0x1000u},
    {Position::kVerticalHeightCenter, 0x2000u},
    {Position::kVerticalHeightRight, 0x4000u},
    {Position::kTopSurroundLeft, 0x8000u},
    {Position::kTopSurroundRight, 0x20000u},
};

std::uint32_t position_bit(Position position) {
    for (const PositionBit& entry : kPositionBits) {
        if (entry.position == position) {
            return entry.bit;
        }
    }
    return 0u;
}

void push_unique(std::vector<Position>* positions, Position position) {
    if (position == Position::kNone) {
        return;
    }
    for (const Position existing : *positions) {
        if (existing == position) {
            return;
        }
    }
    positions->push_back(position);
}

// ADM 10-slot skeleton placement of the extra pair, indexed by joc_dmx_config_idx.
constexpr int kExtraPairSlots[5][2] = {
    {-1, -1}, {6, 7}, {8, 9}, {-1, -1}, {8, 9},
};

Status variant_error(const char* stage_name, const char* name, const std::string& message,
                     std::string* error) {
    const std::string text = std::string(name) + ": " + message;
    if (error != nullptr) {
        *error = text;
    }
    return Status::fail(JOC_ERR_JOC_UNSUPPORTED_VARIANT, stage_name, text);
}

std::string position_name(Position position) {
    switch (position) {
        case Position::kLeft: return "L";
        case Position::kRight: return "R";
        case Position::kCenter: return "C";
        case Position::kLfe: return "LFE";
        case Position::kLeftSurround: return "Ls";
        case Position::kRightSurround: return "Rs";
        case Position::kLeftRearSurround: return "Lrs";
        case Position::kRightRearSurround: return "Rrs";
        case Position::kVerticalHeightLeft: return "Vhl";
        case Position::kVerticalHeightRight: return "Vhr";
        case Position::kTopSurroundLeft: return "Lts";
        case Position::kTopSurroundRight: return "Rts";
        case Position::kCenterSurround: return "Cs";
        case Position::kTopSurround: return "Ts";
        case Position::kVerticalHeightCenter: return "Vhc";
        case Position::kLfe2: return "LFE2";
        default: return "?";
    }
}

// 核心声道位置 → bed 布局声道；返回 kCount 表示本标准表没有对应身份。
BedChannel bed_channel_for_position(Position position) {
    switch (position) {
        case Position::kLeft: return BedChannel::kLeft;
        case Position::kRight: return BedChannel::kRight;
        case Position::kCenter: return BedChannel::kCenter;
        case Position::kLfe: return BedChannel::kLfe;
        case Position::kLfe2: return BedChannel::kLfe2;
        case Position::kLeftSurround: return BedChannel::kLeftSurround;
        case Position::kRightSurround: return BedChannel::kRightSurround;
        case Position::kLeftRearSurround: return BedChannel::kLeftRearSurround;
        case Position::kRightRearSurround: return BedChannel::kRightRearSurround;
        case Position::kVerticalHeightLeft: return BedChannel::kTopFrontLeft;
        case Position::kVerticalHeightRight: return BedChannel::kTopFrontRight;
        case Position::kTopSurroundLeft: return BedChannel::kTopSideLeft;
        case Position::kTopSurroundRight: return BedChannel::kTopSideRight;
        default: return BedChannel::kCount;
    }
}

std::string position_list(const std::vector<Position>& positions) {
    std::string text = "[";
    for (std::size_t index = 0; index < positions.size(); ++index) {
        if (index != 0) {
            text += ",";
        }
        text += position_name(positions[index]);
    }
    return text + "]";
}

}  // namespace

void expand_chanmap(std::uint16_t chanmap, std::vector<Position>* out) {
    if (out == nullptr) {
        return;
    }
    // Table E2.5; bit 0 sits in the field's most significant bit.
    static const Position kSingles[16] = {
        Position::kLeft, Position::kCenter, Position::kRight,
        Position::kLeftSurround, Position::kRightSurround, Position::kNone,
        Position::kNone, Position::kCenterSurround, Position::kTopSurround,
        Position::kNone, Position::kNone, Position::kNone,
        Position::kVerticalHeightCenter, Position::kNone, Position::kLfe2,
        Position::kLfe,
    };
    for (int bit = 0; bit < 16; ++bit) {
        if (((chanmap >> (15 - bit)) & 1u) == 0u) {
            continue;
        }
        switch (bit) {
            case 5:
                out->push_back(Position::kLeftSurround);
                out->push_back(Position::kRightSurround);
                break;
            case 6:
                out->push_back(Position::kLeftRearSurround);
                out->push_back(Position::kRightRearSurround);
                break;
            case 9:
            case 10:
                // Lsd/Rsd and Lw/Rw have no JOC counterpart and stay unmapped.
                break;
            case 11:
                out->push_back(Position::kVerticalHeightLeft);
                out->push_back(Position::kVerticalHeightRight);
                break;
            case 13:
                out->push_back(Position::kTopSurroundLeft);
                out->push_back(Position::kTopSurroundRight);
                break;
            default:
                out->push_back(kSingles[bit]);
                break;
        }
    }
}

void acmod_positions(std::uint8_t acmod, std::vector<Position>* out) {
    if (out == nullptr) {
        return;
    }
    switch (acmod) {
        case 0:
        case 1:
            out->push_back(Position::kCenter);
            break;
        case 2:
            out->push_back(Position::kLeft);
            out->push_back(Position::kRight);
            break;
        case 3:
            out->push_back(Position::kLeft);
            out->push_back(Position::kCenter);
            out->push_back(Position::kRight);
            break;
        case 4:
            out->push_back(Position::kLeft);
            out->push_back(Position::kRight);
            out->push_back(Position::kCenterSurround);
            break;
        case 5:
            out->push_back(Position::kLeft);
            out->push_back(Position::kCenter);
            out->push_back(Position::kRight);
            out->push_back(Position::kCenterSurround);
            break;
        case 6:
            out->push_back(Position::kLeft);
            out->push_back(Position::kRight);
            out->push_back(Position::kLeftSurround);
            out->push_back(Position::kRightSurround);
            break;
        case 7:
        default:
            out->push_back(Position::kLeft);
            out->push_back(Position::kCenter);
            out->push_back(Position::kRight);
            out->push_back(Position::kLeftSurround);
            out->push_back(Position::kRightSurround);
            break;
    }
}

void expected_pair(std::uint8_t dmx_config_idx, std::vector<Position>* out) {
    if (out == nullptr) {
        return;
    }
    if (dmx_config_idx == 1u) {
        out->push_back(Position::kLeftRearSurround);
        out->push_back(Position::kRightRearSurround);
    } else if (dmx_config_idx == 2u || dmx_config_idx == 4u) {
        out->push_back(Position::kVerticalHeightLeft);
        out->push_back(Position::kVerticalHeightRight);
    }
}

void chanmap_extra_pair(const std::vector<Position>& core, std::uint16_t chanmap,
                        std::vector<Position>* out) {
    if (out == nullptr) {
        return;
    }
    std::vector<Position> declared;
    expand_chanmap(chanmap, &declared);
    for (const Position position : declared) {
        bool in_core = false;
        for (const Position existing : core) {
            if (existing == position) {
                in_core = true;
                break;
            }
        }
        if (!in_core) {
            push_unique(out, position);
        }
    }
}

int bed_channels_for_core(int core_channels) {
    return core_channels == 7 ? 8 : 6;
}

Status compute_bed_plan(const eac3::Frame* core_frame, bool has_core, bool has_dependent,
                        bool has_chanmap, std::uint16_t chanmap, std::uint8_t dmx_config_idx,
                        BedPlan* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kJoc, "null bed plan");
    }
    const std::int16_t core_channels = num_channels_for_config(dmx_config_idx);
    if (core_channels < 0) {
        return variant_error("joc", "unsupported_configuration",
                             "unknown JOC downmix configuration " +
                                 std::to_string(dmx_config_idx), nullptr);
    }
    std::vector<Position> core_positions;
    if (has_core && core_frame != nullptr) {
        acmod_positions(core_frame->acmod, &core_positions);
        if (core_frame->lfeon != 0u) {
            core_positions.push_back(Position::kLfe);
        }
    }
    std::vector<Position> extra;
    std::vector<Position> locations = core_positions;
    if (has_dependent && has_chanmap) {
        std::vector<Position> declared;
        expand_chanmap(chanmap, &declared);
        chanmap_extra_pair(core_positions, chanmap, &extra);
        locations.insert(locations.end(), declared.begin(), declared.end());
    }

    std::vector<Position> ordered;
    for (const Position position : locations) {
        push_unique(&ordered, position);
    }
    std::sort(ordered.begin(), ordered.end(), [](Position left, Position right) {
        return position_bit(left) < position_bit(right);
    });
    for (const Position position : ordered) {
        if (position_bit(position) == 0u) {
            return variant_error("eac3", "bed_channel_mapping_unsupported",
                                 "unmapped channel position " + position_name(position),
                                 nullptr);
        }
    }

    BedPlan plan;
    plan.channels = static_cast<int>(ordered.size());
    const int expected_channels = bed_channels_for_core(core_channels);
    if (plan.channels != expected_channels) {
        return variant_error("eac3", "core_channel_count_mismatch",
                             "the bed declares " + std::to_string(plan.channels) +
                                 " channels, configuration " + std::to_string(dmx_config_idx) +
                                 " needs " + std::to_string(expected_channels),
                             nullptr);
    }
    for (int index = 0; index < plan.channels; ++index) {
        if (ordered[index] == Position::kLfe) {
            plan.lfe_channel = index;
        }
    }

    std::vector<Position> wanted = {Position::kLeft, Position::kRight, Position::kCenter,
                                    Position::kLeftSurround, Position::kRightSurround};
    wanted.insert(wanted.end(), extra.begin(), extra.end());
    if (wanted.size() != static_cast<std::size_t>(core_channels)) {
        return variant_error("eac3", "chanmap_cfg_mismatch",
                             "the bed cannot supply " + std::to_string(core_channels) +
                                 " JOC core channels (found " + std::to_string(wanted.size()) +
                                 ")",
                             nullptr);
    }
    for (std::size_t lane = 0; lane < wanted.size(); ++lane) {
        int found = -1;
        for (int index = 0; index < plan.channels; ++index) {
            if (ordered[index] == wanted[lane]) {
                found = index;
                break;
            }
        }
        if (found < 0) {
            return variant_error("eac3", "bed_channel_mapping_unsupported",
                                 "channel " + position_name(wanted[lane]) +
                                     " is missing from the bed " + position_list(ordered),
                                 nullptr);
        }
        plan.lane[lane] = found;
    }
    if (dmx_config_idx < 5u) {
        plan.extra_pair_slot[0] = kExtraPairSlots[dmx_config_idx][0];
        plan.extra_pair_slot[1] = kExtraPairSlots[dmx_config_idx][1];
    }
    *out = plan;
    return Status::success();
}

Status compute_bed_plan_for_config(std::uint8_t dmx_config_idx, BedPlan* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kJoc, "null bed plan");
    }
    const std::int16_t core_channels = num_channels_for_config(dmx_config_idx);
    if (core_channels < 0) {
        return variant_error("joc", "unsupported_configuration",
                             "unknown JOC downmix configuration " +
                                 std::to_string(dmx_config_idx), nullptr);
    }
    std::vector<Position> extra;
    expected_pair(dmx_config_idx, &extra);
    std::vector<Position> wanted = {Position::kLeft, Position::kRight, Position::kCenter,
                                    Position::kLeftSurround, Position::kRightSurround};
    wanted.insert(wanted.end(), extra.begin(), extra.end());

    std::vector<Position> ordered = {Position::kLeft, Position::kRight, Position::kCenter,
                                     Position::kLfe, Position::kLeftSurround,
                                     Position::kRightSurround};
    ordered.insert(ordered.end(), extra.begin(), extra.end());
    std::sort(ordered.begin(), ordered.end(), [](Position left, Position right) {
        return position_bit(left) < position_bit(right);
    });
    ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());

    BedPlan plan;
    plan.channels = static_cast<int>(ordered.size());
    for (int index = 0; index < plan.channels; ++index) {
        if (ordered[index] == Position::kLfe) {
            plan.lfe_channel = index;
        }
    }
    for (std::size_t lane = 0; lane < wanted.size(); ++lane) {
        for (int index = 0; index < plan.channels; ++index) {
            if (ordered[index] == wanted[lane]) {
                plan.lane[lane] = index;
                break;
            }
        }
        if (plan.lane[lane] < 0) {
            return variant_error("eac3", "bed_channel_mapping_unsupported",
                                 "channel " + position_name(wanted[lane]) +
                                     " is missing from the bed " + position_list(ordered),
                                 nullptr);
        }
    }
    if (dmx_config_idx < 5u) {
        plan.extra_pair_slot[0] = kExtraPairSlots[dmx_config_idx][0];
        plan.extra_pair_slot[1] = kExtraPairSlots[dmx_config_idx][1];
    }
    *out = plan;
    return Status::success();
}

Status check_cross_layer(const joc_frame_params& params, const eac3::Frame* core_frame,
                         bool has_core, bool has_dependent, bool has_chanmap,
                         std::uint16_t chanmap,
                         const std::vector<oamd::BedAssignment>* bed_assignments,
                         std::string* error, CrossLayerNotes* notes) {
    const std::uint8_t config_idx = params.dmx_config_idx;
    const int core_channels = params.n_channels;
    std::vector<Position> wanted;
    expected_pair(config_idx, &wanted);

    if (params.ext_config_idx != 0u) {
        return variant_error("joc", "ext_config_idx_unsupported",
                             "joc_ext_config_idx=" + std::to_string(params.ext_config_idx) +
                                 " has no mapping in this implementation",
                             error);
    }
    if (core_channels != JOC_CORE_CHANNELS && core_channels != JOC_MAX_CORE_CHANNELS) {
        return variant_error("joc", "unsupported_configuration",
                             "unsupported JOC core channel count " +
                                 std::to_string(core_channels),
                             error);
    }

    std::vector<Position> core_positions;
    if (has_core && core_frame != nullptr) {
        acmod_positions(core_frame->acmod, &core_positions);
        if (core_frame->lfeon != 0u) {
            core_positions.push_back(Position::kLfe);
        }
    }

    if (core_channels == JOC_MAX_CORE_CHANNELS) {
        if (!has_core || !has_chanmap) {
            return variant_error(
                "eac3", "core_extension_pair",
                "a 7-channel configuration needs both an independent substream and a "
                "dependent-substream chanmap",
                error);
        }
        std::vector<Position> pair;
        chanmap_extra_pair(core_positions, chanmap, &pair);
        if (pair != wanted) {
            // The bed mapping follows the chanmap's channel codes; the name a
            // configuration implies does not reach the DSP, so the frame renders
            // and the disagreement is reported instead.
            if (notes != nullptr) {
                notes->chanmap_pair_mismatch = true;
            }
        }
    } else if (has_chanmap) {
        std::vector<Position> pair;
        chanmap_extra_pair(core_positions, chanmap, &pair);
        if (!pair.empty()) {
            return variant_error("eac3", "core_channel_count_mismatch",
                                 "a 5-channel configuration declares extra dependent "
                                 "channels " + position_list(pair),
                                 error);
        }
    }

    std::vector<Position> declared = core_positions;
    if (has_chanmap) {
        std::vector<Position> expanded;
        expand_chanmap(chanmap, &expanded);
        for (const Position position : expanded) {
            push_unique(&declared, position);
        }
    }
    int full_band = 0;
    for (const Position position : declared) {
        if (position != Position::kLfe && position != Position::kLfe2) {
            ++full_band;
        }
    }
    if (!declared.empty() && full_band != core_channels) {
        return variant_error("eac3", "core_channel_count_mismatch",
                             "the substreams declare " + std::to_string(full_band) +
                                 " full-band channels, joc_dmx_config_idx=" +
                                 std::to_string(config_idx) + " declares " +
                                 std::to_string(core_channels),
                             error);
    }

    if (bed_assignments != nullptr && !bed_assignments->empty()) {
        // OAMD 的 bed 掩码描述的是内容的 bed 布局，可以比核心子流实际编码的声道多
        // （例如 9.1.6 bed 只编 5.1 核心）。只有核心声道在 bed 布局里找不到位置才是
        // 真正的不一致；多出来的 bed 声道是布局的一部分。
        const BedLayout bed = resolve_bed_layout(*bed_assignments, true);
        if (bed.present && bed.complete) {
            bool covered = true;
            for (const Position position : core_positions) {
                const BedChannel channel = bed_channel_for_position(position);
                if (channel == BedChannel::kCount) {
                    continue;
                }
                if (!bed_layout_has(bed, channel)) {
                    covered = false;
                    break;
                }
            }
            if (!covered && notes != nullptr) {
                notes->bed_mask_mismatch = true;
            }
        }
    }
    return Status::success();
}

}  // namespace joc::joc

