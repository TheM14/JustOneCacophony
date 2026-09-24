#include "joc_bitstream/joc_parser.h"

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

    const std::uint32_t n_objects = static_cast<std::uint32_t>(out->num_objects_bits) + 1u;
    const std::int16_t n_channels = num_channels_for_config(out->dmx_config_idx);
    if (n_channels < 0) {
        return syntax_fail("unknown JOC downmix configuration " +
                           std::to_string(out->dmx_config_idx));
    }
    out->n_channels = static_cast<std::uint8_t>(n_channels);
    if (n_objects > JOC_MAX_OBJECTS) {
        // The reference implementation has no check here and fails later inside
        // NumPy; the port reports it explicitly (plan 28.2).
        return Status::fail(JOC_ERR_JOC_UNSUPPORTED_VARIANT, stage::kJoc,
                            "JOC frame declares " + std::to_string(n_objects) +
                                " objects, the ABI supports at most " +
                                std::to_string(static_cast<int>(JOC_MAX_OBJECTS)));
    }
    out->n_objects = static_cast<std::uint8_t>(n_objects);

    out->clipgain_x_bits = static_cast<std::uint8_t>(reader.read(3));
    out->clipgain_y_bits = static_cast<std::uint8_t>(reader.read(5));
    out->seq_count = reader.read(10);
    // clipgain = 1 + (y/32) * 2^(x-4).  The reference multiplies by an exact
    // exactly for the whole legal range.
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

Status parse_eac3_frame(const std::uint8_t* frame, std::size_t frame_size, joc_frame_params* out,
                        emdf::Container* container, FrameSymbols* symbols) {
    if (frame == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kJoc, "null frame or output");
    }
    emdf::Container local;
    const Status status = emdf::find_joc_emdf(frame, frame_size, &local);
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

}  // namespace joc::joc
