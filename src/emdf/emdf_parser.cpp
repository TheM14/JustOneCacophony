#include "emdf/emdf_parser.h"

#include <algorithm>
#include <string>

#include "foundation/bit_reader.h"

namespace joc::emdf {

namespace {

Status syntax_fail(const std::string& message) {
    return Status::fail(JOC_ERR_EMDF_SYNTAX, stage::kEmdf, message);
}

Status truncated_fail(const bits::BitReader& reader) {
    return Status::fail(JOC_ERR_BITSTREAM_TRUNCATED, stage::kEmdf,
                        std::string("EMDF bitstream truncated: ") + reader.error_message());
}

}  // namespace

Status parse_at(const std::uint8_t* data, std::size_t size, std::size_t start_bit, Container* out) {
    if (data == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kEmdf, "null buffer or output");
    }
    if (start_bit + 16u > size * 8u) {
        return syntax_fail("EMDF syncword position beyond buffer");
    }

    bits::BitReader reader;
    reader.reset(data, size, start_bit);

    if (reader.read(16) != kSyncword) {
        return syntax_fail("EMDF syncword mismatch at bit " + std::to_string(start_bit));
    }
    const std::uint32_t length = reader.read(16);
    const std::size_t body_start = reader.position();
    const std::size_t body_end = body_start + static_cast<std::size_t>(length) * 8u;
    if (body_end > reader.limit()) {
        return syntax_fail("EMDF container length " + std::to_string(length) +
                           " exceeds buffer at bit " + std::to_string(start_bit));
    }
    reader.set_limit_bits(body_end);

    std::uint32_t version = reader.read(2);
    if (version == 3u) {
        std::uint32_t extra = 0;
        if (!bits::variable_bits(reader, 2, 8, &extra)) {
            return reader.error() == JOC_ERR_BITSTREAM_TRUNCATED ? truncated_fail(reader)
                                                                 : syntax_fail(reader.error_message());
        }
        version += extra;
    }
    std::uint32_t key_id = reader.read(3);
    if (key_id == 7u) {
        std::uint32_t extra = 0;
        if (!bits::variable_bits(reader, 3, 8, &extra)) {
            return reader.error() == JOC_ERR_BITSTREAM_TRUNCATED ? truncated_fail(reader)
                                                                 : syntax_fail(reader.error_message());
        }
        key_id += extra;
    }
    if (reader.failed()) {
        return truncated_fail(reader);
    }
    // TS 103 420 JOC uses version 0 / key_id 0; the strict check also rejects
    // false 0x5838 markers that happen to sit inside audio data.
    if (version != 0u || key_id != 0u) {
        return syntax_fail("unsupported EMDF version/key_id " + std::to_string(version) + "/" +
                           std::to_string(key_id));
    }

    Container container;
    container.start_bit = start_bit;
    bool terminated = false;

    while (reader.position() + 5u <= body_end) {
        std::uint32_t payload_id = reader.read(5);
        if (reader.failed()) {
            return truncated_fail(reader);
        }
        if (payload_id == 0u) {
            terminated = true;
            break;
        }
        if (payload_id == 0x1Fu) {
            std::uint32_t extra = 0;
            if (!bits::variable_bits(reader, 5, 8, &extra)) {
                return reader.error() == JOC_ERR_BITSTREAM_TRUNCATED ? truncated_fail(reader)
                                                                     : syntax_fail(reader.error_message());
            }
            payload_id += extra;
        }
        for (std::size_t i = 0; i < container.payload_count; ++i) {
            if (container.payloads[i].id == static_cast<std::uint8_t>(payload_id)) {
                return syntax_fail("duplicate EMDF payload id " + std::to_string(payload_id));
            }
        }
        if (container.payload_count >= kMaxPayloads) {
            return syntax_fail("EMDF payload count exceeds " + std::to_string(kMaxPayloads));
        }

        const std::uint32_t has_sample_offset = reader.read(1);
        std::uint16_t sample_offset = 0;
        if (has_sample_offset != 0u) {
            sample_offset = static_cast<std::uint16_t>(reader.read(12) >> 1);
        }
        if (reader.read(1) != 0u) {
            std::uint32_t ignored = 0;
            if (!bits::variable_bits(reader, 11, 8, &ignored)) {
                return reader.error() == JOC_ERR_BITSTREAM_TRUNCATED ? truncated_fail(reader)
                                                                     : syntax_fail(reader.error_message());
            }
        }
        if (reader.read(1) != 0u) {
            std::uint32_t ignored = 0;
            if (!bits::variable_bits(reader, 2, 8, &ignored)) {
                return reader.error() == JOC_ERR_BITSTREAM_TRUNCATED ? truncated_fail(reader)
                                                                     : syntax_fail(reader.error_message());
            }
        }
        if (reader.read(1) != 0u) {
            if (!reader.skip(8)) {
                return truncated_fail(reader);
            }
        }
        if (reader.read(1) == 0u) {
            bool frame_aligned = false;
            if (has_sample_offset == 0u) {
                frame_aligned = reader.read(1) != 0u;
                if (frame_aligned) {
                    if (!reader.skip(2)) {
                        return truncated_fail(reader);
                    }
                }
            }
            if (has_sample_offset != 0u || frame_aligned) {
                if (!reader.skip(7)) {
                    return truncated_fail(reader);
                }
            }
        }
        if (reader.failed()) {
            return truncated_fail(reader);
        }

        std::uint32_t payload_size = 0;
        if (!bits::variable_bits(reader, 8, 8, &payload_size)) {
            return reader.error() == JOC_ERR_BITSTREAM_TRUNCATED ? truncated_fail(reader)
                                                                 : syntax_fail(reader.error_message());
        }
        const std::size_t payload_bits = static_cast<std::size_t>(payload_size) * 8u;
        if (reader.position() + payload_bits > body_end) {
            return syntax_fail("EMDF payload id " + std::to_string(payload_id) +
                               " extends past container body (size " + std::to_string(payload_size) +
                               " at bit " + std::to_string(reader.position()) + ")");
        }

        Payload& entry = container.payloads[container.payload_count++];
        entry.id = static_cast<std::uint8_t>(payload_id);
        entry.sample_offset = sample_offset;
        entry.bit_offset = reader.position();
        entry.size = payload_size;

        if (!reader.skip(payload_bits)) {
            return truncated_fail(reader);
        }
    }

    if (!terminated) {
        return syntax_fail("EMDF container has no payload id 0 terminator");
    }
    container.raw_size = 4u + static_cast<std::size_t>(length);
    *out = container;
    return Status::success();
}

void marker_offsets(const std::uint8_t* data, std::size_t size, std::vector<std::size_t>* out) {
    out->clear();
    if (data == nullptr || size < 4u) {
        return;
    }
    // Eight global bit alignments.  For shift != 0 the reference builds an
    // (n-1)-byte shifted view and only scans pairs inside it, which is what the
    // bounds below reproduce exactly.
    for (std::size_t shift = 0; shift < 8u; ++shift) {
        const std::size_t aligned_len = (shift == 0u) ? size : (size - 1u);
        auto aligned_byte = [&](std::size_t index) -> std::uint8_t {
            if (shift == 0u) {
                return data[index];
            }
            const std::uint16_t high = static_cast<std::uint16_t>(data[index]) << shift;
            const std::uint16_t low = static_cast<std::uint16_t>(data[index + 1u]) >> (8u - shift);
            return static_cast<std::uint8_t>((high | low) & 0xFFu);
        };
        if (aligned_len < 2u) {
            continue;
        }
        for (std::size_t i = 0; i + 1u < aligned_len; ++i) {
            if (aligned_byte(i) == 0x58u && aligned_byte(i + 1u) == 0x38u) {
                out->push_back(i * 8u + shift);
            }
        }
    }
    std::sort(out->begin(), out->end());
}

Status find_joc_emdf(const std::uint8_t* data, std::size_t size, Container* out) {
    if (data == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kEmdf, "null buffer or output");
    }
    std::vector<std::size_t> offsets;
    marker_offsets(data, size, &offsets);

    std::vector<Container> matches;
    std::size_t parse_errors = 0;
    std::string first_parse_error;
    for (const std::size_t start_bit : offsets) {
        Container candidate;
        const Status status = parse_at(data, size, start_bit, &candidate);
        if (!status.ok()) {
            ++parse_errors;
            if (first_parse_error.empty()) {
                first_parse_error = "@bit" + std::to_string(start_bit) + ": " + status.message();
            }
            continue;
        }
        if (candidate.find(kIdOamd) != nullptr && candidate.find(kIdJoc) != nullptr) {
            matches.push_back(candidate);
        }
    }

    if (matches.empty()) {
        // Classification stays at the transport level (identical to the reference
        // implementation, which raises emdf_transport here), but the underlying
        std::string message =
            "no contiguous EMDF container carrying ID11+ID14 in this syncframe (markers=" +
            std::to_string(offsets.size()) + ", parse_failures=" + std::to_string(parse_errors) +
            ")";
        if (!first_parse_error.empty()) {
            message += "; first candidate error " + first_parse_error;
        }
        return Status::fail(JOC_ERR_EMDF_TRANSPORT, stage::kEmdf, message);
    }

    std::sort(matches.begin(), matches.end(),
              [](const Container& a, const Container& b) { return a.start_bit < b.start_bit; });

    // A payload may contain bytes that look like another 0x5838 container; a
    std::vector<Container> top_level;
    for (const Container& candidate : matches) {
        bool nested = false;
        for (const Container& parent : top_level) {
            if (parent.start_bit < candidate.start_bit &&
                candidate.start_bit < parent.start_bit + parent.raw_size * 8u) {
                nested = true;
                break;
            }
        }
        if (!nested) {
            top_level.push_back(candidate);
        }
    }

    if (top_level.size() != 1u) {
        return Status::fail(JOC_ERR_EMDF_TRANSPORT, stage::kEmdf,
                            "multiple top-level JOC EMDF containers (" +
                                std::to_string(top_level.size()) +
                                "); automatic selection is not defined");
    }
    *out = top_level.front();
    return Status::success();
}

void extract_container_bytes(const std::uint8_t* data, std::size_t size, const Container& container,
                             std::vector<std::uint8_t>* out) {
    out->assign(container.raw_size, 0u);
    if (out->empty()) {
        return;
    }
    bits::BitReader reader;
    reader.reset(data, size, container.start_bit);
    reader.read_bytes(out->data(), out->size());
}

Status extract_payload_bytes(const std::uint8_t* data, std::size_t size, const Payload& payload,
                             std::vector<std::uint8_t>* out) {
    if (data == nullptr || out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kEmdf, "null buffer or output");
    }
    out->assign(payload.size, 0u);
    if (out->empty()) {
        return Status::success();
    }
    bits::BitReader reader;
    reader.reset(data, size, payload.bit_offset);
    if (!reader.read_bytes(out->data(), out->size())) {
        return Status::fail(JOC_ERR_BITSTREAM_TRUNCATED, stage::kEmdf,
                            "payload bytes extend past the syncframe");
    }
    return Status::success();
}

}  // namespace joc::emdf
