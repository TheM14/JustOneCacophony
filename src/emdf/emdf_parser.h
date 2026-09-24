
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "joc_core.h"
#include "foundation/status.h"

namespace joc::emdf {

inline constexpr std::uint16_t kSyncword = 0x5838;
inline constexpr std::uint8_t kIdOamd = 11;
inline constexpr std::uint8_t kIdJoc = 14;
inline constexpr std::size_t kMaxPayloads = JOC_MAX_EMDF_PAYLOADS;

struct Payload {
    std::uint8_t id = 0;
    std::uint16_t sample_offset = 0;
    std::size_t bit_offset = 0;  // MSB-first bit position of the payload bytes
    std::size_t size = 0;        // payload byte count
};

struct Container {
    std::size_t start_bit = 0;
    std::size_t raw_size = 0;
    std::size_t payload_count = 0;
    Payload payloads[kMaxPayloads] = {};

    const Payload* find(std::uint8_t id) const {
        for (std::size_t i = 0; i < payload_count; ++i) {
            if (payloads[i].id == id) {
                return &payloads[i];
            }
        }
        return nullptr;
    }
};

Status parse_at(const std::uint8_t* data, std::size_t size, std::size_t start_bit, Container* out);

// All candidate 0x5838 bit offsets over the eight alignments, ascending.
void marker_offsets(const std::uint8_t* data, std::size_t size, std::vector<std::size_t>* out);

Status find_joc_emdf(const std::uint8_t* data, std::size_t size, Container* out);

// Extract the container's bytes exactly as the bit reader sees them (identical
// to a memcpy for byte-aligned containers).
void extract_container_bytes(const std::uint8_t* data, std::size_t size, const Container& container,
                             std::vector<std::uint8_t>* out);

// Extract one payload's bytes with the same MSB-first semantics.
Status extract_payload_bytes(const std::uint8_t* data, std::size_t size, const Payload& payload,
                             std::vector<std::uint8_t>* out);

}  // namespace joc::emdf
