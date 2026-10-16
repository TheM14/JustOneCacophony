
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
    // Payload configuration data (ETSI TS 102 366 H.2.2.3).  Spec section 8.2 and
    // its Table 56 associate the OAMD and JOC payloads of one frame by an equal
    // group id, which is how several containers in one syncframe are told apart.
    std::uint8_t group_id = 0;
    std::uint8_t has_group_id = 0;
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

// Scores how well a candidate container can serve as the frame's metadata when a
// syncframe carries more than one: 0 = not at all, higher = more credible.  A null
// predicate keeps the strict behaviour: several top-level candidates are an error,
// never a choice.
using ContainerPredicate = int (*)(const std::uint8_t* data, std::size_t size,
                                   const Container& container);

// When a syncframe holds several top-level candidates and `usable` leaves exactly
// one with the highest non-zero score, that one is returned and `ignored_extra` is
// set; otherwise the frame stays an ambiguity error.
Status find_joc_emdf(const std::uint8_t* data, std::size_t size, Container* out,
                     ContainerPredicate usable = nullptr, bool* ignored_extra = nullptr);

// Like find_joc_emdf(), but a syncframe that carries no EMDF marker at all is a
// normal "absent" result rather than an error.  A syncframe that does carry
// markers but no usable JOC container still fails, so a real transport problem
// is never mistaken for a core substream.
Status find_joc_emdf_optional(const std::uint8_t* data, std::size_t size, bool* present,
                              Container* out, ContainerPredicate usable = nullptr,
                              bool* ignored_extra = nullptr);

// Extract the container's bytes exactly as the bit reader sees them (identical
// to a memcpy for byte-aligned containers).
void extract_container_bytes(const std::uint8_t* data, std::size_t size, const Container& container,
                             std::vector<std::uint8_t>* out);

// Extract one payload's bytes with the same MSB-first semantics.
Status extract_payload_bytes(const std::uint8_t* data, std::size_t size, const Payload& payload,
                             std::vector<std::uint8_t>* out);

}  // namespace joc::emdf
