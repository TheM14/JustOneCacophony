// Port of src/oamd_tracks.py align_metadata_sample: an OAMD position update is
// scheduled on the renderer's processing-block grid, the same 32-sample grid the
// speaker control blocks use.  It lives in foundation because both the ADM track
// builder and the object position timeline need it, and the timeline must not
// depend on output.
#pragma once

#include <cstdint>

namespace joc::timing {

inline constexpr std::int64_t kMetadataBlockSamples = 32;

// Nearest block boundary, ties towards the earlier block; the negative branch
// floors like the reference's Python `//`.
inline std::int64_t align_metadata_sample(std::int64_t sample,
                                          std::int64_t block_samples = kMetadataBlockSamples) {
    const std::int64_t numerator = sample + block_samples / 2 - 1;
    const std::int64_t quotient =
        numerator >= 0 ? numerator / block_samples
                       : -((-numerator + block_samples - 1) / block_samples);
    return block_samples * quotient;
}

}  // namespace joc::timing
