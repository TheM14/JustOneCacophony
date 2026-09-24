#include "foundation/bit_reader.h"

namespace joc::bits {

bool variable_bits(BitReader& reader, unsigned width, unsigned max_groups, std::uint32_t* out_value) {
    std::uint32_t value = 0;
    for (unsigned group = 0; group < max_groups; ++group) {
        value += reader.read(width);
        if (reader.failed()) {
            return false;
        }
        const std::uint32_t more = reader.read(1);
        if (reader.failed()) {
            return false;
        }
        if (more == 0u) {
            if (out_value != nullptr) {
                *out_value = value;
            }
            return true;
        }
        value = (value + 1u) << width;
    }
    // Same failure mode as the reference implementation: an extension chain
    // that never terminates is a syntax error, not a truncation.
    reader.fail(JOC_ERR_EMDF_SYNTAX, "variable_bits extension groups exceeded");
    return false;
}

}  // namespace joc::bits
