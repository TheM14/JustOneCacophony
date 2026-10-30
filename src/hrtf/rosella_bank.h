#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "hrtf/public_filterbank.h"

namespace joc::hrtf {

// Which 64-QMF / 77-hybrid bank the Rosella renderer runs.  The two are not
// interchangeable: the QMF prototype, the hybrid tap count, the high-band slot
// delay and the analysis/synthesis latency all differ, so each bank keeps its own
// tables and its own state.
enum class RosellaBankKind : std::int32_t {
    Public = 1,    // shared bank: 13-tap hybrid, 6-slot delay, 961 samples
    Captured = 2,  // Rosella bank: 11-tap hybrid, 3-slot delay, 448 samples
};

// Analysis/synthesis latency of `kind` at 48 kHz, in samples.
std::uint32_t rosella_bank_latency(RosellaBankKind kind);

// The four stateful half-steps of either bank, in the order the reference runs
// them.  Hops and QMF data are [slots, channels, 64], hybrid data is
// [slots, channels, 77].  State is per instance, so the analysis bank of a
// runtime and its synthesis bank never share buffers.
class RosellaBank {
public:
    // `kind` must name one of the enumerators; anything else selects Public.
    static std::unique_ptr<RosellaBank> create(RosellaBankKind kind, std::size_t channels);

    RosellaBank() = default;
    RosellaBank(const RosellaBank&) = delete;
    RosellaBank& operator=(const RosellaBank&) = delete;
    virtual ~RosellaBank() = default;

    virtual void reset() = 0;

    virtual void analyze_qmf(const std::vector<double>& hops, std::size_t slots,
                             std::vector<Complex>* qmf) = 0;
    virtual void analyze_hybrid(const std::vector<Complex>& qmf, std::size_t slots,
                                std::vector<Complex>* hybrid) = 0;
    virtual void synthesize_hybrid(const std::vector<Complex>& hybrid, std::size_t slots,
                                   std::vector<Complex>* qmf) = 0;
    virtual void synthesize_qmf(const std::vector<Complex>& qmf, std::size_t slots,
                                std::vector<double>* hops) = 0;
};

}  // namespace joc::hrtf
