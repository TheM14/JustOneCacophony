#include "hrtf/rosella_bank.h"

#include <utility>

#include "hrtf/rosella_filterbank.h"

namespace joc::hrtf {
namespace {

// The shared public bank.  It is also what the SOFA field compiler runs, and it
// is the bank the Rosella renderer used before the Rosella-specific one was
// captured.
class PublicBank final : public RosellaBank {
public:
    explicit PublicBank(std::size_t channels) : bank_(builtin_kernels(), channels) {}

    void reset() override { bank_.reset(); }

    void analyze_qmf(const std::vector<double>& hops, std::size_t slots,
                     std::vector<Complex>* qmf) override {
        bank_.analyze_qmf(hops, slots, qmf);
    }

    void analyze_hybrid(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<Complex>* hybrid) override {
        bank_.analyze_hybrid(qmf, slots, hybrid);
    }

    void synthesize_hybrid(const std::vector<Complex>& hybrid, std::size_t slots,
                           std::vector<Complex>* qmf) override {
        bank_.synthesize_hybrid(hybrid, slots, qmf);
    }

    void synthesize_qmf(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<double>* hops) override {
        bank_.synthesize_qmf(qmf, slots, hops);
    }

private:
    PublicFilterbank bank_;
};

// The bank captured from the Rosella headphone renderer.
class CapturedBank final : public RosellaBank {
public:
    explicit CapturedBank(std::size_t channels) : bank_(channels) {}

    void reset() override { bank_.reset(); }

    void analyze_qmf(const std::vector<double>& hops, std::size_t slots,
                     std::vector<Complex>* qmf) override {
        bank_.analyze_qmf(hops, slots, qmf);
    }

    void analyze_hybrid(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<Complex>* hybrid) override {
        bank_.analyze_hybrid(qmf, slots, hybrid);
    }

    void synthesize_hybrid(const std::vector<Complex>& hybrid, std::size_t slots,
                           std::vector<Complex>* qmf) override {
        bank_.synthesize_hybrid(hybrid, slots, qmf);
    }

    void synthesize_qmf(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<double>* hops) override {
        bank_.synthesize_qmf(qmf, slots, hops);
    }

private:
    RosellaFilterbank bank_;
};

}  // namespace

std::uint32_t rosella_bank_latency(RosellaBankKind kind) {
    return kind == RosellaBankKind::Captured
               ? static_cast<std::uint32_t>(RosellaFilterbank::kLatencySamples)
               : static_cast<std::uint32_t>(kLatencySamples);
}

std::unique_ptr<RosellaBank> RosellaBank::create(RosellaBankKind kind, std::size_t channels) {
    if (kind == RosellaBankKind::Captured) {
        return std::make_unique<CapturedBank>(channels);
    }
    return std::make_unique<PublicBank>(channels);
}

}  // namespace joc::hrtf
