#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "hrtf/public_filterbank.h"
#include "hrtf/rosella_bank.h"
#include "hrtf/rosella_filterbank.h"
#include "rosella_filterbank_vectors.h"

namespace {

int failures = 0;

void check(bool passed, const char* name) {
    if (!passed) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

using joc::hrtf::Complex;
using joc::hrtf::PublicFilterbank;
using joc::hrtf::RosellaBank;
using joc::hrtf::RosellaBankKind;
using joc::hrtf::RosellaFilterbank;

template <typename Value, typename Expected>
void check_samples(const std::vector<Value>& actual, const Expected& expected, const char* name) {
    for (const auto& sample : expected) {
        check(std::abs(actual.at(sample.index) - sample.value) < 1e-10, name);
    }
}

void test_reference_and_streaming() {
    constexpr std::size_t slots = 24;
    constexpr std::size_t channels = 2;
    std::vector<double> input(slots * channels * 64);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = (static_cast<int>((index * 37 + 11) % 257) - 128) / 128.0;
    }
    RosellaFilterbank bank(channels);
    std::vector<Complex> qmf, hybrid, folded;
    std::vector<double> time;
    bank.analyze_qmf(input, slots, &qmf);
    bank.analyze_hybrid(qmf, slots, &hybrid);
    check_samples(qmf, rosella_test_vectors::kAnalysis, "analysis reference");
    check_samples(hybrid, rosella_test_vectors::kHybrid, "hybrid reference");

    std::vector<Complex> synthetic(slots * channels * 77);
    for (std::size_t index = 0; index < synthetic.size(); ++index) {
        synthetic[index] = Complex((static_cast<int>((index * 17 + 3) % 101) - 50) / 64.0,
                                   (static_cast<int>((index * 13 + 7) % 97) - 48) / 64.0);
    }
    bank.synthesize_hybrid(synthetic, slots, &folded);
    bank.synthesize_qmf(folded, slots, &time);
    check_samples(folded, rosella_test_vectors::kFold, "fold reference");
    check_samples(time, rosella_test_vectors::kSynthesis, "synthesis reference");

    for (int repeat = 0; repeat < 2; ++repeat) {
        bank.reset();
        std::vector<Complex> joined_qmf, joined_hybrid, joined_folded;
        std::vector<double> joined_time;
        std::size_t first = 0;
        for (const std::size_t count : {0u, 1u, 7u, 0u, 3u, 13u}) {
            std::vector<double> part_input(input.begin() + first * channels * 64,
                                           input.begin() + (first + count) * channels * 64);
            std::vector<Complex> part_synthetic(synthetic.begin() + first * channels * 77,
                                                synthetic.begin() +
                                                    (first + count) * channels * 77);
            std::vector<Complex> part_qmf, part_hybrid, part_folded;
            std::vector<double> part_time;
            bank.analyze_qmf(part_input, count, &part_qmf);
            bank.analyze_hybrid(part_qmf, count, &part_hybrid);
            bank.synthesize_hybrid(part_synthetic, count, &part_folded);
            bank.synthesize_qmf(part_folded, count, &part_time);
            joined_qmf.insert(joined_qmf.end(), part_qmf.begin(), part_qmf.end());
            joined_hybrid.insert(joined_hybrid.end(), part_hybrid.begin(), part_hybrid.end());
            joined_folded.insert(joined_folded.end(), part_folded.begin(), part_folded.end());
            joined_time.insert(joined_time.end(), part_time.begin(), part_time.end());
            first += count;
        }
        check(joined_qmf == qmf, "analysis chunk/reset invariance");
        check(joined_hybrid == hybrid, "hybrid chunk/reset invariance");
        check(joined_folded == folded, "fold chunk/reset invariance");
        check(joined_time == time, "synthesis chunk/reset invariance");
    }

    for (std::size_t channel = 0; channel < channels; ++channel) {
        RosellaFilterbank mono(1);
        std::vector<double> mono_input;
        for (std::size_t slot = 0; slot < slots; ++slot) {
            const auto first = input.begin() + (slot * channels + channel) * 64;
            mono_input.insert(mono_input.end(), first, first + 64);
        }
        std::vector<Complex> mono_qmf, mono_hybrid;
        mono.analyze_qmf(mono_input, slots, &mono_qmf);
        mono.analyze_hybrid(mono_qmf, slots, &mono_hybrid);
        for (std::size_t slot = 0; slot < slots; ++slot) {
            check(std::equal(mono_hybrid.begin() + slot * 77, mono_hybrid.begin() + (slot + 1) * 77,
                             hybrid.begin() + (slot * channels + channel) * 77),
                  "channel isolation");
        }
    }
}

void test_latency() {
    RosellaFilterbank bank(1);
    std::vector<double> impulse(64 * 64);
    impulse[0] = 1.0;
    std::vector<Complex> qmf, hybrid;
    std::vector<double> output;
    bank.analyze_qmf(impulse, 64, &qmf);
    bank.analyze_hybrid(qmf, 64, &hybrid);
    bank.synthesize_hybrid(hybrid, 64, &qmf);
    bank.synthesize_qmf(qmf, 64, &output);
    const auto peak = std::max_element(output.begin(), output.end(), [](double left, double right) {
        return std::abs(left) < std::abs(right);
    });
    check(peak - output.begin() == RosellaFilterbank::kLatencySamples, "roundtrip latency");
    check(std::abs(*peak - 0.999989847495873) < 1e-10, "roundtrip gain");
}

// One pass of the four half-steps, split into the given slot counts; the
// concatenation is what a single call with the total count produces.
struct BankTrace {
    std::vector<Complex> qmf;
    std::vector<Complex> hybrid;
    std::vector<Complex> folded;
    std::vector<double> time;
};

BankTrace run_bank(RosellaBank& bank, const std::vector<double>& hops,
                   const std::vector<Complex>& synthetic, const std::vector<std::size_t>& chunks,
                   std::size_t channels) {
    BankTrace trace;
    std::size_t first = 0;
    for (const std::size_t count : chunks) {
        const std::vector<double> part_hops(hops.begin() + first * channels * 64,
                                            hops.begin() + (first + count) * channels * 64);
        const std::vector<Complex> part_synthetic(
            synthetic.begin() + first * channels * 77,
            synthetic.begin() + (first + count) * channels * 77);
        std::vector<Complex> qmf, hybrid, folded;
        std::vector<double> time;
        bank.analyze_qmf(part_hops, count, &qmf);
        bank.analyze_hybrid(qmf, count, &hybrid);
        bank.synthesize_hybrid(part_synthetic, count, &folded);
        bank.synthesize_qmf(folded, count, &time);
        trace.qmf.insert(trace.qmf.end(), qmf.begin(), qmf.end());
        trace.hybrid.insert(trace.hybrid.end(), hybrid.begin(), hybrid.end());
        trace.folded.insert(trace.folded.end(), folded.begin(), folded.end());
        trace.time.insert(trace.time.end(), time.begin(), time.end());
        first += count;
    }
    return trace;
}

bool same_trace(const BankTrace& left, const BankTrace& right) {
    return left.qmf == right.qmf && left.hybrid == right.hybrid &&
           left.folded == right.folded && left.time == right.time;
}

void test_bank_selection() {
    constexpr std::size_t slots = 24;
    constexpr std::size_t channels = 2;
    std::vector<double> hops(slots * channels * 64);
    for (std::size_t index = 0; index < hops.size(); ++index) {
        hops[index] = (static_cast<int>((index * 37 + 11) % 257) - 128) / 128.0;
    }
    std::vector<Complex> synthetic(slots * channels * 77);
    for (std::size_t index = 0; index < synthetic.size(); ++index) {
        synthetic[index] = Complex((static_cast<int>((index * 17 + 3) % 101) - 50) / 64.0,
                                   (static_cast<int>((index * 13 + 7) % 97) - 48) / 64.0);
    }

    // The selector is only a choice of implementation: each kind must reproduce
    // exactly what its own class computes, and the two banks must stay separate.
    std::vector<Complex> public_hybrid;
    std::vector<Complex> captured_hybrid;
    for (const RosellaBankKind kind : {RosellaBankKind::Public, RosellaBankKind::Captured}) {
        std::vector<Complex> direct_qmf, direct_hybrid;
        if (kind == RosellaBankKind::Public) {
            PublicFilterbank bank(joc::hrtf::builtin_kernels(), channels);
            bank.analyze_qmf(hops, slots, &direct_qmf);
            bank.analyze_hybrid(direct_qmf, slots, &direct_hybrid);
        } else {
            RosellaFilterbank bank(channels);
            bank.analyze_qmf(hops, slots, &direct_qmf);
            bank.analyze_hybrid(direct_qmf, slots, &direct_hybrid);
        }
        std::unique_ptr<RosellaBank> selected = RosellaBank::create(kind, channels);
        std::vector<Complex> selected_qmf, selected_hybrid;
        selected->analyze_qmf(hops, slots, &selected_qmf);
        selected->analyze_hybrid(selected_qmf, slots, &selected_hybrid);
        check(selected_qmf == direct_qmf, "selector analysis matches the bank");
        check(selected_hybrid == direct_hybrid, "selector hybrid matches the bank");
        if (kind == RosellaBankKind::Public) {
            public_hybrid = selected_hybrid;
        } else {
            captured_hybrid = selected_hybrid;
        }

        BankTrace whole;
        {
            std::unique_ptr<RosellaBank> fresh = RosellaBank::create(kind, channels);
            whole = run_bank(*fresh, hops, synthetic, {slots}, channels);
        }
        check(whole.qmf == selected_qmf, "whole chunk analysis");
        check(whole.hybrid == selected_hybrid, "whole chunk hybrid");
        for (int repeat = 0; repeat < 2; ++repeat) {
            std::unique_ptr<RosellaBank> split = RosellaBank::create(kind, channels);
            const BankTrace parts =
                run_bank(*split, hops, synthetic, {0u, 5u, 0u, 7u, 12u}, channels);
            check(same_trace(whole, parts), "chunk/reset invariance");
        }
    }
    check(public_hybrid != captured_hybrid, "the two banks produce different hybrid data");
}

// Distance from the impulse to the absolute peak of the analysis/synthesis
// roundtrip, which is the latency the runtime skips at the head of the stream.
std::size_t roundtrip_peak(RosellaBank& bank, std::size_t slots) {
    std::vector<double> impulse(slots * 64, 0.0);
    impulse[0] = 1.0;
    std::vector<Complex> qmf, hybrid;
    std::vector<double> output;
    bank.analyze_qmf(impulse, slots, &qmf);
    bank.analyze_hybrid(qmf, slots, &hybrid);
    bank.synthesize_hybrid(hybrid, slots, &qmf);
    bank.synthesize_qmf(qmf, slots, &output);
    const auto peak = std::max_element(output.begin(), output.end(), [](double left, double right) {
        return std::abs(left) < std::abs(right);
    });
    return static_cast<std::size_t>(peak - output.begin());
}

void test_bank_latency_and_reset() {
    check(joc::hrtf::rosella_bank_latency(RosellaBankKind::Public) ==
              static_cast<std::uint32_t>(joc::hrtf::kLatencySamples),
          "public latency is the public filterbank latency");
    check(joc::hrtf::rosella_bank_latency(RosellaBankKind::Captured) ==
              static_cast<std::uint32_t>(RosellaFilterbank::kLatencySamples),
          "captured latency is the Rosella filterbank latency");
    check(joc::hrtf::rosella_bank_latency(RosellaBankKind::Public) == 961u,
          "public latency is 961 samples");
    check(joc::hrtf::rosella_bank_latency(RosellaBankKind::Captured) == 448u,
          "captured latency is 448 samples");
    for (const RosellaBankKind kind : {RosellaBankKind::Public, RosellaBankKind::Captured}) {
        std::unique_ptr<RosellaBank> bank = RosellaBank::create(kind, 1);
        const std::size_t first = roundtrip_peak(*bank, 64);
        check(first == joc::hrtf::rosella_bank_latency(kind), "bank roundtrip latency");
        bank->reset();
        check(roundtrip_peak(*bank, 64) == first, "bank reset restores the startup delay");
    }
}

}

int main() {
    test_reference_and_streaming();
    test_latency();
    test_bank_selection();
    test_bank_latency_and_reset();
    std::printf("Rosella filterbank: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
