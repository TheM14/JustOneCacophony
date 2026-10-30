#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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

}

int main() {
    test_reference_and_streaming();
    test_latency();
    std::printf("Rosella filterbank: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
