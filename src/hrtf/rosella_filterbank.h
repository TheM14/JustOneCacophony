#pragma once

#include <cstddef>
#include <vector>

#include "hrtf/public_filterbank.h"

namespace joc::hrtf {

class RosellaFilterbank {
public:
    static constexpr int kLatencySamples = 448;

    explicit RosellaFilterbank(std::size_t channels);

    void reset();
    void analyze_qmf(const std::vector<double>& hops, std::size_t slots, std::vector<Complex>* qmf);
    void analyze_hybrid(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<Complex>* hybrid);
    void synthesize_hybrid(const std::vector<Complex>& hybrid, std::size_t slots,
                           std::vector<Complex>* qmf);
    void synthesize_qmf(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<double>* hops);

private:
    std::size_t channels_;
    std::size_t analysis_position_ = 0;
    std::size_t hybrid_position_ = 0;
    std::size_t synthesis_position_ = 0;
    std::vector<double> analysis_history_;
    std::vector<Complex> hybrid_history_;
    std::vector<Complex> synthesis_history_;
    std::vector<Complex> spectrum_;
    std::vector<Complex> hybrid_eight_;
    std::vector<Complex> hybrid_four_;
};

}
