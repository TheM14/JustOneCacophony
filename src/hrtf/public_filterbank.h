#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "hrtf/jochrtf.h"

// Public 64-QMF / 77-hybrid filterbank, shared by the SOFA field compiler and the
// Rosella renderer (upstream public_filterbank.py and rosella_filterbank.py are
// the same bank).  Everything is float64/complex128, as the reference computes it,
// and the stateful half-steps are exposed because Rosella drives them directly.
namespace joc::hrtf {

inline constexpr int kQmfBands = 64;
inline constexpr int kQmfHop = 64;
inline constexpr int kHybridLow = 16;
inline constexpr int kHybridBandCount = 77;
inline constexpr int kLatencySamples = 961;

using Complex = std::complex<double>;

class PublicFilterbank {
public:
    PublicFilterbank(const Kernels& kernels, std::size_t channels);
    ~PublicFilterbank();
    PublicFilterbank(const PublicFilterbank&) = delete;
    PublicFilterbank& operator=(const PublicFilterbank&) = delete;

    void reset();

    // Full-rate [slots*64, channels] -> hybrid [slots, channels, 77].
    void analyze_full_rate(const std::vector<double>& samples, std::size_t slots,
                           std::vector<Complex>* hybrid);
    // Hybrid [slots, channels, 77] -> full-rate [slots*64, channels].
    void synthesize_full_rate(const std::vector<Complex>& hybrid, std::size_t slots,
                              std::vector<double>* time);

    // The stateful half-steps, in the order the reference runs them.
    void analyze_qmf(const std::vector<double>& hops, std::size_t slots,
                     std::vector<Complex>* qmf);
    void analyze_hybrid(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<Complex>* hybrid);
    void synthesize_hybrid(const std::vector<Complex>& hybrid, std::size_t slots,
                           std::vector<Complex>* qmf);
    void synthesize_qmf(const std::vector<Complex>& qmf, std::size_t slots,
                        std::vector<double>* time);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joc::hrtf
