#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

// Complex transforms shared by the HRTF and Rosella DSP cores.  The convention is
// NumPy's: the forward transform is unnormalised and the inverse scales by 1/N,
// so a ported pipeline keeps the reference's arithmetic bit for bit.
namespace joc::dsp {

using Complex = std::complex<double>;

inline constexpr std::size_t kQmfFftSize = 128;

// Precomputed radix-2 plan for one size and direction.
//
// The transform derives each butterfly's twiddle by multiplying the previous one
// by the stage step, so the twiddle at offset k is `step` multiplied k times in
// that order, independently of the group.  Materialising that exact recurrence --
// and the bit-reversal permutation -- removes one complex multiply and a
// (length/2)-deep serial dependency from every stage's inner loop.  The table
// entries are the recurrence's own values, so the transform is bit-identical.
//
// The 128-point cascade is executed by the runtime-dispatched SIMD kernel
// (src/simd/simd.h): it computes independent butterflies in parallel lanes,
// which leaves both the table and every output's summation order untouched.
class FftPlan {
public:
    FftPlan(std::size_t size, bool inverse);

    std::size_t size() const { return size_; }
    bool inverse() const { return inverse_; }

private:
    template <typename Container>
    void apply(Container* data) const;

    friend void fft_radix2(std::vector<Complex>* data, const FftPlan& plan);
    friend void fft_radix2(std::array<Complex, kQmfFftSize>* data, const FftPlan& plan);

    std::size_t size_ = 0;
    bool inverse_ = false;
    std::vector<std::size_t> reverse_;      // bit-reversal permutation, [size]
    std::vector<std::size_t> stage_begin_;  // twiddle offset of each stage
    std::vector<Complex> twiddle_;          // per stage, length/2 entries, concatenated
};

// In-place radix-2 transform; the size must be a power of two.
void fft_radix2(std::vector<Complex>* data, bool inverse);
void fft_radix2(std::array<Complex, kQmfFftSize>* data, bool inverse);

// Plan-driven forms: the plan carries the size and the direction, so a caller that
// transforms the same length repeatedly builds it once.
void fft_radix2(std::vector<Complex>* data, const FftPlan& plan);
void fft_radix2(std::array<Complex, kQmfFftSize>* data, const FftPlan& plan);

// Exact-length transform: radix-2 when the size allows it, Bluestein otherwise.
// scipy/numpy use a mixed-radix transform, which is the same transform.
void fft_any(const std::vector<Complex>& input, bool inverse, std::vector<Complex>* output);

// scipy's next_fast_len: the smallest 5-smooth number that is not smaller.
std::size_t next_fast_len(std::size_t value);

std::size_t next_power_of_two(std::size_t value);

}  // namespace joc::dsp
