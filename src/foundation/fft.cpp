#include "foundation/fft.h"

#include <cmath>

#include "simd/simd.h"

namespace joc::dsp {

// The dispatched kernels read and write the spectrum as interleaved doubles, and
// an array of std::complex<double> is exactly that: two doubles per element, no
// padding, no vtable.
static_assert(sizeof(Complex) == 2u * sizeof(double), "complex layout");

namespace {

constexpr double kPi = 3.14159265358979323846;

template <typename Container>
void fft_in_place(Container* data, bool inverse) {
    const std::size_t count = data->size();
    if (count < 2u) {
        return;
    }
    for (std::size_t index = 1u, reversed = 0u; index < count; ++index) {
        std::size_t bit = count >> 1u;
        for (; (reversed & bit) != 0u; bit >>= 1u) {
            reversed ^= bit;
        }
        reversed ^= bit;
        if (index < reversed) {
            std::swap((*data)[index], (*data)[reversed]);
        }
    }
    for (std::size_t length = 2u; length <= count; length <<= 1u) {
        const double angle = (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(length);
        const Complex step(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0u; start < count; start += length) {
            Complex factor(1.0, 0.0);
            for (std::size_t offset = 0u; offset < length / 2u; ++offset) {
                const Complex even = (*data)[start + offset];
                const Complex odd = (*data)[start + offset + length / 2u] * factor;
                (*data)[start + offset] = even + odd;
                (*data)[start + offset + length / 2u] = even - odd;
                factor *= step;
            }
        }
    }
    if (inverse) {
        for (Complex& value : *data) {
            value /= static_cast<double>(count);
        }
    }
}

bool is_power_of_two(std::size_t value) { return value != 0u && (value & (value - 1u)) == 0u; }

}  // namespace

FftPlan::FftPlan(std::size_t size, bool inverse) : size_(size), inverse_(inverse) {
    reverse_.resize(size);
    for (std::size_t index = 1u, reversed = 0u; index < size; ++index) {
        std::size_t bit = size >> 1u;
        for (; (reversed & bit) != 0u; bit >>= 1u) {
            reversed ^= bit;
        }
        reversed ^= bit;
        reverse_[index] = reversed;
    }
    for (std::size_t length = 2u; length <= size; length <<= 1u) {
        const double angle = (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(length);
        const Complex step(std::cos(angle), std::sin(angle));
        stage_begin_.push_back(twiddle_.size());
        Complex factor(1.0, 0.0);
        for (std::size_t offset = 0u; offset < length / 2u; ++offset) {
            twiddle_.push_back(factor);
            factor *= step;
        }
    }
}

// Exactly the operations fft_in_place performs, in the same order, with the
// twiddles read from the precomputed recurrence instead of being re-derived.
template <typename Container>
void FftPlan::apply(Container* data) const {
    const std::size_t count = data->size();
    if (count < 2u) {
        return;
    }
    const std::size_t* reverse = reverse_.data();
    for (std::size_t index = 1u; index < count; ++index) {
        const std::size_t reversed = reverse[index];
        if (index < reversed) {
            std::swap((*data)[index], (*data)[reversed]);
        }
    }
    // The cascade is dispatched for every power-of-two size the kernels can pack
    // whole groups into a vector (JOC_SIMD pins one tier for verification).  A
    // kernel only ever puts independent butterflies in the same vector, so every
    // output keeps the operation sequence and the roundings written below; small
    // transforms -- and the caller's own table -- keep the portable loop.
    if (count >= simd::kMinVectorFftSize && (count & (count - 1u)) == 0u) {
        simd::fft_butterflies(reinterpret_cast<double*>(data->data()), count,
                              reinterpret_cast<const double*>(twiddle_.data()),
                              stage_begin_.data());
    } else {
        std::size_t stage = 0u;
        for (std::size_t length = 2u; length <= count; length <<= 1u, ++stage) {
            const Complex* table = twiddle_.data() + stage_begin_[stage];
            for (std::size_t start = 0u; start < count; start += length) {
                for (std::size_t offset = 0u; offset < length / 2u; ++offset) {
                    const Complex even = (*data)[start + offset];
                    const Complex odd = (*data)[start + offset + length / 2u] * table[offset];
                    (*data)[start + offset] = even + odd;
                    (*data)[start + offset + length / 2u] = even - odd;
                }
            }
        }
    }
    if (inverse_) {
        for (Complex& value : *data) {
            value /= static_cast<double>(count);
        }
    }
}

void fft_radix2(std::vector<Complex>* data, const FftPlan& plan) { plan.apply(data); }

void fft_radix2(std::array<Complex, kQmfFftSize>* data, const FftPlan& plan) { plan.apply(data); }

void fft_radix2(std::vector<Complex>* data, bool inverse) { fft_in_place(data, inverse); }

void fft_radix2(std::array<Complex, kQmfFftSize>* data, bool inverse) {
    fft_in_place(data, inverse);
}

void fft_any(const std::vector<Complex>& input, bool inverse, std::vector<Complex>* output) {
    const std::size_t count = input.size();
    if (is_power_of_two(count)) {
        *output = input;
        fft_radix2(output, inverse);
        return;
    }
    std::size_t size = 1u;
    while (size < 2u * count + 1u) {
        size <<= 1u;
    }
    const double sign = inverse ? 1.0 : -1.0;
    std::vector<Complex> left(size, Complex(0.0, 0.0));
    std::vector<Complex> right(size, Complex(0.0, 0.0));
    for (std::size_t index = 0u; index < count; ++index) {
        const std::size_t wrapped = (index * index) % (2u * count);
        const double angle = kPi * static_cast<double>(wrapped) / static_cast<double>(count);
        const Complex chirp(std::cos(angle), sign * std::sin(angle));
        left[index] = input[index] * chirp;
        right[index] = std::conj(chirp);
        if (index != 0u) {
            right[size - index] = std::conj(chirp);
        }
    }
    fft_radix2(&left, false);
    fft_radix2(&right, false);
    for (std::size_t index = 0u; index < size; ++index) {
        left[index] *= right[index];
    }
    fft_radix2(&left, true);
    output->resize(count);
    for (std::size_t index = 0u; index < count; ++index) {
        const std::size_t wrapped = (index * index) % (2u * count);
        const double angle = kPi * static_cast<double>(wrapped) / static_cast<double>(count);
        const Complex chirp(std::cos(angle), sign * std::sin(angle));
        (*output)[index] = left[index] * chirp;
        if (inverse) {
            (*output)[index] /= static_cast<double>(count);
        }
    }
}

std::size_t next_fast_len(std::size_t value) {
    if (value <= 6u) {
        return value;
    }
    std::size_t best = value;
    for (std::size_t power2 = 1u; power2 < value * 2u; power2 *= 2u) {
        for (std::size_t power3 = power2; power3 < value * 2u; power3 *= 3u) {
            std::size_t power5 = power3;
            while (power5 < value) {
                power5 *= 5u;
            }
            best = std::min(best, power5);
            if (power3 >= value) {
                break;
            }
        }
    }
    return best;
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 1u;
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

}  // namespace joc::dsp
