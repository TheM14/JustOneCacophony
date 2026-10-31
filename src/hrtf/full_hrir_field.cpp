
#include "hrtf/full_hrir_field.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>

#include "foundation/fft.h"
#include "hrtf/sofa_field.h"

namespace joc::hrtf {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSoundSpeed = 343.3;
constexpr double kScanTolerance = 1.0e-10;
constexpr int kMcaBands = 41;

Status field_error(const char* message) {
    return Status::fail(JOC_ERR_HRTF_FORMAT, joc::stage::kRender, message);
}

Status argument_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kRender, message);
}

// Unnormalised forward transform of a real block, first n/2+1 bins (numpy rfft).
void real_fft(const double* time, std::size_t count, std::vector<std::complex<double>>* out) {
    std::vector<std::complex<double>> buffer(count);
    for (std::size_t index = 0; index < count; ++index) {
        buffer[index] = std::complex<double>(time[index], 0.0);
    }
    dsp::fft_radix2(&buffer, false);
    out->assign(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(count / 2 + 1));
}

// In-place phase unwrapping (numpy.unwrap with the default 2*pi period).
void unwrap_phase(const std::vector<double>& wrapped, std::vector<double>* out) {
    out->assign(wrapped.size(), 0.0);
    if (wrapped.empty()) {
        return;
    }
    (*out)[0] = wrapped[0];
    for (std::size_t index = 1; index < wrapped.size(); ++index) {
        double delta = wrapped[index] - wrapped[index - 1];
        delta -= 2.0 * kPi * std::round(delta / (2.0 * kPi));
        (*out)[index] = (*out)[index - 1] + delta;
    }
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                     values.end());
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    const double upper = values[middle];
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle - 1),
                     values.end());
    return 0.5 * (values[middle - 1] + upper);
}

// scipy.signal._cubic edge/interior slopes of PchipInterpolator.
double pchip_edge(double h0, double h1, double m0, double m1) {
    double derivative = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    const bool opposite = std::signbit(derivative) != std::signbit(m0);
    const bool clamp = (std::signbit(m0) != std::signbit(m1)) &&
                       (std::fabs(derivative) > 3.0 * std::fabs(m0));
    if (opposite) {
        derivative = 0.0;
    } else if (clamp) {
        derivative = 3.0 * m0;
    }
    return derivative;
}

class PchipEvaluator {
public:
    void build(const std::vector<double>& x, const std::vector<double>& y) {
        x_ = x;
        y_ = y;
        const std::size_t count = x.size();
        d_.assign(count, 0.0);
        std::vector<double> h(count - 1);
        std::vector<double> delta(count - 1);
        for (std::size_t index = 0; index + 1 < count; ++index) {
            h[index] = x[index + 1] - x[index];
            delta[index] = h[index] != 0.0 ? (y[index + 1] - y[index]) / h[index] : 0.0;
        }
        if (count == 2) {
            d_[0] = delta[0];
            d_[1] = delta[0];
            return;
        }
        d_[0] = pchip_edge(h[0], h[1], delta[0], delta[1]);
        d_[count - 1] = pchip_edge(h[count - 2], h[count - 3], delta[count - 2], delta[count - 3]);
        for (std::size_t index = 1; index + 1 < count; ++index) {
            if (delta[index - 1] * delta[index] <= 0.0) {
                d_[index] = 0.0;
                continue;
            }
            const double w1 = 2.0 * h[index] + h[index - 1];
            const double w2 = h[index] + 2.0 * h[index - 1];
            d_[index] = (w1 + w2) / (w1 / delta[index - 1] + w2 / delta[index]);
        }
    }

    double evaluate(double value) const {
        const std::size_t count = x_.size();
        if (count < 2) {
            return count == 1 ? y_[0] : 0.0;
        }
        std::size_t index = 0;
        if (value >= x_.back()) {
            index = count - 2;
        } else if (value > x_.front()) {
            index = static_cast<std::size_t>(
                        std::upper_bound(x_.begin(), x_.end(), value) - x_.begin()) - 1;
        }
        const double h = x_[index + 1] - x_[index];
        if (h == 0.0) {
            return y_[index];
        }
        const double t = (value - x_[index]) / h;
        const double t2 = t * t;
        const double t3 = t2 * t;
        return (2.0 * t3 - 3.0 * t2 + 1.0) * y_[index] +
               (t3 - 2.0 * t2 + t) * h * d_[index] +
               (-2.0 * t3 + 3.0 * t2) * y_[index + 1] +
               (t3 - t2) * h * d_[index + 1];
    }

private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> d_;
};

}  // namespace

Status sphere_transfer(const std::vector<double>& frequencies, double radius, double head_radius,
                       const std::vector<double>& cosines,
                       std::vector<std::complex<double>>* out) {
    if (out == nullptr || !(radius > head_radius) || head_radius <= 0.0) {
        return argument_error("sphere transfer needs a source outside the head");
    }
    const std::size_t bands = frequencies.size();
    out->assign(cosines.size() * bands, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> x(bands);
    std::vector<std::complex<double>> z(bands);
    std::vector<std::complex<double>> ratio_a(bands);
    std::vector<std::complex<double>> ratio_r(bands);
    std::vector<std::complex<double>> product(bands, std::complex<double>(1.0, 0.0));
    double maximum_x = 0.0;
    for (std::size_t band = 0; band < bands; ++band) {
        const double wave = 2.0 * kPi * frequencies[band] * head_radius / kSoundSpeed;
        maximum_x = std::max(maximum_x, wave);
        x[band] = std::complex<double>(wave, 0.0);
        z[band] = std::complex<double>(wave * radius / head_radius, 0.0);
        ratio_a[band] = x[band] / (1.0 + std::complex<double>(0.0, 1.0) * x[band]);
        ratio_r[band] = z[band] / (1.0 + std::complex<double>(0.0, 1.0) * z[band]);
    }
    const double geometric = std::ceil(std::log(kScanTolerance) / std::log(head_radius / radius)) + 16.0;
    const int order = static_cast<int>(std::max(std::ceil(maximum_x) + 32.0, geometric));
    if (order > 4096) {
        return argument_error("source too close to the sphere surface");
    }
    std::vector<double> legendre_previous(cosines.size(), 1.0);
    std::vector<double> legendre(cosines.size(), 0.0);
    std::vector<double> mu(cosines.size());
    for (std::size_t row = 0; row < cosines.size(); ++row) {
        mu[row] = std::clamp(cosines[row], -1.0, 1.0);
        legendre[row] = mu[row];
        for (std::size_t band = 0; band < bands; ++band) {
            const std::complex<double> value = 1.0 / (1.0 + std::complex<double>(0.0, 1.0) * x[band]);
            (*out)[row * bands + band] = value;
        }
    }
    std::vector<std::complex<double>> term(bands);
    for (int degree = 1; degree <= order; ++degree) {
        for (std::size_t band = 0; band < bands; ++band) {
            const std::complex<double> quotient =
                std::abs(z[band]) != 0.0 ? ratio_a[band] / ratio_r[band]
                                         : std::complex<double>(head_radius / radius, 0.0);
            product[band] *= quotient;
            term[band] = product[band] /
                         ((static_cast<double>(degree) + 1.0) -
                          x[band] * ratio_a[band]);
        }
        for (std::size_t row = 0; row < cosines.size(); ++row) {
            const double factor = (2.0 * static_cast<double>(degree) + 1.0) * legendre[row];
            for (std::size_t band = 0; band < bands; ++band) {
                (*out)[row * bands + band] += factor * term[band];
            }
        }
        // Legendre recurrence P_{n+1} = ((2n+1) mu P_n - n P_{n-1}) / (n+1).
        for (std::size_t row = 0; row < cosines.size(); ++row) {
            const double next =
                ((2.0 * static_cast<double>(degree) + 1.0) * mu[row] * legendre[row] -
                 static_cast<double>(degree) * legendre_previous[row]) /
                (static_cast<double>(degree) + 1.0);
            legendre_previous[row] = legendre[row];
            legendre[row] = next;
        }
        for (std::size_t band = 0; band < bands; ++band) {
            const double denominator_a =
                (2.0 * static_cast<double>(degree) + 1.0);
            ratio_a[band] = x[band] / (denominator_a - x[band] * ratio_a[band]);
            ratio_r[band] = z[band] / (denominator_a - z[band] * ratio_r[band]);
        }
    }
    for (std::size_t band = 0; band < bands; ++band) {
        const std::complex<double> phase = std::exp(std::complex<double>(0.0, 1.0) * x[band]);
        for (std::size_t row = 0; row < cosines.size(); ++row) {
            (*out)[row * bands + band] *= phase;
        }
    }
    for (const std::complex<double>& value : *out) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            return field_error("sphere series failed to converge");
        }
    }
    return Status::success();
}

Status FullHrirField::build(const CanonicalHrtf& canonical, const FullHrirOptions& options) {
    method_ = options.method;
    if (std::fabs(canonical.sample_rate_hz - 48000.0) > 1e-9) {
        return field_error("full HRIR renderer requires 48000 Hz");
    }
    if (canonical.measurements == 0 || canonical.taps == 0) {
        return field_error("canonical HRIR set is empty");
    }
    if (canonical.hrir.size() < static_cast<std::size_t>(canonical.measurements) * 2u *
                                     canonical.taps ||
        canonical.delay_samples.size() < static_cast<std::size_t>(canonical.measurements) * 2u) {
        return field_error("canonical HRIR set is truncated");
    }

    // Shell selection: an explicit radius wins, otherwise a single shell is required.
    std::vector<double> shells(canonical.measurement_radius_m.begin(),
                               canonical.measurement_radius_m.end());
    std::sort(shells.begin(), shells.end());
    shells.erase(std::unique(shells.begin(), shells.end(),
                             [](double a, double b) { return std::fabs(a - b) < 1e-9; }),
                 shells.end());
    if (!options.has_shell_radius && shells.size() != 1) {
        return field_error("multi-radius SOFA requires an explicit shell radius");
    }
    if (options.has_shell_radius && (!std::isfinite(options.shell_radius_m) ||
                                     options.shell_radius_m <= 0.0)) {
        return field_error("shell radius must be finite and positive");
    }
    const double requested = options.has_shell_radius ? options.shell_radius_m : shells[0];
    double actual_radius = 0.0;
    const std::vector<std::size_t> selected =
        canonical_shell_indices(canonical, requested, &actual_radius);

    // Every selected ear must carry energy; delays size the FFT budget.
    double maximum_delay = 0.0;
    for (const std::size_t row : selected) {
        for (int ear = 0; ear < 2; ++ear) {
            double energy = 0.0;
            const double* taps = canonical.hrir.data() +
                                 (row * 2u + static_cast<std::size_t>(ear)) * canonical.taps;
            for (std::uint32_t tap = 0; tap < canonical.taps; ++tap) {
                energy += taps[tap] * taps[tap];
            }
            if (!(energy > 0.0)) {
                return field_error("each selected HRIR ear must contain signal energy");
            }
            maximum_delay = std::max(maximum_delay,
                                     canonical.delay_samples[row * 2u +
                                                             static_cast<std::size_t>(ear)]);
        }
    }
    taps_ = canonical.taps;
    std::vector<double> radii;
    radii.reserve(selected.size());
    for (const std::size_t row : selected) {
        radii.push_back(canonical.measurement_radius_m[row]);
    }
    measurement_radius_m_ = median(radii);
    const double support = static_cast<double>(canonical.taps) +
                           std::ceil(maximum_delay) + 1.0;
    std::size_t fft_size = options.fft_size;
    if (fft_size == 0) {
        fft_size = 2048;
        while (fft_size < static_cast<std::size_t>(2.0 * support - 1.0)) {
            fft_size <<= 1u;
        }
        if (fft_size > 16384) {
            return field_error("HRIR and Data.Delay exceed the automatic FFT budget");
        }
    }
    if (fft_size < 64 || fft_size % 2 != 0 ||
        static_cast<double>(canonical.taps) + maximum_delay >=
            static_cast<double>(fft_size / 2)) {
        return field_error("HRIR and Data.Delay exceed the selected FFT time budget");
    }
    fft_size_ = fft_size;
    bins_ = fft_size / 2 + 1;

    // Receiver geometry: a fixed pair, centred on the listener, one radius.
    if (canonical.receiver_position_cartesian_m.size() < 6u) {
        return field_error("ReceiverPosition is missing");
    }
    const double* left = canonical.receiver_position_cartesian_m.data();
    const double* right = left + 3;
    const double mean[3] = {(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5,
                            (left[2] + right[2]) * 0.5};
    if (std::sqrt(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]) > 1e-6) {
        return field_error("radial model requires ears centred on the listener");
    }
    head_radius_m_ = std::sqrt(left[0] * left[0] + left[1] * left[1] + left[2] * left[2]);
    const double right_radius = std::sqrt(right[0] * right[0] + right[1] * right[1] +
                                          right[2] * right[2]);
    if (!(head_radius_m_ > 0.04 && head_radius_m_ < 0.15) ||
        std::fabs(right_radius - head_radius_m_) > 1e-6) {
        return field_error("unsupported receiver geometry for the radial model");
    }
    ear_directions_.assign(6, 0.0);
    for (int axis = 0; axis < 3; ++axis) {
        ear_directions_[static_cast<std::size_t>(axis)] = left[axis] / head_radius_m_;
        ear_directions_[3u + static_cast<std::size_t>(axis)] = right[axis] / head_radius_m_;
    }

    // Frequencies and phase-fit grid.
    frequencies_.resize(bins_);
    omega_.resize(bins_);
    for (std::size_t bin = 0; bin < bins_; ++bin) {
        frequencies_[bin] = static_cast<double>(bin) * 48000.0 / static_cast<double>(fft_size_);
        omega_[bin] = 2.0 * kPi * frequencies_[bin] / 48000.0;
    }
    std::size_t fit_begin = 0;
    std::size_t fit_end = 0;
    double omega_mean = 0.0;
    for (std::size_t bin = 0; bin < bins_; ++bin) {
        if (frequencies_[bin] >= 200.0 && frequencies_[bin] <= 1500.0) {
            if (fit_end == fit_begin) {
                fit_begin = bin;
            }
            fit_end = bin + 1;
        }
    }
    if (fit_end <= fit_begin) {
        return field_error("the phase-fit band is empty");
    }
    for (std::size_t bin = fit_begin; bin < fit_end; ++bin) {
        omega_mean += omega_[bin];
    }
    omega_mean /= static_cast<double>(fit_end - fit_begin);
    double x_squares = 0.0;
    std::vector<double> x_fit(fit_end - fit_begin);
    for (std::size_t bin = fit_begin; bin < fit_end; ++bin) {
        x_fit[bin - fit_begin] = omega_[bin] - omega_mean;
        x_squares += x_fit[bin - fit_begin] * x_fit[bin - fit_begin];
    }

    // Raw spectra, then delay estimation and alignment.
    const std::size_t rows = selected.size();
    std::vector<std::complex<double>> raw(rows * 2 * bins_);
    for (std::size_t row = 0; row < rows; ++row) {
        for (int ear = 0; ear < 2; ++ear) {
            const double* taps = canonical.hrir.data() +
                                 (selected[row] * 2u + static_cast<std::size_t>(ear)) *
                                     canonical.taps;
            std::vector<double> padded(fft_size_, 0.0);
            std::memcpy(padded.data(), taps, sizeof(double) * canonical.taps);
            std::vector<std::complex<double>> spectrum;
            real_fft(padded.data(), fft_size_, &spectrum);
            std::copy(spectrum.begin(), spectrum.end(),
                      raw.begin() + static_cast<std::ptrdiff_t>((row * 2u +
                                                                 static_cast<std::size_t>(ear)) *
                                                                bins_));
        }
    }
    delays_.assign(rows * 2, 0.0);
    std::vector<double> residuals;
    std::vector<double> wrapped(bins_);
    std::vector<double> unwrapped;
    for (std::size_t row = 0; row < rows; ++row) {
        for (int ear = 0; ear < 2; ++ear) {
            const std::complex<double>* spectrum =
                raw.data() + (row * 2u + static_cast<std::size_t>(ear)) * bins_;
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                wrapped[bin] = std::arg(spectrum[bin]);
            }
            unwrap_phase(wrapped, &unwrapped);
            double numerator = 0.0;
            for (std::size_t bin = fit_begin; bin < fit_end; ++bin) {
                numerator += x_fit[bin - fit_begin] * unwrapped[bin];
            }
            const double data_delay =
                canonical.delay_samples[selected[row] * 2u + static_cast<std::size_t>(ear)];
            delays_[row * 2u + static_cast<std::size_t>(ear)] =
                -numerator / x_squares + data_delay;
        }
    }
    // Residual phase spread of the estimate (reported, not used downstream).
    for (std::size_t row = 0; row < rows; ++row) {
        for (int ear = 0; ear < 2; ++ear) {
            const std::complex<double>* spectrum =
                raw.data() + (row * 2u + static_cast<std::size_t>(ear)) * bins_;
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                wrapped[bin] = std::arg(spectrum[bin]);
            }
            unwrap_phase(wrapped, &unwrapped);
            const double data_delay =
                canonical.delay_samples[selected[row] * 2u + static_cast<std::size_t>(ear)];
            const double delay = delays_[row * 2u + static_cast<std::size_t>(ear)];
            double mean = 0.0;
            std::vector<double> residual(fit_end - fit_begin);
            for (std::size_t bin = fit_begin; bin < fit_end; ++bin) {
                residual[bin - fit_begin] = unwrapped[bin] + (delay - data_delay) * omega_[bin];
                mean += residual[bin - fit_begin];
            }
            mean /= static_cast<double>(residual.size());
            double squares = 0.0;
            for (double& value : residual) {
                value -= mean;
                squares += value * value;
            }
            residuals.push_back(std::sqrt(squares / static_cast<double>(residual.size())));
        }
    }
    std::sort(residuals.begin(), residuals.end());
    phase_fit_rms_rad_p95_ = residuals.empty()
                                 ? 0.0
                                 : residuals[static_cast<std::size_t>(
                                       std::floor(0.95 * static_cast<double>(residuals.size())))];

    // Recording origin: the median mean delay over the measurements.
    std::vector<double> row_means(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        row_means[row] = 0.5 * (delays_[row * 2u] + delays_[row * 2u + 1u]);
    }
    recording_origin_samples_ = static_cast<int>(std::llround(median(row_means)));

    // Physical spectra (Data.Delay removed) and the alignment delay.
    std::vector<std::complex<double>> physical(raw.size());
    double maximum = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (int ear = 0; ear < 2; ++ear) {
            const std::size_t base = (row * 2u + static_cast<std::size_t>(ear)) * bins_;
            const double data_delay =
                canonical.delay_samples[selected[row] * 2u + static_cast<std::size_t>(ear)];
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                const std::complex<double> rotation =
                    std::exp(std::complex<double>(0.0, -1.0) * omega_[bin] * data_delay);
                physical[base + bin] = raw[base + bin] * rotation;
                maximum = std::max(maximum, std::abs(physical[base + bin]));
            }
        }
    }
    floor_ = std::max(maximum * 1e-6, 1e-15);

    // Directions: deduplicate coincident measurements, keeping the first, then the
    // mesh over that set.
    std::vector<double> directions;
    directions.reserve(rows * 3);
    std::vector<std::size_t> kept_rows;
    for (std::size_t row = 0; row < rows; ++row) {
        const double* unit = canonical.unit_directions.data() + selected[row] * 3u;
        bool merged = false;
        for (std::size_t existing = 0; existing < kept_rows.size(); ++existing) {
            const double* other = canonical.unit_directions.data() + kept_rows[existing] * 3u;
            if (std::round(unit[0] * 1e10) == std::round(other[0] * 1e10) &&
                std::round(unit[1] * 1e10) == std::round(other[1] * 1e10) &&
                std::round(unit[2] * 1e10) == std::round(other[2] * 1e10)) {
                // Coincident measurements must agree, as the reference requires.
                for (std::size_t tap = 0; tap < canonical.taps * 2u; ++tap) {
                    const double a = canonical.hrir[selected[row] * 2u * canonical.taps + tap];
                    const double b = canonical.hrir[kept_rows[existing] * 2u * canonical.taps + tap];
                    if (std::fabs(a - b) > 1e-10 + 1e-8 * std::fabs(b)) {
                        return field_error("coincident SOFA measurements disagree");
                    }
                }
                merged = true;
                break;
            }
        }
        if (merged) {
            continue;
        }
        kept_rows.push_back(selected[row]);
        directions.push_back(unit[0]);
        directions.push_back(unit[1]);
        directions.push_back(unit[2]);
    }
    const Status mesh_status = mesh_.build(directions, kept_rows.size());
    if (!mesh_status.ok()) {
        return mesh_status;
    }
    // The measurements must enclose the listener: every hull face must sit in
    // front of the unit sphere's centre.
    for (std::size_t face = 0; face < mesh_.face_count(); ++face) {
        const int* ids = mesh_.faces().data() + face * 3;
        const double* a = mesh_.directions().data() + ids[0] * 3;
        const double* b = mesh_.directions().data() + ids[1] * 3;
        const double* c = mesh_.directions().data() + ids[2] * 3;
        double one[3];
        double two[3];
        for (int axis = 0; axis < 3; ++axis) {
            one[axis] = b[axis] - a[axis];
            two[axis] = c[axis] - a[axis];
        }
        double normal[3] = {one[1] * two[2] - one[2] * two[1],
                            one[2] * two[0] - one[0] * two[2],
                            one[0] * two[1] - one[1] * two[0]};
        const double length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                        normal[2] * normal[2]);
        if (!(length > 0.0)) {
            return field_error("ill-conditioned spherical interpolation mesh");
        }
        for (double& value : normal) {
            value /= length;
        }
        const double distance = normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2];
        if (std::fabs(distance) < 1e-9) {
            return field_error("SOFA must enclose the listener");
        }
        if (distance < 0.0) {
            return field_error("ill-conditioned spherical interpolation mesh");
        }
    }

    // Centre spectrum from the spherical Voronoi weights.
    std::vector<double> areas;
    mesh_.voronoi_areas(&areas);
    std::vector<double> energy(bins_, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        const double weight = row < areas.size() ? areas[row] : 0.0;
        for (int ear = 0; ear < 2; ++ear) {
            const std::size_t base = (row * 2u + static_cast<std::size_t>(ear)) * bins_;
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                energy[bin] += weight * std::norm(physical[base + bin]);
            }
        }
    }
    centre_spectrum_.assign(bins_, std::complex<double>(0.0, 0.0));
    for (std::size_t bin = 0; bin < bins_; ++bin) {
        const double magnitude = std::sqrt(energy[bin] * 0.5);
        centre_spectrum_[bin] = magnitude * std::exp(std::complex<double>(
                                                   0.0, -omega_[bin] *
                                                            static_cast<double>(
                                                                recording_origin_samples_)));
    }

    // Aligned spectra and the magnitude representation used by the interpolation.
    aligned_.assign(rows * 2 * bins_, std::complex<double>(0.0, 0.0));
    log_magnitude_.assign(rows * 2 * bins_, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (int ear = 0; ear < 2; ++ear) {
            const std::size_t index = row * 2u + static_cast<std::size_t>(ear);
            const double delay = delays_[index];
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                aligned_[index * bins_ + bin] =
                    physical[index * bins_ + bin] *
                    std::exp(std::complex<double>(0.0, omega_[bin] * delay));
                log_magnitude_[index * bins_ + bin] =
                    std::log(std::max(std::abs(physical[index * bins_ + bin]), floor_));
            }
        }
    }

    if (method_ == HrirInterpolation::kMcaZero) {
        // 41 auditory bands (reference: erb-spaced centres, rounded exponential bank).
        centres_hz_.resize(kMcaBands);
        const auto erb = [](double frequency) {
            return 21.4 * std::log10(1.0 + 0.00437 * frequency);
        };
        const double low = erb(50.0);
        const double high = erb(20000.0);
        for (int band = 0; band < kMcaBands; ++band) {
            const double position = low + (high - low) * band / (kMcaBands - 1);
            centres_hz_[static_cast<std::size_t>(band)] =
                (std::pow(10.0, position / 21.4) - 1.0) / 0.00437;
        }
        bank_.assign(kMcaBands * bins_, 0.0);
        for (int band = 0; band < kMcaBands; ++band) {
            const double centre = centres_hz_[static_cast<std::size_t>(band)];
            const double width = 1.019 * 24.7 * (1.0 + 0.00437 * centre);
            double total = 0.0;
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                const double ratio = (frequencies_[bin] - centre) / width;
                double value = std::pow(1.0 + ratio * ratio, -4.0);
                if (bin != 0 && bin + 1 != bins_) {
                    value *= 2.0;
                }
                bank_[static_cast<std::size_t>(band) * bins_ + bin] = value;
                total += value;
            }
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                bank_[static_cast<std::size_t>(band) * bins_ + bin] /= total;
            }
        }
        auditory_.assign(rows * 2 * kMcaBands, 0.0);
        std::vector<double> power(bins_, 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (int ear = 0; ear < 2; ++ear) {
                const std::size_t index = row * 2u + static_cast<std::size_t>(ear);
                for (std::size_t bin = 0; bin < bins_; ++bin) {
                    power[bin] = std::norm(physical[index * bins_ + bin]);
                }
                for (int band = 0; band < kMcaBands; ++band) {
                    double sum = 0.0;
                    for (std::size_t bin = 0; bin < bins_; ++bin) {
                        sum += power[bin] *
                               bank_[static_cast<std::size_t>(band) * bins_ + bin];
                    }
                    auditory_[index * kMcaBands + static_cast<std::size_t>(band)] =
                        10.0 * std::log10(std::max(sum, 1e-24));
                }
            }
        }
    }
    return Status::success();
}

Status FullHrirField::radial_ratio(const double direction[3], double radius,
                                   std::vector<std::complex<double>>* out) const {
    if (out == nullptr) {
        return argument_error("radial ratio output is null");
    }
    if (std::fabs(radius - measurement_radius_m_) < 1e-10) {
        out->assign(2 * bins_, std::complex<double>(1.0, 0.0));
        return Status::success();
    }
    if (!(radius > 0.0) || !std::isfinite(radius)) {
        return argument_error("radial ratio needs a positive finite radius");
    }
    std::vector<double> cosines(2, 0.0);
    for (int ear = 0; ear < 2; ++ear) {
        cosines[static_cast<std::size_t>(ear)] =
            ear_directions_[static_cast<std::size_t>(ear) * 3u] * direction[0] +
            ear_directions_[static_cast<std::size_t>(ear) * 3u + 1u] * direction[1] +
            ear_directions_[static_cast<std::size_t>(ear) * 3u + 2u] * direction[2];
    }
    std::vector<std::complex<double>> source;
    std::vector<std::complex<double>> target;
    Status status = sphere_transfer(frequencies_, measurement_radius_m_, head_radius_m_, cosines,
                                    &source);
    if (!status.ok()) {
        return status;
    }
    status = sphere_transfer(frequencies_, radius, head_radius_m_, cosines, &target);
    if (!status.ok()) {
        return status;
    }
    for (const std::complex<double>& value : source) {
        if (std::abs(value) < 1e-10) {
            return field_error("singular radial reference response");
        }
    }
    out->resize(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) {
        (*out)[index] = target[index] / source[index];
    }
    return Status::success();
}

Status FullHrirField::spectrum(const double direction[3],
                               std::vector<std::complex<double>>* out) {
    if (out == nullptr) {
        return argument_error("spectrum output is null");
    }
    std::vector<int> ids;
    std::vector<double> weights;
    const Status status = mesh_.weights(direction, &ids, &weights);
    if (!status.ok()) {
        return status;
    }
    if (ids.empty()) {
        return field_error("no interpolation weights for the direction");
    }
    double delay[2] = {0.0, 0.0};
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const std::size_t id = static_cast<std::size_t>(ids[index]);
        for (int ear = 0; ear < 2; ++ear) {
            delay[ear] += weights[index] * delays_[id * 2u + static_cast<std::size_t>(ear)];
        }
    }
    out->assign(2 * bins_, std::complex<double>(0.0, 0.0));
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const std::size_t id = static_cast<std::size_t>(ids[index]);
        for (int ear = 0; ear < 2; ++ear) {
            const std::complex<double>* aligned =
                aligned_.data() + (id * 2u + static_cast<std::size_t>(ear)) * bins_;
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                (*out)[static_cast<std::size_t>(ear) * bins_ + bin] +=
                    weights[index] * aligned[bin];
            }
        }
    }
    for (int ear = 0; ear < 2; ++ear) {
        for (std::size_t bin = 0; bin < bins_; ++bin) {
            (*out)[static_cast<std::size_t>(ear) * bins_ + bin] *=
                std::exp(std::complex<double>(0.0, -omega_[bin] *
                                                       delay[static_cast<std::size_t>(ear)]));
        }
    }
    if (ids.size() > 1) {
        if (method_ == HrirInterpolation::kLogMagnitudePhase) {
            for (int ear = 0; ear < 2; ++ear) {
                std::vector<double> magnitude(bins_, 0.0);
                for (std::size_t index = 0; index < ids.size(); ++index) {
                    const std::size_t id = static_cast<std::size_t>(ids[index]);
                    const double* values =
                        log_magnitude_.data() +
                        (id * 2u + static_cast<std::size_t>(ear)) * bins_;
                    for (std::size_t bin = 0; bin < bins_; ++bin) {
                        magnitude[bin] += weights[index] * values[bin];
                    }
                }
                for (std::size_t bin = 0; bin < bins_; ++bin) {
                    std::complex<double>& value =
                        (*out)[static_cast<std::size_t>(ear) * bins_ + bin];
                    const double target = std::exp(magnitude[bin]);
                    value = target * value /
                            std::max(std::abs(value), floor_);
                }
            }
        } else {
            for (int ear = 0; ear < 2; ++ear) {
                std::vector<double> target(kMcaBands, 0.0);
                for (std::size_t index = 0; index < ids.size(); ++index) {
                    const std::size_t id = static_cast<std::size_t>(ids[index]);
                    const double* values =
                        auditory_.data() +
                        (id * 2u + static_cast<std::size_t>(ear)) * kMcaBands;
                    for (int band = 0; band < kMcaBands; ++band) {
                        target[static_cast<std::size_t>(band)] += weights[index] * values[band];
                    }
                }
                // Auditory band levels of the interpolated spectrum.
                std::vector<double> power(bins_, 0.0);
                for (std::size_t bin = 0; bin < bins_; ++bin) {
                    power[bin] = std::norm((*out)[static_cast<std::size_t>(ear) * bins_ + bin]);
                }
                std::vector<double> difference(kMcaBands, 0.0);
                for (int band = 0; band < kMcaBands; ++band) {
                    double sum = 0.0;
                    for (std::size_t bin = 0; bin < bins_; ++bin) {
                        sum += power[bin] *
                               bank_[static_cast<std::size_t>(band) * bins_ + bin];
                    }
                    difference[static_cast<std::size_t>(band)] =
                        target[static_cast<std::size_t>(band)] -
                        10.0 * std::log10(std::max(sum, 1e-24));
                }
                PchipEvaluator interpolator;
                interpolator.build(centres_hz_, difference);
                for (std::size_t bin = 0; bin < bins_; ++bin) {
                    const double frequency =
                        std::clamp(frequencies_[bin], centres_hz_.front(), centres_hz_.back());
                    const double correction = std::clamp(interpolator.evaluate(frequency),
                                                         -12.0, 12.0);
                    (*out)[static_cast<std::size_t>(ear) * bins_ + bin] *=
                        std::pow(10.0, correction / 20.0);
                }
            }
        }
    }
    return Status::success();
}

}  // namespace joc::hrtf
