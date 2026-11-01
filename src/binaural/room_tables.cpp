#include "binaural/room_tables.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace joc::binaural {
namespace {

Status room_error(const char* message) {
    return Status::fail(JOC_ERR_HRTF_FORMAT, joc::stage::kRender, message);
}

bool is_prime(int value) {
    if (value < 2) {
        return false;
    }
    if (value % 2 == 0) {
        return value == 2;
    }
    for (int divisor = 3; divisor * divisor <= value; divisor += 2) {
        if (value % divisor == 0) {
            return false;
        }
    }
    return true;
}

int next_prime(int value) {
    int candidate = std::max(2, value);
    while (!is_prime(candidate)) {
        ++candidate;
    }
    return candidate;
}

// One Schroeder all-pass section; the sign convention follows the reference.
class SchroederAllpass {
public:
    void configure(int delay_samples, double gain) {
        delay_ = std::max(delay_samples, 1);
        gain_ = gain;
        buffer_.assign(static_cast<std::size_t>(delay_), 0.0);
        position_ = 0;
    }

    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        position_ = 0;
    }

    void process(const double* input, double* output, std::size_t count) {
        for (std::size_t index = 0; index < count; ++index) {
            const double value = input[index];
            const double delayed = buffer_[static_cast<std::size_t>(position_)];
            const double result = delayed - gain_ * value;
            buffer_[static_cast<std::size_t>(position_)] = value + gain_ * result;
            position_ = (position_ + 1) % delay_;
            output[index] = result;
        }
    }

private:
    int delay_ = 1;
    double gain_ = 0.0;
    std::vector<double> buffer_;
    int position_ = 0;
};

constexpr double kHadamard4[4][4] = {
    {0.5, 0.5, 0.5, 0.5},
    {0.5, -0.5, 0.5, -0.5},
    {0.5, 0.5, -0.5, -0.5},
    {0.5, -0.5, -0.5, 0.5},
};

constexpr double kInputVector[4] = {0.5, -0.5, 0.5, 0.5};
constexpr double kOutputMatrix[2][4] = {
    {0.5, 0.5, -0.5, -0.5},
    {0.5, -0.5, 0.5, -0.5},
};

struct Fdn {
    std::array<std::vector<double>, 4> buffers;
    std::array<int, 4> positions = {};
    std::array<int, 4> delays = {};
    std::array<double, 4> feedback_gain = {};
    std::array<double, 4> damping_state = {};
    SchroederAllpass diffusers[2];
    double damping = 0.0;
    double output_gain = 0.0;

    void reset() {
        for (std::vector<double>& buffer : buffers) {
            std::fill(buffer.begin(), buffer.end(), 0.0);
        }
        positions.fill(0);
        damping_state.fill(0.0);
        for (SchroederAllpass& diffuser : diffusers) {
            diffuser.reset();
        }
    }

    void process(const double* input, double* left, double* right, std::size_t count) {
        std::vector<double> diffused(input, input + count);
        std::vector<double> stage(count);
        for (SchroederAllpass& diffuser : diffusers) {
            diffuser.process(diffused.data(), stage.data(), count);
            diffused.swap(stage);
        }
        for (std::size_t sample = 0; sample < count; ++sample) {
            std::array<double, 4> delayed = {};
            for (int line = 0; line < 4; ++line) {
                delayed[static_cast<std::size_t>(line)] =
                    buffers[static_cast<std::size_t>(line)]
                           [static_cast<std::size_t>(positions[static_cast<std::size_t>(line)])];
            }
            for (int line = 0; line < 4; ++line) {
                const std::size_t index = static_cast<std::size_t>(line);
                damping_state[index] =
                    damping * damping_state[index] + (1.0 - damping) * delayed[index];
            }
            left[sample] = output_gain * (kOutputMatrix[0][0] * damping_state[0] +
                                          kOutputMatrix[0][1] * damping_state[1] +
                                          kOutputMatrix[0][2] * damping_state[2] +
                                          kOutputMatrix[0][3] * damping_state[3]);
            right[sample] = output_gain * (kOutputMatrix[1][0] * damping_state[0] +
                                           kOutputMatrix[1][1] * damping_state[1] +
                                           kOutputMatrix[1][2] * damping_state[2] +
                                           kOutputMatrix[1][3] * damping_state[3]);
            std::array<double, 4> feedback = {};
            for (int row = 0; row < 4; ++row) {
                double sum = 0.0;
                for (int column = 0; column < 4; ++column) {
                    sum += kHadamard4[row][column] *
                           (damping_state[static_cast<std::size_t>(column)] *
                            feedback_gain[static_cast<std::size_t>(column)]);
                }
                feedback[static_cast<std::size_t>(row)] = sum;
            }
            for (int line = 0; line < 4; ++line) {
                const std::size_t index = static_cast<std::size_t>(line);
                const double write =
                    kInputVector[index] * diffused[sample] + feedback[index];
                buffers[index][static_cast<std::size_t>(positions[index])] = write;
                positions[index] = (positions[index] + 1) % delays[index];
            }
        }
    }
};

double volume_over_area(const std::array<double, 3>& dimensions) {
    const double volume = dimensions[0] * dimensions[1] * dimensions[2];
    const double area = 2.0 * (dimensions[0] * dimensions[1] +
                               dimensions[0] * dimensions[2] +
                               dimensions[1] * dimensions[2]);
    return volume / area;
}

struct FdnConfig {
    double sample_rate_hz = 48000.0;
    double rt60_seconds = 0.85;
    double damping = 0.32;
    double output_gain = 0.22;
    std::array<double, 4> delay_seconds = {0.0297, 0.0371, 0.0411, 0.0437};
    std::array<double, 2> allpass_seconds = {0.0023, 0.0067};
    std::array<double, 2> allpass_gain = {0.63, 0.51};
};

}  // namespace

Status render_late_fir(double sample_rate_hz, double rt60_seconds, double damping,
                       double output_gain, const std::array<double, 4>& delay_seconds,
                       const std::array<double, 2>& allpass_seconds,
                       const std::array<double, 2>& allpass_gain, std::size_t impulse_samples,
                       std::vector<double>* left, std::vector<double>* right) {
    if (left == nullptr || right == nullptr || impulse_samples == 0u) {
        return room_error("late FDN output is invalid");
    }
    if (!(sample_rate_hz > 0.0) || !(rt60_seconds > 0.0) || !std::isfinite(damping) ||
        damping < 0.0 || damping >= 1.0 || !std::isfinite(output_gain)) {
        return room_error("invalid FDN sample rate, RT60, damping or output gain");
    }
    Fdn fdn;
    fdn.damping = damping;
    fdn.output_gain = output_gain;
    for (int line = 0; line < 4; ++line) {
        const double seconds = delay_seconds[static_cast<std::size_t>(line)];
        if (!(seconds > 0.0)) {
            return room_error("FDN requires four positive delay times");
        }
        const int delay = next_prime(
            static_cast<int>(std::lround(seconds * sample_rate_hz)));
        fdn.delays[static_cast<std::size_t>(line)] = delay;
        fdn.feedback_gain[static_cast<std::size_t>(line)] =
            std::pow(10.0, -3.0 * static_cast<double>(delay) / (rt60_seconds * sample_rate_hz));
        fdn.buffers[static_cast<std::size_t>(line)].assign(static_cast<std::size_t>(delay), 0.0);
    }
    for (int stage = 0; stage < 2; ++stage) {
        const double seconds = allpass_seconds[static_cast<std::size_t>(stage)];
        const double gain = allpass_gain[static_cast<std::size_t>(stage)];
        if (!(seconds > 0.0) || !(std::fabs(gain) < 1.0)) {
            return room_error("FDN requires two positive all-pass delays and gains");
        }
        fdn.diffusers[stage].configure(
            next_prime(static_cast<int>(std::lround(seconds * sample_rate_hz))), gain);
    }
    // The probe is the part beyond the FIR that the reference discards.
    const std::size_t probe =
        static_cast<std::size_t>(std::ceil(rt60_seconds * sample_rate_hz));
    std::vector<double> impulse(impulse_samples + probe, 0.0);
    impulse[0] = 1.0;
    left->assign(impulse.size(), 0.0);
    right->assign(impulse.size(), 0.0);
    fdn.process(impulse.data(), left->data(), right->data(), impulse.size());
    left->resize(impulse_samples);
    right->resize(impulse_samples);
    return Status::success();
}

Status build_room_preset(const std::string& preset, RoomPreset* out) {
    if (out == nullptr) {
        return room_error("room preset output is null");
    }
    FdnConfig config;
    ShoeboxRoomConfig early;
    double scale = 1.0;
    if (preset == "compact") {
        const std::array<double, 3> dimensions = {8.5, 11.3, 5.2};
        early.dimensions_m = dimensions;
        early.listener_position_m = {dimensions[0] * 0.5, dimensions[1] * 0.5,
                                     dimensions[2] * 0.5};
        scale = volume_over_area(dimensions) /
                volume_over_area(std::array<double, 3>{18.0, 18.0, 14.0});
        config.rt60_seconds *= scale;
        for (double& value : config.delay_seconds) {
            value *= scale;
        }
        for (double& value : config.allpass_seconds) {
            value *= scale;
        }
    } else if (preset != "legacy") {
        return room_error("room preset must be compact or legacy");
    }
    const Status validated = early.validate();
    if (!validated.ok()) {
        return validated;
    }
    const std::size_t taps = static_cast<std::size_t>(
        std::ceil(2.0 * config.rt60_seconds * config.sample_rate_hz));
    std::vector<double> left;
    std::vector<double> right;
    const Status rendered = render_late_fir(
        config.sample_rate_hz, config.rt60_seconds, config.damping, config.output_gain,
        config.delay_seconds, config.allpass_seconds, config.allpass_gain, taps, &left, &right);
    if (!rendered.ok()) {
        return rendered;
    }
    out->early = early;
    out->late_fir[0] = std::move(left);
    out->late_fir[1] = std::move(right);
    out->rt60_seconds = config.rt60_seconds;
    out->size_time_scale = scale;
    return Status::success();
}

}  // namespace joc::binaural
