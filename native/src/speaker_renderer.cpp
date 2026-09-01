#define EJOC_BUILD_DLL
#include "eac3joc_core.h"
#include "speaker_layouts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>

namespace ejoc::speaker {

using speaker_tables::AxisGroup;
using speaker_tables::LayoutGeometry;
using speaker_tables::RegionGeometry;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kQ15Scale = 32768.0;
constexpr double kQ15Max = 32767.0 / kQ15Scale;
constexpr double kGainSnapThreshold = 1.0e-4;
constexpr std::size_t kObjects = EJOC_MAX_OBJECTS;
constexpr std::size_t kChannels = EJOC_OUTPUT_CHANNELS;
constexpr std::size_t kBlock = EJOC_SPEAKER_BLOCK_SAMPLES;

using PointGains = std::array<double, speaker_tables::kMaxPoints>;
using ChannelGains = std::array<double, kChannels>;
using ObjectChannelGains = std::array<ChannelGains, kObjects>;
using RemainingCounts = std::array<std::array<std::uint32_t, kChannels>, kObjects>;

inline double clamp(const double value, const double low, const double high) noexcept {
    return value < low ? low : (value > high ? high : value);
}

inline double coordinate(const RegionGeometry& region, const std::size_t point,
                         const std::size_t component) noexcept {
    return static_cast<double>(region.points[point].coordinate_q15[component]) / kQ15Scale;
}

std::uint64_t expand_speaker_bitfield(const std::uint32_t compact) noexcept {
    constexpr std::array<std::uint64_t, 22> expansions{{
        0x00000003ULL, 0x00000004ULL, 0x00000008ULL, 0x00000030ULL,
        0x000000C0ULL, 0x00000100ULL, 0x00000600ULL, 0x00001800ULL,
        0x00006000ULL, 0x00018000ULL, 0x00060000ULL, 0x00180000ULL,
        0x00600000ULL, 0x01800000ULL, 0x06000000ULL, 0x18000000ULL,
        0x60000000ULL, 0x080000000ULL, 0x600000000ULL, 0x800000000ULL,
        0x1000000000ULL, 0x2000000000ULL,
    }};
    std::uint64_t expanded = 0;
    for (std::size_t bit_index = 0; bit_index < expansions.size(); ++bit_index) {
        if ((compact & (1u << bit_index)) != 0) {
            expanded |= expansions[bit_index];
        }
    }
    return expanded;
}

inline int bit(const std::uint64_t value, const unsigned index) noexcept {
    return static_cast<int>((value >> index) & 1ULL);
}

double layout_attenuation_db(const std::uint32_t compact) noexcept {
    const std::uint64_t expanded = expand_speaker_bitfield(compact);
    const int height_channels = 2 * (
        bit(expanded, 13) + bit(expanded, 15) + bit(expanded, 17) +
        bit(expanded, 19) + bit(expanded, 21));
    const int floor_channels = bit(expanded, 8) + 2 * (
        bit(expanded, 31) + bit(expanded, 4) + bit(expanded, 6) +
        bit(expanded, 11) + bit(expanded, 25) + bit(expanded, 27) +
        bit(expanded, 29) + bit(expanded, 33));
    const double height_factor = std::min(static_cast<double>(height_channels) / 4.0, 1.0);
    const double floor_factor = std::min(static_cast<double>(floor_channels) / 4.0, 1.0);
    return -std::max(4.5 - 1.5 * height_factor - 3.0 * floor_factor, 0.0);
}

int floor_y_exponent(const std::uint32_t compact) noexcept {
    const std::uint32_t low = static_cast<std::uint32_t>(expand_speaker_bitfield(compact));
    return ((low & 0x130u) != 0 && (low & 0x18C0u) == 0) ? 1 : 0;
}

inline void equal_power_pair(const double position, double& lower, double& upper) noexcept {
    const double angle = (kPi * 0.5) * position;
    lower = std::cos(angle);
    upper = std::sin(angle);
}

void axis0_gains(const RegionGeometry& region,
                 const std::array<AxisGroup, speaker_tables::kMaxGroups>& groups,
                 const std::uint8_t group_count,
                 const double value,
                 PointGains& output) noexcept {
    output.fill(0.0);
    for (std::size_t row = 0; row < group_count; ++row) {
        const auto& group = groups[row];
        if (group.size == 0) {
            continue;
        }
        const std::size_t first = group.indices[0];
        const std::size_t last = group.indices[group.size - 1];
        const double first_value = coordinate(region, first, 0);
        const double last_value = coordinate(region, last, 0);
        if (value <= first_value) {
            output[first] = 1.0;
            continue;
        }
        if (value >= last_value) {
            output[last] = 1.0;
            continue;
        }
        for (std::size_t index = 0; index + 1 < group.size; ++index) {
            const std::size_t lower_index = group.indices[index];
            const std::size_t upper_index = group.indices[index + 1];
            const double lower_value = coordinate(region, lower_index, 0);
            const double upper_value = coordinate(region, upper_index, 0);
            if (value > lower_value && value <= upper_value) {
                const double position = (value - lower_value) / (upper_value - lower_value);
                equal_power_pair(position, output[lower_index], output[upper_index]);
                break;
            }
        }
    }
}

void axis1_gains(const RegionGeometry& region,
                 const std::array<AxisGroup, speaker_tables::kMaxGroups>& groups,
                 const std::uint8_t group_count,
                 const double value,
                 PointGains& output) noexcept {
    output.fill(0.0);
    if (group_count == 0) {
        return;
    }
    const auto& first_group = groups[0];
    const auto& last_group = groups[group_count - 1];
    const double first_value = coordinate(region, first_group.indices[0], 1);
    const double last_value = coordinate(region, last_group.indices[0], 1);
    if (value <= first_value) {
        for (std::size_t index = 0; index < first_group.size; ++index) {
            output[first_group.indices[index]] = 1.0;
        }
        return;
    }
    if (value > last_value) {
        for (std::size_t index = 0; index < last_group.size; ++index) {
            output[last_group.indices[index]] = 1.0;
        }
        return;
    }
    for (std::size_t row = 0; row + 1 < group_count; ++row) {
        const auto& lower_group = groups[row];
        const auto& upper_group = groups[row + 1];
        const double lower_value = coordinate(region, lower_group.indices[0], 1);
        const double upper_value = coordinate(region, upper_group.indices[0], 1);
        if (value >= lower_value && value <= upper_value) {
            const double position = (value - lower_value) / (upper_value - lower_value);
            double lower_gain = 0.0;
            double upper_gain = 0.0;
            equal_power_pair(position, lower_gain, upper_gain);
            for (std::size_t index = 0; index < lower_group.size; ++index) {
                output[lower_group.indices[index]] = lower_gain;
            }
            for (std::size_t index = 0; index < upper_group.size; ++index) {
                output[upper_group.indices[index]] = upper_gain;
            }
            return;
        }
    }
}

void plane_gains(const RegionGeometry& region,
                 const std::array<AxisGroup, speaker_tables::kMaxGroups>& groups,
                 const std::uint8_t group_count,
                 const double u,
                 const double v,
                 const std::uint8_t mode,
                 PointGains& output) noexcept {
    axis0_gains(region, groups, group_count, u, output);
    if (mode >= 2) {
        PointGains vertical{};
        axis1_gains(region, groups, group_count, v, vertical);
        for (std::size_t point = 0; point < region.point_count; ++point) {
            output[point] *= vertical[point];
        }
    }
}

class Renderer {
public:
    explicit Renderer(const LayoutGeometry* layout) noexcept : layout_(layout) {
        for (std::size_t standard = 0; standard < layout_->channel_count; ++standard) {
            standard_index_for_internal_[layout_->standard_from_internal[standard]] =
                static_cast<std::uint8_t>(standard);
        }
        attenuation_db_ = layout_attenuation_db(layout_->speaker_bitfield);
        floor_y_exponent_ = floor_y_exponent(layout_->speaker_bitfield);
        reset_state();
    }

    int reset() noexcept {
        reset_state();
        error_[0] = '\0';
        return 0;
    }

    const char* last_error() const noexcept {
        return error_[0] ? error_.data() : "";
    }

    int process(const float* input,
                const std::uint32_t sample_count,
                const std::uint32_t metadata_count,
                const std::uint32_t* metadata_offsets,
                const std::uint32_t* ramp_durations,
                const std::uint16_t* positions_q15,
                const std::uint8_t* region_indices,
                const std::uint8_t* height_enabled,
                const double* object_gains,
                double* output) noexcept {
        error_[0] = '\0';
        if (!input || !output) {
            return fail("null PCM pointer passed to ejoc_speaker_renderer_process");
        }
        if ((sample_count % kBlock) != 0) {
            return fail("sample_count must be a multiple of 32");
        }
        if (metadata_count && (!metadata_offsets || !ramp_durations || !positions_q15)) {
            return fail("metadata arrays are null while metadata_count is nonzero");
        }
        if (!validate_metadata(sample_count, metadata_count, metadata_offsets,
                               positions_q15, region_indices, object_gains)) {
            return -1;
        }

        std::fill(output, output + static_cast<std::size_t>(sample_count) * layout_->channel_count, 0.0);
        const bool has_lfe = (layout_->speaker_bitfield & 0x4u) != 0;
        const std::size_t total_blocks = sample_count / kBlock;
        std::size_t event = 0;
        for (std::size_t block = 0; block < total_blocks; ++block) {
            while (event < metadata_count && aligned_block(metadata_offsets[event]) == block) {
                apply_event(event, ramp_durations, positions_q15, region_indices,
                            height_enabled, object_gains);
                ++event;
            }
            mix_block(input, output, block, has_lfe);
        }
        while (event < metadata_count && aligned_block(metadata_offsets[event]) == total_blocks) {
            apply_event(event, ramp_durations, positions_q15, region_indices,
                        height_enabled, object_gains);
            ++event;
        }
        if (event != metadata_count) {
            return fail("metadata alignment produced an event outside this process call");
        }
        return 0;
    }

private:
    void reset_state() noexcept {
        for (auto& row : current_) row.fill(0.0);
        for (auto& row : target_) row.fill(0.0);
        for (auto& row : step_) row.fill(0.0);
        for (auto& row : remaining_) row.fill(0);
    }

    int fail(const char* message) noexcept {
        std::snprintf(error_.data(), error_.size(), "%s", message);
        return -1;
    }

    bool validate_metadata(const std::uint32_t sample_count,
                           const std::uint32_t metadata_count,
                           const std::uint32_t* metadata_offsets,
                           const std::uint16_t* positions_q15,
                           const std::uint8_t* region_indices,
                           const double* object_gains) noexcept {
        for (std::size_t event = 0; event < metadata_count; ++event) {
            if (metadata_offsets[event] > sample_count) {
                fail("metadata offset exceeds sample_count");
                return false;
            }
            if (event && metadata_offsets[event] < metadata_offsets[event - 1]) {
                fail("metadata offsets must be nondecreasing");
                return false;
            }
            for (std::size_t object = 0; object < kObjects; ++object) {
                const std::size_t object_event = event * kObjects + object;
                if (region_indices && region_indices[object_event] >= 7) {
                    fail("region index is above 6");
                    return false;
                }
                if (object_gains && !std::isfinite(object_gains[object_event])) {
                    fail("object gain is not finite");
                    return false;
                }
                const std::size_t coordinate_base = object_event * EJOC_SPEAKER_COORDINATES;
                for (std::size_t component = 0; component < EJOC_SPEAKER_COORDINATES; ++component) {
                    if (positions_q15[coordinate_base + component] > 32767u) {
                        fail("Q15 object coordinate is above 32767");
                        return false;
                    }
                }
            }
        }
        return true;
    }

    static std::size_t aligned_block(const std::uint32_t sample) noexcept {
        return (static_cast<std::size_t>(sample) + kBlock / 2 - 1) / kBlock;
    }

    static std::uint32_t ramp_blocks(const std::uint32_t duration) noexcept {
        return static_cast<std::uint32_t>(
            (static_cast<std::size_t>(duration) + kBlock / 2 - 1) / kBlock);
    }

    void render_point(const std::uint16_t* position,
                      const std::uint8_t region_index,
                      const bool enable_height,
                      const double object_gain,
                      ChannelGains& output) const noexcept {
        output.fill(0.0);
        const auto& region = layout_->regions[region_index];
        const double u = static_cast<double>(position[0]) / kQ15Scale;
        const double v = static_cast<double>(position[1]) / kQ15Scale;
        const double w = static_cast<double>(position[2]) / kQ15Scale;
        const double floor_v = clamp(std::ldexp(v, floor_y_exponent_), 0.0, 1.0);
        PointGains floor{};
        plane_gains(region, region.axis0_groups, region.axis0_group_count,
                    u, floor_v, region.mode, floor);
        PointGains point = floor;
        if (region.mode == 3) {
            PointGains height{};
            plane_gains(region, region.axis1_groups, region.axis1_group_count,
                        u, v, 3, height);
            const double z = enable_height ? clamp(w, 0.0, kQ15Max) : 0.0;
            if (z >= kQ15Max) {
                point = height;
            } else if (z > 0.0) {
                double floor_weight = 0.0;
                double height_weight = 0.0;
                equal_power_pair(z, floor_weight, height_weight);
                for (std::size_t index = 0; index < region.point_count; ++index) {
                    point[index] = floor[index] * floor_weight + height[index] * height_weight;
                }
            }
        }
        const double y_term = clamp(v / 0.6, 0.0, 1.0);
        const double z_term = clamp((w - 0.2) / 0.8, 0.0, 1.0);
        const double amount = clamp(y_term + z_term, 0.0, 1.0);
        const double gain = std::pow(10.0, attenuation_db_ * amount / 20.0) * object_gain;
        for (std::size_t index = 0; index < region.point_count; ++index) {
            output[region.points[index].speaker_id] = point[index] * gain;
        }
    }

    void apply_event(const std::size_t event,
                     const std::uint32_t* ramp_durations,
                     const std::uint16_t* positions_q15,
                     const std::uint8_t* region_indices,
                     const std::uint8_t* height_enabled,
                     const double* object_gains) noexcept {
        const std::uint32_t blocks = ramp_blocks(ramp_durations[event]);
        for (std::size_t object = 0; object < kObjects; ++object) {
            const std::size_t object_event = event * kObjects + object;
            const auto* position = positions_q15 + object_event * EJOC_SPEAKER_COORDINATES;
            const std::uint8_t region = region_indices ? region_indices[object_event] : 0;
            const bool height = !height_enabled || height_enabled[object_event] != 0;
            const double object_gain = object_gains ? object_gains[object_event] : 1.0;
            ChannelGains next{};
            render_point(position, region, height, object_gain, next);
            for (std::size_t channel = 0; channel < layout_->channel_count; ++channel) {
                const double difference = next[channel] - current_[object][channel];
                target_[object][channel] = next[channel];
                if (std::abs(difference) >= kGainSnapThreshold && blocks != 0) {
                    step_[object][channel] = difference / static_cast<double>(blocks);
                    remaining_[object][channel] = blocks;
                } else {
                    current_[object][channel] = next[channel];
                    step_[object][channel] = 0.0;
                    remaining_[object][channel] = 0;
                }
            }
        }
    }

    void mix_block(const float* input, double* output, const std::size_t block,
                   const bool has_lfe) noexcept {
        const std::size_t start = block * kBlock;
        if (has_lfe) {
            for (std::size_t sample = 0; sample < kBlock; ++sample) {
                output[(start + sample) * layout_->channel_count + 3] =
                    static_cast<double>(input[(start + sample) * EJOC_OUTPUT_CHANNELS]);
            }
        }
        for (std::size_t object = 0; object < kObjects; ++object) {
            for (std::size_t internal = 0; internal < layout_->channel_count; ++internal) {
                const bool active = remaining_[object][internal] != 0;
                const double fixed_gain = target_[object][internal];
                if (!active && fixed_gain == 0.0) {
                    continue;
                }
                const std::size_t standard = standard_index_for_internal_[internal];
                for (std::size_t sample = 0; sample < kBlock; ++sample) {
                    const double gain = active
                        ? current_[object][internal] +
                          (static_cast<double>(sample) / static_cast<double>(kBlock)) *
                          step_[object][internal]
                        : fixed_gain;
                    output[(start + sample) * layout_->channel_count + standard] +=
                        static_cast<double>(
                            input[(start + sample) * EJOC_OUTPUT_CHANNELS + object + 1]) * gain;
                }
                if (active) {
                    current_[object][internal] += step_[object][internal];
                    --remaining_[object][internal];
                    if (remaining_[object][internal] == 0) {
                        current_[object][internal] = target_[object][internal];
                    }
                } else {
                    current_[object][internal] = target_[object][internal];
                }
            }
        }
    }

    const LayoutGeometry* layout_;
    double attenuation_db_{};
    int floor_y_exponent_{};
    ObjectChannelGains current_{};
    ObjectChannelGains target_{};
    ObjectChannelGains step_{};
    RemainingCounts remaining_{};
    std::array<std::uint8_t, kChannels> standard_index_for_internal_{};
    std::array<char, 256> error_{};
};

}  // namespace ejoc::speaker

extern "C" {

uint32_t EJOC_CALL ejoc_speaker_layout_channel_count(const uint32_t speaker_bitfield) {
    const auto* layout = ejoc::speaker_tables::find_layout(speaker_bitfield);
    return layout ? layout->channel_count : 0;
}

ejoc_speaker_renderer_handle EJOC_CALL ejoc_speaker_renderer_create(
    const uint32_t speaker_bitfield) {
    const auto* layout = ejoc::speaker_tables::find_layout(speaker_bitfield);
    if (!layout) {
        return nullptr;
    }
    return new (std::nothrow) ejoc::speaker::Renderer(layout);
}

void EJOC_CALL ejoc_speaker_renderer_destroy(ejoc_speaker_renderer_handle handle) {
    delete static_cast<ejoc::speaker::Renderer*>(handle);
}

int EJOC_CALL ejoc_speaker_renderer_reset(ejoc_speaker_renderer_handle handle) {
    if (!handle) {
        return -1;
    }
    return static_cast<ejoc::speaker::Renderer*>(handle)->reset();
}

const char* EJOC_CALL ejoc_speaker_renderer_last_error(ejoc_speaker_renderer_handle handle) {
    if (!handle) {
        return "speaker renderer handle is null";
    }
    return static_cast<ejoc::speaker::Renderer*>(handle)->last_error();
}

int EJOC_CALL ejoc_speaker_renderer_process(
    ejoc_speaker_renderer_handle handle,
    const float* objects16_interleaved,
    const uint32_t sample_count,
    const uint32_t metadata_count,
    const uint32_t* metadata_offsets,
    const uint32_t* ramp_durations,
    const uint16_t* positions_q15,
    const uint8_t* region_indices,
    const uint8_t* height_enabled,
    const double* object_gains,
    double* output_interleaved) {
    if (!handle) {
        return -1;
    }
    return static_cast<ejoc::speaker::Renderer*>(handle)->process(
        objects16_interleaved, sample_count, metadata_count,
        metadata_offsets, ramp_durations, positions_q15,
        region_indices, height_enabled, object_gains, output_interleaved);
}

}  // extern "C"
