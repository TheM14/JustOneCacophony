// The shared library compiles this unit with JOC_BUILD_DLL and exports the C
// entry points; the static joc_core_impl build compiles it with EJOC_STATIC and
// keeps them local, so the two libraries can be linked into the same tool.
#if defined(JOC_BUILD_DLL) && !defined(EJOC_STATIC)
#define EJOC_BUILD_DLL
#endif
#include "eac3joc_core.h"
#include "joc_bitstream/bed_layout_map.h"
#include "oamd/oamd_math.h"
#include "speaker/speaker_meta.h"
#include "speaker/speaker_trim.h"
#include "speaker_layouts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace ejoc::speaker {

using speaker_tables::AxisGroup;
using speaker_tables::LayoutGeometry;
using speaker_tables::RegionGeometry;
using speaker_tables::SpeakerPoint;

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
        target_floor_y_exponent_ = floor_y_exponent(layout_->speaker_bitfield);
        configure_surround_fold();
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
                const ejoc_speaker_controls* controls,
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
        if (!validate_controls(controls)) {
            return -1;
        }
        if (!validate_metadata(sample_count, metadata_count, metadata_offsets,
                               positions_q15, region_indices, object_gains)) {
            return -1;
        }

        // A caller that passes no controls gets warp 0, no trim and no balance.
        ejoc_speaker_controls fallback{};
        fallback.struct_size = sizeof(fallback);
        fallback.struct_version = EJOC_SPEAKER_CONTROLS_VERSION;
        fallback.trim_mode = EJOC_SPEAKER_TRIM_DISABLED;
        const ejoc_speaker_controls& active = controls != nullptr ? *controls : fallback;

        std::fill(output, output + static_cast<std::size_t>(sample_count) * layout_->channel_count, 0.0);
        const bool has_lfe = (layout_->speaker_bitfield & 0x4u) != 0;
        const std::size_t total_blocks = sample_count / kBlock;
        std::size_t event = 0;
        for (std::size_t block = 0; block < total_blocks; ++block) {
            while (event < metadata_count && aligned_block(metadata_offsets[event]) == block) {
                apply_event(event, ramp_durations, positions_q15, region_indices,
                            height_enabled, object_gains, active);
                ++event;
            }
            mix_block(input, output, block, has_lfe);
        }
        while (event < metadata_count && aligned_block(metadata_offsets[event]) == total_blocks) {
            apply_event(event, ramp_durations, positions_q15, region_indices,
                        height_enabled, object_gains, active);
            ++event;
        }
        if (event != metadata_count) {
            return fail("metadata alignment produced an event outside this process call");
        }
        return 0;
    }

    // Renders one 32-sample legacy block into an already zeroed interleaved
    // output, using the same point-renderer state as process().  The metadata
    // path consumes this for the blocks that precede its first targets.
    void render_legacy_block(const float* input, const std::size_t block,
                             double* output) noexcept {
        mix_block(input, output, block, (layout_->speaker_bitfield & 0x4u) != 0);
    }

private:
    // Warp 2/3 render the 5.1-family and 4.0 layouts through an expanded
    // surround layout, then fold the rear pair into the target surround pair.
    void configure_surround_fold() noexcept {
        fold_layout_ = layout_;
        fold_source_channels_ = layout_->channel_count;
        fold_source_floor_y_exponent_ = target_floor_y_exponent_;
        fold_available_ = false;
        if (target_floor_y_exponent_ == 0) {
            return;
        }
        const LayoutGeometry* source = nullptr;
        switch (layout_->speaker_bitfield) {
            case 15u: source = speaker_tables::find_layout(31u); break;
            case 1039u: source = speaker_tables::find_layout(1055u); break;
            case 2575u: source = speaker_tables::find_layout(2591u); break;
            case 9u:
                build_fold_source_40();
                source = &fold_layout_storage_;
                break;
            default: return;
        }
        if (source == nullptr || source->channel_count == 0) {
            return;
        }
        fold_layout_ = source;
        fold_source_channels_ = source->channel_count;
        fold_source_floor_y_exponent_ = floor_y_exponent(source->speaker_bitfield);
        build_fold_matrix(source->speaker_bitfield, layout_->speaker_bitfield);
        fold_available_ = true;
    }

    // 4.0 expands to six full-band speakers without adding centre or LFE.
    void build_fold_source_40() noexcept {
        const LayoutGeometry* base = speaker_tables::find_layout(31u);
        if (base == nullptr) {
            return;
        }
        LayoutGeometry& out = fold_layout_storage_;
        out = *base;
        out.speaker_bitfield = 25u;  // 4.0's pair plus the rear pair
        out.out_ch_config = 255u;
        out.channel_count = 6;
        for (std::size_t slot = 0; slot < out.standard_from_internal.size(); ++slot) {
            out.standard_from_internal[slot] = static_cast<std::uint8_t>(slot);
        }
        constexpr int kMapped[8] = {0, 1, -1, -1, 2, 3, 4, 5};
        for (RegionGeometry& region : out.regions) {
            int index_map[speaker_tables::kMaxPoints];
            for (int& value : index_map) {
                value = -1;
            }
            std::size_t kept = 0;
            for (std::size_t point = 0; point < region.point_count; ++point) {
                const int speaker = region.points[point].speaker_id;
                const int mapped = speaker >= 0 && speaker < 8 ? kMapped[speaker] : -1;
                if (mapped < 0) {
                    continue;
                }
                region.points[kept] = region.points[point];
                region.points[kept].speaker_id = static_cast<std::uint8_t>(mapped);
                index_map[point] = static_cast<int>(kept);
                ++kept;
            }
            for (std::size_t point = kept; point < region.point_count; ++point) {
                region.points[point] = SpeakerPoint{};
            }
            region.point_count = static_cast<std::uint8_t>(kept);
            filter_groups(region.axis0_groups, region.axis0_group_count, index_map);
            filter_groups(region.axis1_groups, region.axis1_group_count, index_map);
        }
    }

    static void filter_groups(std::array<AxisGroup, speaker_tables::kMaxGroups>& groups,
                              std::uint8_t& count,
                              const int (&index_map)[speaker_tables::kMaxPoints]) noexcept {
        std::array<AxisGroup, speaker_tables::kMaxGroups> filtered{};
        std::uint8_t kept_groups = 0;
        for (std::size_t row = 0; row < count; ++row) {
            AxisGroup group{};
            for (std::size_t index = 0; index < groups[row].size; ++index) {
                const int mapped = index_map[groups[row].indices[index]];
                if (mapped < 0) {
                    continue;
                }
                group.indices[group.size++] = static_cast<std::uint8_t>(mapped);
            }
            if (group.size != 0) {
                filtered[kept_groups++] = group;
            }
        }
        groups = filtered;
        count = kept_groups;
    }

    // Shared speakers pass through; speaker IDs 6/7 are the rear pair and fold
    // into the same-side (and, for PLIIx, the cross-side) surround.
    void build_fold_matrix(std::uint32_t source_bitfield,
                           std::uint32_t target_bitfield) noexcept {
        int source_ids[64];
        int target_ids[64];
        std::size_t source_count = 0;
        std::size_t target_count = 0;
        const std::uint64_t source_mask = expand_speaker_bitfield(source_bitfield);
        const std::uint64_t target_mask = expand_speaker_bitfield(target_bitfield);
        for (int id = 0; id < 64; ++id) {
            if (((source_mask >> id) & 1ULL) != 0u) {
                source_ids[source_count++] = id;
            }
            if (((target_mask >> id) & 1ULL) != 0u) {
                target_ids[target_count++] = id;
            }
        }
        for (auto& row : fold_pl2x_) {
            row.fill(0.0);
        }
        for (auto& row : fold_loro_) {
            row.fill(0.0);
        }
        constexpr double kSqrt3Over2 = 0.86602540378443864676;
        for (std::size_t column = 0; column < source_count; ++column) {
            const int id = source_ids[column];
            const std::size_t direct = target_position(target_ids, target_count, id);
            if (direct < kChannels) {
                fold_pl2x_[direct][column] = 1.0;
                fold_loro_[direct][column] = 1.0;
                continue;
            }
            if (id != 6 && id != 7) {
                continue;
            }
            const std::size_t same = target_position(target_ids, target_count, id - 2);
            const std::size_t cross = target_position(target_ids, target_count, id == 6 ? 5 : 4);
            if (same < kChannels) {
                fold_pl2x_[same][column] = kSqrt3Over2;
                fold_loro_[same][column] = 1.0;
            }
            if (cross < kChannels) {
                fold_pl2x_[cross][column] = 0.5;
                fold_loro_[cross][column] = 0.0;
            }
        }
    }

    static std::size_t target_position(const int (&ids)[64], std::size_t count,
                                       int id) noexcept {
        for (std::size_t index = 0; index < count; ++index) {
            if (ids[index] == id) {
                return index;
            }
        }
        return kChannels;
    }

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

    bool validate_controls(const ejoc_speaker_controls* controls) noexcept {
        if (controls == nullptr) {
            return true;
        }
        if (controls->struct_size != sizeof(ejoc_speaker_controls)) {
            fail("speaker controls struct_size mismatch");
            return false;
        }
        if (controls->struct_version != EJOC_SPEAKER_CONTROLS_VERSION) {
            fail("unknown speaker controls struct_version");
            return false;
        }
        if (controls->warp_mode > 3u) {
            fail("unknown speaker warp mode");
            return false;
        }
        if (controls->trim_mode > EJOC_SPEAKER_TRIM_EXPLICIT) {
            fail("unknown speaker trim mode");
            return false;
        }
        for (const double level :
             {controls->centre_db, controls->surround_db, controls->height_db}) {
            if (!std::isfinite(level)) {
                fail("trim level is not finite");
                return false;
            }
        }
        for (const double balance : {controls->listener_balance, controls->overhead_balance}) {
            if (!std::isfinite(balance) || balance < -1.0 || balance > 1.0) {
                fail("balance is not finite or outside [-1, 1]");
                return false;
            }
        }
        return true;
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
                      const ejoc_speaker_controls& controls,
                      const bool bypass_trim,
                      ChannelGains& output) const noexcept {
        output.fill(0.0);
        const bool folding = fold_available_ && controls.warp_mode >= 2u;
        const auto& region = (folding ? *fold_layout_ : *layout_).regions[region_index];
        const double u = static_cast<double>(position[0]) / kQ15Scale;
        const double v = static_cast<double>(position[1]) / kQ15Scale;
        const double w = std::max(static_cast<double>(position[2]) / kQ15Scale, 0.0);
        const int shift = controls.warp_mode == 0u
                              ? 0
                              : (folding ? fold_source_floor_y_exponent_
                                         : target_floor_y_exponent_);
        double floor_v = clamp(std::ldexp(v, shift), 0.0, 1.0);
        double top_v = clamp(v, 0.0, 1.0);
        double trim_v = v;
        if (controls.trim_mode == EJOC_SPEAKER_TRIM_EXPLICIT && !bypass_trim) {
            // Warp feeds the floor coordinate only: the height and trim
            // coordinates see the balance map on the original Y.
            floor_v = joc::speaker::apply_balance(floor_v, w, controls.listener_balance,
                                                  controls.overhead_balance);
            top_v = joc::speaker::apply_balance(top_v, w, controls.listener_balance,
                                                controls.overhead_balance);
            trim_v = joc::speaker::apply_balance(v, w, controls.listener_balance,
                                                 controls.overhead_balance);
        }
        PointGains floor{};
        plane_gains(region, region.axis0_groups, region.axis0_group_count,
                    u, floor_v, region.mode, floor);
        PointGains point = floor;
        if (region.mode == 3) {
            PointGains height{};
            plane_gains(region, region.axis1_groups, region.axis1_group_count,
                        u, top_v, 3, height);
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
        double gain = object_gain;
        if (!bypass_trim) {
            if (controls.trim_mode == EJOC_SPEAKER_TRIM_AUTOMATIC) {
                gain *= position_gain(trim_v, w);
            } else if (controls.trim_mode == EJOC_SPEAKER_TRIM_EXPLICIT) {
                gain *= joc::speaker::explicit_position_gain(controls, u, trim_v, w);
            }
        }
        ChannelGains internal{};
        for (std::size_t index = 0; index < region.point_count; ++index) {
            internal[region.points[index].speaker_id] = point[index] * gain;
        }
        if (!folding) {
            output = internal;
            return;
        }
        const auto& matrix = controls.warp_mode == 2u ? fold_pl2x_ : fold_loro_;
        for (std::size_t target = 0; target < layout_->channel_count; ++target) {
            double sum = 0.0;
            for (std::size_t source = 0; source < fold_source_channels_; ++source) {
                sum += matrix[target][source] * internal[source];
            }
            output[target] = sum;
        }
    }

    double position_gain(const double v, const double w) const noexcept {
        const double y_term = clamp(v / 0.6, 0.0, 1.0);
        const double z_term = clamp((w - 0.2) / 0.8, 0.0, 1.0);
        const double amount = clamp(y_term + z_term, 0.0, 1.0);
        return std::pow(10.0, attenuation_db_ * amount / 20.0);
    }

    void apply_event(const std::size_t event,
                     const std::uint32_t* ramp_durations,
                     const std::uint16_t* positions_q15,
                     const std::uint8_t* region_indices,
                     const std::uint8_t* height_enabled,
                     const double* object_gains,
                     const ejoc_speaker_controls& controls) noexcept {
        const std::uint32_t blocks = ramp_blocks(ramp_durations[event]);
        for (std::size_t object = 0; object < kObjects; ++object) {
            const std::size_t object_event = event * kObjects + object;
            const auto* position = positions_q15 + object_event * EJOC_SPEAKER_COORDINATES;
            const std::uint8_t region = region_indices ? region_indices[object_event] : 0;
            const bool height = !height_enabled || height_enabled[object_event] != 0;
            const double object_gain = object_gains ? object_gains[object_event] : 1.0;
            const bool bypass = controls.trim_bypass[object + 1] != 0u;
            ChannelGains next{};
            render_point(position, region, height, object_gain, controls, bypass, next);
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
    const LayoutGeometry* fold_layout_ = nullptr;
    LayoutGeometry fold_layout_storage_{};
    bool fold_available_ = false;
    std::uint8_t fold_source_channels_ = 0;
    int fold_source_floor_y_exponent_ = 0;
    std::array<ChannelGains, kChannels> fold_pl2x_{};
    std::array<ChannelGains, kChannels> fold_loro_{};
    double attenuation_db_{};
    int target_floor_y_exponent_{};
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
    const ejoc_speaker_controls* controls,
    double* output_interleaved) {
    if (!handle) {
        return -1;
    }
    return static_cast<ejoc::speaker::Renderer*>(handle)->process(
        objects16_interleaved, sample_count, metadata_count,
        metadata_offsets, ramp_durations, positions_q15,
        region_indices, height_enabled, object_gains, controls, output_interleaved);
}

}  // extern "C"

// Metadata application path (src/speaker_renderer.py HEAD).  The legacy point
// renderer above stays untouched: this section owns the block-state path.
#include "speaker/speaker_meta.h"

#include "joc_bitstream/bed_layout_map.h"
#include "oamd/oamd_math.h"

namespace joc::speaker {

namespace {

constexpr int kBlockSamples = EJOC_SPEAKER_BLOCK_SAMPLES;
constexpr int kOutputChannels = EJOC_OUTPUT_CHANNELS;
constexpr double kQ15 = 32768.0;
constexpr double kQ15MaxValue = 32767.0 / 32768.0;
constexpr double kGainSnapThreshold = 1.0e-4;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kZoneToRegion[8] = {0, 1, 2, 3, 4, 5, 6, 0};
constexpr int kLfeBedLabel = 4;    // Python BED_LABEL_INDEX["RC_LFE"]
constexpr int kLfe2BedLabel = 17;  // Python BED_LABEL_INDEX["RC_LFE2"]

inline double clamp(const double value, const double low, const double high) {
    return value < low ? low : (value > high ? high : value);
}

// The reference's _stable_norm: tiny-norm floor and unit clamp in float64.
std::vector<double> stable_norm(const std::vector<double>& vector) {
    double total = 0.0;
    for (const double value : vector) {
        total += value * value;
    }
    std::vector<double> result(vector.size(), 0.0);
    if (total == 0.0) {
        return result;
    }
    const double denominator = std::max(std::sqrt(total), 1.0e-5);
    for (std::size_t index = 0; index < vector.size(); ++index) {
        result[index] = std::min(vector[index] / denominator, 1.0);
    }
    return result;
}

void equal_power_pair(const double position, double& lower, double& upper) {
    const double angle = kPi * 0.5 * position;
    lower = std::cos(angle);
    upper = std::sin(angle);
}

std::uint32_t align_metadata_sample(const std::uint32_t sample) {
    return static_cast<std::uint32_t>(
        kBlockSamples * ((static_cast<std::uint64_t>(sample) + kBlockSamples / 2 - 1) /
                         kBlockSamples));
}

std::uint32_t ramp_block_count(const std::uint32_t duration_samples) {
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(duration_samples) + kBlockSamples / 2 - 1) /
        kBlockSamples);
}

// One 16x16 channel matrix, rows indexed by the target layout's channel id.
struct ChannelMatrix {
    double value[EJOC_OUTPUT_CHANNELS][EJOC_OUTPUT_CHANNELS] = {};
};

void make_identity(ChannelMatrix& matrix) {
    for (int row = 0; row < kOutputChannels; ++row) {
        for (int column = 0; column < kOutputChannels; ++column) {
            matrix.value[row][column] = row == column ? 1.0 : 0.0;
        }
    }
}

// Copies shared speakers and folds the rear pair into the target surround pair,
// exactly like the reference's _surround_fold_matrix.
void make_fold_matrix(const std::uint32_t source_bitfield,
                      const std::uint32_t target_bitfield, const int warp_mode,
                      ChannelMatrix& matrix) {
    make_identity(matrix);
    const std::uint64_t source_mask = ejoc::speaker::expand_speaker_bitfield(source_bitfield);
    const std::uint64_t target_mask = ejoc::speaker::expand_speaker_bitfield(target_bitfield);
    int source_ids[64];
    int target_ids[64];
    std::size_t source_count = 0;
    std::size_t target_count = 0;
    for (int id = 0; id < 64; ++id) {
        if (((source_mask >> id) & 1ULL) != 0u) {
            source_ids[source_count++] = id;
        }
        if (((target_mask >> id) & 1ULL) != 0u) {
            target_ids[target_count++] = id;
        }
    }
    for (auto& row : matrix.value) {
        for (double& value : row) {
            value = 0.0;
        }
    }
    const auto position = [&](const int id) {
        for (std::size_t index = 0; index < target_count; ++index) {
            if (target_ids[index] == id) {
                return static_cast<int>(index);
            }
        }
        return -1;
    };
    const double same_side = warp_mode == 2 ? std::sqrt(3.0) / 2.0 : 1.0;
    const double cross_side = warp_mode == 2 ? 0.5 : 0.0;
    for (std::size_t column = 0; column < source_count; ++column) {
        const int id = source_ids[column];
        const int direct = position(id);
        if (direct >= 0) {
            matrix.value[direct][column] = 1.0;
            continue;
        }
        if (id != 6 && id != 7) {
            continue;
        }
        const int same = position(id - 2);
        const int cross = position(id == 6 ? 5 : 4);
        if (same >= 0) {
            matrix.value[same][column] = same_side;
        }
        if (cross >= 0) {
            matrix.value[cross][column] = cross_side;
        }
    }
}

// Uncompensated point-source vector in internal channel order (the reference's
// _region_vector).
void region_vector(const ejoc::speaker_tables::LayoutGeometry& layout,
                   const int region_index, const double u, const double floor_v,
                   const double top_v, const double w, const bool enable_height,
                   std::vector<double>& vector) {
    using PointGains = std::array<double, ejoc::speaker_tables::kMaxPoints>;
    vector.assign(layout.channel_count, 0.0);
    const auto& region = layout.regions[static_cast<std::size_t>(region_index)];
    PointGains floor{};
    ejoc::speaker::plane_gains(region, region.axis0_groups, region.axis0_group_count, u,
                               floor_v, region.mode, floor);
    PointGains gains = floor;
    if (region.mode == 3) {
        PointGains top{};
        ejoc::speaker::plane_gains(region, region.axis1_groups, region.axis1_group_count, u,
                                   top_v, 3, top);
        const double height =
            enable_height ? clamp(w, 0.0, kQ15MaxValue) : 0.0;
        if (height >= kQ15MaxValue) {
            gains = top;
        } else if (height > 0.0) {
            double floor_weight = 0.0;
            double height_weight = 0.0;
            equal_power_pair(height, floor_weight, height_weight);
            for (std::size_t index = 0; index < region.point_count; ++index) {
                gains[index] = floor[index] * floor_weight + top[index] * height_weight;
            }
        }
    }
    for (std::size_t point = 0; point < region.point_count; ++point) {
        vector[region.points[point].speaker_id] = gains[point];
    }
}

double layout_attenuation(const std::uint32_t bitfield) {
    return ejoc::speaker::layout_attenuation_db(bitfield);
}

// The reference's position_gain: float64 layout compensation of one point.
double point_position_gain(const std::uint32_t bitfield, const double v, const double w) {
    const double y_term = clamp(v / 0.6, 0.0, 1.0);
    const double z_term = clamp((w - 0.2) / 0.8, 0.0, 1.0);
    const double amount = clamp(y_term + z_term, 0.0, 1.0);
    return std::pow(10.0, layout_attenuation(bitfield) * amount / 20.0);
}

int warp_floor_exponent(const std::uint32_t bitfield, const std::uint32_t warp_mode) {
    if (warp_mode == 0) {
        return 0;
    }
    return ejoc::speaker::floor_y_exponent(bitfield);
}

int zone_region(const std::uint8_t zone_idx) {
    return kZoneToRegion[static_cast<int>(zone_idx) & 7];
}

}  // namespace

struct MetaRenderer::Impl {
    struct PendingTargets {
        std::vector<double> targets;
        std::uint32_t ramp_samples = 0;
    };

    bool configured = false;
    LayoutInfo info{};
    const ejoc::speaker_tables::LayoutGeometry* geometry = nullptr;
    const ejoc::speaker_tables::LayoutGeometry* calc = nullptr;
    LayoutInfo calc_info{};
    bool fold_possible = false;  // the final layout expands for warp 2/3
    ChannelMatrix fold_pl2x{};
    ChannelMatrix fold_loro{};
    int channels = 0;
    int calc_channels = 0;
    // Output (standard) channel index of every internal channel.  The layout
    // table stores internal_to_standard, so the renderer needs its inverse.
    std::array<std::uint8_t, kOutputChannels> standard_for_internal{};
    double window[kBlockSamples] = {};

    bool meta_mode = false;
    bool meta_ready = false;
    std::uint32_t n_meta = 0;
    std::uint32_t raw_trim_mode = 0;
    std::vector<double> current;
    std::vector<double> target;
    std::vector<double> step;
    std::vector<std::int32_t> remaining;
    std::vector<int> slots;
    std::vector<int> labels;
    std::unordered_map<int, oamd::IsfBasis> isf_cache;
    std::map<std::uint32_t, std::vector<PendingTargets>> pending;
    std::string error;

    Status fail(const joc_error code, const char* reason, const std::string& message) {
        error = std::string("speaker: ") + reason + ": " + message;
        return Status::fail(code, stage::kRender, error);
    }

    Status configure(const LayoutInfo& layout);
    void reset_state();
    bool bypassed(const oamd::OamdTrim* trim, const int object_index) const;
    Status init_meta(const oamd::FrameState& frame);
    Status compute_targets(const oamd::FrameState& frame, const std::uint32_t block_index,
                           const ejoc_speaker_controls& controls,
                           const oamd::OamdTrim* trim, const std::uint32_t size_budget,
                           std::vector<double>* targets);
    void dynamic_spatial(const oamd::ObjectRenderState& state, const std::uint32_t warp_mode,
                         const ejoc_speaker_controls& controls, const oamd::OamdTrim* trim,
                         const int object_index, const bool force_zero_size,
                         std::vector<double>& spatial);
    void bed_spatial(const oamd::ProgramState& program, const int label,
                     const std::uint32_t warp_mode, const ejoc_speaker_controls& controls,
                     const oamd::OamdTrim* trim, const int object_index,
                     std::vector<double>& spatial) const;
    void isf_spatial(const oamd::ProgramState& program, const int object_index,
                     const std::uint32_t warp_mode, const oamd::OamdTrim* trim,
                     std::vector<double>& spatial);
    void fold_vector(const std::vector<double>& input, const std::uint32_t warp_mode,
                     std::vector<double>& output) const;
    void apply_targets(const std::vector<double>& targets, const std::uint32_t ramp_samples);
    void render_meta_block(const float* input, const std::uint32_t block, double* output);
    Status schedule(const oamd::OamdUpdate& update, const std::uint32_t event_sample,
                    const ejoc_speaker_controls& controls, const oamd::OamdTrim* trim,
                    const std::uint32_t size_budget);
};

Status MetaRenderer::Impl::configure(const LayoutInfo& layout) {
    configured = false;
    info = layout;
    geometry = ejoc::speaker_tables::find_layout(layout.speaker_bitfield);
    if (geometry == nullptr || geometry->channel_count == 0) {
        return fail(JOC_ERR_LAYOUT_UNSUPPORTED, "layout_unavailable",
                    std::string("no geometry for layout ") + layout.name);
    }
    channels = geometry->channel_count;
    if (channels > kOutputChannels) {
        return fail(JOC_ERR_LAYOUT_UNSUPPORTED, "layout_unavailable",
                    "speaker layout has too many channels");
    }
    for (int index = 0; index < kBlockSamples; ++index) {
        window[index] = static_cast<double>(index) / static_cast<double>(kBlockSamples);
    }
    for (int internal = 0; internal < channels; ++internal) {
        standard_for_internal[static_cast<std::size_t>(internal)] =
            static_cast<std::uint8_t>(internal);
    }
    for (int standard = 0; standard < channels; ++standard) {
        standard_for_internal[geometry->standard_from_internal[static_cast<std::size_t>(
            standard)]] = static_cast<std::uint8_t>(standard);
    }

    // Warp 2/3 render the 5.1-family and 4.0 layouts through the expanded
    // surround layout and fold the rear pair back (reference
    // _surround_fold_layout).  Everything else renders directly.
    calc = geometry;
    calc_info = info;
    fold_possible = false;
    if (ejoc::speaker::floor_y_exponent(layout.speaker_bitfield) != 0) {
        std::uint32_t source_bitfield = 0;
        switch (layout.speaker_bitfield) {
            case 0xFu: source_bitfield = 0x1Fu; break;   // 5.1 -> 7.1
            case 0x40Fu: source_bitfield = 0x41Fu; break;  // 5.1.2 -> 7.1.2
            case 0xA0Fu: source_bitfield = 0xA1Fu; break;  // 5.1.4 -> 7.1.4
            case 0x9u: source_bitfield = 0x19u; break;    // 4.0 -> 6.0
            default:
                return fail(JOC_ERR_LAYOUT_UNSUPPORTED, "surround_fold_layout",
                            std::string("unsupported surround fold layout: ") + layout.name);
        }
        const auto* source = ejoc::speaker_tables::find_layout(source_bitfield);
        if (source == nullptr) {
            return fail(JOC_ERR_LAYOUT_UNSUPPORTED, "surround_fold_layout",
                        "missing surround fold geometry");
        }
        calc = source;
        calc_channels = source->channel_count;
        fold_possible = true;
        if (!layout_by_bitfield(source_bitfield, &calc_info)) {
            return fail(JOC_ERR_LAYOUT_UNSUPPORTED, "surround_fold_layout",
                        "missing surround fold layout info");
        }
        make_fold_matrix(source_bitfield, layout.speaker_bitfield, 2, fold_pl2x);
        make_fold_matrix(source_bitfield, layout.speaker_bitfield, 3, fold_loro);
    } else {
        calc_channels = channels;
    }
    configured = true;
    return Status::success();
}

void MetaRenderer::Impl::reset_state() {
    meta_mode = false;
    meta_ready = false;
    n_meta = 0;
    raw_trim_mode = 0;
    current.clear();
    target.clear();
    step.clear();
    remaining.clear();
    slots.clear();
    labels.clear();
    isf_cache.clear();
    pending.clear();
    error.clear();
}

bool MetaRenderer::Impl::bypassed(const oamd::OamdTrim* trim, const int object_index) const {
    if (trim == nullptr || object_index < 0 || object_index >= oamd::kMaxSlots) {
        return false;
    }
    return trim->disable_per_object[object_index];
}

Status MetaRenderer::Impl::init_meta(const oamd::FrameState& frame) {
    const oamd::ProgramState& program = frame.program;
    std::vector<int> declared_labels;
    declared_labels.reserve(program.bed_labels.size());
    for (const ::joc::joc::BedChannel channel : program.bed_labels) {
        declared_labels.push_back(static_cast<int>(channel) + 1);
    }
    const std::uint32_t num_bed = program.num_bed_objects;
    if (declared_labels.size() != num_bed) {
        if (program.dynamic_object_only) {
            declared_labels.clear();
            if (program.lfe_present) {
                declared_labels.push_back(kLfeBedLabel);
            }
        }
        if (declared_labels.size() != num_bed) {
            return fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT, "bed_declaration_mismatch",
                        "OAMD bed declaration does not match the programme section "
                        "(declared=" +
                            std::to_string(declared_labels.size()) + ", num_bed_objects=" +
                            std::to_string(num_bed) + ")");
        }
    }

    std::vector<int> object_slots;
    object_slots.reserve(frame.object_count);
    int slot = 1;
    for (std::uint32_t object = 0; object < frame.object_count; ++object) {
        const int label = object < declared_labels.size()
                              ? declared_labels[object]
                              : -1;
        const oamd::ObjectKind kind =
            object < frame.objects.size() ? frame.objects[object].kind
                                          : oamd::ObjectKind::kDynamic;
        if (kind == oamd::ObjectKind::kBed &&
            (label == kLfeBedLabel || label == kLfe2BedLabel)) {
            object_slots.push_back(0);
        } else {
            if (slot > 15) {
                return fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT, "object_slot_overflow",
                            "OAMD metadata carries more than 15 JOC object slots "
                            "(object_count=" +
                                std::to_string(frame.object_count) + ")");
            }
            object_slots.push_back(slot);
            ++slot;
        }
    }
    if (!program.dynamic_object_only) {
        const std::int32_t dynamic_objects = program.num_dynamic_objects;
        if (dynamic_objects != -1 &&
            static_cast<std::int64_t>(num_bed) +
                    static_cast<std::int64_t>(program.num_isf_objects) +
                    static_cast<std::int64_t>(dynamic_objects) !=
                static_cast<std::int64_t>(frame.object_count)) {
            return fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT, "program_object_count_mismatch",
                        "OAMD programme object sections do not add up to the object count");
        }
    }
    if (program.num_isf_objects != 0u) {
        const oamd::MetadataParams* params = oamd::metadata_params();
        const oamd::IsfConfig* config =
            params != nullptr && program.isf_idx >= 0 &&
                    program.isf_idx < static_cast<std::int32_t>(params->configs.size())
                ? &params->configs[static_cast<std::size_t>(program.isf_idx)]
                : nullptr;
        if (config == nullptr ||
            config->row_count() != static_cast<int>(program.num_isf_objects)) {
            return fail(JOC_ERR_OAMD_UNSUPPORTED_VARIANT, "isf_configuration_mismatch",
                        "ISF configuration does not match the declared ISF object count");
        }
    }
    if (frame.object_count != n_meta || object_slots != slots) {
        n_meta = frame.object_count;
        slots = object_slots;
        const std::size_t total =
            static_cast<std::size_t>(n_meta) * static_cast<std::size_t>(channels);
        current.assign(total, 0.0);
        target.assign(total, 0.0);
        step.assign(total, 0.0);
        remaining.assign(total, 0);
    }
    labels.assign(n_meta, -1);
    for (std::uint32_t object = 0; object < n_meta; ++object) {
        if (object < declared_labels.size()) {
            labels[object] = declared_labels[object];
        }
    }
    meta_mode = true;
    return Status::success();
}

Status MetaRenderer::Impl::compute_targets(const oamd::FrameState& frame,
                                           const std::uint32_t block_index,
                                           const ejoc_speaker_controls& controls,
                                           const oamd::OamdTrim* trim,
                                           const std::uint32_t size_budget,
                                           std::vector<double>* targets) {
    const std::uint32_t count = frame.object_count;
    targets->assign(static_cast<std::size_t>(count) * static_cast<std::size_t>(channels),
                    0.0);
    if (block_index >= frame.block_count) {
        return Status::success();
    }
    std::vector<std::uint32_t> size_objects;
    for (std::uint32_t object = 0; object < count; ++object) {
        if (object >= frame.objects.size() ||
            block_index >= frame.objects[object].blocks.size()) {
            continue;
        }
        const oamd::ObjectRenderState& state =
            frame.objects[object].blocks[block_index].render;
        if (frame.objects[object].kind == oamd::ObjectKind::kDynamic &&
            !state.not_active && (state.size_mode == 1u || state.size_mode == 2u) &&
            !state.snap && state.divergence_q == 0u) {
            size_objects.push_back(object);
        }
    }
    std::vector<char> zero_size(count, 0);
    if (size_budget > 0u && size_budget < size_objects.size()) {
        std::vector<std::uint32_t> ranked = size_objects;
        std::stable_sort(ranked.begin(), ranked.end(),
                         [&](const std::uint32_t left, const std::uint32_t right) {
                             const auto bucket = [&](const std::uint32_t object) {
                                 return (frame.objects[object].blocks[block_index].render
                                             .priority_q +
                                         512) >> 10;
                             };
                             const int left_bucket = bucket(left);
                             const int right_bucket = bucket(right);
                             if (left_bucket != right_bucket) {
                                 return left_bucket > right_bucket;
                             }
                             return left < right;
                         });
        for (std::size_t index = size_budget; index < ranked.size(); ++index) {
            zero_size[ranked[index]] = 1;
        }
    }
    std::vector<double> spatial;
    for (std::uint32_t object = 0; object < count; ++object) {
        if (object >= frame.objects.size() ||
            block_index >= frame.objects[object].blocks.size()) {
            continue;
        }
        const oamd::ObjectRenderState& state =
            frame.objects[object].blocks[block_index].render;
        if (state.not_active) {
            continue;
        }
        const oamd::ObjectKind kind = frame.objects[object].kind;
        const std::uint32_t warp_mode = controls.warp_mode;
        if (kind == oamd::ObjectKind::kDynamic) {
            dynamic_spatial(state, warp_mode, controls, trim, static_cast<int>(object),
                            zero_size[object] != 0, spatial);
        } else if (kind == oamd::ObjectKind::kBed) {
            bed_spatial(frame.program, labels[object], warp_mode, controls, trim,
                        static_cast<int>(object), spatial);
        } else if (kind == oamd::ObjectKind::kIsf) {
            isf_spatial(frame.program, static_cast<int>(object), warp_mode, trim,
                        spatial);
        } else {
            spatial.assign(static_cast<std::size_t>(channels), 0.0);
        }
        const double gain = static_cast<double>(state.gain_q) / 4096.0;
        const std::size_t base =
            static_cast<std::size_t>(object) * static_cast<std::size_t>(channels);
        for (int channel = 0; channel < channels; ++channel) {
            (*targets)[base + static_cast<std::size_t>(channel)] =
                spatial[static_cast<std::size_t>(channel)] * gain;
        }
    }
    return Status::success();
}

void MetaRenderer::Impl::dynamic_spatial(const oamd::ObjectRenderState& state,
                                         const std::uint32_t warp_mode,
                                         const ejoc_speaker_controls& controls,
                                         const oamd::OamdTrim* trim, const int object_index,
                                         const bool force_zero_size,
                                         std::vector<double>& spatial) {
    const bool folding = fold_possible && warp_mode >= 2u;
    const ejoc::speaker_tables::LayoutGeometry& layout = folding ? *calc : *geometry;
    const int region_index = zone_region(state.zone_idx);
    double u = static_cast<double>(state.qx) / kQ15;
    const double v = static_cast<double>(state.qy) / kQ15;
    double w = std::max(static_cast<double>(state.qz) / kQ15, 0.0);
    const bool enable_height = state.enable_elevation;
    if (!enable_height) {
        w = 0.0;
    }
    const bool bypass = bypassed(trim, object_index);
    double floor_v = clamp(std::ldexp(v, warp_floor_exponent(layout.speaker_bitfield,
                                                             warp_mode)),
                           0.0, 1.0);
    double top_v = clamp(v, 0.0, 1.0);
    double trim_v = v;
    if (controls.trim_mode == EJOC_SPEAKER_TRIM_EXPLICIT && !bypass) {
        floor_v = apply_balance(floor_v, w, controls.listener_balance,
                                               controls.overhead_balance);
        top_v = apply_balance(top_v, w, controls.listener_balance,
                                             controls.overhead_balance);
        trim_v = apply_balance(v, w, controls.listener_balance,
                                              controls.overhead_balance);
    }
    const auto& region = layout.regions[static_cast<std::size_t>(region_index)];
    std::vector<double> vector;
    double comp_u = u;
    double comp_v = trim_v;
    double comp_w = w;
    if (state.snap) {
        std::vector<double> point;
        region_vector(layout, region_index, u, floor_v, top_v, w, enable_height, point);
        double best = -1.0;
        int chosen = 0;
        for (std::size_t index = 0; index < region.point_count; ++index) {
            const double value =
                point[region.points[index].speaker_id];
            if (value > best) {
                best = value;
                chosen = static_cast<int>(index);
            }
        }
        const double speaker_u =
            static_cast<double>(region.points[static_cast<std::size_t>(chosen)]
                                    .coordinate_q15[0]) /
            kQ15;
        const double speaker_v =
            static_cast<double>(region.points[static_cast<std::size_t>(chosen)]
                                    .coordinate_q15[1]) /
            kQ15;
        const double speaker_w =
            static_cast<double>(region.points[static_cast<std::size_t>(chosen)]
                                    .coordinate_q15[2]) /
            kQ15;
        const double scale = 1.0;
        const double distance =
            (u - speaker_u) * (u - speaker_u) +
            (std::min(v, scale) - scale * speaker_v) *
                (std::min(v, scale) - scale * speaker_v) +
            (w - speaker_w) * (w - speaker_w);
        if (distance < 0.04) {
            vector.assign(static_cast<std::size_t>(layout.channel_count), 0.0);
            vector[region.points[static_cast<std::size_t>(chosen)].speaker_id] = 1.0;
            comp_u = speaker_u;
            comp_v = speaker_v;
            comp_w = speaker_w;
        } else {
            vector = point;
        }
    } else if (state.divergence_q > 0u) {
        const double spread =
            std::min(static_cast<double>(state.divergence_q) / kQ15, std::min(u, 1.0 - u));
        std::vector<double> left;
        std::vector<double> right;
        region_vector(layout, region_index, u - spread, floor_v, top_v, w,
                      enable_height, left);
        region_vector(layout, region_index, u + spread, floor_v, top_v, w,
                      enable_height, right);
        for (std::size_t index = 0; index < left.size(); ++index) {
            left[index] += right[index];
        }
        vector = stable_norm(left);
    } else if ((state.size_mode == 1u || state.size_mode == 2u) && !force_zero_size) {
        const std::uint16_t* size_q = state.size_q;
        int q_size = 0;
        if (size_q[0] == size_q[1] && size_q[1] == size_q[2]) {
            q_size = size_q[0];
        } else {
            q_size = (static_cast<int>(size_q[0]) + static_cast<int>(size_q[1]) +
                      static_cast<int>(size_q[2])) /
                     3;
        }
        const double size = static_cast<double>(q_size) / kQ15;
        const float t = oamd::size_curve_value(size);
        std::vector<double> point;
        region_vector(layout, region_index, u, floor_v, top_v, w, enable_height, point);
        if (t < 0.000025f) {
            vector = point;
        } else {
            const double query[3] = {u, (1.0 - w) * floor_v + w * top_v, w};
            const std::vector<float> spread_float = oamd::size_targets(
                *oamd::metadata_params(), layout, region_index, query, size, enable_height);
            std::vector<double> spread(spread_float.size());
            for (std::size_t index = 0; index < spread_float.size(); ++index) {
                spread[index] = static_cast<double>(spread_float[index]);
            }
            if (t > 0.05f) {
                vector = spread;
            } else {
                const double angle = 20.0 * static_cast<double>(t);
                double cosine = 0.0;
                double sine = 0.0;
                if (angle >= 0.99996948) {
                    cosine = 0.0;
                    sine = 1.0;
                } else {
                    cosine = std::cos(angle * kPi / 2.0);
                    sine = std::sin(angle * kPi / 2.0);
                }
                std::vector<double> mixed(point.size(), 0.0);
                for (std::size_t index = 0; index < point.size() && index < spread.size();
                     ++index) {
                    mixed[index] = 0.5 * cosine * point[index] + 0.5 * sine * spread[index];
                }
                vector = stable_norm(mixed);
            }
        }
    } else {
        region_vector(layout, region_index, u, floor_v, top_v, w, enable_height, vector);
    }
    double compensation = 1.0;
    if (!bypass) {
        if (controls.trim_mode == EJOC_SPEAKER_TRIM_AUTOMATIC) {
            compensation = point_position_gain(geometry->speaker_bitfield, comp_v, comp_w);
        } else if (controls.trim_mode == EJOC_SPEAKER_TRIM_EXPLICIT) {
            compensation = explicit_position_gain(controls, comp_u, comp_v,
                                                                 comp_w);
        }
    }
    fold_vector(vector, warp_mode, spatial);
    for (double& value : spatial) {
        value *= compensation;
    }
}

void MetaRenderer::Impl::bed_spatial(const oamd::ProgramState& program, const int label,
                                     const std::uint32_t warp_mode,
                                     const ejoc_speaker_controls& controls,
                                     const oamd::OamdTrim* trim, const int object_index,
                                     std::vector<double>& spatial) const {
    spatial.assign(static_cast<std::size_t>(channels), 0.0);
    if (label < 1 || label > oamd::kBedLabelCount) {
        return;
    }
    const bool folding = fold_possible && warp_mode >= 2u;
    const LayoutInfo& calc_layout = folding ? calc_info : info;
    const ejoc::speaker_tables::LayoutGeometry& layout = folding ? *calc : *geometry;
    const oamd::MetadataParams* params = oamd::metadata_params();
    if (params == nullptr) {
        return;
    }
    const oamd::BedDistribution distribution = oamd::bed_distribution(
        calc_layout, label, program.bed_mask, program.b_bed_chan_distribute);
    const bool bypass = bypassed(trim, object_index);
    int mode = 1;
    if (controls.trim_mode == EJOC_SPEAKER_TRIM_AUTOMATIC) {
        mode = 0;
    } else if (controls.trim_mode == EJOC_SPEAKER_TRIM_EXPLICIT) {
        mode = 2;
    }
    std::vector<double> vector;
    double compensation = 1.0;
    if (distribution.route != -1) {
        int position[3] = {0, 0, 0};
        oamd::bed_reference_position(distribution.route, position);
        int qx = position[0];
        const int qy = position[1];
        const int qz = position[2];
        if (qz > 0) {
            qx = qx > 16384 ? 32767 : 0;
        }
        const double u = static_cast<double>(qx) / kQ15;
        const double v = static_cast<double>(qy) / kQ15;
        const double w = std::max(qz, 0) / kQ15;
        const double floor_v = clamp(
            std::ldexp(v, warp_floor_exponent(layout.speaker_bitfield, warp_mode)), 0.0,
            1.0);
        const double top_v = clamp(v, 0.0, 1.0);
        region_vector(layout, 0, u, floor_v, top_v, w, true, vector);
        if (!bypass && controls.trim_mode != EJOC_SPEAKER_TRIM_DISABLED) {
            if (controls.trim_mode == EJOC_SPEAKER_TRIM_AUTOMATIC) {
                compensation = point_position_gain(geometry->speaker_bitfield, v, w);
            } else {
                compensation = explicit_position_gain(controls, u, v, w);
            }
        }
    } else {
        vector.assign(distribution.weights.size(), 0.0);
        for (std::size_t index = 0; index < distribution.weights.size(); ++index) {
            vector[index] = static_cast<double>(distribution.weights[index]);
        }
        const double levels[3] = {controls.surround_db, controls.height_db,
                                  controls.centre_db};
        compensation =
            static_cast<double>(oamd::bed_compensation(info, label, mode, bypass, levels));
    }
    fold_vector(vector, warp_mode, spatial);
    for (double& value : spatial) {
        value *= compensation;
    }
}

void MetaRenderer::Impl::isf_spatial(const oamd::ProgramState& program,
                                     const int object_index, const std::uint32_t warp_mode,
                                     const oamd::OamdTrim* trim,
                                     std::vector<double>& spatial) {
    spatial.assign(static_cast<std::size_t>(channels), 0.0);
    const oamd::MetadataParams* params = oamd::metadata_params();
    if (params == nullptr || program.isf_idx < 0) {
        return;
    }
    const int isf_idx = program.isf_idx;
    const int row = object_index - static_cast<int>(program.num_bed_objects);
    const bool folding = fold_possible && warp_mode >= 2u;
    const ejoc::speaker_tables::LayoutGeometry& layout = folding ? *calc : *geometry;
    const std::string& calc_key =
        folding ? std::string(calc_info.compact_name) : std::string(info.compact_name);
    auto found = isf_cache.find(isf_idx);
    if (found == isf_cache.end()) {
        found = isf_cache.emplace(isf_idx,
                                  oamd::assemble_isf(*params, calc_key.c_str(), isf_idx))
                    .first;
    }
    const oamd::IsfBasis& basis = found->second;
    if (row < 0 || row >= basis.rows) {
        return;
    }
    const bool bypass = bypassed(trim, object_index);
    const bool disabled = raw_trim_mode == 1u || bypass;
    std::vector<double> vector(static_cast<std::size_t>(layout.channel_count), 0.0);
    for (int column = 0; column < basis.columns && column < layout.channel_count;
         ++column) {
        vector[static_cast<std::size_t>(column)] =
            static_cast<double>(basis.at(row, column));
    }
    const double compensation = static_cast<double>(oamd::isf_compensation(
        *params, info.compact_name, isf_idx, row, disabled));
    fold_vector(vector, warp_mode, spatial);
    for (double& value : spatial) {
        value *= compensation;
    }
}

void MetaRenderer::Impl::fold_vector(const std::vector<double>& input,
                                     const std::uint32_t warp_mode,
                                     std::vector<double>& output) const {
    const bool folding = fold_possible && warp_mode >= 2u;
    if (!folding) {
        output = input;
        if (output.size() > static_cast<std::size_t>(channels)) {
            output.resize(static_cast<std::size_t>(channels));
        }
        output.resize(static_cast<std::size_t>(channels), 0.0);
        return;
    }
    output.assign(static_cast<std::size_t>(channels), 0.0);
    const ChannelMatrix& matrix = warp_mode == 2 ? fold_pl2x : fold_loro;
    for (int row = 0; row < channels; ++row) {
        double sum = 0.0;
        const int source_count = std::min<int>(calc_channels, static_cast<int>(input.size()));
        for (int source = 0; source < source_count; ++source) {
            sum += matrix.value[row][source] * input[static_cast<std::size_t>(source)];
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
}

void MetaRenderer::Impl::apply_targets(const std::vector<double>& targets,
                                       const std::uint32_t ramp_samples) {
    if (current.size() != targets.size()) {
        current.assign(targets.size(), 0.0);
        target.assign(targets.size(), 0.0);
        step.assign(targets.size(), 0.0);
        remaining.assign(targets.size(), 0);
    }
    const std::uint32_t blocks = ramp_block_count(ramp_samples);
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const double difference = targets[index] - current[index];
        const bool significant = std::fabs(difference) >= kGainSnapThreshold;
        target[index] = targets[index];
        if (!significant) {
            current[index] = target[index];
            step[index] = 0.0;
            remaining[index] = 0;
            continue;
        }
        if (blocks != 0u) {
            step[index] = difference / static_cast<double>(blocks);
            remaining[index] = static_cast<std::int32_t>(blocks);
        } else {
            current[index] = target[index];
            step[index] = 0.0;
            remaining[index] = 0;
        }
    }
    meta_ready = true;
}

void MetaRenderer::Impl::render_meta_block(const float* input, const std::uint32_t block,
                                           double* output) {
    const std::uint32_t start = block * static_cast<std::uint32_t>(kBlockSamples);
    const std::uint8_t* standard_map = standard_for_internal.data();
    for (std::uint32_t object = 0; object < n_meta; ++object) {
        const int slot = slots[object];
        if (slot < 0) {
            continue;
        }
        const std::size_t base =
            static_cast<std::size_t>(object) * static_cast<std::size_t>(channels);
        for (int channel = 0; channel < channels; ++channel) {
            const std::size_t index = base + static_cast<std::size_t>(channel);
            const bool active = remaining[index] > 0;
            const double fixed_gain = target[index];
            if (!active && fixed_gain == 0.0) {
                continue;
            }
            const std::size_t standard =
                standard_map[static_cast<std::size_t>(channel)];
            for (int sample = 0; sample < kBlockSamples; ++sample) {
                const double gain =
                    active ? current[index] +
                                 window[static_cast<std::size_t>(sample)] * step[index]
                           : fixed_gain;
                output[(static_cast<std::size_t>(start) +
                        static_cast<std::size_t>(sample)) *
                           static_cast<std::size_t>(channels) +
                       standard] +=
                    static_cast<double>(
                        input[(static_cast<std::size_t>(start) +
                               static_cast<std::size_t>(sample)) *
                                  static_cast<std::size_t>(kOutputChannels) +
                              static_cast<std::size_t>(slot)]) *
                    gain;
            }
        }
        for (int channel = 0; channel < channels; ++channel) {
            const std::size_t index = base + static_cast<std::size_t>(channel);
            if (remaining[index] > 0) {
                current[index] += step[index];
                --remaining[index];
            }
            if (remaining[index] == 0) {
                current[index] = target[index];
            }
        }
    }
}

Status MetaRenderer::Impl::schedule(const oamd::OamdUpdate& update,
                                    const std::uint32_t event_sample,
                                    const ejoc_speaker_controls& controls,
                                    const oamd::OamdTrim* trim,
                                    const std::uint32_t size_budget) {
    const oamd::FrameState& frame = update.frame_state;
    const Status initialized = init_meta(frame);
    if (!initialized.ok()) {
        return initialized;
    }
    const std::size_t block_count =
        std::min<std::size_t>(frame.block_count, frame.blocks.size());
    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
        // The reference accumulates the element sample offset twice here: once
        // through this addition and once through block_offset_samples, which
        // already contains it.  Keep the same arithmetic.
        const std::uint32_t sample =
            event_sample + frame.sample_offset + frame.blocks[block_index].block_offset_samples;
        const std::uint32_t aligned = align_metadata_sample(sample);
        std::vector<double> targets;
        const Status computed = compute_targets(frame, static_cast<std::uint32_t>(block_index),
                                                controls, trim, size_budget, &targets);
        if (!computed.ok()) {
            return computed;
        }
        pending[aligned / static_cast<std::uint32_t>(kBlockSamples)].push_back(
            PendingTargets{std::move(targets),
                           frame.blocks[block_index].ramp_duration_samples});
    }
    return Status::success();
}

MetaRenderer::MetaRenderer() : impl_(std::make_unique<Impl>()) {}

MetaRenderer::~MetaRenderer() = default;

MetaRenderer::MetaRenderer(MetaRenderer&&) noexcept = default;

MetaRenderer& MetaRenderer::operator=(MetaRenderer&&) noexcept = default;

Status MetaRenderer::configure(const LayoutInfo& layout) {
    const Status status = impl_->configure(layout);
    impl_->reset_state();
    return status;
}

Status MetaRenderer::reset() {
    impl_->reset_state();
    return Status::success();
}

bool MetaRenderer::configured() const { return impl_->configured; }

bool MetaRenderer::engaged() const { return impl_->meta_mode; }

const char* MetaRenderer::last_error() const {
    return impl_->error.empty() ? "" : impl_->error.c_str();
}

Status MetaRenderer::block_targets(const oamd::OamdUpdate& update,
                                   const std::uint32_t block_index,
                                   const ejoc_speaker_controls& controls,
                                   const oamd::OamdTrim* trim,
                                   const std::uint32_t size_budget,
                                   MetadataTargets* out) const {
    if (!impl_->configured || out == nullptr) {
        return Status::fail(JOC_ERR_STATE, stage::kRender,
                            "speaker: metadata renderer is not configured");
    }
    // block_targets must not disturb the live renderer state, so it validates
    // and computes on a copy of the object bookkeeping only.
    Impl scratch = *impl_;
    scratch.meta_mode = true;
    scratch.current.clear();
    scratch.target.clear();
    scratch.step.clear();
    scratch.remaining.clear();
    scratch.pending.clear();
    const Status initialized = scratch.init_meta(update.frame_state);
    if (!initialized.ok()) {
        return initialized;
    }
    std::vector<double> targets;
    const Status computed =
        scratch.compute_targets(update.frame_state, block_index, controls, trim,
                                size_budget, &targets);
    if (!computed.ok()) {
        return computed;
    }
    out->object_count = update.frame_state.object_count;
    out->channel_count = static_cast<std::uint32_t>(scratch.channels);
    out->values = std::move(targets);
    return Status::success();
}

Status MetaRenderer::process_frame(
    ejoc_speaker_renderer_handle legacy, const float* objects16_interleaved,
    const std::uint32_t sample_count, const oamd::OamdUpdate* update,
    const std::uint32_t metadata_offset, const ejoc_speaker_controls& controls,
    const oamd::OamdTrim* trim, const std::uint32_t raw_trim_mode,
    const std::uint32_t size_budget, double* output_interleaved) {
    Impl& state = *impl_;
    state.error.clear();
    if (!state.configured) {
        return Status::fail(JOC_ERR_STATE, stage::kRender,
                            "speaker: metadata renderer is not configured");
    }
    if (!objects16_interleaved || !output_interleaved) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "speaker: null PCM pointer passed to the metadata renderer");
    }
    if (sample_count == 0u || sample_count % static_cast<std::uint32_t>(kBlockSamples) != 0u) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "speaker: sample_count must be a multiple of 32");
    }
    if (metadata_offset > sample_count) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kRender,
                            "speaker: metadata offset outside process call");
    }
    state.raw_trim_mode = raw_trim_mode;
    const std::size_t total_samples = sample_count;
    std::fill(output_interleaved,
              output_interleaved + total_samples * static_cast<std::size_t>(state.channels),
              0.0);

    const std::uint32_t total_blocks = sample_count / static_cast<std::uint32_t>(kBlockSamples);
    const std::uint32_t event_block =
        align_metadata_sample(metadata_offset) / static_cast<std::uint32_t>(kBlockSamples);
    const bool has_metadata =
        update != nullptr && update->has_frame_state && update->frame_state.object_count != 0u;
    bool scheduled = false;
    for (std::uint32_t block = 0; block < total_blocks; ++block) {
        if (has_metadata && !scheduled && block == event_block) {
            const Status status = state.schedule(*update, metadata_offset, controls, trim,
                                                 size_budget);
            if (!status.ok()) {
                return status;
            }
            scheduled = true;
        }
        const auto found = state.pending.find(block);
        if (found != state.pending.end()) {
            for (const Impl::PendingTargets& targets : found->second) {
                state.apply_targets(targets.targets, targets.ramp_samples);
            }
            state.pending.erase(found);
        }
        if (state.meta_mode && state.meta_ready) {
            state.render_meta_block(objects16_interleaved, block, output_interleaved);
        } else if (legacy != nullptr) {
            static_cast<ejoc::speaker::Renderer*>(legacy)->render_legacy_block(
                objects16_interleaved, block, output_interleaved);
        }
    }
    if (has_metadata && !scheduled) {
        // The reference only handles events exactly at the end of the call.
        if (event_block == total_blocks) {
            const Status status = state.schedule(*update, metadata_offset, controls, trim,
                                                 size_budget);
            if (!status.ok()) {
                return status;
            }
        }
    }
    const auto tail = state.pending.find(total_blocks);
    if (tail != state.pending.end()) {
        for (const Impl::PendingTargets& targets : tail->second) {
            state.apply_targets(targets.targets, targets.ramp_samples);
        }
        state.pending.erase(tail);
    }
    for (auto& entry : state.pending) {
        for (const Impl::PendingTargets& targets : entry.second) {
            state.apply_targets(targets.targets, targets.ramp_samples);
        }
    }
    state.pending.clear();
    return Status::success();
}

Status metadata_block_targets(const LayoutInfo& layout, const oamd::OamdUpdate& update,
                              const std::uint32_t block_index,
                              const ejoc_speaker_controls& controls,
                              const oamd::OamdTrim* trim, const std::uint32_t size_budget,
                              MetadataTargets* out) {
    MetaRenderer renderer;
    const Status configured = renderer.configure(layout);
    if (!configured.ok()) {
        return configured;
    }
    return renderer.block_targets(update, block_index, controls, trim, size_budget, out);
}

}  // namespace joc::speaker
