#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ejoc::speaker_tables {

inline constexpr std::size_t kMaxPoints = 15;
inline constexpr std::size_t kMaxGroups = 4;
inline constexpr std::size_t kMaxGroupSize = 3;
inline constexpr std::size_t kMaxChannels = 16;

struct SpeakerPoint {
    std::array<std::uint16_t, 3> coordinate_q15{};
    std::uint8_t speaker_id{};
};

struct AxisGroup {
    std::uint8_t size{};
    std::array<std::uint8_t, kMaxGroupSize> indices{};
};

struct RegionGeometry {
    std::uint8_t point_count{};
    std::uint8_t mode{};
    std::uint8_t axis0_group_count{};
    std::uint8_t axis1_group_count{};
    std::array<SpeakerPoint, kMaxPoints> points{};
    std::array<AxisGroup, kMaxGroups> axis0_groups{};
    std::array<AxisGroup, kMaxGroups> axis1_groups{};
};

struct LayoutGeometry {
    std::uint32_t speaker_bitfield{};
    std::uint8_t out_ch_config{};
    std::uint8_t channel_count{};
    std::array<std::uint8_t, kMaxChannels> standard_from_internal{};
    std::array<RegionGeometry, 7> regions{};
};

inline constexpr std::array<LayoutGeometry, 10> kLayouts{{
    LayoutGeometry{
        0x1u, 0, 2,
        {{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0x7u, 3, 4,
        {{0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            1, 1, 1, 0,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0xFu, 7, 6,
        {{0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            5, 2, 2, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 2, 2, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 2, 2, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 2, 2, 0,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            2, 1, 1, 0,
            {{SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0x1Fu, 11, 8,
        {{0, 1, 2, 3, 6, 7, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            7, 2, 3, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 2, 2, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 2, 2, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 2, 2, 0,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            4, 2, 2, 0,
            {{SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{2, {2, 3, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0x40Fu, 13, 8,
        {{0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            7, 3, 2, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 16384, 32767}, 6}, SpeakerPoint{{24840, 16384, 32767}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 16384, 32767}, 6}, SpeakerPoint{{24840, 16384, 32767}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 16384, 32767}, 6}, SpeakerPoint{{24840, 16384, 32767}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 3, 2, 1,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 16384, 32767}, 6}, SpeakerPoint{{24840, 16384, 32767}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 3, 1, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 16384, 32767}, 6}, SpeakerPoint{{24840, 16384, 32767}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            4, 3, 1, 1,
            {{SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 16384, 32767}, 6}, SpeakerPoint{{24840, 16384, 32767}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {2, 3, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0xA0Fu, 14, 10,
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            9, 3, 2, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 7928, 32767}, 6}, SpeakerPoint{{24840, 7928, 32767}, 7}, SpeakerPoint{{7928, 24840, 32767}, 8}, SpeakerPoint{{24840, 24840, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            9, 3, 2, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 7928, 32767}, 6}, SpeakerPoint{{24840, 7928, 32767}, 7}, SpeakerPoint{{7928, 24840, 32767}, 8}, SpeakerPoint{{24840, 24840, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            9, 3, 2, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 7928, 32767}, 6}, SpeakerPoint{{24840, 7928, 32767}, 7}, SpeakerPoint{{7928, 24840, 32767}, 8}, SpeakerPoint{{24840, 24840, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 2,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 7928, 32767}, 6}, SpeakerPoint{{24840, 7928, 32767}, 7}, SpeakerPoint{{7928, 24840, 32767}, 8}, SpeakerPoint{{24840, 24840, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 1, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 7928, 32767}, 6}, SpeakerPoint{{24840, 7928, 32767}, 7}, SpeakerPoint{{7928, 24840, 32767}, 8}, SpeakerPoint{{24840, 24840, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            6, 3, 1, 2,
            {{SpeakerPoint{{0, 32767, 0}, 4}, SpeakerPoint{{32767, 32767, 0}, 5}, SpeakerPoint{{7928, 7928, 32767}, 6}, SpeakerPoint{{24840, 7928, 32767}, 7}, SpeakerPoint{{7928, 24840, 32767}, 8}, SpeakerPoint{{24840, 24840, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {2, 3, 0}}, AxisGroup{2, {4, 5, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 3, 1, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 7928, 32767}, 6}, SpeakerPoint{{24840, 7928, 32767}, 7}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0x41Fu, 15, 10,
        {{0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 0, 0, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            9, 3, 3, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 16384, 32767}, 8}, SpeakerPoint{{24840, 16384, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}}},
            {{AxisGroup{2, {7, 8, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{7928, 16384, 32767}, 8}, SpeakerPoint{{24840, 16384, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 16384, 32767}, 8}, SpeakerPoint{{24840, 16384, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 3, 2, 1,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 16384, 32767}, 8}, SpeakerPoint{{24840, 16384, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 3, 1, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 16384, 32767}, 8}, SpeakerPoint{{24840, 16384, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            6, 3, 2, 1,
            {{SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 16384, 32767}, 8}, SpeakerPoint{{24840, 16384, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{2, {2, 3, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {4, 5, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            3, 1, 1, 0,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0xA1Fu, 16, 12,
        {{0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11, 0, 0, 0, 0}},
        {{
        RegionGeometry{
            11, 3, 3, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 8}, SpeakerPoint{{24840, 7928, 32767}, 9}, SpeakerPoint{{7928, 24840, 32767}, 10}, SpeakerPoint{{24840, 24840, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}}},
            {{AxisGroup{2, {7, 8, 0}}, AxisGroup{2, {9, 10, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            9, 3, 2, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{7928, 7928, 32767}, 8}, SpeakerPoint{{24840, 7928, 32767}, 9}, SpeakerPoint{{7928, 24840, 32767}, 10}, SpeakerPoint{{24840, 24840, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            9, 3, 2, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 8}, SpeakerPoint{{24840, 7928, 32767}, 9}, SpeakerPoint{{7928, 24840, 32767}, 10}, SpeakerPoint{{24840, 24840, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 2,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 8}, SpeakerPoint{{24840, 7928, 32767}, 9}, SpeakerPoint{{7928, 24840, 32767}, 10}, SpeakerPoint{{24840, 24840, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 1, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 7928, 32767}, 8}, SpeakerPoint{{24840, 7928, 32767}, 9}, SpeakerPoint{{7928, 24840, 32767}, 10}, SpeakerPoint{{24840, 24840, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            8, 3, 2, 2,
            {{SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 8}, SpeakerPoint{{24840, 7928, 32767}, 9}, SpeakerPoint{{7928, 24840, 32767}, 10}, SpeakerPoint{{24840, 24840, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{2, {2, 3, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {4, 5, 0}}, AxisGroup{2, {6, 7, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            5, 3, 1, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 7928, 32767}, 8}, SpeakerPoint{{24840, 7928, 32767}, 9}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0xA9Fu, 19, 14,
        {{0, 1, 2, 3, 6, 7, 4, 5, 10, 11, 12, 13, 8, 9, 0, 0}},
        {{
        RegionGeometry{
            13, 3, 4, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{0, 5285, 0}, 8}, SpeakerPoint{{32767, 5285, 0}, 9}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 24840, 32767}, 12}, SpeakerPoint{{24840, 24840, 32767}, 13}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}}},
            {{AxisGroup{2, {9, 10, 0}}, AxisGroup{2, {11, 12, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            11, 3, 3, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 5285, 0}, 8}, SpeakerPoint{{32767, 5285, 0}, 9}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 24840, 32767}, 12}, SpeakerPoint{{24840, 24840, 32767}, 13}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}}},
            {{AxisGroup{2, {7, 8, 0}}, AxisGroup{2, {9, 10, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            9, 3, 2, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 24840, 32767}, 12}, SpeakerPoint{{24840, 24840, 32767}, 13}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 2,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 24840, 32767}, 12}, SpeakerPoint{{24840, 24840, 32767}, 13}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 1, 2,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 24840, 32767}, 12}, SpeakerPoint{{24840, 24840, 32767}, 13}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            8, 3, 2, 2,
            {{SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 24840, 32767}, 12}, SpeakerPoint{{24840, 24840, 32767}, 13}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{2, {2, 3, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {4, 5, 0}}, AxisGroup{2, {6, 7, 0}}, AxisGroup{}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 5285, 0}, 8}, SpeakerPoint{{32767, 5285, 0}, 9}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    },
    LayoutGeometry{
        0xE9Fu, 20, 16,
        {{0, 1, 2, 3, 6, 7, 4, 5, 10, 11, 14, 15, 12, 13, 8, 9}},
        {{
        RegionGeometry{
            15, 3, 4, 3,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{0, 5285, 0}, 8}, SpeakerPoint{{32767, 5285, 0}, 9}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 16384, 32767}, 12}, SpeakerPoint{{24840, 16384, 32767}, 13}, SpeakerPoint{{7928, 24840, 32767}, 14}, SpeakerPoint{{24840, 24840, 32767}, 15}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}}},
            {{AxisGroup{2, {9, 10, 0}}, AxisGroup{2, {11, 12, 0}}, AxisGroup{2, {13, 14, 0}}, AxisGroup{}}}
        },
        RegionGeometry{
            13, 3, 3, 3,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 5285, 0}, 8}, SpeakerPoint{{32767, 5285, 0}, 9}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 16384, 32767}, 12}, SpeakerPoint{{24840, 16384, 32767}, 13}, SpeakerPoint{{7928, 24840, 32767}, 14}, SpeakerPoint{{24840, 24840, 32767}, 15}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}}},
            {{AxisGroup{2, {7, 8, 0}}, AxisGroup{2, {9, 10, 0}}, AxisGroup{2, {11, 12, 0}}, AxisGroup{}}}
        },
        RegionGeometry{
            11, 3, 2, 3,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 16384, 32767}, 12}, SpeakerPoint{{24840, 16384, 32767}, 13}, SpeakerPoint{{7928, 24840, 32767}, 14}, SpeakerPoint{{24840, 24840, 32767}, 15}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{2, {9, 10, 0}}, AxisGroup{}}}
        },
        RegionGeometry{
            9, 3, 2, 3,
            {{SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 16384, 32767}, 12}, SpeakerPoint{{24840, 16384, 32767}, 13}, SpeakerPoint{{7928, 24840, 32767}, 14}, SpeakerPoint{{24840, 24840, 32767}, 15}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{1, {0, 0, 0}}, AxisGroup{2, {1, 2, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}}}
        },
        RegionGeometry{
            9, 3, 1, 3,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 16384, 32767}, 12}, SpeakerPoint{{24840, 16384, 32767}, 13}, SpeakerPoint{{7928, 24840, 32767}, 14}, SpeakerPoint{{24840, 24840, 32767}, 15}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {3, 4, 0}}, AxisGroup{2, {5, 6, 0}}, AxisGroup{2, {7, 8, 0}}, AxisGroup{}}}
        },
        RegionGeometry{
            10, 3, 2, 3,
            {{SpeakerPoint{{0, 16384, 0}, 4}, SpeakerPoint{{32767, 16384, 0}, 5}, SpeakerPoint{{0, 32767, 0}, 6}, SpeakerPoint{{32767, 32767, 0}, 7}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{{7928, 16384, 32767}, 12}, SpeakerPoint{{24840, 16384, 32767}, 13}, SpeakerPoint{{7928, 24840, 32767}, 14}, SpeakerPoint{{24840, 24840, 32767}, 15}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{2, {0, 1, 0}}, AxisGroup{2, {2, 3, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {4, 5, 0}}, AxisGroup{2, {6, 7, 0}}, AxisGroup{2, {8, 9, 0}}, AxisGroup{}}}
        },
        RegionGeometry{
            7, 3, 2, 1,
            {{SpeakerPoint{{0, 0, 0}, 0}, SpeakerPoint{{32767, 0, 0}, 1}, SpeakerPoint{{16384, 0, 0}, 2}, SpeakerPoint{{0, 5285, 0}, 8}, SpeakerPoint{{32767, 5285, 0}, 9}, SpeakerPoint{{7928, 7928, 32767}, 10}, SpeakerPoint{{24840, 7928, 32767}, 11}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}, SpeakerPoint{}}},
            {{AxisGroup{3, {0, 2, 1}}, AxisGroup{2, {3, 4, 0}}, AxisGroup{}, AxisGroup{}}},
            {{AxisGroup{2, {5, 6, 0}}, AxisGroup{}, AxisGroup{}, AxisGroup{}}}
        }
        }}
    }
}};

inline constexpr const LayoutGeometry* find_layout(const std::uint32_t speaker_bitfield) noexcept {
    for (const auto& layout : kLayouts) {
        if (layout.speaker_bitfield == speaker_bitfield) {
            return &layout;
        }
    }
    return nullptr;
}

}  // namespace ejoc::speaker_tables
