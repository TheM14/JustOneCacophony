// Port of src/joc_decode.py.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "joc_core.h"
#include "emdf/emdf_parser.h"
#include "foundation/status.h"

namespace joc::joc {

struct ObjectSymbols {
    std::uint8_t present = 0;
    std::uint8_t sparse = 0;
    std::uint8_t n_bands = 0;
    std::uint8_t n_dpoints = 0;
    std::uint8_t n_channels = 0;
    std::int64_t q[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS] = {};
    std::int16_t mtx[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS] = {};
    std::uint8_t idx[JOC_MAX_DPOINTS][JOC_MAX_PARAMETER_BANDS] = {};
    std::int16_t vec[JOC_MAX_DPOINTS][JOC_MAX_PARAMETER_BANDS] = {};
};

struct FrameSymbols {
    std::uint8_t n_objects = 0;
    std::uint8_t n_channels = 0;
    ObjectSymbols objects[JOC_MAX_OBJECTS];
};

Status parse_id14(const std::uint8_t* payload, std::size_t payload_size, joc_frame_params* out,
                  FrameSymbols* symbols);

Status parse_eac3_frame(const std::uint8_t* frame, std::size_t frame_size, joc_frame_params* out,
                        emdf::Container* container, FrameSymbols* symbols);

Status check_id14_padding(const std::uint8_t* payload, std::size_t payload_size,
                          std::uint32_t* out_trailing_bits);

std::int16_t num_channels_for_config(std::uint32_t dmx_config_idx);

std::int16_t num_bands_for_index(std::uint32_t num_bands_idx);

}  // namespace joc::joc
