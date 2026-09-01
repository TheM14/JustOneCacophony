#pragma once

#include <stdint.h>

#if defined(_WIN32)
  #if defined(EJOC_BUILD_DLL)
    #define EJOC_API __declspec(dllexport)
  #else
    #define EJOC_API __declspec(dllimport)
  #endif
  #define EJOC_CALL __cdecl
#else
  #define EJOC_API __attribute__((visibility("default")))
  #define EJOC_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    EJOC_ABI_VERSION = 1,
    EJOC_FRAME_SAMPLES = 1536,
    EJOC_TIMESLOTS = 24,
    EJOC_SUBBANDS = 64,
    EJOC_CORE_CHANNELS = 5,
    EJOC_OUTPUT_CHANNELS = 16,
    EJOC_MAX_OBJECTS = 15,
    EJOC_MAX_DPOINTS = 2,
    EJOC_MAX_PARAMETER_BANDS = 23,
    EJOC_SPEAKER_BLOCK_SAMPLES = 32,
    EJOC_SPEAKER_COORDINATES = 3
};

typedef void* ejoc_renderer_handle;
typedef void* ejoc_speaker_renderer_handle;

/*
Fixed array layouts used by ejoc_renderer_process():
  bed5_planar  [5][1536]
  lfe          [1536] or NULL
  n_bands      [15]
  n_dpoints    [15]
  slope_idx    [15]
  offset_ts    [15][2]
  dq           [15][2][5][23]
  output16     [16][1536]

Only objects selected by object_mask are read from the descriptor arrays.
Sparse JOC must be rejected by the caller; this ABI accepts already dequantized
dense matrix coefficients.
*/

EJOC_API uint32_t EJOC_CALL ejoc_abi_version(void);
EJOC_API const char* EJOC_CALL ejoc_build_info(void);
EJOC_API ejoc_renderer_handle EJOC_CALL ejoc_renderer_create(void);
EJOC_API void EJOC_CALL ejoc_renderer_destroy(ejoc_renderer_handle handle);
EJOC_API int EJOC_CALL ejoc_renderer_reset(ejoc_renderer_handle handle);
EJOC_API int EJOC_CALL ejoc_renderer_set_threads(ejoc_renderer_handle handle, uint32_t total_threads);
EJOC_API uint32_t EJOC_CALL ejoc_renderer_thread_count(ejoc_renderer_handle handle);
EJOC_API const char* EJOC_CALL ejoc_renderer_last_error(ejoc_renderer_handle handle);

EJOC_API int EJOC_CALL ejoc_renderer_process(
    ejoc_renderer_handle handle,
    const float* bed5_planar,
    const float* lfe,
    uint32_t object_mask,
    const uint8_t* n_bands,
    const uint8_t* n_dpoints,
    const uint8_t* slope_idx,
    const uint8_t* offset_ts,
    const double* dq,
    double clipgain,
    float phase_new,
    float output_scale,
    float* output16_planar);

/*
High-precision object-to-speaker renderer.

The renderer consumes interleaved float32 input PCM arranged as:
  objects16_interleaved [sample_count][16]
where channel 0 is LFE and channels 1..15 are point objects. All spatial
calculations, gain ramps, and accumulation use double. Output is interleaved:
  output_interleaved [sample_count][layout_channel_count]

Each metadata entry is a complete object-state snapshot:
  metadata_offsets       [metadata_count], relative to this process call
  ramp_durations         [metadata_count], in samples
  positions_q15          [metadata_count][15][3]
  region_indices         [metadata_count][15] or NULL (all region 0)
  height_enabled         [metadata_count][15] or NULL (all enabled)
  object_gains           [metadata_count][15] or NULL (all 1.0)

metadata_offsets must be nondecreasing and <= sample_count. sample_count must
be a multiple of EJOC_SPEAKER_BLOCK_SAMPLES. State and unfinished ramps are
preserved across calls.
*/
EJOC_API uint32_t EJOC_CALL ejoc_speaker_layout_channel_count(uint32_t speaker_bitfield);
EJOC_API ejoc_speaker_renderer_handle EJOC_CALL ejoc_speaker_renderer_create(uint32_t speaker_bitfield);
EJOC_API void EJOC_CALL ejoc_speaker_renderer_destroy(ejoc_speaker_renderer_handle handle);
EJOC_API int EJOC_CALL ejoc_speaker_renderer_reset(ejoc_speaker_renderer_handle handle);
EJOC_API const char* EJOC_CALL ejoc_speaker_renderer_last_error(ejoc_speaker_renderer_handle handle);
EJOC_API int EJOC_CALL ejoc_speaker_renderer_process(
    ejoc_speaker_renderer_handle handle,
    const float* objects16_interleaved,
    uint32_t sample_count,
    uint32_t metadata_count,
    const uint32_t* metadata_offsets,
    const uint32_t* ramp_durations,
    const uint16_t* positions_q15,
    const uint8_t* region_indices,
    const uint8_t* height_enabled,
    const double* object_gains,
    double* output_interleaved);

#ifdef __cplusplus
}
#endif
