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
    EJOC_SPEAKER_COORDINATES = 3,
    EJOC_BINAURAL_BLOCK_SAMPLES = 512,
    EJOC_BINAURAL_INPUT_CHANNELS = 16,
    EJOC_BINAURAL_OUTPUT_CHANNELS = 2,
    EJOC_BINAURAL_QMF_BANDS = 64,
    EJOC_BINAURAL_HYBRID_BANDS = 77
};

typedef void* ejoc_renderer_handle;
typedef void* ejoc_speaker_renderer_handle;
typedef void* ejoc_binaural_renderer_handle;

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

EJOC_API ejoc_binaural_renderer_handle EJOC_CALL ejoc_binaural_renderer_create(void);
EJOC_API void EJOC_CALL ejoc_binaural_renderer_destroy(ejoc_binaural_renderer_handle handle);
EJOC_API int EJOC_CALL ejoc_binaural_renderer_reset(ejoc_binaural_renderer_handle handle);
EJOC_API const char* EJOC_CALL ejoc_binaural_renderer_last_error(
    ejoc_binaural_renderer_handle handle);
EJOC_API int EJOC_CALL ejoc_binaural_renderer_configure_kernels(
    ejoc_binaural_renderer_handle handle,
    const double* qmf_analysis,
    const double* hybrid_low,
    const int16_t* hybrid_indices,
    const double* hybrid_values,
    uint32_t hybrid_count,
    const double* qmf_basis,
    const double* qmf_taps);
EJOC_API int EJOC_CALL ejoc_binaural_renderer_configure_room(
    ejoc_binaural_renderer_handle handle,
    uint32_t bands,
    uint32_t allpass_count,
    const uint32_t* allpass_delays,
    const double* allpass_gains,
    const uint32_t* fdn_delays,
    const double* fdn_matrix,
    uint32_t output_tap_delay,
    const double* feedback_complex,
    const double* output_taps,
    const double* output_complex,
    uint32_t extra_count,
    const uint32_t* extra_delays,
    const double* extra_fields_complex,
    const double* extra_matrices);
EJOC_API int EJOC_CALL ejoc_binaural_renderer_process(
    ejoc_binaural_renderer_handle handle,
    const double* input16_interleaved,
    const double* gains_complex,
    const double* room_sends,
    double output_gain,
    double* output_stereo_interleaved);

/*
Native SOFA binaural renderer.

The handle owns the complete runtime: 64-QMF/77-hybrid analysis and synthesis,
fifth-order ACN/N3D real spherical-harmonic direction-field evaluation,
per-object whole-QMF-slot delay histories, six first-order image-source early
reflections, the shared unitary-FDN late room, the 120-180 Hz LFE low-pass and
the 961-sample latency compensation.  The caller configures the filterbank
tables, the compiled HRTF field and the room constants once, then per 512-sample
block updates every source with ejoc_sofa_binaural_set_source() and calls
ejoc_sofa_binaural_process().  Process returns the number of trimmed stereo
samples written; the first 961 processed samples across calls are discarded.
*/
typedef void* ejoc_sofa_binaural_handle;

EJOC_API ejoc_sofa_binaural_handle EJOC_CALL ejoc_sofa_binaural_create(void);
EJOC_API void EJOC_CALL ejoc_sofa_binaural_destroy(ejoc_sofa_binaural_handle handle);
EJOC_API int EJOC_CALL ejoc_sofa_binaural_reset(ejoc_sofa_binaural_handle handle);
EJOC_API const char* EJOC_CALL ejoc_sofa_binaural_last_error(
    ejoc_sofa_binaural_handle handle);
EJOC_API int EJOC_CALL ejoc_sofa_binaural_configure_kernels(
    ejoc_sofa_binaural_handle handle,
    const double* qmf_analysis,
    const double* hybrid_low,
    const int16_t* hybrid_indices,
    const double* hybrid_values,
    uint32_t hybrid_count,
    const double* qmf_basis,
    const double* qmf_taps);
EJOC_API int EJOC_CALL ejoc_sofa_binaural_configure_field(
    ejoc_sofa_binaural_handle handle,
    const double* coefficients,
    const double* delay_coefficients,
    const double* delay_bounds,
    const double* band_centers,
    double measurement_radius_m);
EJOC_API int EJOC_CALL ejoc_sofa_binaural_configure_room(
    ejoc_sofa_binaural_handle handle,
    const double* room_dims,
    const double* listener_pos,
    const double* wall_gains,
    double speed_of_sound,
    const uint32_t* fdn_delays,
    const double* fdn_feedback,
    double damping,
    double fdn_output_gain,
    const uint32_t* allpass_delays,
    const double* allpass_gains,
    uint32_t enable_early_reflections,
    uint32_t enable_late_room);
EJOC_API int EJOC_CALL ejoc_sofa_binaural_set_source(
    ejoc_sofa_binaural_handle handle,
    uint32_t source,
    const double* position_adm,
    uint32_t profile,
    double gain,
    uint32_t enabled,
    uint32_t special_lfe,
    uint32_t fade);
EJOC_API int EJOC_CALL ejoc_sofa_binaural_process(
    ejoc_sofa_binaural_handle handle,
    const double* input16_interleaved,
    uint32_t sample_count,
    double output_gain,
    double* output_stereo_interleaved);
EJOC_API int EJOC_CALL ejoc_sofa_binaural_finish(
    ejoc_sofa_binaural_handle handle,
    uint32_t flush_samples,
    double* output_stereo_interleaved,
    uint32_t capacity);

#ifdef __cplusplus
}
#endif
