/*
 * joc_core.h -- JustOneCacophony C++ Core, public ABI.  Pure C.
 *
 * This header is the single authoritative definition of the Core's public
 * parameter/error surface.  Frontends (thin Python CLI, joc_dump, foobar2000,
 * MPV, FFmpeg) only ever fill these POD structs and read these POD results;
 * no audio data, no internal DSP concept, and no Python type crosses this
 * boundary.
 *
 * Stability tiers
 * ---------------
 *  [T1] host tier
 *       joc_abi_version / joc_version_string / joc_build_info, joc_error,
 *       joc_error_name / joc_error_stage, joc_last_error_detail.
 *       Stable, versioned, safe for media hosts.
 *
 *  [T2] bitstream / verification tier
 *       joc_parse_eac3_frame, joc_parse_id14, joc_frame_params, joc_emdf_info.
 *       These deliberately expose the dequantized JOC matrix coefficients so the
 *       bitstream front-end can be validated bit-exactly and driven by tooling
 *       (joc_dump) and A/B harnesses.  Media hosts must NOT use this tier; they
 *       use the task/stream API (added in later milestones).
 *
 * Conventions
 * -----------
 *  * every struct's first two fields are struct_size / struct_version;
 *  * all arrays are fixed size and POD, no pointers, no allocation;
 *  * every function returns joc_error (JOC_OK == 0);
 *  * joc_last_error_detail() returns a thread-local message that stays valid
 *    until the next Core call on the same thread.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>

#define JOC_ABI_VERSION 3u
#define JOC_FRAME_PARAMS_VERSION 1u
#define JOC_EMDF_INFO_VERSION 1u
#define JOC_TASK_CONFIG_VERSION 3u
#define JOC_TASK_RESULT_VERSION 2u
#define JOC_EVENT_VERSION 1u

/* JOC_STATIC: this branch exists for building the sources directly into an application, where nothing is imported or exported. */
#if defined(JOC_STATIC)
  #define JOC_API
  #define JOC_CALL __cdecl
#elif defined(_WIN32)
  #if defined(JOC_BUILD_DLL)
    #define JOC_API __declspec(dllexport)
  #else
    #define JOC_API __declspec(dllimport)
  #endif
  #define JOC_CALL __cdecl
#else
  #define JOC_API __attribute__((visibility("default")))
  #define JOC_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Fixed layout constants (single source of truth for every frontend)  */
/* ------------------------------------------------------------------ */
enum {
    JOC_FRAME_SAMPLES = 1536,          /* E-AC-3 frame = 24 * 64          */
    JOC_TIMESLOTS = 24,
    JOC_SUBBANDS = 64,
    JOC_CORE_CHANNELS = 5,             /* L R C Ls Rs (JOC order)         */
    JOC_MAX_CORE_CHANNELS = 7,         /* dmx_config_idx 1/2/4 declare 7  */
    JOC_OUTPUT_CHANNELS = 16,          /* ch0 = LFE, ch1..15 = objects    */
    JOC_MAX_OBJECTS = 15,
    JOC_MAX_DPOINTS = 2,
    JOC_MAX_PARAMETER_BANDS = 23,
    JOC_MAX_EMDF_PAYLOADS = 16,
    JOC_SPEAKER_BLOCK_SAMPLES = 32,
    JOC_BINAURAL_BLOCK_SAMPLES = 512,
    JOC_BINAURAL_QMF_BANDS = 64,
    JOC_BINAURAL_HYBRID_BANDS = 77,
    JOC_QMF_HOP_SAMPLES = 64,
    JOC_BINAURAL_LATENCY_SAMPLES = 961,
    JOC_LFE_DELAY_SAMPLES = 1217
};

/* ------------------------------------------------------------------ */
/* [T1] library / error surface                                        */
/* ------------------------------------------------------------------ */
typedef enum joc_error {
    JOC_OK = 0,
    JOC_ERR_INVALID_ARGUMENT,
    JOC_ERR_INVALID_CONFIG,
    JOC_ERR_OUT_OF_MEMORY,
    JOC_ERR_IO,
    JOC_ERR_UNSUPPORTED_PLATFORM,
    JOC_ERR_LIBRARY_MISSING,
    /* input / bitstream */
    JOC_ERR_INPUT_NOT_FOUND,
    JOC_ERR_INPUT_FORMAT,
    JOC_ERR_EAC3_SYNCFRAME,
    JOC_ERR_EMDF_TRANSPORT,
    JOC_ERR_EMDF_SYNTAX,
    JOC_ERR_JOC_SYNTAX,
    JOC_ERR_JOC_UNSUPPORTED_VARIANT,
    JOC_ERR_OAMD_SYNTAX,
    JOC_ERR_OAMD_UNSUPPORTED_VARIANT,
    JOC_ERR_BITSTREAM_TRUNCATED,
    JOC_ERR_BITSTREAM_PADDING,
    /* resources */
    JOC_ERR_HRTF_NOT_FOUND,
    JOC_ERR_HRTF_FORMAT,
    JOC_ERR_HRTF_VERSION,
    JOC_ERR_HRTF_HASH,
    JOC_ERR_HRTF_UNSUPPORTED_CONVENTION,
    /* rendering / output */
    JOC_ERR_LAYOUT_UNSUPPORTED,
    JOC_ERR_RENDER_FAILED,
    JOC_ERR_OUTPUT_OPEN,
    JOC_ERR_OUTPUT_WRITE,
    JOC_ERR_OUTPUT_CLIP_ABORT,
    JOC_ERR_ADM_VALIDATION,
    /* task / stream */
    JOC_ERR_CANCELLED,
    JOC_ERR_STATE,
    JOC_ERR_NOT_SUPPORTED,
    JOC_ERR_INTERNAL
} joc_error;

JOC_API uint32_t JOC_CALL joc_abi_version(void);
JOC_API const char* JOC_CALL joc_version_string(void);
JOC_API const char* JOC_CALL joc_build_info(void);
/* Struct sizes, so a binding can assert its layout matches the library instead of
 * assuming (mismatches are otherwise silent memory corruption). */
JOC_API uint32_t JOC_CALL joc_event_size(void);
JOC_API uint32_t JOC_CALL joc_task_config_size(void);
JOC_API uint32_t JOC_CALL joc_task_result_size(void);
JOC_API const char* JOC_CALL joc_error_name(joc_error code);
JOC_API const char* JOC_CALL joc_error_stage(joc_error code);
/* Thread-local structured detail for the most recent failing call. */
JOC_API const char* JOC_CALL joc_last_error_detail(void);

/* ------------------------------------------------------------------ */
/* [T2] bitstream / verification tier                                  */
/* ------------------------------------------------------------------ */

/* One EMDF payload directory entry.  bit_offset is the MSB-first bit
 * position of the payload's first byte inside the syncframe, exactly as the
 * EMDF container syntax defines it (payloads are not byte aligned in general). */
typedef struct joc_emdf_payload_info {
    uint8_t  id;
    uint8_t  reserved[3];
    uint16_t sample_offset;      /* EMDF outer smpoffst, 11 bits */
    uint16_t reserved2;
    uint32_t bit_offset;
    uint32_t size;               /* payload bytes */
} joc_emdf_payload_info;

typedef struct joc_emdf_info {
    uint32_t struct_size;
    uint32_t struct_version;
    uint32_t start_bit;          /* container syncword bit position */
    uint32_t container_bytes;    /* 4 + declared length */
    uint32_t payload_count;
    uint32_t reserved;
    joc_emdf_payload_info payloads[JOC_MAX_EMDF_PAYLOADS];
} joc_emdf_info;

/* Per-object ID14/JOC descriptor plus its dequantized matrix.
 * dq[dp][ch][pb] is zero filled outside [0,n_dpoints) x [0,n_channels) x
 * [0,n_bands).  Absent objects are entirely zero. */
typedef struct joc_object_params {
    uint8_t  present;
    uint8_t  num_bands_idx;
    uint8_t  n_bands;
    uint8_t  sparse;             /* 0 = dense (MTX), 1 = sparse (IDX+VEC) */
    uint8_t  quant_idx;          /* 0 = 96 levels, 1 = 192 levels        */
    uint8_t  slope_idx;          /* 0 = interpolate, 1 = step at offset_ts */
    uint8_t  num_dpoints_bits;
    uint8_t  n_dpoints;
    uint8_t  offset_ts[JOC_MAX_DPOINTS];
    uint8_t  reserved[2];
    double   dq[JOC_MAX_DPOINTS][JOC_MAX_CORE_CHANNELS][JOC_MAX_PARAMETER_BANDS];
} joc_object_params;

typedef struct joc_frame_params {
    uint32_t struct_size;
    uint32_t struct_version;
    uint8_t  dmx_config_idx;
    uint8_t  num_objects_bits;
    uint8_t  ext_config_idx;
    uint8_t  n_objects;
    uint8_t  n_channels;         /* 5 or 7 */
    uint8_t  clipgain_x_bits;
    uint8_t  clipgain_y_bits;
    uint8_t  reserved;
    uint32_t seq_count;          /* 10-bit JOC sequence counter, parsed only  */
    uint32_t present_mask;       /* bit i == object i present */
    uint32_t data_end_bits;      /* bit position just after joc_data          */
    uint32_t trailing_bits;      /* bits left after joc_data (padding/ext)    */
    uint8_t  tail_bytes[8];      /* first up to 8 trailing bytes, for A/B     */
    double   clipgain;           /* 1 + (y/32) * 2^(x-4), bit-exact           */
    joc_object_params objects[JOC_MAX_OBJECTS];
} joc_frame_params;

/* Parse an EMDF ID14 (JOC) payload. */
JOC_API joc_error JOC_CALL joc_parse_id14(const uint8_t* payload, size_t payload_size,
                                          joc_frame_params* out_params);

/* Locate the contiguous JOC EMDF container inside one E-AC-3 syncframe and
 * parse its ID14 payload.  out_emdf may be NULL. */
JOC_API joc_error JOC_CALL joc_parse_eac3_frame(const uint8_t* frame, size_t frame_size,
                                                joc_frame_params* out_params,
                                                joc_emdf_info* out_emdf);

/* Copy one payload's bytes out of a syncframe (MSB-first bit extraction, which
 * equals a memcpy for byte-aligned containers).  out_size receives the payload
 * byte count; pass out == NULL to query only the size. */
JOC_API joc_error JOC_CALL joc_extract_payload(const uint8_t* frame, size_t frame_size,
                                               const joc_emdf_payload_info* payload,
                                               uint8_t* out, size_t out_capacity,
                                               size_t* out_size);

/* E-AC-3 syncframe traversal: given the offset of a frame start, report its
 * byte length so a host can walk a bare E-AC-3 stream without duplicating
 * frmsiz logic.  Rejects resynchronisation (no silent recovery). */
JOC_API joc_error JOC_CALL joc_eac3_frame_bytes(const uint8_t* data, size_t size,
                                                size_t offset, size_t* out_frame_bytes);

/* Strict trailing-bit check for the JOC payload (A/B robustness corpus).
 * Returns JOC_ERR_BITSTREAM_PADDING when more than 7 bits are left over or the
 * leftover bits are not zero. */
JOC_API joc_error JOC_CALL joc_check_id14_padding(const uint8_t* payload, size_t payload_size,
                                                  uint32_t* out_trailing_bits);

/* ================================================================== */
/* [T1] host tier: task, telemetry and control                         */
/* ================================================================== */

/* ---- 1. events ---------------------------------------------------- */
/*
 * Events carry state only: frame/sample counters, stage, progress, statistics,
 * warnings, errors and paths.  Audio never travels through an event; a fixed
 * size POD with no pointers keeps that enforceable (see the static assertion in
 * the implementation).
 */
typedef enum joc_event_type {
    JOC_EV_TASK_STARTED = 0x0001,
    JOC_EV_TASK_STATE_CHANGED = 0x0002,
    JOC_EV_TASK_COMPLETED = 0x0003,
    JOC_EV_TASK_FAILED = 0x0004,
    JOC_EV_TASK_CANCELLED = 0x0005,
    JOC_EV_INPUT_OPENED = 0x0101,
    JOC_EV_METADATA_INDEXED = 0x0103,
    JOC_EV_HRTF_LOADED = 0x0110,
    JOC_EV_RENDERER_INITIALIZED = 0x0120,
    JOC_EV_PROGRESS = 0x0201,
    JOC_EV_STAGE_CHANGED = 0x0202,
    JOC_EV_JOC_FRAME_STATS = 0x0301,
    JOC_EV_OAMD_STATS = 0x0302,
    JOC_EV_OUTPUT_STATS = 0x0303,
    JOC_EV_OUTPUT_OPENED = 0x0401,
    JOC_EV_OUTPUT_FORMAT_DECIDED = 0x0402,
    JOC_EV_OUTPUT_FINALIZED = 0x0403,
    JOC_EV_LOG = 0x0501,
    JOC_EV_WARNING = 0x0502,
    JOC_EV_ERROR = 0x0503
} joc_event_type;

typedef enum joc_stage {
    JOC_STAGE_IDLE = 0,
    JOC_STAGE_INPUT = 1,
    JOC_STAGE_METADATA = 2,
    JOC_STAGE_DECODE = 3,
    JOC_STAGE_JOC = 4,
    JOC_STAGE_RENDER = 5,
    JOC_STAGE_OUTPUT = 6,
    JOC_STAGE_DONE = 7
} joc_stage;

enum { JOC_LOG_TRACE = 0, JOC_LOG_DEBUG = 1, JOC_LOG_INFO = 2, JOC_LOG_NOTICE = 3,
       JOC_LOG_WARNING = 4, JOC_LOG_ERROR = 5, JOC_LOG_FATAL = 6 };

typedef struct joc_event {
    uint32_t struct_size;
    uint32_t type;
    uint64_t sequence;
    uint64_t timestamp_us;
    uint64_t current_frame;
    uint64_t total_frames;
    uint64_t current_sample;
    uint64_t total_samples;
    uint32_t stage;
    uint32_t backend;
    double progress;              /* 0..1, -1 = unknown */
    double elapsed_seconds;
    double realtime_factor;
    uint64_t output_samples;
    uint64_t output_bytes;
    double output_duration_seconds;
    joc_error error_code;
    uint32_t log_level;
    char stage_name[32];
    char message[256];
} joc_event;

typedef void (JOC_CALL *joc_event_fn)(void* user, const joc_event* event);

typedef struct joc_event_sink {
    uint32_t struct_size;
    joc_event_fn callback;
    void* user;
    uint32_t min_type;   /* 0 = no filter */
    uint32_t max_type;   /* 0 = no filter */
} joc_event_sink;

/* ---- 2. cancellation ---------------------------------------------- */
typedef struct joc_cancel_token joc_cancel_token;

JOC_API joc_cancel_token* JOC_CALL joc_cancel_token_create(void);
JOC_API void JOC_CALL joc_cancel_token_request(joc_cancel_token* token);
JOC_API int32_t JOC_CALL joc_cancel_token_is_requested(const joc_cancel_token* token);
JOC_API void JOC_CALL joc_cancel_token_destroy(joc_cancel_token* token);

/* ---- 3. task configuration and results ---------------------------- */
typedef enum joc_operation {
    JOC_OP_ADM_BWF = 0,
    JOC_OP_SPEAKER = 1,
    JOC_OP_BINAURAL = 2
} joc_operation;

typedef enum joc_output_format {
    JOC_FORMAT_FLOAT32 = 0,
    JOC_FORMAT_PCM24 = 1
} joc_output_format;

typedef enum joc_binaural_mode {
    JOC_BINAURAL_OFF = 0,
    JOC_BINAURAL_NEAR = 1,
    JOC_BINAURAL_FAR = 2,
    JOC_BINAURAL_MID = 3
} joc_binaural_mode;

typedef enum joc_trajectory_mode {
    JOC_TRAJECTORY_COMPACT = 0,
    JOC_TRAJECTORY_DENSE64 = 1
} joc_trajectory_mode;

/* What to do when an int24 WAV would clip (peak outside [-1, 1]).  Mirrors the
 * reference CLI's --clip-action; ADM BWF output is always int24 and does not
 * consult this because no alternative format exists there. */
typedef enum joc_clip_action {
    JOC_CLIP_ASK = 0,      /* prompt on stdin; an error when stdin is not a terminal */
    JOC_CLIP_CONTINUE = 1, /* write int24, truncating out-of-range values */
    JOC_CLIP_FLOAT32 = 2,  /* switch the output to float32 */
    JOC_CLIP_ABORT = 3     /* fail the task */
} joc_clip_action;

/* Compiled-HRTF cache policy for a SOFA input; the .jochrtf itself is an
 * internal artifact of the compile step. */
typedef enum joc_hrtf_cache_policy {
    JOC_HRTF_CACHE_NONE = 0,   /* compile and discard */
    JOC_HRTF_CACHE_MEMORY = 1, /* compile and keep in this process (default) */
    JOC_HRTF_CACHE_DISK = 2    /* compile, reuse and write <cache_dir>/<name>.jochrtf */
} joc_hrtf_cache_policy;

enum { JOC_TASK_F_SKIP_SHA256 = 1u, JOC_TASK_F_KEEP_INTERMEDIATE = 2u,
       JOC_TASK_F_QUIET = 4u, JOC_TASK_F_METADATA_ONLY = 8u };

typedef struct joc_task_config {
    uint32_t struct_size;
    uint32_t struct_version;

    /* input */
    const char* input_path;        /* .eac3/.ec3/.m4a/... */
    const char* ffmpeg_path;       /* NULL = "ffmpeg" from PATH */
    const char* bed_path;          /* NULL = decode the core PCM with ffmpeg */
    const char* work_dir;          /* NULL = a temporary directory */
    double eac3_drc_scale;         /* 0 = DRC off (reference default) */
    int32_t eac3_target_level;     /* -31..0, 0 = not applied */

    /* output */
    uint32_t operation;
    const char* output_path;
    uint32_t output_format;        /* requested format for speaker/binaural */
    uint32_t flags;
    uint32_t clip_action;          /* joc_clip_action; JOC_CLIP_ASK by default */

    /* rendering */
    const char* speaker_layout_name;
    uint32_t speaker_metadata_offset;   /* default 1473 */
    uint32_t binaural_mode;
    const char* hrtf_path;              /* .jochrtf */
    const char* kernels_path;           /* rosella_kernels.npz */
    double binaural_tail_seconds;       /* default 5.0 */
    double binaural_tail_threshold;     /* binaural only; 0 disables trimming */
    uint32_t binaural_chunk_frames;     /* accepted for CLI parity; no effect */

    /* ADM */
    uint32_t adm_binaural_mode;         /* DBMD segment 10 encoding */
    uint32_t trajectory_mode;

    /* generic */
    uint32_t object_delay_samples;      /* default 1473 */
    double gain_db;                     /* default 0 */
    uint64_t duration_frames;           /* 0 = the whole stream */
    uint32_t progress_interval_frames;  /* default 1000 */
    uint32_t native_threads;            /* 0 = automatic */

    /* diagnostics */
    uint32_t print_metadata;            /* 0 none, 1 summary, 2 per frame */
    const char* metadata_json_path;     /* NULL = no JSON summary */

    joc_cancel_token* cancel;           /* optional */

    /* Binaural HRTF input (config version 2): when hrtf_sofa_path is set the
     * library compiles it with hrtf_cache_policy / hrtf_cache_dir / hrtf_radius_m
     * and hrtf_path is unused.  hrtf_path stays the advanced override that reads
     * a .jochrtf directly. */
    const char* hrtf_sofa_path;         /* SOFA SimpleFreeFieldHRIR input */
    const char* personalized_headphone_path; /* Rosella .personalized_headphone input */
    const char* hrtf_cache_dir;         /* disk policy directory */
    uint32_t hrtf_cache_policy;         /* joc_hrtf_cache_policy */
    uint32_t reserved0;
    double hrtf_radius_m;               /* SOFA measurement-radius shell */
} joc_task_config;

typedef enum joc_task_status {
    JOC_TASK_OK = 0,
    JOC_TASK_FAILED = 1,
    JOC_TASK_CANCELLED = 2
} joc_task_status;

typedef struct joc_task_result {
    uint32_t struct_size;
    uint32_t struct_version;
    uint32_t status;
    uint32_t error_code;
    char error_message[512];
    char error_stage[32];
    uint64_t input_frames;
    uint64_t output_samples;
    double duration_sec;
    uint32_t output_format_actual;
    double output_peak;
    uint64_t output_over_unity_values;
    uint64_t output_file_bytes;
    char output_sha256[65];       /* empty when skipped */
    uint64_t oamd_payloads;
    uint64_t oamd_transitions;
    double t_decode_bed;
    double t_render;
    double t_write;
    double t_total;
    double t_render_dsp;          /* speaker/binaural DSP calls only; 0 for ADM */
    double t_write_file;          /* disk writes including the finalize; >= t_write */
} joc_task_result;

typedef struct joc_validation_issue {
    joc_error code;
    uint32_t severity;            /* 0 = info, 1 = warning, 2 = error */
    char field[48];
    char message[256];
} joc_validation_issue;

/* ---- 4. entry points ---------------------------------------------- */
/* Fills `issues` (up to `capacity`) and reports how many were produced.
 * Returns JOC_OK when no *error*-severity issue was found. */
JOC_API joc_error JOC_CALL joc_task_validate(const joc_task_config* config,
                                             joc_validation_issue* issues, uint32_t capacity,
                                             uint32_t* count);

/* Runs the whole task on the calling thread.  `sink` may be NULL; `out` may be
 * NULL.  On cancellation the partial output is removed and JOC_ERR_CANCELLED is
 * returned with out->status = JOC_TASK_CANCELLED. */
JOC_API joc_error JOC_CALL joc_task_execute(const joc_task_config* config,
                                            const joc_event_sink* sink, joc_task_result* out);

/* Serialises the stable subset of the result as JSON (no environment fields -
 * those belong to the frontend).  `needed` receives the required size including
 * the terminator; a NULL buffer queries only the size. */
JOC_API joc_error JOC_CALL joc_task_result_to_json(const joc_task_result* result, char* buffer,
                                                   size_t capacity, size_t* needed);

/* The streaming (push/pull) tier lives in "joc_stream.h" so an embedder can
 * include the narrow surface without the bitstream/verification tier. */

#ifdef __cplusplus
} /* extern "C" */
#endif
