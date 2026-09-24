/*
 * joc_stream.h -- streaming (push/pull) interface of the JustOneCacophony core.
 *
 * This is the narrow, embedder-facing surface: a player or decoder component
 * includes only this header.  It is the same shared library as joc_core.h, split
 * so an integrator never has to see the bitstream/verification tier.
 *
 * A stream is a stateful instance for hosts that cannot wait for a whole file:
 * the caller feeds E-AC-3 bytes and the core PCM of the same frames (or already
 * rebuilt objects16) and pulls rendered PCM as soon as it is available.  It
 * mirrors the two library shapes a decoder library normally offers: this
 * push/pull form for host-owned I/O, and joc_task_execute() in joc_core.h for
 * library-owned file I/O.
 *
 * Contract:
 *   - all state is instance-private, so several streams coexist;
 *   - a stream is NOT thread safe: push and pull must come from one thread;
 *   - rendering is stateful (matrix interpolation, gain ramps, room tail), so a
 *     new position on the timeline requires decoding to continue from the start
 *     of the stream; there is no seek in this version;
 *   - the kernel latency is 961 samples for speaker/binaural output: the first
 *     pull after two 512-sample blocks, and flush() drains the binaural tail.
 */
#pragma once

#include "joc_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct joc_stream joc_stream;

typedef enum joc_stream_input {
    JOC_STREAM_IN_EAC3 = 0,          /* bare E-AC-3 syncframes (the metadata stream) */
    JOC_STREAM_IN_PCM_OBJECTS16 = 1, /* 16-channel objects16, decoded by the host   */
    JOC_STREAM_IN_CORE_PCM = 3       /* the 5.1 core PCM of the pushed E-AC-3 frames */
} joc_stream_input;

typedef enum joc_stream_output {
    JOC_STREAM_OUT_PCM_OBJECTS16 = 0, /* planar [16][samples] float32        */
    JOC_STREAM_OUT_SPEAKER = 1,       /* interleaved [samples][channels] f32 */
    JOC_STREAM_OUT_BINAURAL = 2       /* interleaved [samples][2] f32        */
} joc_stream_output;

typedef struct joc_stream_config {
    uint32_t struct_size;
    uint32_t struct_version;
    uint32_t input;
    uint32_t output;
    const char* speaker_layout_name;
    uint32_t speaker_metadata_offset; /* default 1473 */
    uint32_t binaural_mode;           /* near|mid|far; 0 means mid */
    const char* hrtf_path;            /* binaural only */
    const char* kernels_path;         /* binaural only */
    double binaural_tail_seconds;     /* default 5.0 */
    uint32_t object_delay_samples;    /* default 1473 */
    double gain_db;                   /* default 0 */
    uint32_t native_threads;
    uint32_t reserved;
} joc_stream_config;

typedef struct joc_stream_buffer {
    uint32_t struct_size;
    uint32_t struct_version;
    uint32_t kind;         /* which joc_stream_input/output this buffer carries */
    uint32_t channels;
    uint32_t sample_rate;
    uint32_t sample_count; /* in: capacity / out: produced (per channel) */
    uint32_t byte_count;   /* in: capacity / out: consumed or produced bytes */
    uint32_t reserved;
    const uint8_t* bytes;  /* EAC3 input */
    const float* pcm;      /* PCM input */
    uint8_t* out_bytes;    /* reserved for encoded outputs */
    float* out_pcm;        /* PCM output */
} joc_stream_buffer;

typedef struct joc_stream_status_info {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t frames_in;
    uint64_t frames_out;
    uint64_t samples_in;
    uint64_t samples_out;
    uint64_t bytes_in;
    uint64_t buffered_samples; /* rendered but not yet pulled, per channel */
    uint64_t oamd_payloads;
    uint64_t oamd_transitions;
    uint32_t output_channels;
    uint32_t ended; /* 1 after flush() */
} joc_stream_status_info;

JOC_API joc_error JOC_CALL joc_stream_create(const joc_stream_config* config, joc_stream** out);

/* Feeds one buffer.  Kind selects the path: EAC3 bytes, the core PCM of those
 * frames, or objects16.  Consumed counts are reported so a caller can resume
 * from a partial push. */
JOC_API joc_error JOC_CALL joc_stream_push(joc_stream* stream, const joc_stream_buffer* input,
                                           uint32_t* consumed_samples, uint32_t* consumed_bytes);

/* Copies as many rendered samples as fit into `output` (interleaved, or planar
 * for PCM_OBJECTS16) and reports how many were produced; 0 means "push more". */
JOC_API joc_error JOC_CALL joc_stream_pull(joc_stream* stream, joc_stream_buffer* output,
                                           uint32_t* produced_samples);

/* Marks the end of input and drains whatever the renderer still holds (the
 * binaural room tail); pull the remaining samples afterwards. */
JOC_API joc_error JOC_CALL joc_stream_flush(joc_stream* stream);

/* Returns the instance to its initial state (kernel, ramps, timeline, room,
 * parser, counters) so the same stream can be reused for another pass. */
JOC_API joc_error JOC_CALL joc_stream_reset(joc_stream* stream);

JOC_API joc_error JOC_CALL joc_stream_status(const joc_stream* stream,
                                             joc_stream_status_info* out);
JOC_API joc_error JOC_CALL joc_stream_destroy(joc_stream* stream);

#ifdef __cplusplus
} /* extern "C" */
#endif
