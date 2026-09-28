#include <cstring>
#include <string>

#include "foundation/status.h"
#include "joc_core.h"
#include "joc_stream.h"
#include "stream/stream.h"

struct joc_stream {
    joc::stream::Stream instance;
};

namespace {

joc_error finish_stream(const joc::Status& status) {
    return status.code();
}

joc::stream::Config to_config(const joc_stream_config& config) {
    joc::stream::Config out;
    out.input = config.input;
    out.output = config.output;
    if (config.speaker_layout_name != nullptr) { out.layout = config.speaker_layout_name; }
    if (config.speaker_metadata_offset != 0u) {
        out.metadata_offset = config.speaker_metadata_offset;
    }
    out.binaural_mode = config.binaural_mode != 0u ? config.binaural_mode : JOC_BINAURAL_MID;
    if (config.hrtf_path != nullptr) { out.hrtf_path = config.hrtf_path; }
    if (config.kernels_path != nullptr) { out.kernels_path = config.kernels_path; }
    if (config.binaural_tail_seconds > 0.0) { out.tail_seconds = config.binaural_tail_seconds; }
    if (config.object_delay_samples != 0u) {
        out.object_delay_samples = config.object_delay_samples;
    }
    out.gain_db = config.gain_db;
    out.native_threads = config.native_threads;
    // The binaural HRTF inputs of joc_task_config, copied with the same defaults:
    // the policy is taken verbatim (0 is "none", a real choice, not "unset") and
    // the radius keeps its documented default of 1.0 when the field is not set.
    if (config.hrtf_sofa_path != nullptr) { out.hrtf_sofa_path = config.hrtf_sofa_path; }
    if (config.personalized_headphone_path != nullptr) {
        out.personalized_headphone_path = config.personalized_headphone_path;
    }
    if (config.hrtf_cache_dir != nullptr) { out.hrtf_cache_dir = config.hrtf_cache_dir; }
    out.hrtf_cache_policy = config.hrtf_cache_policy;
    if (config.hrtf_radius_m > 0.0) { out.hrtf_radius_m = config.hrtf_radius_m; }
    return out;
}

}  // namespace

extern "C" {

joc_error JOC_CALL joc_stream_create(const joc_stream_config* config, joc_stream** out) {
    if (config == nullptr || out == nullptr || config->struct_size != sizeof(joc_stream_config)) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    auto* stream = new (std::nothrow) joc_stream();
    if (stream == nullptr) {
        return JOC_ERR_OUT_OF_MEMORY;
    }
    const joc::Status status = stream->instance.create(to_config(*config));
    if (!status.ok()) {
        delete stream;
        return finish_stream(status);
    }
    *out = stream;
    return JOC_OK;
}

joc_error JOC_CALL joc_stream_push(joc_stream* stream, const joc_stream_buffer* input,
                                   std::uint32_t* consumed_samples, std::uint32_t* consumed_bytes) {
    if (stream == nullptr || input == nullptr ||
        input->struct_size != sizeof(joc_stream_buffer)) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    const joc::Status status = [&] {
        if (input->kind == JOC_STREAM_IN_EAC3) {
            std::size_t consumed = 0;
            const joc::Status pushed =
                stream->instance.push_eac3(input->bytes, input->byte_count, &consumed);
            if (consumed_bytes != nullptr) {
                *consumed_bytes = static_cast<std::uint32_t>(consumed);
            }
            return pushed;
        }
        if (input->kind == JOC_STREAM_IN_PCM_OBJECTS16) {
            std::size_t consumed = 0;
            const joc::Status pushed =
                stream->instance.push_objects16(input->pcm, input->sample_count, &consumed);
            if (consumed_samples != nullptr) {
                *consumed_samples = static_cast<std::uint32_t>(consumed);
            }
            return pushed;
        }
        std::size_t consumed = 0;
        const joc::Status pushed =
            stream->instance.push_bed(input->pcm, input->sample_count, &consumed);
        if (consumed_samples != nullptr) {
            *consumed_samples = static_cast<std::uint32_t>(consumed);
        }
        return pushed;    }();
    return finish_stream(status);
}

joc_error JOC_CALL joc_stream_pull(joc_stream* stream, joc_stream_buffer* output,
                                   std::uint32_t* produced_samples) {
    if (stream == nullptr || output == nullptr || output->out_pcm == nullptr ||
        output->struct_size != sizeof(joc_stream_buffer)) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    std::size_t produced = 0;
    const joc::Status status =
        stream->instance.pull(output->out_pcm, output->sample_count, &produced);
    if (!status.ok()) {
        return finish_stream(status);
    }
    if (produced_samples != nullptr) {
        *produced_samples = static_cast<std::uint32_t>(produced);
    }
    output->sample_count = static_cast<std::uint32_t>(produced);
    return JOC_OK;
}

joc_error JOC_CALL joc_stream_flush(joc_stream* stream) {
    if (stream == nullptr) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    return finish_stream(stream->instance.flush());
}

joc_error JOC_CALL joc_stream_reset(joc_stream* stream) {
    if (stream == nullptr) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    return finish_stream(stream->instance.reset());
}

joc_error JOC_CALL joc_stream_status(const joc_stream* stream, joc_stream_status_info* out) {
    if (stream == nullptr || out == nullptr ||
        out->struct_size != sizeof(joc_stream_status_info)) {
        return JOC_ERR_INVALID_ARGUMENT;
    }
    const joc::stream::Info& info = stream->instance.info();
    out->frames_in = info.frames_in;
    out->frames_out = info.frames_out;
    out->samples_in = info.samples_in;
    out->samples_out = info.samples_out;
    out->bytes_in = info.bytes_in;
    out->buffered_samples = stream->instance.buffered_samples();
    out->oamd_payloads = info.oamd_payloads;
    out->oamd_transitions = info.oamd_transitions;
    out->output_channels = info.output_channels;
    out->ended = info.ended;
    return JOC_OK;
}

joc_error JOC_CALL joc_stream_destroy(joc_stream* stream) {
    delete stream;
    return JOC_OK;
}

}  // extern "C"
