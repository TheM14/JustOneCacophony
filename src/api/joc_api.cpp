
#include "joc_core.h"

#include <cstring>
#include <string>

#include "eac3_transport/eac3_reader.h"
#include "emdf/emdf_parser.h"
#include "foundation/status.h"
#include "joc_bitstream/joc_parser.h"

namespace {

thread_local std::string g_detail;

joc_error finish(const joc::Status& status) {
    if (status.ok()) {
        g_detail.clear();
        return JOC_OK;
    }
    g_detail.assign(status.stage());
    g_detail.append(": ");
    g_detail.append(status.message());
    return status.code();
}

joc_error arg_fail(const char* message) {
    return finish(joc::Status::fail(JOC_ERR_INVALID_ARGUMENT, "core", message));
}

void fill_emdf_info(const joc::emdf::Container& container, joc_emdf_info* out) {
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(joc_emdf_info);
    out->struct_version = JOC_EMDF_INFO_VERSION;
    out->start_bit = static_cast<std::uint32_t>(container.start_bit);
    out->container_bytes = static_cast<std::uint32_t>(container.raw_size);
    out->payload_count = static_cast<std::uint32_t>(container.payload_count);
    for (std::size_t i = 0; i < container.payload_count; ++i) {
        out->payloads[i].id = container.payloads[i].id;
        out->payloads[i].sample_offset = container.payloads[i].sample_offset;
        out->payloads[i].bit_offset = static_cast<std::uint32_t>(container.payloads[i].bit_offset);
        out->payloads[i].size = static_cast<std::uint32_t>(container.payloads[i].size);
    }
}

}  // namespace

extern "C" {

std::uint32_t JOC_CALL joc_abi_version(void) { return JOC_ABI_VERSION; }

const char* JOC_CALL joc_version_string(void) { return "0.1.0-m1"; }

std::uint32_t JOC_CALL joc_event_size(void) { return static_cast<std::uint32_t>(sizeof(joc_event)); }
std::uint32_t JOC_CALL joc_task_config_size(void) {
    return static_cast<std::uint32_t>(sizeof(joc_task_config));
}
std::uint32_t JOC_CALL joc_task_result_size(void) {
    return static_cast<std::uint32_t>(sizeof(joc_task_result));
}

const char* JOC_CALL joc_build_info(void) {
    static const std::string info = [] {
        std::string text = "joc_core 0.1.0-m1 (";
#if defined(_MSC_VER)
        text += "msvc " + std::to_string(_MSC_VER);
#elif defined(__clang__)
        text += std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
        text += "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
        text += "unknown-compiler";
#endif
#if defined(_M_AMD64) || defined(__x86_64__)
        text += ", x64";
#elif defined(_M_ARM64) || defined(__aarch64__)
        text += ", arm64";
#elif defined(_M_IX86) || defined(__i386__)
        text += ", x86";
#endif
        text += ", c++";
        text += std::to_string(static_cast<long long>(__cplusplus / 100 % 100));
        text += ")";
        return text;
    }();
    return info.c_str();
}

const char* JOC_CALL joc_error_name(joc_error code) {
    switch (code) {
        case JOC_OK: return "JOC_OK";
        case JOC_ERR_INVALID_ARGUMENT: return "JOC_ERR_INVALID_ARGUMENT";
        case JOC_ERR_INVALID_CONFIG: return "JOC_ERR_INVALID_CONFIG";
        case JOC_ERR_OUT_OF_MEMORY: return "JOC_ERR_OUT_OF_MEMORY";
        case JOC_ERR_IO: return "JOC_ERR_IO";
        case JOC_ERR_UNSUPPORTED_PLATFORM: return "JOC_ERR_UNSUPPORTED_PLATFORM";
        case JOC_ERR_LIBRARY_MISSING: return "JOC_ERR_LIBRARY_MISSING";
        case JOC_ERR_INPUT_NOT_FOUND: return "JOC_ERR_INPUT_NOT_FOUND";
        case JOC_ERR_INPUT_FORMAT: return "JOC_ERR_INPUT_FORMAT";
        case JOC_ERR_EAC3_SYNCFRAME: return "JOC_ERR_EAC3_SYNCFRAME";
        case JOC_ERR_EMDF_TRANSPORT: return "JOC_ERR_EMDF_TRANSPORT";
        case JOC_ERR_EMDF_SYNTAX: return "JOC_ERR_EMDF_SYNTAX";
        case JOC_ERR_JOC_SYNTAX: return "JOC_ERR_JOC_SYNTAX";
        case JOC_ERR_JOC_UNSUPPORTED_VARIANT: return "JOC_ERR_JOC_UNSUPPORTED_VARIANT";
        case JOC_ERR_OAMD_SYNTAX: return "JOC_ERR_OAMD_SYNTAX";
        case JOC_ERR_OAMD_UNSUPPORTED_VARIANT: return "JOC_ERR_OAMD_UNSUPPORTED_VARIANT";
        case JOC_ERR_BITSTREAM_TRUNCATED: return "JOC_ERR_BITSTREAM_TRUNCATED";
        case JOC_ERR_BITSTREAM_PADDING: return "JOC_ERR_BITSTREAM_PADDING";
        case JOC_ERR_HRTF_NOT_FOUND: return "JOC_ERR_HRTF_NOT_FOUND";
        case JOC_ERR_HRTF_FORMAT: return "JOC_ERR_HRTF_FORMAT";
        case JOC_ERR_HRTF_VERSION: return "JOC_ERR_HRTF_VERSION";
        case JOC_ERR_HRTF_HASH: return "JOC_ERR_HRTF_HASH";
        case JOC_ERR_HRTF_UNSUPPORTED_CONVENTION: return "JOC_ERR_HRTF_UNSUPPORTED_CONVENTION";
        case JOC_ERR_LAYOUT_UNSUPPORTED: return "JOC_ERR_LAYOUT_UNSUPPORTED";
        case JOC_ERR_RENDER_FAILED: return "JOC_ERR_RENDER_FAILED";
        case JOC_ERR_OUTPUT_OPEN: return "JOC_ERR_OUTPUT_OPEN";
        case JOC_ERR_OUTPUT_WRITE: return "JOC_ERR_OUTPUT_WRITE";
        case JOC_ERR_OUTPUT_CLIP_ABORT: return "JOC_ERR_OUTPUT_CLIP_ABORT";
        case JOC_ERR_ADM_VALIDATION: return "JOC_ERR_ADM_VALIDATION";
        case JOC_ERR_CANCELLED: return "JOC_ERR_CANCELLED";
        case JOC_ERR_STATE: return "JOC_ERR_STATE";
        case JOC_ERR_NOT_SUPPORTED: return "JOC_ERR_NOT_SUPPORTED";
        case JOC_ERR_INTERNAL: return "JOC_ERR_INTERNAL";
        default: return "JOC_ERR_UNKNOWN";
    }
}

const char* JOC_CALL joc_error_stage(joc_error code) {
    switch (code) {
        case JOC_ERR_EAC3_SYNCFRAME:
        case JOC_ERR_INPUT_NOT_FOUND:
        case JOC_ERR_INPUT_FORMAT:
            return "eac3_transport";
        case JOC_ERR_EMDF_TRANSPORT:
        case JOC_ERR_EMDF_SYNTAX:
            return "emdf";
        case JOC_ERR_JOC_SYNTAX:
        case JOC_ERR_JOC_UNSUPPORTED_VARIANT:
            return "joc";
        case JOC_ERR_OAMD_SYNTAX:
        case JOC_ERR_OAMD_UNSUPPORTED_VARIANT:
            return "oamd";
        case JOC_ERR_BITSTREAM_TRUNCATED:
        case JOC_ERR_BITSTREAM_PADDING:
            return "bitstream";
        case JOC_ERR_HRTF_NOT_FOUND:
        case JOC_ERR_HRTF_FORMAT:
        case JOC_ERR_HRTF_VERSION:
        case JOC_ERR_HRTF_HASH:
        case JOC_ERR_HRTF_UNSUPPORTED_CONVENTION:
            return "hrtf";
        case JOC_ERR_LAYOUT_UNSUPPORTED:
        case JOC_ERR_RENDER_FAILED:
            return "render";
        case JOC_ERR_OUTPUT_OPEN:
        case JOC_ERR_OUTPUT_WRITE:
        case JOC_ERR_OUTPUT_CLIP_ABORT:
        case JOC_ERR_ADM_VALIDATION:
            return "output";
        case JOC_ERR_CANCELLED:
        case JOC_ERR_STATE:
        case JOC_ERR_NOT_SUPPORTED:
        case JOC_ERR_INTERNAL:
            return "task";
        default:
            return "core";
    }
}

const char* JOC_CALL joc_last_error_detail(void) { return g_detail.c_str(); }

joc_error JOC_CALL joc_parse_id14(const std::uint8_t* payload, std::size_t payload_size,
                                 joc_frame_params* out_params) {
    if (payload == nullptr || out_params == nullptr || payload_size == 0) {
        return arg_fail("joc_parse_id14 requires a non-empty payload and an output struct");
    }
    return finish(joc::joc::parse_id14(payload, payload_size, out_params, nullptr));
}

joc_error JOC_CALL joc_parse_eac3_frame(const std::uint8_t* frame, std::size_t frame_size,
                                       joc_frame_params* out_params, joc_emdf_info* out_emdf) {
    if (frame == nullptr || out_params == nullptr || frame_size == 0) {
        return arg_fail("joc_parse_eac3_frame requires a frame and an output struct");
    }
    joc::emdf::Container container;
    const joc::Status status =
        joc::joc::parse_eac3_frame(frame, frame_size, out_params, &container, nullptr);
    if (!status.ok()) {
        return finish(status);
    }
    if (out_emdf != nullptr) {
        fill_emdf_info(container, out_emdf);
    }
    return JOC_OK;
}

joc_error JOC_CALL joc_extract_payload(const std::uint8_t* frame, std::size_t frame_size,
                                      const joc_emdf_payload_info* payload, std::uint8_t* out,
                                      std::size_t out_capacity, std::size_t* out_size) {
    if (frame == nullptr || payload == nullptr || out_size == nullptr) {
        return arg_fail("joc_extract_payload requires frame, payload and out_size");
    }
    *out_size = payload->size;
    if (out == nullptr) {
        return JOC_OK;
    }
    if (out_capacity < payload->size) {
        return arg_fail("joc_extract_payload output buffer too small");
    }
    joc::emdf::Payload entry;
    entry.id = payload->id;
    entry.sample_offset = payload->sample_offset;
    entry.bit_offset = payload->bit_offset;
    entry.size = payload->size;
    std::vector<std::uint8_t> bytes;
    const joc::Status status = joc::emdf::extract_payload_bytes(frame, frame_size, entry, &bytes);
    if (!status.ok()) {
        return finish(status);
    }
    if (!bytes.empty()) {
        std::memcpy(out, bytes.data(), bytes.size());
    }
    *out_size = bytes.size();
    return JOC_OK;
}

joc_error JOC_CALL joc_eac3_frame_bytes(const std::uint8_t* data, std::size_t size, std::size_t offset,
                                       std::size_t* out_frame_bytes) {
    if (data == nullptr || out_frame_bytes == nullptr) {
        return arg_fail("joc_eac3_frame_bytes requires data and out_frame_bytes");
    }
    const joc_error code = joc::eac3::FrameReader::frame_bytes(data, size, offset, out_frame_bytes);
    if (code != JOC_OK) {
        return finish(joc::Status::fail(code, joc::stage::kEac3,
                                        "invalid or truncated E-AC-3 syncframe at byte " +
                                            std::to_string(offset)));
    }
    return JOC_OK;
}

joc_error JOC_CALL joc_check_id14_padding(const std::uint8_t* payload, std::size_t payload_size,
                                         std::uint32_t* out_trailing_bits) {
    if (payload == nullptr || payload_size == 0) {
        return arg_fail("joc_check_id14_padding requires a payload");
    }
    return finish(joc::joc::check_id14_padding(payload, payload_size, out_trailing_bits));
}

}  // extern "C"
