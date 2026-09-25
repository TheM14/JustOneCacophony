#include "task/task.h"
#include "foundation/fs_utf8.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "adm/adm_metadata.h"
#include "adm/adm_tracks.h"
#include "binaural/binaural_runtime.h"
#include "eac3_transport/eac3_reader.h"
#include "emdf/emdf_parser.h"
#include "foundation/sha256.h"
#include "hrtf/jochrtf.h"
#include "hrtf/rosella_model.h"
#include "hrtf/rosella_renderer.h"
#include "hrtf/sofa_cache.h"
#include "io/adm_writer.h"
#include "io/process.h"
#include "io/wav_writer.h"
#include "joc_bitstream/joc_parser.h"
#include "joc_core/objects16.h"
#include "oamd/oamd_parser.h"
#include "speaker/speaker_step.h"
#include "telemetry/event_bus.h"

namespace joc::task {

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kBedFrameBytes = static_cast<std::size_t>(JOC_FRAME_SAMPLES) * 6u * 4u;
constexpr std::size_t kReadChunk = 1u << 20;
constexpr double kDefaultTailThreshold = 1.0e-8;
constexpr int kCoreChannels[5] = {0, 1, 2, 4, 5};
constexpr int kLfeChannel = 3;

std::string lower(std::string text) {
    for (char& c : text) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return text;
}

bool ends_with(const std::string& text, const char* suffix) {
    const std::size_t length = std::strlen(suffix);
    return text.size() >= length && text.compare(text.size() - length, length, suffix) == 0;
}

std::string fmt(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return std::string(buffer);
}

struct Settings {
    std::string input_path;    std::string ffmpeg_path = "ffmpeg";
    std::string bed_path;
    std::string work_dir;
    std::string output_path;
    std::string layout;
    std::string hrtf_path;
    std::string kernels_path;
    std::string hrtf_sofa_path;
    std::string personalized_headphone_path;
    std::string hrtf_cache_dir;
    std::uint32_t hrtf_cache_policy = JOC_HRTF_CACHE_MEMORY;
    double hrtf_radius_m = 1.0;
    std::uint32_t operation = JOC_OP_ADM_BWF;
    std::uint32_t output_format = JOC_FORMAT_FLOAT32;
    std::uint32_t adm_binaural_mode = JOC_BINAURAL_MID;
    std::uint32_t trajectory_mode = JOC_TRAJECTORY_COMPACT;
    std::uint32_t object_delay_samples = 1473;
    std::uint32_t speaker_metadata_offset = 1473;
    std::uint32_t binaural_mode = JOC_BINAURAL_MID;
    std::uint32_t progress_interval = 1000;
    std::uint32_t native_threads = 0;
    std::uint32_t flags = 0;
    std::uint32_t clip_action = JOC_CLIP_ASK;
    std::uint32_t print_metadata = 0;
    std::string metadata_json_path;
    bool metadata_only = false;
    double gain_db = 0.0;
    double tail_seconds = 5.0;
    double tail_threshold = kDefaultTailThreshold;
    double drc_scale = 0.0;
    std::int32_t target_level = 0;
    std::uint64_t duration_frames = 0;
    const joc_cancel_token* cancel = nullptr;
};

Settings normalise(const joc_task_config& raw) {
    Settings cfg;
    if (raw.input_path != nullptr) { cfg.input_path = raw.input_path; }
    if (raw.ffmpeg_path != nullptr && raw.ffmpeg_path[0] != '\0') { cfg.ffmpeg_path = raw.ffmpeg_path; }
    if (raw.bed_path != nullptr) { cfg.bed_path = raw.bed_path; }
    if (raw.work_dir != nullptr) { cfg.work_dir = raw.work_dir; }
    if (raw.output_path != nullptr) { cfg.output_path = raw.output_path; }
    if (raw.speaker_layout_name != nullptr) { cfg.layout = raw.speaker_layout_name; }
    if (raw.hrtf_path != nullptr) { cfg.hrtf_path = raw.hrtf_path; }
    if (raw.hrtf_sofa_path != nullptr) { cfg.hrtf_sofa_path = raw.hrtf_sofa_path; }
    if (raw.personalized_headphone_path != nullptr) {
        cfg.personalized_headphone_path = raw.personalized_headphone_path;
    }
    if (raw.hrtf_cache_dir != nullptr) { cfg.hrtf_cache_dir = raw.hrtf_cache_dir; }
    cfg.hrtf_cache_policy = raw.hrtf_cache_policy;
    cfg.hrtf_radius_m = raw.hrtf_radius_m;
    if (raw.kernels_path != nullptr) { cfg.kernels_path = raw.kernels_path; }
    cfg.operation = raw.operation;
    cfg.output_format = raw.output_format;
    cfg.flags = raw.flags;
    cfg.clip_action = raw.clip_action;
    cfg.print_metadata = raw.print_metadata;
    cfg.metadata_only = (raw.flags & JOC_TASK_F_METADATA_ONLY) != 0u;
    if (raw.metadata_json_path != nullptr) { cfg.metadata_json_path = raw.metadata_json_path; }
    cfg.gain_db = raw.gain_db;
    cfg.drc_scale = raw.eac3_drc_scale;
    cfg.target_level = raw.eac3_target_level;
    cfg.duration_frames = raw.duration_frames;
    cfg.cancel = raw.cancel;
    // config still produces the reference behaviour (plan 26.1).
    if (raw.object_delay_samples != 0u) { cfg.object_delay_samples = raw.object_delay_samples; }
    if (raw.speaker_metadata_offset != 0u) {
        cfg.speaker_metadata_offset = raw.speaker_metadata_offset;
    }
    if (raw.progress_interval_frames != 0u) { cfg.progress_interval = raw.progress_interval_frames; }
    if (raw.binaural_tail_seconds > 0.0) { cfg.tail_seconds = raw.binaural_tail_seconds; }
    if (raw.binaural_tail_threshold > 0.0) { cfg.tail_threshold = raw.binaural_tail_threshold; }
    if (raw.native_threads != 0u) { cfg.native_threads = raw.native_threads; }
    if (raw.operation == JOC_OP_BINAURAL) {
        cfg.binaural_mode = raw.binaural_mode != 0u ? raw.binaural_mode : JOC_BINAURAL_MID;
        cfg.adm_binaural_mode = raw.adm_binaural_mode;
    } else {
        cfg.binaural_mode = raw.binaural_mode != 0u ? raw.binaural_mode : JOC_BINAURAL_MID;
        cfg.adm_binaural_mode =
            raw.adm_binaural_mode != 0u ? raw.adm_binaural_mode : JOC_BINAURAL_MID;
    }
    if (raw.trajectory_mode != 0u) { cfg.trajectory_mode = raw.trajectory_mode; }
    return cfg;
}

void add_issue(std::vector<joc_validation_issue>* issues, joc_error code, std::uint32_t severity,
               const char* field, const std::string& message) {
    if (issues == nullptr) {
        return;
    }
    joc_validation_issue issue{};
    issue.code = code;
    issue.severity = severity;
    std::snprintf(issue.field, sizeof(issue.field), "%s", field);
    std::snprintf(issue.message, sizeof(issue.message), "%s", message.c_str());
    issues->push_back(issue);
}

bool cancelled(const Settings& cfg) {
    return cfg.cancel != nullptr && joc_cancel_token_is_requested(cfg.cancel) != 0;
}

std::string create_temp_dir(const Settings& cfg) {    const fs::path base = cfg.work_dir.empty() ? fs::temp_directory_path()
                                               : fs_utf8::to_path(cfg.work_dir);
    const fs::path target = base / fmt("joc_task_%u", static_cast<unsigned>(std::random_device{}()));
    std::error_code error;
    fs::create_directories(target, error);
    return fs_utf8::from_path(target);
}

std::uint64_t count_frames(const std::string& path) {
    std::ifstream input = fs_utf8::open_input(path);
    if (!input) {
        return 0;
    }
    ::joc::eac3::FrameReader reader;
    std::vector<char> chunk(kReadChunk);
    std::uint64_t frames = 0;
    bool done = false;
    while (!done) {
        input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = input.gcount();
        if (got > 0) {
            reader.push(reinterpret_cast<const std::uint8_t*>(chunk.data()),
                        static_cast<std::size_t>(got));
        } else {
            reader.finish();
            done = true;
        }
        for (;;) {
            ::joc::eac3::Frame frame;
            const ::joc::eac3::FrameReader::Next state = reader.next(&frame);
            if (state == ::joc::eac3::FrameReader::Next::Ok) {
                ++frames;
                continue;
            }
            if (state == ::joc::eac3::FrameReader::Next::Fail) {
                return frames;
            }
            break;
        }
    }
    return frames;
}

bool stdin_is_interactive() {
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

// Rendered PCM is spooled to a float32 file so the output format can be decided
// once the peak is known, which is what makes --clip-action possible; the
// reference CLI spools for the same reason.
class PcmSpool {
public:
    ~PcmSpool() { close(); }

    Status open(const std::string& path, std::uint32_t channels, bool trim, double threshold) {
        path_ = path;
        channels_ = channels;
        trim_ = trim;
        threshold_ = threshold;
        // A spool left behind by a killed run would otherwise be appended to by the
        // "wb+" open below; only this exact path is ever removed.
        if (fs_utf8::exists(path)) {
            std::error_code ignored;
            std::filesystem::remove(fs_utf8::to_path(path), ignored);
        }
        file_ = fs_utf8::fopen_spool(path);
        if (file_ == nullptr) {
            return Status::fail(JOC_ERR_OUTPUT_OPEN, stage::kOutput, "cannot open " + path);
        }
        return Status::success();
    }

    // The declared sample budget; write() refuses to grow past it.
    void set_limit(std::uint64_t frames) { limit_frames_ = frames; }

    Status write(const float* interleaved, std::size_t frames) {
        if (frames == 0u || file_ == nullptr) {
            return Status::success();
        }
        // The declared budget is a hard stop: a renderer that overproduces must fail
        // here, not after writing an unbounded spool.
        if (limit_frames_ != 0u && position_ + frames > limit_frames_) {
            return Status::fail(JOC_ERR_RENDER_FAILED, stage::kOutput,
                                "binaural output exceeded its declared budget: " +
                                    std::to_string(position_ + frames) + " > " +
                                    std::to_string(limit_frames_) + " samples");
        }
        for (std::size_t index = 0; index < frames; ++index) {
            double per_sample = 0.0;
            for (std::uint32_t channel = 0; channel < channels_; ++channel) {
                const double value =
                    std::abs(static_cast<double>(interleaved[index * channels_ + channel]));
                per_sample = std::max(per_sample, value);
            }
            peak_ = std::max(peak_, per_sample);
            if (per_sample > 1.0) {
                ++clipped_;
            }
            if (trim_ && per_sample > threshold_) {
                last_above_ = position_ + index;
                any_above_ = true;
            }
        }
        const std::size_t count = frames * channels_;
        if (std::fwrite(interleaved, sizeof(float), count, file_) != count) {
            return Status::fail(JOC_ERR_OUTPUT_WRITE, stage::kOutput, "cannot write the PCM spool");
        }
        position_ += frames;
        return Status::success();
    }

    // Trailing near-silence is dropped, but never below `minimum` frames.
    std::uint64_t kept(std::uint64_t minimum) const {
        if (!trim_) {
            return position_;
        }
        const std::uint64_t last = any_above_ ? last_above_ + 1u : 0u;
        return std::min(position_, std::max(minimum, last));
    }

    // `report` is called with (frames written so far, total) so a long finalize is
    // visibly alive instead of looking hung.
    Status copy_to(io::WavWriter* writer, std::uint64_t keep,
                   const std::function<void(std::uint64_t, std::uint64_t)>& report = {}) {
        if (file_ == nullptr) {
            return Status::fail(JOC_ERR_STATE, stage::kOutput, "the PCM spool is closed");
        }
        std::fflush(file_);
        if (std::fseek(file_, 0, SEEK_SET) != 0) {
            return Status::fail(JOC_ERR_IO, stage::kOutput, "cannot rewind the PCM spool");
        }
        constexpr std::size_t kBlockFrames = 4096;
        std::vector<float> block(kBlockFrames * channels_);
        std::vector<double> converted(kBlockFrames * channels_);
        std::uint64_t remaining = keep;
        while (remaining > 0u) {
            const std::size_t frames =
                static_cast<std::size_t>(std::min<std::uint64_t>(remaining, kBlockFrames));
            const std::size_t count = frames * channels_;
            if (std::fread(block.data(), sizeof(float), count, file_) != count) {
                return Status::fail(JOC_ERR_IO, stage::kOutput, "cannot read the PCM spool");
            }
            for (std::size_t index = 0; index < count; ++index) {
                converted[index] = static_cast<double>(block[index]);
            }
            const Status written = writer->write(converted.data(), frames);
            if (!written.ok()) {
                return written;
            }
            remaining -= frames;
            if (report) {
                report(keep - remaining, keep);
            }
        }
        return Status::success();
    }

    double peak() const { return peak_; }
    std::uint64_t clipped() const { return clipped_; }
    std::uint64_t position() const { return position_; }

    void close() {
        if (file_ != nullptr) {
            std::fclose(file_);
            file_ = nullptr;
            fs_utf8::remove(path_);
        }
    }

private:
    std::FILE* file_ = nullptr;
    std::string path_;
    std::uint32_t channels_ = 0;
    bool trim_ = false;
    double threshold_ = kDefaultTailThreshold;
    double peak_ = 0.0;
    std::uint64_t clipped_ = 0;
    std::uint64_t position_ = 0;
    std::uint64_t last_above_ = 0;
    std::uint64_t limit_frames_ = 0;
    bool any_above_ = false;
};

// Mirrors the reference CLI's choose_pcm_output_format.
Status decide_output_format(const Settings& cfg, telemetry::EventBus* bus, double peak,
                            std::uint64_t clipped, std::uint32_t* actual) {
    *actual = cfg.output_format;
    if (cfg.output_format != JOC_FORMAT_PCM24 || clipped == 0u) {
        return Status::success();
    }
    const std::string notice =
        fmt("[clip] int24 将发生削波：peak=%.9g，超出 [-1,1] 的样本值=%llu", peak,
            static_cast<unsigned long long>(clipped));
    std::fprintf(stderr, "%s\n", notice.c_str());
    bus->warning(notice);
    std::uint32_t action = cfg.clip_action;
    if (action == JOC_CLIP_ASK) {
        if (!stdin_is_interactive()) {
            return Status::fail(JOC_ERR_OUTPUT_CLIP_ABORT, stage::kOutput,
                                "检测到 int24 削波，但当前不是交互终端；请使用 "
                                "--clip-action continue、--clip-action float32 或 "
                                "--clip-action abort");
        }
        for (;;) {
            std::fputs("继续写 int24 并截断 [i] / 改为 float32 [f，默认] / 取消 [a]：", stderr);
            std::fflush(stderr);
            char line[64] = {0};
            if (std::fgets(line, sizeof(line), stdin) == nullptr) {
                return Status::fail(JOC_ERR_OUTPUT_CLIP_ABORT, stage::kOutput,
                                    "用户因 int24 削波取消输出");
            }
            std::string answer(line);
            while (!answer.empty() && (answer.back() == '\n' || answer.back() == '\r' ||
                                       answer.back() == ' ')) {
                answer.pop_back();
            }
            for (char& character : answer) {
                character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
            }
            if (answer.empty() || answer == "f" || answer == "float" || answer == "float32") {
                action = JOC_CLIP_FLOAT32;
                break;
            }
            if (answer == "i" || answer == "int" || answer == "int24" || answer == "c" ||
                answer == "continue") {
                action = JOC_CLIP_CONTINUE;
                break;
            }
            if (answer == "a" || answer == "abort" || answer == "q" || answer == "quit" ||
                answer == "n" || answer == "no") {
                action = JOC_CLIP_ABORT;
                break;
            }
            std::fputs("请输入 i、f 或 a。\n", stderr);
        }
    }
    if (action == JOC_CLIP_CONTINUE) {
        std::fputs("[clip] 将继续写 int24，超范围值会截断到 [-1,1]。\n", stdout);
        *actual = JOC_FORMAT_PCM24;
        return Status::success();
    }
    if (action == JOC_CLIP_FLOAT32) {
        std::fputs("[clip] 已切换为 float32 WAV，不执行截断。\n", stdout);
        *actual = JOC_FORMAT_FLOAT32;
        return Status::success();
    }
    if (action == JOC_CLIP_ABORT) {
        return Status::fail(JOC_ERR_OUTPUT_CLIP_ABORT, stage::kOutput,
                            "用户因 int24 削波取消输出");
    }
    return Status::fail(JOC_ERR_INVALID_CONFIG, stage::kOutput, "unknown clip action");
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8u);
    for (const char character : text) {
        switch (character) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20u) {
                    out += fmt("\\u%04x", static_cast<unsigned>(character));
                } else {
                    out.push_back(character);
                }
        }
    }
    return out;
}

// Parses the bitstream without decoding or rendering: --metadata-only and the
// diagnostic metadata reports.
Status report_metadata(const Settings& cfg, telemetry::EventBus* bus, const std::string& eac3_path,
                       std::uint64_t total_frames) {
    std::ifstream input = fs_utf8::open_input(eac3_path);
    if (!input) {
        return Status::fail(JOC_ERR_INPUT_NOT_FOUND, stage::kEac3,
                            "cannot open the E-AC-3 stream: " + eac3_path);
    }
    ::joc::eac3::FrameReader reader;
    std::vector<char> chunk(kReadChunk);
    std::vector<std::string> rows;
    std::uint64_t frames = 0;
    std::uint64_t oamd_payloads = 0;
    std::uint64_t objects_total = 0;
    std::uint32_t objects_min = 0xFFFFFFFFu;
    std::uint32_t objects_max = 0u;
    double clipgain_min = 0.0;
    double clipgain_max = 0.0;
    bool stream_done = false;
    Status failure = Status::success();
    while (!stream_done && frames < total_frames) {
        input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = input.gcount();
        if (got > 0) {
            reader.push(reinterpret_cast<const std::uint8_t*>(chunk.data()),
                        static_cast<std::size_t>(got));
        } else {
            reader.finish();
            stream_done = true;
        }
        for (;;) {
            ::joc::eac3::Frame frame;
            const ::joc::eac3::FrameReader::Next state = reader.next(&frame);
            if (state == ::joc::eac3::FrameReader::Next::End) {
                break;
            }
            if (state == ::joc::eac3::FrameReader::Next::Fail) {
                return Status::fail(reader.error(), stage::kEac3, reader.error_message());
            }
            if (frames >= total_frames) {
                stream_done = true;
                break;
            }
            joc_frame_params params{};
            ::joc::emdf::Container container;
            const Status parsed = ::joc::joc::parse_eac3_frame(frame.data, frame.size, &params,
                                                              &container, nullptr);
            if (!parsed.ok()) {
                return Status::fail(parsed.code(), parsed.stage(),
                                    "frame " + std::to_string(frames) + ": " + parsed.message());
            }
            std::uint32_t oamd_objects = 0;
            std::uint32_t oamd_updates = 0;
            if (const ::joc::emdf::Payload* payload = container.find(::joc::emdf::kIdOamd)) {
                std::vector<std::uint8_t> bytes;
                const Status extracted =
                    ::joc::emdf::extract_payload_bytes(frame.data, frame.size, *payload, &bytes);
                if (!extracted.ok()) {
                    return Status::fail(extracted.code(), extracted.stage(), extracted.message());
                }
                ::joc::oamd::OamdUpdate update;
                const Status oamd = ::joc::oamd::parse_id11(bytes.data(), bytes.size(), &update);
                if (!oamd.ok()) {
                    return Status::fail(oamd.code(), oamd.stage(),
                                        "frame " + std::to_string(frames) + ": " + oamd.message());
                }
                ++oamd_payloads;
                oamd_updates = update.object_count;
                oamd_objects = update.object_count;
            }
            objects_total += params.n_objects;
            objects_min = std::min(objects_min, static_cast<std::uint32_t>(params.n_objects));
            objects_max = std::max(objects_max, static_cast<std::uint32_t>(params.n_objects));
            if (frames == 0u) {
                clipgain_min = params.clipgain;
                clipgain_max = params.clipgain;
            } else {
                clipgain_min = std::min(clipgain_min, params.clipgain);
                clipgain_max = std::max(clipgain_max, params.clipgain);
            }
            if (cfg.print_metadata >= 2u) {
                rows.push_back(fmt("frame %6llu: bytes=%5u containers=%u objects=%u clipgain=%.9g "
                                   "oamd=%s(%u updates)",
                                   static_cast<unsigned long long>(frames), frame.size,
                                   static_cast<unsigned>(container.payload_count),
                                   static_cast<unsigned>(params.n_objects), params.clipgain,
                                   oamd_objects != 0u ? "yes" : "no", oamd_updates));
            }
            ++frames;
        }
    }
    if (cfg.print_metadata >= 2u) {
        for (const std::string& row : rows) {
            std::printf("%s\n", row.c_str());
        }
    }
    const double mean_objects = frames != 0u
                                    ? static_cast<double>(objects_total) / static_cast<double>(frames)
                                    : 0.0;
    if (cfg.print_metadata >= 1u) {
        std::printf("metadata summary: frames=%llu objects(min/mean/max)=%u/%.3f/%u "
                    "clipgain=%.9g..%.9g oamd_payloads=%llu\n",
                    static_cast<unsigned long long>(frames),
                    objects_min == 0xFFFFFFFFu ? 0u : objects_min, mean_objects, objects_max,
                    clipgain_min, clipgain_max,
                    static_cast<unsigned long long>(oamd_payloads));
    }
    if (!cfg.metadata_json_path.empty()) {
        std::string json = fmt(
            "{\n  \"input\": \"%s\",\n  \"frames\": %llu,\n  \"objects_min\": %u,\n"
            "  \"objects_mean\": %.6f,\n  \"objects_max\": %u,\n  \"clipgain_min\": %.9g,\n"
            "  \"clipgain_max\": %.9g,\n  \"oamd_payloads\": %llu\n}\n",
            json_escape(cfg.input_path).c_str(), static_cast<unsigned long long>(frames),
            objects_min == 0xFFFFFFFFu ? 0u : objects_min, mean_objects, objects_max,
            clipgain_min, clipgain_max, static_cast<unsigned long long>(oamd_payloads));
        std::ofstream output = fs_utf8::open_output(cfg.metadata_json_path);
        if (!output) {
            return Status::fail(JOC_ERR_OUTPUT_OPEN, stage::kOutput,
                                "cannot write " + cfg.metadata_json_path);
        }
        output << json;
    }
    bus->emit(JOC_EV_METADATA_INDEXED, JOC_STAGE_METADATA, JOC_LOG_INFO,
              fmt("metadata: %llu frames, oamd payloads %llu",
                  static_cast<unsigned long long>(frames),
                  static_cast<unsigned long long>(oamd_payloads)));
    std::fflush(stdout);
    return failure;
}

class BedSource {
public:
    bool open(const std::string& path) {
        file_.open(path, std::ios::binary);
        return static_cast<bool>(file_);
    }

    bool read_frame(std::uint64_t index, std::vector<float>* bed5, std::vector<float>* lfe) {
        bed5->assign(static_cast<std::size_t>(JOC_CORE_CHANNELS) * JOC_FRAME_SAMPLES, 0.0f);
        lfe->assign(JOC_FRAME_SAMPLES, 0.0f);
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(index * kBedFrameBytes), std::ios::beg);
        std::vector<float> interleaved(static_cast<std::size_t>(JOC_FRAME_SAMPLES) * 6u);
        file_.read(reinterpret_cast<char*>(interleaved.data()),
                   static_cast<std::streamsize>(interleaved.size() * sizeof(float)));
        if (file_.gcount() != static_cast<std::streamsize>(interleaved.size() * sizeof(float))) {
            return false;
        }
        for (std::size_t n = 0; n < JOC_FRAME_SAMPLES; ++n) {
            for (std::size_t c = 0; c < JOC_CORE_CHANNELS; ++c) {
                (*bed5)[c * JOC_FRAME_SAMPLES + n] = interleaved[n * 6u + kCoreChannels[c]];
            }
            (*lfe)[n] = interleaved[n * 6u + kLfeChannel];
        }
        return true;
    }

private:
    std::ifstream file_;
};

// ---------------------------------------------------------------------------
// Overlapped frame pipeline
//
// The frame loop used to run its three stages back to back on the calling
// thread: read + decode + JOC rebuild, then the DSP, then the write.  Only the
// DSP is long, so the other two were pure waiting time on the critical path.
// The same three stages now run on three threads and hand frames over through
// two small bounded rings:
//
//   producer thread : E-AC-3 read -> core PCM read -> JOC rebuild   (t_render)
//   calling thread  : speaker / binaural DSP, events, progress      (t_render_dsp)
//   writer thread   : spool / ADM BWF / objects16 intermediate      (t_write)
//
// No stage work changed.  The DSP still consumes frames in strictly increasing
// order on one thread, the writer still writes them in that order, and every
// stage sees the same input it saw before, so the produced bytes are the bytes
// the serial loop produced.  What did change is the order of the stages in time
// -- and therefore the error handling, which now has to arbitrate between
// stages that fail at different moments.  The rule is the serial loop's rule:
// the failure of the lowest frame index wins, ties broken by stage order.
// ---------------------------------------------------------------------------

// Frames in flight in the bitstream ring.  The DSP does not consume at a
// constant rate: the Rosella runtime renders in 64-frame chunks, so one frame in
// 64 costs ~115 ms while the other 63 are nearly free.  A ring smaller than a
// chunk starves the DSP during the cheap frames -- measured: with 3 slots the
// overlap collapsed to zero (the producer had to rebuild every frame on demand
// at 0.41 ms against a DSP that wanted them at 0.04 ms).  One chunk of slack
// lets the bitstream stage do its work inside the render burst instead.  This is
// the only unbounded-looking number here: 64 x 98 KB ~ 6.3 MB, independent of
// the length of the file.
constexpr std::size_t kBitstreamRingDepth = 64;

// The output stage is neither bursty nor slow (it is a buffered write per
// frame), so its ring only has to cover the writer's scheduling latency.
constexpr std::size_t kOutputRingDepth = 4;

// One frame on its way from the bitstream stage to the DSP.
struct JocJob {
    std::uint64_t index = 0;
    std::vector<float> objects16;  // channel-major, 16 x 1536
    oamd::OamdUpdate update;
    bool has_update = false;
    std::int64_t outer_offset = 0;
    unsigned objects = 0;  // for the per-frame trace event
    double clipgain = 0.0;
};

// One rendered frame on its way to disk.  It owns everything the writer needs,
// so the DSP thread can move on while the writer is still on this frame.
struct OutputJob {
    std::uint64_t index = 0;
    std::vector<double> dsp;  // interleaved render, the spool payload
    std::size_t dsp_frames = 0;
    std::vector<float> objects16;  // only when the ADM writer or --keep-raw needs it
    oamd::OamdUpdate update;
    bool has_update = false;
    std::int64_t outer_offset = 0;
};

// One producer, one consumer, fixed capacity.  Payloads are handed over by
// swapping, so both sides keep a buffer and the steady state allocates nothing.
// The producer blocks while the ring is full, which is what makes the buffering
// bounded; the consumer blocks while it is empty.
template <typename Payload>
class FrameRing {
public:
    explicit FrameRing(std::size_t capacity) : slots_(capacity == 0u ? 1u : capacity) {}

    FrameRing(const FrameRing&) = delete;
    FrameRing& operator=(const FrameRing&) = delete;

    // Producer side.  False once the peer stopped (failure, cancellation or the
    // normal end), in which case the payload is left untouched.
    bool push(Payload* payload) {
        std::unique_lock<std::mutex> lock(mutex_);
        has_space_.wait(lock, [this] { return count_ < slots_.size() || closed_; });
        if (closed_) {
            return false;
        }
        std::swap(slots_[(head_ + count_) % slots_.size()], *payload);
        ++count_;
        has_item_.notify_one();
        return true;
    }

    // Consumer side.  False once the producer closed the ring and it drained,
    // or once the ring was aborted.
    bool pop(Payload* payload) {
        std::unique_lock<std::mutex> lock(mutex_);
        has_item_.wait(lock, [this] { return count_ != 0u || closed_; });
        if (count_ == 0u || aborted_) {
            return false;
        }
        std::swap(slots_[head_], *payload);
        head_ = (head_ + 1u) % slots_.size();
        --count_;
        has_space_.notify_one();
        return true;
    }

    // Normal end: drain what is queued, then stop.
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        has_item_.notify_all();
        has_space_.notify_all();
    }

    // Failure or cancellation: stop now and drop what is queued, so a peer
    // blocked in push() or pop() is released immediately.
    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        aborted_ = true;
        has_item_.notify_all();
        has_space_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable has_item_;
    std::condition_variable has_space_;
    std::vector<Payload> slots_;
    std::size_t head_ = 0;
    std::size_t count_ = 0;
    bool closed_ = false;
    bool aborted_ = false;
};

// The order the serial loop would have met two failures at the same frame.
enum class StageRank : int { kJoc = 0, kDsp = 1, kWrite = 2 };

struct StageFailure {
    bool valid = false;
    std::uint64_t frame = 0;
    int rank = 0;
    joc_error code = JOC_OK;
    std::string stage_text;
    std::string message;
};

// Where each stage leaves the failure that stopped it.  The serial loop always
// reported the failure of the lowest frame index, so that is the one recorded
// and the one every thread reports, whichever stage happens to notice first.
class FailureBoard {
public:
    void record(std::uint64_t frame, StageRank rank, joc_error code,
                const std::string& stage_text, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_.valid &&
            (failure_.frame < frame ||
             (failure_.frame == frame && failure_.rank <= static_cast<int>(rank)))) {
            return;
        }
        failure_.valid = true;
        failure_.frame = frame;
        failure_.rank = static_cast<int>(rank);
        failure_.code = code;
        failure_.stage_text = stage_text;
        failure_.message = message;
    }

    // The earliest failure recorded so far, if there is one.
    bool earliest(StageFailure* out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.valid) {
            return false;
        }
        *out = failure_;
        return true;
    }

    // True when a stage already failed on a frame the caller has not reached.
    // The serial loop would have stopped there, so the caller stops too.
    bool before(std::uint64_t frame, StageFailure* out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.valid || failure_.frame >= frame) {
            return false;
        }
        *out = failure_;
        return true;
    }

private:
    mutable std::mutex mutex_;
    StageFailure failure_;
};

// The stop flag every stage polls between frames.  A plain bool behind a mutex
// rather than an atomic: the stages touch it once per frame, so the lock costs
// nothing next to a frame of DSP.
class StopFlag {
public:
    void request() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }

    bool requested() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stop_;
    }

private:
    mutable std::mutex mutex_;
    bool stop_ = false;
};

// The serial loop emitted its progress line after that frame's write, so a
// progress boundary waits for the writer to reach the frame first: the line --
// and any write failure hiding behind it -- is then exactly the serial one.
// The wait costs one handoff every `progress_interval` frames.
class WriteProgress {
public:
    void complete(std::uint64_t frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        completed_ = std::max(completed_, frame + 1u);
        cv_.notify_all();
    }

    // The writer is done: nobody will complete another frame.
    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();
    }

    void wait_for(std::uint64_t frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, frame] { return finished_ || completed_ > frame; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::uint64_t completed_ = 0;  // frames written so far
    bool finished_ = false;
};

// Wakes and joins both stage threads.  It is declared after every resource the
// threads touch, so on every exit path -- normal end, early return, failure,
// cancellation or an exception out of the frame loop -- the threads are joined
// before those resources are destroyed.  Without it, a throwing exit would
// destroy a joinable std::thread and terminate.
class StageJoiner {
public:
    StageJoiner(FrameRing<JocJob>* joc_ring, FrameRing<OutputJob>* output_ring, StopFlag* stop,
                std::thread* producer, std::thread* writer)
        : joc_ring_(joc_ring), output_ring_(output_ring), stop_(stop), producer_(producer),
          writer_(writer) {}

    ~StageJoiner() { shutdown(true); }

    StageJoiner(const StageJoiner&) = delete;
    StageJoiner& operator=(const StageJoiner&) = delete;

    // `discard` drops whatever is still queued (failure, cancellation); without
    // it the writer drains the ring first, which is what the normal end wants.
    void shutdown(bool discard) {
        stop_->request();
        if (discard) {
            joc_ring_->abort();
            output_ring_->abort();
        } else {
            joc_ring_->close();
            output_ring_->close();
        }
        if (producer_->joinable()) {
            producer_->join();
        }
        if (writer_->joinable()) {
            writer_->join();
        }
    }

private:
    FrameRing<JocJob>* joc_ring_;
    FrameRing<OutputJob>* output_ring_;
    StopFlag* stop_;
    std::thread* producer_;
    std::thread* writer_;
};

}  // namespace

Status validate(const joc_task_config& raw, std::vector<joc_validation_issue>* issues,
                std::uint32_t* error_count) {
    const Settings cfg = normalise(raw);
    std::uint32_t errors = 0;
    auto fail = [&](joc_error code, const char* field, const std::string& message) {
        add_issue(issues, code, 2u, field, message);
        ++errors;
    };
    auto warn = [&](joc_error code, const char* field, const std::string& message) {
        add_issue(issues, code, 1u, field, message);
    };

    if (cfg.input_path.empty()) {
        fail(JOC_ERR_INVALID_CONFIG, "input_path", "an input path is required");
    } else if (!fs_utf8::exists(cfg.input_path)) {
        fail(JOC_ERR_INPUT_NOT_FOUND, "input_path", "input not found: " + cfg.input_path);
    }
    if (cfg.output_path.empty()) {
        fail(JOC_ERR_INVALID_CONFIG, "output_path", "an output path is required");
    }
    if (cfg.operation > JOC_OP_BINAURAL) {
        fail(JOC_ERR_INVALID_CONFIG, "operation", "operation must be ADM_BWF, SPEAKER or BINAURAL");
    }
    if (cfg.output_format > JOC_FORMAT_PCM24) {
        fail(JOC_ERR_INVALID_CONFIG, "output_format", "output_format must be FLOAT32 or PCM24");
    }
    if (cfg.clip_action > JOC_CLIP_ABORT) {
        fail(JOC_ERR_INVALID_CONFIG, "clip_action", "clip_action must be ask/continue/float32/abort");
    }
    if (!std::isfinite(cfg.tail_threshold) || cfg.tail_threshold < 0.0) {
        fail(JOC_ERR_INVALID_CONFIG, "binaural_tail_threshold",
             "binaural_tail_threshold must be a finite non-negative value");
    }
    if (cfg.operation == JOC_OP_ADM_BWF) {
        // ADM BWF is inherently 24-bit and this mode has no format option, so a
        // float32 request is simply unused: not a warning, just the mode's nature.
        if (cfg.output_format > JOC_FORMAT_PCM24) {
            fail(JOC_ERR_INVALID_CONFIG, "output_format",
                 "output_format must be FLOAT32 or PCM24");
        }
    }
    if (cfg.operation == JOC_OP_SPEAKER) {
        ::joc::speaker::LayoutInfo layout;
        if (cfg.layout.empty()) {
            fail(JOC_ERR_INVALID_CONFIG, "speaker_layout_name",
                 "a speaker layout is required for the speaker operation");
        } else if (!::joc::speaker::layout_by_name(cfg.layout.c_str(), &layout)) {
            fail(JOC_ERR_LAYOUT_UNSUPPORTED, "speaker_layout_name",
                 "unknown speaker layout: " + cfg.layout);
        }
    }
    if (cfg.operation == JOC_OP_BINAURAL) {
        if (cfg.hrtf_path.empty() && cfg.hrtf_sofa_path.empty() &&
            cfg.personalized_headphone_path.empty()) {
            fail(JOC_ERR_INVALID_CONFIG, "hrtf_path",
                 "a SOFA input, a Rosella model or a compiled HRTF (.jochrtf) is required for the "
                 "binaural operation");
        } else if (!cfg.personalized_headphone_path.empty() &&
                   !fs_utf8::exists(cfg.personalized_headphone_path)) {
            fail(JOC_ERR_HRTF_NOT_FOUND, "personalized_headphone_path",
                 "personalized headphone model not found: " + cfg.personalized_headphone_path);
        } else if (!cfg.hrtf_sofa_path.empty() && !fs_utf8::exists(cfg.hrtf_sofa_path)) {
            fail(JOC_ERR_HRTF_NOT_FOUND, "hrtf_sofa_path",
                 "SOFA HRTF not found: " + cfg.hrtf_sofa_path);
        } else if (cfg.hrtf_sofa_path.empty() && cfg.personalized_headphone_path.empty() &&
                   !fs_utf8::exists(cfg.hrtf_path)) {
            fail(JOC_ERR_HRTF_NOT_FOUND, "hrtf_path", "HRTF cache not found: " + cfg.hrtf_path);
        }
        if (!cfg.hrtf_sofa_path.empty()) {
            if (cfg.hrtf_cache_policy > JOC_HRTF_CACHE_DISK) {
                fail(JOC_ERR_INVALID_CONFIG, "hrtf_cache_policy",
                     "hrtf_cache_policy must be none, memory or disk");
            }
            if (cfg.hrtf_cache_policy == JOC_HRTF_CACHE_DISK && cfg.hrtf_cache_dir.empty()) {
                fail(JOC_ERR_INVALID_CONFIG, "hrtf_cache_dir",
                     "the disk cache policy needs a cache directory");
            }
            if (!std::isfinite(cfg.hrtf_radius_m) || cfg.hrtf_radius_m <= 0.0) {
                fail(JOC_ERR_INVALID_CONFIG, "hrtf_radius_m",
                     "hrtf_radius_m must be positive and finite");
            }
        }
        // The filterbank tables are compiled in; a path is only a verification override.
        if (!cfg.kernels_path.empty() && !fs_utf8::exists(cfg.kernels_path)) {
            fail(JOC_ERR_HRTF_NOT_FOUND, "kernels_path",
                 "kernel tables not found: " + cfg.kernels_path);
        }
        if (cfg.binaural_mode == JOC_BINAURAL_OFF) {
            fail(JOC_ERR_INVALID_CONFIG, "binaural_mode",
                 "binaural_mode off is only meaningful for ADM output");
        }
    }
    if (cfg.operation == JOC_OP_ADM_BWF) {
        if (cfg.adm_binaural_mode > 4u) {
            fail(JOC_ERR_INVALID_CONFIG, "adm_binaural_mode",
                 "adm binaural mode must be off/near/far/mid/unspecified");
        }
        if (cfg.trajectory_mode > JOC_TRAJECTORY_DENSE64) {
            fail(JOC_ERR_INVALID_CONFIG, "trajectory_mode",
                 "trajectory mode must be compact or dense64");
        }
    }
    if (!std::isfinite(cfg.gain_db) || cfg.gain_db < -200.0 || cfg.gain_db > 200.0) {
        fail(JOC_ERR_INVALID_CONFIG, "gain_db", "gain_db must be a finite value in [-200, 200]");
    }
    if (cfg.native_threads > 15u) {
        warn(JOC_ERR_INVALID_CONFIG, "native_threads",
             "native_threads is clamped to the 15 JOC objects");
    }
    if (cfg.bed_path.empty() && !fs_utf8::exists(cfg.ffmpeg_path)) {
        warn(JOC_ERR_LIBRARY_MISSING, "ffmpeg_path",
             "bed PCM is not supplied, so ffmpeg must be reachable: " + cfg.ffmpeg_path);
    }
    if (error_count != nullptr) {
        *error_count = errors;
    }
    return errors == 0 ? Status::success()
                       : Status::fail(JOC_ERR_INVALID_CONFIG, stage::kOutput,
                                      std::to_string(errors) + " configuration error(s)");
}

Status run(const joc_task_config& raw, const joc_event_sink* sink, joc_task_result* out) {
    const Settings cfg = normalise(raw);
    const auto started = Clock::now();
    telemetry::EventBus bus(sink);
    bus.emit(JOC_EV_TASK_STARTED, JOC_STAGE_INPUT, JOC_LOG_INFO, "task started");

    if (out != nullptr) {
        std::memset(out, 0, sizeof(*out));
        out->struct_size = sizeof(joc_task_result);
        out->struct_version = JOC_TASK_RESULT_VERSION;
    }

    std::vector<joc_validation_issue> issues;
    std::uint32_t error_count = 0;
    const Status validated = validate(raw, &issues, &error_count);
    for (const joc_validation_issue& issue : issues) {
        if (issue.severity >= 2u) {
            bus.error(issue.code, issue.field, issue.message);
        } else {
            bus.warning(std::string(issue.field) + ": " + issue.message);
        }
    }
    auto finalize_with_error = [&](joc_error code, const std::string& stage_text,
                                   const std::string& message) -> Status {
        bus.error(code, stage_text, message);
        if (out != nullptr) {
            out->status = JOC_TASK_FAILED;
            out->error_code = static_cast<std::uint32_t>(code);
            std::snprintf(out->error_stage, sizeof(out->error_stage), "%s", stage_text.c_str());
            std::snprintf(out->error_message, sizeof(out->error_message), "%s", message.c_str());
            out->t_total = std::chrono::duration<double>(Clock::now() - started).count();
        }
        bus.emit(JOC_EV_TASK_FAILED, JOC_STAGE_DONE, JOC_LOG_ERROR, message, code);
        return Status::fail(code, stage_text, message);
    };
    if (!validated.ok()) {
        return finalize_with_error(JOC_ERR_INVALID_CONFIG, "config",
                                   issues.empty() ? validated.message() : issues.front().message);
    }

    const auto decode_started = Clock::now();
    std::string temp_dir;
    std::string eac3_path = cfg.input_path;
    const std::string lowered = lower(cfg.input_path);
    if (!ends_with(lowered, ".eac3") && !ends_with(lowered, ".ec3")) {
        temp_dir = create_temp_dir(cfg);
        eac3_path = temp_dir + "\\input.eac3";
        bus.stage(JOC_STAGE_INPUT, "extracting the E-AC-3 elementary stream");
        const io::ProcessResult extracted = [&] {
            io::ProcessResult result;
            run_process({cfg.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i",
                         cfg.input_path, "-map", "0:a:0", "-vn", "-c:a", "copy", "-f", "eac3",
                         eac3_path},
                        &result);
            return result;
        }();
        if (!fs_utf8::exists(eac3_path)) {
            return finalize_with_error(JOC_ERR_INPUT_FORMAT, "input",
                                       "ffmpeg could not extract E-AC-3: " +
                                           (extracted.output.empty() ? "no output"
                                                                     : extracted.output));
        }
    }

    std::string bed_path = cfg.bed_path;
    if (bed_path.empty() && !cfg.metadata_only) {
        if (temp_dir.empty()) {
            temp_dir = create_temp_dir(cfg);
        }
        bed_path = temp_dir + "\\core51_f32le.raw";
        bus.stage(JOC_STAGE_DECODE, "decoding the 5.1 core PCM with ffmpeg");
        std::vector<std::string> argv = {cfg.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
                                         "-i", eac3_path};
        argv.insert(argv.end(), {"-map", "0:a:0", "-vn", "-ac", "6", "-ar", "48000",
                                 "-c:a", "pcm_f32le", "-f", "f32le", bed_path});
        argv.insert(argv.begin() + 4, {"-drc_scale", fmt("%.10g", cfg.drc_scale)});
        if (cfg.target_level != 0) {
            argv.insert(argv.begin() + 4, {"-target_level", std::to_string(cfg.target_level)});
        }
        io::ProcessResult decoded;
        const Status status = io::run_process(argv, &decoded);
        if (!status.ok()) {
            return finalize_with_error(JOC_ERR_INPUT_FORMAT, "decode",
                                       "ffmpeg could not decode the core PCM: " + decoded.output);
        }
    }
    if (!bed_path.empty() && !fs_utf8::exists(bed_path)) {
        return finalize_with_error(JOC_ERR_INPUT_NOT_FOUND, "bed_path",
                                   "core PCM not found: " + bed_path);
    }
    const double t_decode = std::chrono::duration<double>(Clock::now() - decode_started).count();

    std::uint64_t total_frames = count_frames(eac3_path);
    if (total_frames == 0u) {
        return finalize_with_error(JOC_ERR_INPUT_FORMAT, "input", "no E-AC-3 frames found");
    }
    if (cfg.duration_frames != 0u && cfg.duration_frames < total_frames) {
        total_frames = cfg.duration_frames;
    }
    bus.set_totals(total_frames, total_frames * static_cast<std::uint64_t>(JOC_FRAME_SAMPLES));
    bus.emit(JOC_EV_INPUT_OPENED, JOC_STAGE_INPUT, JOC_LOG_INFO, "input: " + cfg.input_path);
    bus.emit(JOC_EV_METADATA_INDEXED, JOC_STAGE_METADATA, JOC_LOG_INFO,
             fmt("%llu frames, %.2f s", static_cast<unsigned long long>(total_frames),
                 static_cast<double>(total_frames) *
                     static_cast<double>(JOC_FRAME_SAMPLES) / 48000.0));

    if (cfg.metadata_only || cfg.print_metadata != 0u || !cfg.metadata_json_path.empty()) {
        const Status reported = report_metadata(cfg, &bus, eac3_path, total_frames);
        if (!reported.ok()) {
            return finalize_with_error(reported.code(), reported.stage(), reported.message());
        }
    }
    if (cfg.metadata_only) {
        if (out != nullptr) {
            out->status = JOC_TASK_OK;
            out->input_frames = total_frames;
            out->duration_sec = static_cast<double>(total_frames) *
                                static_cast<double>(JOC_FRAME_SAMPLES) / 48000.0;
            out->t_decode_bed = t_decode;
            out->t_total = std::chrono::duration<double>(Clock::now() - started).count();
        }
        if (!temp_dir.empty()) {
            std::error_code ignored;
            fs::remove_all(fs_utf8::to_path(temp_dir), ignored);
        }
        bus.emit(JOC_EV_TASK_COMPLETED, JOC_STAGE_DONE, JOC_LOG_INFO,
                 fmt("metadata only: %llu frames",
                     static_cast<unsigned long long>(total_frames)));
        return Status::success();
    }

    const float gain = static_cast<float>(std::pow(10.0, cfg.gain_db / 20.0));
    ejoc_renderer_handle rebuilder = ejoc_renderer_create();
    if (rebuilder == nullptr) {
        return finalize_with_error(JOC_ERR_OUT_OF_MEMORY, "joc", "cannot create the JOC kernel");
    }
    struct KernelGuard {
        ejoc_renderer_handle handle;
        ~KernelGuard() { ejoc_renderer_destroy(handle); }
    } guard{rebuilder};
    if (cfg.native_threads != 0u) {
        ejoc_renderer_set_threads(rebuilder, cfg.native_threads);
    }

    ::joc::speaker::SpeakerStep speaker;
    const bool speaker_enabled = cfg.operation == JOC_OP_SPEAKER;
    if (speaker_enabled) {
        if (!::joc::speaker::layout_by_name(cfg.layout.c_str(), &speaker.layout)) {
            return finalize_with_error(JOC_ERR_LAYOUT_UNSUPPORTED, "speaker_layout_name",
                                       "unknown speaker layout: " + cfg.layout);
        }
        speaker.metadata_offset = cfg.speaker_metadata_offset;
        speaker.handle = ejoc_speaker_renderer_create(speaker.layout.speaker_bitfield);
        if (speaker.handle == nullptr) {
            return finalize_with_error(JOC_ERR_RENDER_FAILED, "render",
                                       "cannot create the speaker renderer");
        }
        bus.emit(JOC_EV_RENDERER_INITIALIZED, JOC_STAGE_RENDER, JOC_LOG_INFO,
                 fmt("speaker renderer: %s, %u channels", speaker.layout.name,
                     speaker.layout.channel_count));
    }

    ::joc::binaural::SofaBinauralRuntime binaural;
    ::joc::hrtf::RosellaRuntime rosella;
    const bool binaural_enabled = cfg.operation == JOC_OP_BINAURAL;
    const bool rosella_enabled = !cfg.personalized_headphone_path.empty();
    if (binaural_enabled) {
        if (rosella_enabled) {
            ::joc::hrtf::RosellaModel model;
            Status status =
                ::joc::hrtf::load_personalized_headphone(cfg.personalized_headphone_path, &model);
            ::joc::hrtf::RosellaRenderOptions render_options;
            if (cfg.binaural_mode == JOC_BINAURAL_NEAR) {
                render_options.profile = ::joc::hrtf::RosellaProfile::Near;
            } else if (cfg.binaural_mode == JOC_BINAURAL_FAR) {
                render_options.profile = ::joc::hrtf::RosellaProfile::Far;
            }
            render_options.object_delay_samples = cfg.object_delay_samples;
            render_options.tail_seconds = cfg.tail_seconds;
            render_options.output_gain = 1.0;
            if (status.ok()) {
                status = rosella.open(model, render_options);
            }
            if (status.ok()) {
                bus.emit(JOC_EV_HRTF_LOADED, JOC_STAGE_RENDER, JOC_LOG_INFO, model.summary());
                bus.emit(JOC_EV_RENDERER_INITIALIZED, JOC_STAGE_RENDER, JOC_LOG_INFO,
                         "Rosella binaural runtime ready (the first 961 samples are the "
                         "filterbank latency)");
            }
            if (!status.ok()) {
                return finalize_with_error(status.code(), "render",
                                           "binaural setup failed: " + status.message());
            }
        }
        if (!rosella_enabled) {
        ::joc::hrtf::Field field;
        ::joc::hrtf::Kernels kernels;
        ::joc::binaural::Profile profile = ::joc::binaural::Profile::Mid;
        Status status = Status::success();
        if (!cfg.hrtf_sofa_path.empty()) {
            // The .jochrtf is an internal cache: the SOFA is the user-facing input.
            ::joc::hrtf::SofaFieldRequest request;
            request.sofa_path = cfg.hrtf_sofa_path;
            request.options.shell_radius_m = cfg.hrtf_radius_m;
            request.cache_dir = cfg.hrtf_cache_dir;
            switch (cfg.hrtf_cache_policy) {
                case JOC_HRTF_CACHE_NONE: request.policy = ::joc::hrtf::CachePolicy::None; break;
                case JOC_HRTF_CACHE_DISK: request.policy = ::joc::hrtf::CachePolicy::Disk; break;
                default: request.policy = ::joc::hrtf::CachePolicy::Memory; break;
            }
            std::string cache_path;
            status = ::joc::hrtf::load_or_compile_sofa_field(request, &field, &cache_path);
            if (status.ok()) {
                bus.emit(JOC_EV_HRTF_LOADED, JOC_STAGE_RENDER, JOC_LOG_INFO,
                         "SOFA HRTF compiled: " + field.source_sha256.substr(0, 16) +
                             (cache_path.empty() ? std::string() : " -> " + cache_path));
            }
        } else {
            status = ::joc::hrtf::load_jochrtf(cfg.hrtf_path, &field);
            if (status.ok()) {
                bus.emit(JOC_EV_HRTF_LOADED, JOC_STAGE_RENDER, JOC_LOG_INFO,
                         "compiled HRTF: " + field.source_sha256.substr(0, 16));
            }
        }
        if (status.ok()) {
            status = cfg.kernels_path.empty()
                         ? (kernels = ::joc::hrtf::builtin_kernels(), Status::success())
                         : ::joc::hrtf::load_kernels(cfg.kernels_path, &kernels);
        }
        if (status.ok() && cfg.binaural_mode == JOC_BINAURAL_NEAR) {
            profile = ::joc::binaural::Profile::Near;
        } else if (status.ok() && cfg.binaural_mode == JOC_BINAURAL_FAR) {
            profile = ::joc::binaural::Profile::Far;
        }
        if (status.ok()) {
            status = binaural.open(field, kernels, profile);
        }
        if (!status.ok()) {
            return finalize_with_error(status.code(), "render",
                                       "binaural setup failed: " + status.message());
        }
        bus.emit(JOC_EV_RENDERER_INITIALIZED, JOC_STAGE_RENDER, JOC_LOG_INFO,
                 "binaural runtime ready (the first 961 samples are the kernel latency)");
        }
    }

    ::joc::io::AdmBwfWriter adm_writer;
    ::joc::adm::TrajectoryBuilder trajectory(48000, 64, cfg.object_delay_samples);
    const bool adm_enabled = cfg.operation == JOC_OP_ADM_BWF;
    if (adm_enabled) {
        const Status status = adm_writer.open(cfg.output_path);
        if (!status.ok()) {
            return finalize_with_error(status.code(), "output", status.message());
        }
    }

    PcmSpool spool;
    ::joc::io::WavWriter wav_writer;
    std::uint32_t wav_channels = 0;
    bool wav_output = speaker_enabled || binaural_enabled;
    if (wav_output) {
        wav_channels = binaural_enabled ? 2u : static_cast<std::uint32_t>(speaker.layout.channel_count);
        const Status status = spool.open(cfg.output_path + ".spool.f32", wav_channels,
                                         binaural_enabled, cfg.tail_threshold);
        if (!status.ok()) {
            return finalize_with_error(status.code(), "output", status.message());
        }
        bus.emit(JOC_EV_OUTPUT_OPENED, JOC_STAGE_OUTPUT, JOC_LOG_INFO, "output: " + cfg.output_path);
        bus.emit(JOC_EV_OUTPUT_FORMAT_DECIDED, JOC_STAGE_OUTPUT, JOC_LOG_INFO,
                 cfg.output_format == JOC_FORMAT_PCM24 ? "int24 (clip policy applies)"
                                                       : "float32");
    } else {
        bus.emit(JOC_EV_OUTPUT_OPENED, JOC_STAGE_OUTPUT, JOC_LOG_INFO,
                 "output: " + cfg.output_path + " (25-channel RF64 ADM BWF)");
    }

    std::ofstream raw_objects;
    if ((cfg.flags & JOC_TASK_F_KEEP_INTERMEDIATE) != 0u) {
        raw_objects = fs_utf8::open_output(cfg.output_path + ".objects16.f32le");
        if (!raw_objects) {
            return finalize_with_error(JOC_ERR_OUTPUT_OPEN, "output",
                                       "cannot write the objects16 intermediate");
        }
    }

    BedSource bed;
    if (!bed.open(bed_path)) {
        return finalize_with_error(JOC_ERR_INPUT_NOT_FOUND, "bed_path",
                                   "cannot open the core PCM: " + bed_path);
    }
    std::ifstream input = fs_utf8::open_input(eac3_path);
    if (!input) {
        return finalize_with_error(JOC_ERR_INPUT_NOT_FOUND, "input",
                                   "cannot open the E-AC-3 stream: " + eac3_path);
    }

    double dsp_seconds = 0.0;
    double write_seconds = 0.0;        // the writer thread's own write time
    double write_setup_seconds = 0.0;  // the caller's share of the same stage
    double joc_seconds = 0.0;          // the bitstream stage's own time
    std::uint64_t written_samples = 0;
    std::uint64_t file_bytes = 0;
    std::uint32_t output_format_actual = cfg.output_format;
    double peak = 0.0;
    std::uint64_t over_unity = 0;
    std::uint64_t processed = 0;
    ::joc::eac3::FrameReader reader;
    std::vector<char> chunk(kReadChunk);
    std::vector<float> bed5;
    std::vector<float> lfe;
    std::vector<float> objects16;

    FrameRing<JocJob> joc_ring(kBitstreamRingDepth);
    FrameRing<OutputJob> output_ring(kOutputRingDepth);
    StopFlag stop;
    FailureBoard failures;
    WriteProgress write_progress;
    std::thread producer;
    std::thread writer;
    StageJoiner joiner(&joc_ring, &output_ring, &stop, &producer, &writer);

    // ---- stage 1: the bitstream, on its own thread ------------------------ //
    // The serial loop's read/decode/JOC section, verbatim apart from its error
    // paths: it records failures instead of returning, because the calling
    // thread owns the return value, the result struct and the event sink.
    producer = std::thread([&]() {
        std::uint64_t produced = 0;
        bool stream_done = false;
        double seconds = 0.0;
        JocJob job;
        try {
            while (!stream_done && !stop.requested()) {
                auto stage_started = Clock::now();
                input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                const std::streamsize got = input.gcount();
                if (got > 0) {
                    reader.push(reinterpret_cast<const std::uint8_t*>(chunk.data()),
                                static_cast<std::size_t>(got));
                } else {
                    reader.finish();
                    stream_done = true;
                }
                for (;;) {
                    ::joc::eac3::Frame frame;
                    const ::joc::eac3::FrameReader::Next state = reader.next(&frame);
                    if (state == ::joc::eac3::FrameReader::Next::End) {
                        break;
                    }
                    if (state == ::joc::eac3::FrameReader::Next::Fail) {
                        failures.record(produced, StageRank::kJoc, reader.error(), "eac3_transport",
                                        reader.error_message());
                        stream_done = true;
                        break;
                    }
                    if (produced >= total_frames) {
                        stream_done = true;
                        break;
                    }
                    if (!bed.read_frame(produced, &bed5, &lfe)) {
                        failures.record(produced, StageRank::kJoc, JOC_ERR_INPUT_FORMAT, "bed_path",
                                        "core PCM ended before frame " + std::to_string(produced));
                        stream_done = true;
                        break;
                    }
                    joc_frame_params params{};
                    ::joc::emdf::Container container;
                    const Status parsed = ::joc::joc::parse_eac3_frame(frame.data, frame.size,
                                                                       &params, &container, nullptr);
                    if (!parsed.ok()) {
                        failures.record(produced, StageRank::kJoc, parsed.code(), parsed.stage(),
                                        "frame " + std::to_string(produced) + ": " +
                                            parsed.message());
                        stream_done = true;
                        break;
                    }
                    job.update = ::joc::oamd::OamdUpdate{};
                    job.has_update = false;
                    job.outer_offset = 0;
                    if (const ::joc::emdf::Payload* payload = container.find(::joc::emdf::kIdOamd)) {
                        std::vector<std::uint8_t> bytes;
                        const Status extracted = ::joc::emdf::extract_payload_bytes(
                            frame.data, frame.size, *payload, &bytes);
                        if (!extracted.ok()) {
                            failures.record(produced, StageRank::kJoc, extracted.code(),
                                            extracted.stage(), extracted.message());
                            stream_done = true;
                            break;
                        }
                        const Status oamd =
                            ::joc::oamd::parse_id11(bytes.data(), bytes.size(), &job.update);
                        if (!oamd.ok()) {
                            failures.record(produced, StageRank::kJoc, oamd.code(), oamd.stage(),
                                            "frame " + std::to_string(produced) + ": " +
                                                oamd.message());
                            stream_done = true;
                            break;
                        }
                        job.has_update = true;
                        job.outer_offset = static_cast<std::int64_t>(payload->sample_offset);
                    }
                    std::string dsp_error;
                    const Status rebuilt = ::joc::joc::rebuild_objects16(
                        rebuilder, params, bed5.data(), lfe.data(), gain, &objects16, &dsp_error);
                    if (!rebuilt.ok()) {
                        failures.record(produced, StageRank::kJoc, rebuilt.code(), "joc",
                                        "frame " + std::to_string(produced) + ": " + dsp_error);
                        stream_done = true;
                        break;
                    }
                    job.index = produced;
                    job.objects = params.n_objects;
                    job.clipgain = params.clipgain;
                    job.objects16.swap(objects16);
                    // The clock stops before the handoff, so the stage's number is
                    // its own work (including the chunk read) and never the time it
                    // spent waiting for the DSP to make room in the ring.
                    seconds += std::chrono::duration<double>(Clock::now() - stage_started).count();
                    if (!joc_ring.push(&job)) {
                        stream_done = true;
                        break;
                    }
                    ++produced;
                    stage_started = Clock::now();
                }
            }
        } catch (...) {
            // On the calling thread this would have propagated out of run(); on a
            // stage thread it has to become a failure, or the thread terminates.
            failures.record(produced, StageRank::kJoc, JOC_ERR_OUT_OF_MEMORY, "joc",
                            "the bitstream stage ran out of memory at frame " +
                                std::to_string(produced));
        }
        joc_seconds = seconds;
        joc_ring.close();
    });

    // ---- stage 3: the output, on its own thread --------------------------- //
    // The serial loop's write section, verbatim apart from its error paths.
    writer = std::thread([&]() {
        OutputJob job;
        std::vector<float> interleaved;
        double seconds = 0.0;
        try {
            while (output_ring.pop(&job)) {
                const auto write_started = Clock::now();
                Status status = Status::success();
                if (raw_objects.is_open() && !job.objects16.empty()) {
                    // 16 objects, interleaved per sample, exactly like the reference
                    // intermediate (frame_count, 1536, 16) float32.
                    interleaved.resize(static_cast<std::size_t>(JOC_OUTPUT_CHANNELS) *
                                       JOC_FRAME_SAMPLES);
                    for (std::size_t sample = 0; sample < JOC_FRAME_SAMPLES; ++sample) {
                        for (std::size_t channel = 0; channel < JOC_OUTPUT_CHANNELS; ++channel) {
                            interleaved[sample * JOC_OUTPUT_CHANNELS + channel] =
                                job.objects16[channel * JOC_FRAME_SAMPLES + sample];
                        }
                    }
                    raw_objects.write(reinterpret_cast<const char*>(interleaved.data()),
                                      static_cast<std::streamsize>(interleaved.size() *
                                                                   sizeof(float)));
                }
                if (adm_enabled && !job.objects16.empty()) {
                    trajectory.submit_frame(static_cast<std::int64_t>(job.index),
                                            job.has_update ? &job.update : nullptr,
                                            job.outer_offset);
                    status = adm_writer.write_objects16(job.objects16.data());
                } else if (speaker_enabled && !job.dsp.empty()) {
                    interleaved.resize(job.dsp.size());
                    for (std::size_t index = 0; index < job.dsp.size(); ++index) {
                        interleaved[index] = static_cast<float>(job.dsp[index]);
                    }
                    status = spool.write(interleaved.data(), JOC_FRAME_SAMPLES);
                } else if (binaural_enabled && !job.dsp.empty()) {
                    interleaved.resize(job.dsp.size());
                    for (std::size_t index = 0; index < job.dsp.size(); ++index) {
                        interleaved[index] = static_cast<float>(job.dsp[index]);
                    }
                    status = spool.write(interleaved.data(), job.dsp.size() / 2u);
                }
                seconds += std::chrono::duration<double>(Clock::now() - write_started).count();
                if (!status.ok()) {
                    failures.record(job.index, StageRank::kWrite, status.code(), status.stage(),
                                    status.message());
                    break;
                }
                write_progress.complete(job.index);
            }
        } catch (...) {
            failures.record(job.index, StageRank::kWrite, JOC_ERR_OUT_OF_MEMORY, "output",
                            "the output stage ran out of memory at frame " +
                                std::to_string(job.index));
        }
        write_seconds = seconds;
        write_progress.finish();
        // Released whether the writer finished or failed, so the calling thread
        // can never be left waiting for a frame that will not be written.
        output_ring.abort();
    });

    // ---- stage 2: the DSP, on the calling thread -------------------------- //
    JocJob frame;
    OutputJob pending;
    StageFailure pending_failure;
    bool failed = false;
    for (;;) {
        if (!joc_ring.pop(&frame)) {
            // The producer finished the stream -- or failed, which the shutdown
            // below and the failure board decide.
            break;
        }
        // A stage that already failed on an earlier frame stops the run here,
        // which is where the serial loop would have stopped.
        if (failures.before(frame.index, &pending_failure)) {
            failed = true;
            break;
        }
        if (cancelled(cfg)) {
            // Stop and join both stages before the ADM writer is aborted, so
            // nothing can still be writing when the abort runs.
            joiner.shutdown(true);
            if (adm_enabled) {
                adm_writer.abort();
            }
            if (out != nullptr) {
                out->status = JOC_TASK_CANCELLED;
                out->error_code = static_cast<std::uint32_t>(JOC_ERR_CANCELLED);
                out->input_frames = processed;
                out->t_total = std::chrono::duration<double>(Clock::now() - started).count();
            }
            bus.emit(JOC_EV_TASK_CANCELLED, JOC_STAGE_DONE, JOC_LOG_WARNING,
                     "cancelled at frame " + std::to_string(processed), JOC_ERR_CANCELLED);
            return Status::fail(JOC_ERR_CANCELLED, stage::kOutput, "task cancelled");
        }

        // The bitstream stage already rebuilt this frame.  The trace event stays
        // here, on the thread that owns the event sink, in the same place it had
        // in the serial loop: after the rebuild, before the DSP.
        bus.emit(JOC_EV_JOC_FRAME_STATS, JOC_STAGE_JOC, JOC_LOG_TRACE,
                 fmt("frame %llu objects=%u clipgain=%.6f",
                     static_cast<unsigned long long>(frame.index),
                     static_cast<unsigned>(frame.objects), frame.clipgain));

        // The DSP and the output write are timed apart: the DSP dominates the
        // frame loop, so charging it to the write made that number meaningless.
        double frame_dsp_seconds = 0.0;
        if (!adm_enabled) {
            // ADM has no DSP stage (objects16 goes straight into the BWF), so it
            // must not be charged any DSP time at all.
            const auto dsp_started = Clock::now();
            if (speaker_enabled) {
                std::string speaker_error;
                const Status stepped = ::joc::speaker::step(
                    &speaker, frame.objects16, frame.has_update ? &frame.update : nullptr,
                    &speaker_error);
                if (!stepped.ok()) {
                    failures.record(frame.index, StageRank::kDsp, stepped.code(), "render",
                                    "frame " + std::to_string(frame.index) + ": " + speaker_error);
                    failed = true;
                }
            } else {
                const Status submitted =
                    rosella_enabled
                        ? rosella.submit_frame(frame.objects16.data(),
                                               frame.has_update ? &frame.update : nullptr,
                                               static_cast<std::int64_t>(frame.index),
                                               frame.outer_offset,
                                               static_cast<std::int64_t>(cfg.object_delay_samples))
                        : binaural.submit_frame(frame.objects16.data(),
                                                frame.has_update ? &frame.update : nullptr,
                                                static_cast<std::int64_t>(frame.index),
                                                frame.outer_offset,
                                                static_cast<std::int64_t>(
                                                    cfg.object_delay_samples));
                if (!submitted.ok()) {
                    failures.record(frame.index, StageRank::kDsp, submitted.code(), "render",
                                    "frame " + std::to_string(frame.index) + ": " +
                                        submitted.message());
                    failed = true;
                }
            }
            frame_dsp_seconds = std::chrono::duration<double>(Clock::now() - dsp_started).count();
            dsp_seconds += frame_dsp_seconds;
        }
        if (failed) {
            break;
        }

        // Hand the rendered frame to the writer.  The job owns its payload, so the
        // DSP can start the next frame while this one is still being written; the
        // bounded ring is what limits how far ahead that can run.
        const auto write_started = Clock::now();
        pending.index = frame.index;
        pending.objects16.clear();
        pending.dsp.clear();
        pending.dsp_frames = 0;
        pending.has_update = false;
        pending.outer_offset = 0;
        if (adm_enabled) {
            // objects16 goes straight into the BWF, so the writer owns it now.
            pending.objects16.swap(frame.objects16);
            pending.update = frame.update;
            pending.has_update = frame.has_update;
            pending.outer_offset = frame.outer_offset;
        } else {
            if (raw_objects.is_open()) {
                // --keep-raw interleaves objects16 for every operation.
                pending.objects16.swap(frame.objects16);
            }
            if (speaker_enabled) {
                pending.dsp = speaker.output;
                pending.dsp_frames = JOC_FRAME_SAMPLES;
            } else {
                if (rosella_enabled) {
                    rosella.take_output(&pending.dsp);
                } else {
                    binaural.take_output(&pending.dsp);
                }
                pending.dsp_frames = pending.dsp.size() / 2u;
            }
        }
        // The spool accounting is known as soon as the frame is rendered, so the
        // progress line reports exactly what the serial loop reported without
        // waiting for the write itself.
        const std::uint64_t frame_samples =
            (adm_enabled || speaker_enabled) ? static_cast<std::uint64_t>(JOC_FRAME_SAMPLES)
                                             : static_cast<std::uint64_t>(pending.dsp_frames);
        write_setup_seconds +=
            std::chrono::duration<double>(Clock::now() - write_started).count();
        if (!output_ring.push(&pending)) {
            // The writer stopped; it recorded the failure that stopped it.
            failures.earliest(&pending_failure);
            failed = true;
            break;
        }
        written_samples += frame_samples;

        ++processed;
        if (cfg.progress_interval != 0u && processed % cfg.progress_interval == 0u) {
            // The serial loop printed this line after that frame's write, so wait
            // for the writer to reach the frame first: the line, and any write
            // failure hiding behind it, is then exactly the serial one.  One
            // handoff per progress interval, so the overlap is not given back.
            write_progress.wait_for(processed - 1u);
            if (failures.before(processed, &pending_failure)) {
                failed = true;
                break;
            }
            bus.progress(processed, processed * static_cast<std::uint64_t>(JOC_FRAME_SAMPLES),
                         written_samples, file_bytes,
                         static_cast<double>(written_samples) / 48000.0);
            bus.emit(JOC_EV_OAMD_STATS, JOC_STAGE_METADATA, JOC_LOG_DEBUG,
                     fmt("oamd payloads=%llu transitions=%llu",
                         static_cast<unsigned long long>(binaural_enabled
                                                             ? binaural.timeline().payload_count()
                                                             : 0ull),
                         static_cast<unsigned long long>(binaural_enabled
                                                             ? binaural.timeline()
                                                                   .transition_count()
                                                             : 0ull)));
        }
    }

    if (failed) {
        // The calling thread stopped the run -- its own failure, or one an
        // earlier frame's stage already recorded.  Stop both stages and drop
        // whatever is still queued; the board holds the earliest failure, which
        // is the one the serial loop would have reported.
        joiner.shutdown(true);
        StageFailure reported = pending_failure;
        if (!failures.earliest(&reported) && !reported.valid) {
            // Unreachable: a stage always records before it stops.  A lost error
            // would be worse than a synthetic one, so the code stays non-zero.
            reported.valid = true;
            reported.code = JOC_ERR_INTERNAL;
            reported.stage_text = "output";
            reported.message = "the render pipeline stopped without reporting a failure";
        }
        return finalize_with_error(reported.code, reported.stage_text, reported.message);
    }
    // Normal end: the producer closed its ring, so let the writer drain the last
    // frames and join it.  Every in-loop write is finished before the finalize
    // (tail flush, spool -> WAV, SHA-256, report) looks at the output.
    joiner.shutdown(false);
    StageFailure recorded;
    if (failures.earliest(&recorded)) {
        return finalize_with_error(recorded.code, recorded.stage_text, recorded.message);
    }

    // The stages overlap now, so t_render can no longer be "the frame loop minus
    // the renderer and the disk": that difference is whatever the DSP left over,
    // not the JOC stage's time.  The fields keep their meaning and are measured
    // where the work happens -- t_render is the bitstream stage's own time (read
    // + decode + JOC rebuild, with no waiting for the other stages), and t_write
    // is the output stage's own time: the writer thread's writes plus the
    // caller's assembly of the jobs it writes.
    const double t_render = joc_seconds;
    write_seconds += write_setup_seconds;

    // The finalize (tail drain, spool -> WAV conversion, writer finalize) is disk
    // work, so it is added to the file-write total rather than left untimed.
    const auto finalize_started = Clock::now();
    bus.stage(JOC_STAGE_OUTPUT, "finalizing the output");
    std::string failure;
    if (adm_enabled) {
        std::vector<::joc::adm::Track> tracks;
        const Status built = trajectory.build(
            static_cast<std::int64_t>(processed) * static_cast<std::int64_t>(JOC_FRAME_SAMPLES),
            cfg.trajectory_mode == JOC_TRAJECTORY_DENSE64 ? ::joc::adm::TrajectoryMode::Dense64
                                                          : ::joc::adm::TrajectoryMode::Compact,
            &tracks);
        if (!built.ok()) {
            failure = built.message();
        } else {
            const double duration_sec =
                static_cast<double>(processed) * static_cast<std::int64_t>(JOC_FRAME_SAMPLES) / 48000.0;
            const std::string axml = ::joc::adm::build_axml(tracks, duration_sec, 48000);
            const std::string chna = ::joc::adm::build_chna();
            std::string dbmd;
            const ::joc::adm::BinauralMode mode =
                static_cast<::joc::adm::BinauralMode>(cfg.adm_binaural_mode);
            const Status dbmd_status =
                ::joc::adm::build_dbmd(::joc::io::AdmBwfWriter::kChannels, mode, &dbmd);
            if (!dbmd_status.ok()) {
                failure = dbmd_status.message();
            } else {
                const Status finalized = adm_writer.finalize(axml, chna, dbmd);
                if (!finalized.ok()) {
                    failure = finalized.message();
                } else if (out != nullptr) {
                    out->oamd_payloads = 0;
                    for (const ::joc::adm::Track& track : tracks) {
                        out->oamd_transitions += track.blocks.size();
                    }
                }
            }
        }
        if (!failure.empty()) {
            adm_writer.abort();
            return finalize_with_error(JOC_ERR_OUTPUT_WRITE, "output", failure);
        }
        written_samples = processed * static_cast<std::uint64_t>(JOC_FRAME_SAMPLES);
        output_format_actual = JOC_FORMAT_PCM24;
    } else {
        const std::uint64_t program_samples = written_samples;
        if (binaural_enabled) {
            std::vector<double> tail;
            const Status drained_tail =
                rosella_enabled
                    ? rosella.finish(rosella.finish_capacity(cfg.tail_seconds), &tail)
                    : binaural.finish(binaural.finish_capacity(cfg.tail_seconds), &tail);
            if (!drained_tail.ok()) {
                return finalize_with_error(drained_tail.code(), "output",
                                           drained_tail.message());
            }
            const std::size_t samples = tail.size() / 2u;
            std::vector<float> interleaved(tail.size());
            for (std::size_t index = 0; index < tail.size(); ++index) {
                interleaved[index] = static_cast<float>(tail[index]);
            }
            bus.emit(JOC_EV_OUTPUT_STATS, JOC_STAGE_OUTPUT, JOC_LOG_INFO,
                     fmt("binaural: %llu programme samples + %llu tail samples = %llu total",
                         static_cast<unsigned long long>(program_samples),
                         static_cast<unsigned long long>(samples),
                         static_cast<unsigned long long>(program_samples + samples)));
            const Status written = spool.write(interleaved.data(), samples);
            if (!written.ok()) {
                return finalize_with_error(written.code(), "output", written.message());
            }
        }
        // The spool holds float32; the final format is decided now that the peak
        // is known, then the requested WAV is written from it.
        const std::uint64_t keep = spool.kept(program_samples);
        std::uint32_t actual_format = cfg.output_format;
        const Status decided =
            decide_output_format(cfg, &bus, spool.peak(), spool.clipped(), &actual_format);
        if (!decided.ok()) {
            return finalize_with_error(decided.code(), decided.stage(), decided.message());
        }
        if (keep != spool.position()) {
            bus.emit(JOC_EV_OUTPUT_STATS, JOC_STAGE_OUTPUT, JOC_LOG_INFO,
                     fmt("tail trimmed: %llu -> %llu samples (threshold %.3g)",
                         static_cast<unsigned long long>(spool.position()),
                         static_cast<unsigned long long>(keep), cfg.tail_threshold));
        }
        const io::SampleFormat format = actual_format == JOC_FORMAT_PCM24
                                            ? io::SampleFormat::Int24
                                            : io::SampleFormat::Float32;
        Status status = wav_writer.open(cfg.output_path, wav_channels, 48000, format, keep);
        if (status.ok()) {
            // Report on the same cadence as the render loop, so one knob
            // (`--progress-every`, default 1000 frames) governs both.  The frame
            // field stays in E-AC-3 frames so the frontend's "frame/total" keeps
            // its meaning (and stays monotone).
            const std::uint64_t quantum =
                static_cast<std::uint64_t>(cfg.progress_interval) * JOC_FRAME_SAMPLES;
            std::uint64_t last_reported = 0u;
            status = spool.copy_to(&wav_writer, keep, [&](std::uint64_t done, std::uint64_t total) {
                if (quantum == 0u || (done != total && done < last_reported + quantum)) {
                    return;
                }
                last_reported = done;
                const std::uint64_t frames =
                    std::min<std::uint64_t>(processed, done / JOC_FRAME_SAMPLES);
                bus.progress(frames, done, done, 0u, static_cast<double>(done) / 48000.0);
            });
        }
        if (status.ok()) {
            status = wav_writer.finalize();
        }
        if (!status.ok()) {
            return finalize_with_error(status.code(), "output", status.message());
        }
        written_samples = keep;
        peak = spool.peak();
        over_unity = spool.clipped();
        output_format_actual = actual_format;
    }
    spool.close();
    if (raw_objects.is_open()) {
        raw_objects.close();
        bus.emit(JOC_EV_OUTPUT_STATS, JOC_STAGE_OUTPUT, JOC_LOG_INFO,
                 "objects16 intermediate: " + cfg.output_path + ".objects16.f32le");
    }
    const double finalize_seconds =
        std::chrono::duration<double>(Clock::now() - finalize_started).count();

    std::error_code size_error;
    file_bytes = fs_utf8::file_size(cfg.output_path, size_error);
    std::string digest;
    if ((cfg.flags & JOC_TASK_F_SKIP_SHA256) == 0u) {
        std::ifstream output = fs_utf8::open_input(cfg.output_path);
        if (output) {
            crypto::Sha256 hash;
            std::vector<char> block(1u << 20);
            for (;;) {
                output.read(block.data(), static_cast<std::streamsize>(block.size()));
                const std::streamsize count = output.gcount();
                if (count <= 0) {
                    break;
                }
                hash.update(block.data(), static_cast<std::size_t>(count));
            }
            digest = hash.finish_hex();
        }
    }
    if (out != nullptr) {
        out->status = JOC_TASK_OK;
        out->error_code = 0;
        out->input_frames = processed;
        out->output_samples = written_samples;
        out->duration_sec = static_cast<double>(processed) * static_cast<std::int64_t>(JOC_FRAME_SAMPLES) / 48000.0;
        out->output_format_actual = output_format_actual;
        out->output_peak = peak;
        out->output_over_unity_values = over_unity;
        out->output_file_bytes = file_bytes;
        std::snprintf(out->output_sha256, sizeof(out->output_sha256), "%s", digest.c_str());
        out->oamd_payloads = binaural_enabled ? binaural.timeline().payload_count() : out->oamd_payloads;
        out->oamd_transitions =
            binaural_enabled ? binaural.timeline().transition_count() : out->oamd_transitions;
        out->t_decode_bed = t_decode;
        out->t_render = t_render;
        out->t_write = write_seconds;
        out->t_render_dsp = dsp_seconds;
        out->t_write_file = write_seconds + finalize_seconds;
        out->t_total = std::chrono::duration<double>(Clock::now() - started).count();
    }
    if (speaker.handle != nullptr) {
        ejoc_speaker_renderer_destroy(speaker.handle);
        speaker.handle = nullptr;
    }
    if (!temp_dir.empty()) {
        std::error_code ignored;
        fs::remove_all(fs_utf8::to_path(temp_dir), ignored);
    }
    bus.emit(JOC_EV_OUTPUT_FINALIZED, JOC_STAGE_OUTPUT, JOC_LOG_INFO,
             fmt("%s (%llu bytes, sha256 %s)", cfg.output_path.c_str(),
                 static_cast<unsigned long long>(file_bytes),
                 digest.empty() ? "skipped" : digest.substr(0, 16).c_str()));
    bus.emit(JOC_EV_TASK_COMPLETED, JOC_STAGE_DONE, JOC_LOG_INFO,
             fmt("done: %llu frames, %llu output samples, %.2f s wall clock",
                 static_cast<unsigned long long>(processed),
                 static_cast<unsigned long long>(written_samples),
                 std::chrono::duration<double>(Clock::now() - started).count()));
    return Status::success();
}

Status result_to_json(const joc_task_result& result, std::string* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput, "null output");
    }
    *out = fmt(
        "{\"status\":%u,\"error_code\":%u,\"error_stage\":\"%s\",\"error_message\":\"%s\","
        "\"input_frames\":%llu,\"output_samples\":%llu,\"duration_sec\":%.6f,"
        "\"output_format_actual\":%u,\"output_peak\":%.9g,\"output_over_unity_values\":%llu,"
        "\"output_file_bytes\":%llu,\"output_sha256\":\"%s\",\"oamd_payloads\":%llu,"
        "\"oamd_transitions\":%llu,\"t_decode_bed\":%.6f,\"t_render\":%.6f,\"t_write\":%.6f,"
        "\"t_render_dsp\":%.6f,\"t_write_file\":%.6f,\"t_total\":%.6f}",
        result.status, result.error_code, result.error_stage, result.error_message,
        static_cast<unsigned long long>(result.input_frames),
        static_cast<unsigned long long>(result.output_samples), result.duration_sec,
        result.output_format_actual, result.output_peak,
        static_cast<unsigned long long>(result.output_over_unity_values),
        static_cast<unsigned long long>(result.output_file_bytes), result.output_sha256,
        static_cast<unsigned long long>(result.oamd_payloads),
        static_cast<unsigned long long>(result.oamd_transitions), result.t_decode_bed,
        result.t_render, result.t_write, result.t_render_dsp, result.t_write_file,
        result.t_total);
    return Status::success();
}

}  // namespace joc::task
