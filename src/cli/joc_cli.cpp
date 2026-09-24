// joc_cli -- command line frontend, argument-compatible with the reference
// Python CLI (main.py): the same positional input, the same mode selection
// (ADM BWF by default, --speaker-layout or --binaural), the same option names,
// choices and defaults, and the same default output naming under output/.
//
// Options that exist only because this build has no Python side or no Rosella
// import chain (--backend python, --sofa-hrtf, --personalized-headphone,
// metadata sidecars) fail with an explicit message instead of being ignored.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "foundation/fs_utf8.h"
#include "joc_core.h"

namespace {

namespace fs = std::filesystem;
namespace fs_utf8 = joc::fs_utf8;

constexpr double kRate = 48000.0;
constexpr int kFrameSamples = 1536;

struct Options {
    std::string input;
    std::string output;
    std::string speaker_output;
    std::string binaural_output;
    std::string speaker_layout;
    bool binaural = false;
    std::string speaker_format = "float32";
    std::string binaural_format = "float32";
    std::string clip_action = "ask";
    int speaker_metadata_offset = 1473;
    std::string binaural_mode = "mid";
    std::string sofa_hrtf;
    std::string compiled_hrtf_cache;
    std::string personalized_headphone;
    bool personalized_headphone_used = false;
    std::string hrtf_cache_policy;
    std::string hrtf_cache_dir;
    double hrtf_radius_m = 1.0;
    double binaural_tail_seconds = 5.0;
    double binaural_tail_threshold = 1.0e-8;
    int binaural_chunk_frames = 64;
    double gain_db = 0.0;
    double duration = 0.0;
    bool duration_set = false;
    int object_delay_samples = 1473;
    std::string trajectory_mode = "compact";
    std::string ffmpeg;
    double eac3_drc_scale = 0.0;
    int eac3_target_level = 0;
    std::string backend = "auto";
    std::string native_library;
    int native_threads = 0;
    bool native_threads_set = false;
    std::string metadata_dir;
    std::string metadata_cache;
    std::string metadata_backend = "auto";
    std::string print_metadata = "none";
    std::string metadata_json;
    bool metadata_only = false;
    bool keep_raw = false;
    bool skip_sha256 = false;
    int progress_every = 1000;
    // C++-side additions (documented as such; the Python CLI has no equivalent).
    std::string bed;
    std::string kernels;
    std::string work_dir;
    std::string report_json;
    bool report_json_set = false;
    bool dry_run = false;
    bool quiet = false;
    bool help = false;
};

const char* kLayoutChoices =
    "2.0 3.0 3.1 4.0 5.0 5.1 5.1.2 5.1.4 6.1 7.0 7.1 7.1.2 7.1.4 9.1.4 9.1.6 22.2";

void print_usage() {
    std::printf(
        "usage: joc_cli [options] input\n"
        "\n"
        "JustOneCacophony (JOC)：E-AC-3 JOC → 25ch ADM BWF、扬声器 WAV 或双耳 WAV\n"
        "\n"
        "位置参数:\n"
        "  input                 输入 .m4a/.eac3/.ec3\n"
        "\n"
        "模式（默认输出 ADM BWF）:\n"
        "  --speaker-layout L    直接扬声器渲染布局，例如 2.0、5.1、7.1.2\n"
        "                        可选值: %s\n"
        "  --binaural            直接双耳渲染；不生成临时 ADM BWF\n"
        "\n"
        "输出:\n"
        "  -o, --output PATH     输出文件；默认 output/<名称>.adm.wav、\n"
        "                        output/<名称>.<布局>.wav 或 output/<名称>.binaural.wav\n"
        "  --speaker-output PATH 扬声器 WAV 路径；仅与 --speaker-layout 一起使用\n"
        "  --binaural-output PATH 双耳 WAV 路径；仅与 --binaural 一起使用\n"
        "  --speaker-format F    扬声器 WAV 格式 float32|int24，默认 float32\n"
        "  --binaural-format F   双耳 WAV 格式 float32|int24，默认 float32\n"
        "  --clip-action A       int24 削波处理 ask|continue|float32|abort，默认 ask\n"
        "\n"
        "渲染:\n"
        "  --speaker-metadata-offset N  扬声器渲染 metadata 相对帧偏移，默认 1473 samples\n"
        "  --binaural-mode M     双耳渲染模式 off|near|mid|far，默认 mid；\n"
        "                        off 仅用于 ADM BWF（关闭 DBMD 双耳提示）\n"
        "  --sofa-hrtf PATH      SOFA SimpleFreeFieldHRIR 输入；.jochrtf 由本工具内部编译\n"
        "  --personalized-headphone [PATH]  Rosella 个性化模型，默认 "
        "HRTF/binaural.personalized_headphone\n"
        "  --compiled-hrtf-cache PATH   直接读取 .jochrtf（高级用法，跳过 SOFA 编译）\n"
        "  --hrtf-cache-policy P SOFA 编译缓存策略 none|memory|disk，默认 memory\n"
        "  --hrtf-cache-dir DIR  disk cache 目录，默认 <exe>/output/hrtf-cache\n"
        "  --hrtf-radius-m R     选择最近的 SOFA measurement-radius shell，默认 1.0 m\n"
        "  --binaural-tail-seconds S    双耳 room/filterbank flush 上限，默认 5 秒\n"
        "  --binaural-tail-threshold T  双耳尾声裁切阈值，默认 1e-8；主体至少保留原时长\n"
        "  --binaural-chunk-frames N    双耳内部批处理帧数，默认 64（本构建按 512 块渲染，\n"
        "                               取值不影响输出）\n"
        "  --gain-db X           成品增益 dB，默认 0；双耳路径以 float64 应用\n"
        "  --duration S          只处理开头指定秒数\n"
        "  --object-delay-samples N     对象 PCM/OAMD 时间补偿，默认 1473 samples\n"
        "  --trajectory-mode M   ADM 对象轨迹表示 compact|dense64，默认 compact\n"
        "\n"
        "输入与解码:\n"
        "  --ffmpeg PATH         ffmpeg 可执行文件，默认取 FFMPEG 环境变量或 PATH\n"
        "  --eac3-drc-scale X    E-AC-3 解码器 -drc_scale，0=关闭码流 dynrng，默认 0\n"
        "  --eac3-target-level N E-AC-3 解码器 -target_level，0=不施加，默认 0\n"
        "  --backend B           JOC/扬声器 DSP 后端 auto|native；本构建无 python 后端\n"
        "  --native-threads N    原生 DSP 总线程数；默认在 4 核以上使用 2\n"
        "\n"
        "诊断:\n"
        "  --print-metadata M    诊断元数据输出 none|summary|frames，默认 none\n"
        "  --metadata-json PATH  元数据汇总 JSON 路径\n"
        "  --metadata-only       解析/打印元数据后退出\n"
        "  --keep-raw            额外保留 16ch f32le 对象中间文件\n"
        "  --skip-sha256         跳过最终文件 SHA-256 全量复扫\n"
        "  --progress-every N    进度输出间隔，默认 1000 帧（渲染与收尾写盘同一节奏）\n"
        "\n"
        "本构建特有（Python 版没有对应参数）:\n"
        "  --bed PATH            已解码的 6 通道 float32 PCM；给出后不调用 ffmpeg 解码\n"
        "  --kernels PATH        双耳滤波器组表 rosella_kernels.npz\n"
        "  --work-dir DIR        临时目录\n"
        "  --report-json PATH    结果 JSON 路径；默认 <输出>.report.json\n"
        "  --dry-run             只校验配置\n"
        "  --quiet               只输出警告与错误\n",
        kLayoutChoices);
}

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::string require_value(const std::vector<std::string>& arguments, int* index) {
    if (static_cast<std::size_t>(*index) + 1u >= arguments.size()) {
        fail("argument " + arguments[static_cast<std::size_t>(*index)] +
             ": expected one argument");
    }
    return arguments[static_cast<std::size_t>(++(*index))];
}

double to_double(const std::string& text, const char* name) {
    try {
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        if (used != text.size()) {
            fail(std::string(name) + ": invalid float value: " + text);
        }
        return value;
    } catch (const std::exception&) {
        fail(std::string(name) + ": invalid float value: " + text);
    }
}

long long to_int(const std::string& text, const char* name) {
    try {
        std::size_t used = 0;
        const long long value = std::stoll(text, &used);
        if (used != text.size()) {
            fail(std::string(name) + ": invalid int value: " + text);
        }
        return value;
    } catch (const std::exception&) {
        fail(std::string(name) + ": invalid int value: " + text);
    }
}

void check_choice(const std::string& value, const char* name,
                  std::initializer_list<const char*> allowed) {
    for (const char* candidate : allowed) {
        if (value == candidate) {
            return;
        }
    }
    std::string list;
    for (const char* candidate : allowed) {
        list += list.empty() ? candidate : (", " + std::string(candidate));
    }
    fail(std::string(name) + ": invalid choice: '" + value + "' (choose from " + list + ")");
}

void parse_args(const std::vector<std::string>& arguments, Options* options) {
    std::vector<std::string> positional;
    const int argc = static_cast<int>(arguments.size());
    for (int index = 1; index < argc; ++index) {
        const std::string arg = arguments[static_cast<std::size_t>(index)];
        if (arg == "-h" || arg == "--help") { options->help = true; }
        else if (arg == "-o" || arg == "--output") { options->output = require_value(arguments, &index); }
        else if (arg == "--speaker-output") { options->speaker_output = require_value(arguments, &index); }
        else if (arg == "--binaural-output") { options->binaural_output = require_value(arguments, &index); }
        else if (arg == "--speaker-layout") { options->speaker_layout = require_value(arguments, &index); }
        else if (arg == "--binaural") { options->binaural = true; }
        else if (arg == "--speaker-format") { options->speaker_format = require_value(arguments, &index); }
        else if (arg == "--binaural-format") { options->binaural_format = require_value(arguments, &index); }
        else if (arg == "--clip-action") { options->clip_action = require_value(arguments, &index); }
        else if (arg == "--speaker-metadata-offset") {
            options->speaker_metadata_offset = static_cast<int>(
                to_int(require_value(arguments, &index), "--speaker-metadata-offset"));
        }
        else if (arg == "--binaural-mode") { options->binaural_mode = require_value(arguments, &index); }
        else if (arg == "--sofa-hrtf") { options->sofa_hrtf = require_value(arguments, &index); }
        else if (arg == "--compiled-hrtf-cache") { options->compiled_hrtf_cache = require_value(arguments, &index); }
        else if (arg == "--personalized-headphone") {
            options->personalized_headphone_used = true;
            // nargs="?": the path is optional, so the next token may be the input.
            // Without a path the executable-anchored default is resolved later.
            if (static_cast<std::size_t>(index) + 1u < arguments.size() &&
                arguments[static_cast<std::size_t>(index) + 1u][0] != '-') {
                options->personalized_headphone = require_value(arguments, &index);
            }
        }
        else if (arg == "--hrtf-cache-policy") { options->hrtf_cache_policy = require_value(arguments, &index); }
        else if (arg == "--hrtf-cache-dir") { options->hrtf_cache_dir = require_value(arguments, &index); }
        else if (arg == "--hrtf-radius-m") { options->hrtf_radius_m = to_double(require_value(arguments, &index), "--hrtf-radius-m"); }
        else if (arg == "--binaural-tail-seconds") { options->binaural_tail_seconds = to_double(require_value(arguments, &index), "--binaural-tail-seconds"); }
        else if (arg == "--binaural-tail-threshold") { options->binaural_tail_threshold = to_double(require_value(arguments, &index), "--binaural-tail-threshold"); }
        else if (arg == "--binaural-chunk-frames") { options->binaural_chunk_frames = static_cast<int>(to_int(require_value(arguments, &index), "--binaural-chunk-frames")); }
        else if (arg == "--gain-db") { options->gain_db = to_double(require_value(arguments, &index), "--gain-db"); }
        else if (arg == "--duration") { options->duration = to_double(require_value(arguments, &index), "--duration"); options->duration_set = true; }
        else if (arg == "--object-delay-samples") { options->object_delay_samples = static_cast<int>(to_int(require_value(arguments, &index), "--object-delay-samples")); }
        else if (arg == "--trajectory-mode") { options->trajectory_mode = require_value(arguments, &index); }
        else if (arg == "--ffmpeg") { options->ffmpeg = require_value(arguments, &index); }
        else if (arg == "--eac3-drc-scale") { options->eac3_drc_scale = to_double(require_value(arguments, &index), "--eac3-drc-scale"); }
        else if (arg == "--eac3-target-level") { options->eac3_target_level = static_cast<int>(to_int(require_value(arguments, &index), "--eac3-target-level")); }
        else if (arg == "--backend") { options->backend = require_value(arguments, &index); }
        else if (arg == "--native-library") { options->native_library = require_value(arguments, &index); }
        else if (arg == "--native-threads") { options->native_threads = static_cast<int>(to_int(require_value(arguments, &index), "--native-threads")); options->native_threads_set = true; }
        else if (arg == "--metadata-dir") { options->metadata_dir = require_value(arguments, &index); }
        else if (arg == "--metadata-cache") { options->metadata_cache = require_value(arguments, &index); }
        else if (arg == "--metadata-backend") { options->metadata_backend = require_value(arguments, &index); }
        else if (arg == "--print-metadata") { options->print_metadata = require_value(arguments, &index); }
        else if (arg == "--metadata-json") { options->metadata_json = require_value(arguments, &index); }
        else if (arg == "--metadata-only") { options->metadata_only = true; }
        else if (arg == "--keep-raw") { options->keep_raw = true; }
        else if (arg == "--skip-sha256") { options->skip_sha256 = true; }
        else if (arg == "--progress-every") { options->progress_every = static_cast<int>(to_int(require_value(arguments, &index), "--progress-every")); }
        else if (arg == "--bed") { options->bed = require_value(arguments, &index); }
        else if (arg == "--kernels") { options->kernels = require_value(arguments, &index); }
        else if (arg == "--work-dir") { options->work_dir = require_value(arguments, &index); }
        else if (arg == "--report-json") { options->report_json = require_value(arguments, &index); options->report_json_set = true; }
        else if (arg == "--dry-run") { options->dry_run = true; }
        else if (arg == "--quiet") { options->quiet = true; }
        else if (!arg.empty() && arg[0] == '-' && arg != "-") { fail("unrecognized argument: " + arg); }
        else { positional.push_back(arg); }
    }
    if (positional.size() > 1u) {
        fail("unrecognized extra arguments: " + positional[1] +
             (positional.size() > 2u ? " ..." : ""));
    }
    if (!positional.empty()) {
        options->input = positional.front();
    }
}

// Mirrors the reference resolve_output(): <project>/output plus a mode-specific
// name.  The project directory is the executable's directory, as upstream uses
// the script's directory, so the layout does not depend on the working directory.
std::string resolve_output(const Options& options, const std::string& source,
                           const std::string& executable_directory) {
    const std::string requested = !options.speaker_output.empty() ? options.speaker_output
                                  : !options.binaural_output.empty() ? options.binaural_output
                                                                     : options.output;
    if (!requested.empty()) {
        std::error_code error;
        const fs::path absolute = fs::absolute(fs_utf8::to_path(requested), error);
        return error ? requested : fs_utf8::from_path(absolute);
    }
    const fs::path directory = fs_utf8::to_path(executable_directory) / "output";
    const std::string stem = fs_utf8::from_path(fs_utf8::to_path(source).stem());
    if (!options.speaker_layout.empty()) {
        return fs_utf8::from_path(directory /
                                  fs_utf8::to_path(stem + "." + options.speaker_layout + ".wav"));
    }
    if (options.binaural) {
        return fs_utf8::from_path(directory / fs_utf8::to_path(stem + ".binaural.wav"));
    }
    return fs_utf8::from_path(directory / fs_utf8::to_path(stem + ".adm.wav"));
}

// The project directory the reference anchors its defaults at: the directory of
// the running executable, never the working directory.
std::string executable_dir(const std::string& argv0) {
    const std::string own_path = fs_utf8::executable_path();
    if (!own_path.empty()) {
        const fs::path path = fs_utf8::to_path(own_path);
        if (path.has_parent_path()) {
            return fs_utf8::from_path(path.parent_path());
        }
    }
    if (argv0.empty()) {
        return ".";
    }
    std::error_code error;
    const fs::path path = fs::absolute(fs_utf8::to_path(argv0), error);
    if (error || path.empty()) {
        return ".";
    }
    return fs_utf8::from_path(path.parent_path());
}

std::string find_kernels(const Options& options, const std::string& argv0) {
    (void)argv0;
    if (!options.kernels.empty() && !fs_utf8::exists(options.kernels)) {
        fail("--kernels 指向的文件不存在: " + options.kernels);
    }
    // Empty means the tables compiled into the library.
    return options.kernels;
}

// Mirrors the reference binaural HRTF resolution (main.py:89-160): the SOFA file
// is the user-facing input and the .jochrtf is only its compiled cache.  Paths
// are anchored at the executable directory, as the reference anchors them at the
// project directory.
struct HrtfInput {
    std::string sofa_path;      // compile this
    std::string compiled_path;  // or read this .jochrtf directly
    std::string cache_dir;      // disk policy directory
    std::string personalized_path;  // Rosella .personalized_headphone
    bool disk = false;
};

std::string resolve_compiled_hrtf(const Options& options, const std::string& project_directory) {
    if (!options.compiled_hrtf_cache.empty()) {
        return options.compiled_hrtf_cache;
    }
    const std::string directory_utf8 =
        options.hrtf_cache_dir.empty()
            ? fs_utf8::from_path(fs_utf8::to_path(project_directory) / "output" / "hrtf-cache")
            : options.hrtf_cache_dir;
    if (!fs_utf8::is_directory(directory_utf8)) {
        return std::string();
    }
    const fs::path directory = fs_utf8::to_path(directory_utf8);
    std::vector<fs::path> candidates;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jochrtf") {
            candidates.push_back(entry.path());
        }
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() > 1u) {
        fail(directory_utf8 +
             " 下有多个 .jochrtf 缓存，无法自动选择；请用 --sofa-hrtf PATH 或 "
             "--compiled-hrtf-cache PATH 显式指定");
    }
    return candidates.empty() ? std::string() : fs_utf8::from_path(candidates.front());
}

HrtfInput resolve_hrtf_input(const Options& options, const std::string& project_directory) {
    HrtfInput input;
    const std::string default_sofa =
        fs_utf8::from_path(fs_utf8::to_path(project_directory) / "HRTF" / "binaural.sofa");
    const std::string default_private = fs_utf8::from_path(
        fs_utf8::to_path(project_directory) / "HRTF" / "binaural.personalized_headphone");
    const std::string default_cache_dir =
        fs_utf8::from_path(fs_utf8::to_path(project_directory) / "output" / "hrtf-cache");

    if (!options.compiled_hrtf_cache.empty() && !options.hrtf_cache_policy.empty()) {
        fail("显式 .jochrtf 输入不能再指定 --hrtf-cache-policy");
    }
    if (!options.compiled_hrtf_cache.empty() && options.hrtf_radius_m != 1.0) {
        fail("显式 .jochrtf 输入不能再选择 SOFA radius shell");
    }
    if (options.personalized_headphone_used &&
        (!options.hrtf_cache_policy.empty() || !options.hrtf_cache_dir.empty() ||
         options.hrtf_radius_m != 1.0)) {
        fail("Rosella 模型输入不能使用 --hrtf-cache-policy/--hrtf-cache-dir/--hrtf-radius-m");
    }

    std::string sofa = options.sofa_hrtf;
    std::string compiled = options.compiled_hrtf_cache;
    std::string personalized =
        options.personalized_headphone_used ? options.personalized_headphone : std::string();
    if (options.personalized_headphone_used && personalized.empty()) {
        // "--personalized-headphone" without a path means the project default.
        personalized = default_private;
    }
    if (sofa.empty() && compiled.empty() && personalized.empty()) {
        // The reference order: the SOFA file, then the unique compiled cache, then the
        // personalized model.
        if (fs_utf8::exists(default_sofa)) {
            sofa = default_sofa;
        } else {
            compiled = resolve_compiled_hrtf(options, project_directory);
            if (compiled.empty() && fs_utf8::exists(default_private)) {
                personalized = default_private;
            }
        }
    }
    if (!personalized.empty()) {
        if (!fs_utf8::exists(personalized)) {
            fail("双耳模型不存在: " + personalized);
        }
        input.personalized_path = personalized;
        return input;
    }
    if (sofa.empty() && compiled.empty()) {
        if (!options.hrtf_cache_policy.empty() || !options.hrtf_cache_dir.empty() ||
            options.hrtf_radius_m != 1.0) {
            fail("HRTF cache/radius 选项需要 --sofa-hrtf");
        }
        fail("--binaural 未找到 HRTF 输入：默认 " + default_sofa + "、" + default_private +
             " 或 " + default_cache_dir +
             " 下的 .jochrtf 都不存在，请用 --sofa-hrtf PATH、--personalized-headphone PATH "
             "或 --compiled-hrtf-cache PATH 指定");
    }
    if (sofa.empty() && (!options.hrtf_cache_policy.empty() || !options.hrtf_cache_dir.empty() ||
                         options.hrtf_radius_m != 1.0)) {
        fail("HRTF cache/radius 选项需要 --sofa-hrtf");
    }
    if (sofa.empty()) {
        input.compiled_path = compiled;
        return input;
    }
    const std::string effective_policy =
        options.hrtf_cache_policy.empty() ? "memory" : options.hrtf_cache_policy;
    if (!options.hrtf_cache_dir.empty() && effective_policy != "disk") {
        fail("--hrtf-cache-dir 需要 SOFA 与 disk cache policy 一起使用");
    }
    input.sofa_path = sofa;
    input.disk = effective_policy == "disk";
    input.cache_dir = options.hrtf_cache_dir.empty() ? default_cache_dir : options.hrtf_cache_dir;
    if (!fs_utf8::exists(input.sofa_path)) {
        fail("SOFA HRTF 不存在: " + input.sofa_path);
    }
    return input;
}

std::uint32_t binaural_mode_value(const std::string& name) {
    if (name == "off") { return JOC_BINAURAL_OFF; }
    if (name == "near") { return JOC_BINAURAL_NEAR; }
    if (name == "far") { return JOC_BINAURAL_FAR; }
    return JOC_BINAURAL_MID;
}

std::uint32_t clip_action_value(const std::string& name) {
    if (name == "continue") { return JOC_CLIP_CONTINUE; }
    if (name == "float32") { return JOC_CLIP_FLOAT32; }
    if (name == "abort") { return JOC_CLIP_ABORT; }
    return JOC_CLIP_ASK;
}

std::string format_eta(double seconds) {
    if (seconds < 0.0 || seconds > 86400.0) {
        return "--";
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.0fs", seconds);
    return buffer;
}

void JOC_CALL on_event(void* user, const joc_event* event) {
    const Options* options = static_cast<const Options*>(user);
    if (event == nullptr) {
        return;
    }
    switch (event->type) {
        case JOC_EV_PROGRESS: {
            if (options->quiet) {
                return;
            }
            const double fraction = event->progress >= 0.0 ? event->progress : 0.0;
            const double remaining =
                fraction > 0.0 ? event->elapsed_seconds * (1.0 - fraction) / fraction : -1.0;
            std::printf("[%s] %llu/%llu  %.1fx realtime  ETA %s\n", event->stage_name,
                        static_cast<unsigned long long>(event->current_frame),
                        static_cast<unsigned long long>(event->total_frames),
                        event->realtime_factor, format_eta(remaining).c_str());
            std::fflush(stdout);
            return;
        }
        case JOC_EV_LOG: {
            if (options->quiet || event->log_level < JOC_LOG_INFO || event->message[0] == '\0') {
                return;
            }
            std::printf("[%s] %s\n", event->stage_name, event->message);
            std::fflush(stdout);
            return;
        }
        case JOC_EV_WARNING:
            std::printf("[warning] %s\n", event->message);
            return;
        case JOC_EV_ERROR:
            std::fprintf(stderr, "[error] %s (%s)\n", event->message,
                         joc_error_name(event->error_code));
            return;
        default:
            if (options->quiet || event->log_level < JOC_LOG_INFO || event->message[0] == '\0') {
                return;
            }
            std::printf("[%s] %s\n", event->stage_name, event->message);
            return;
    }
}

}  // namespace

int main(int argc, char** argv) {
    fs_utf8::configure_console();
    const std::vector<std::string> arguments = fs_utf8::command_line_arguments(argc, argv);
    Options options;
    try {
        parse_args(arguments, &options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "joc_cli: error: %s\n", error.what());
        return 2;
    }
    if (options.help) {
        print_usage();
        return 0;
    }
    if (options.input.empty()) {
        print_usage();
        return 2;
    }

    try {
        check_choice(options.speaker_format, "--speaker-format", {"float32", "int24"});
        check_choice(options.binaural_format, "--binaural-format", {"float32", "int24"});
        check_choice(options.clip_action, "--clip-action",
                     {"ask", "continue", "float32", "abort"});
        check_choice(options.binaural_mode, "--binaural-mode", {"off", "near", "mid", "far"});
        check_choice(options.trajectory_mode, "--trajectory-mode", {"compact", "dense64"});
        check_choice(options.backend, "--backend", {"auto", "native", "python"});
        check_choice(options.print_metadata, "--print-metadata", {"none", "summary", "frames"});
        check_choice(options.metadata_backend, "--metadata-backend", {"auto", "emdf", "sidecar"});
        if (!options.hrtf_cache_policy.empty()) {
            check_choice(options.hrtf_cache_policy, "--hrtf-cache-policy",
                         {"none", "memory", "disk"});
        }

        const bool speaker_mode = !options.speaker_layout.empty();
        const bool binaural_mode = options.binaural;
        if (speaker_mode && binaural_mode) {
            fail("argument --binaural: not allowed with argument --speaker-layout");
        }
        if (options.binaural_mode == "off" && (speaker_mode || binaural_mode)) {
            fail("--binaural-mode off 仅用于 ADM BWF 输出（关闭 DBMD 双耳提示）；"
                 "直接双耳渲染请使用 near/mid/far");
        }
        if (!options.speaker_output.empty() && !speaker_mode) {
            fail("--speaker-output 必须与 --speaker-layout 一起使用");
        }
        if (!options.binaural_output.empty() && !binaural_mode) {
            fail("--binaural-output 必须与 --binaural 一起使用");
        }
        const bool specific_output =
            !options.speaker_output.empty() || !options.binaural_output.empty();
        if (!options.output.empty() && specific_output) {
            fail("-o/--output 与 --speaker-output/--binaural-output 不能同时使用");
        }
        if (!options.speaker_output.empty() && !options.binaural_output.empty()) {
            fail("--speaker-output 与 --binaural-output 不能同时使用");
        }
        if (options.speaker_metadata_offset < 0) {
            fail("speaker-metadata-offset 不能为负数");
        }
        const bool hrtf_options_used =
            !options.sofa_hrtf.empty() || !options.compiled_hrtf_cache.empty() ||
            options.personalized_headphone_used || !options.hrtf_cache_policy.empty() ||
            !options.hrtf_cache_dir.empty() || options.hrtf_radius_m != 1.0;
        if (hrtf_options_used && !binaural_mode) {
            fail("SOFA/HRTF 选项仅与 --binaural 一起使用");
        }
        if (!std::isfinite(options.binaural_tail_seconds) ||
            options.binaural_tail_seconds < 0.0) {
            fail("binaural-tail-seconds 必须是非负有限值");
        }
        if (!std::isfinite(options.binaural_tail_threshold) ||
            options.binaural_tail_threshold < 0.0) {
            fail("binaural-tail-threshold 必须是非负有限值");
        }
        if (options.binaural_chunk_frames <= 0) {
            fail("binaural-chunk-frames 必须大于 0");
        }
        if (!std::isfinite(options.hrtf_radius_m) || options.hrtf_radius_m <= 0.0) {
            fail("hrtf-radius-m 必须是正有限值");
        }
        if (options.duration_set && options.duration <= 0.0) {
            fail("duration 必须大于 0");
        }
        if (options.object_delay_samples < 0) {
            fail("object-delay-samples 不能为负数");
        }
        if (options.native_threads_set && options.native_threads < 1) {
            fail("native-threads 必须大于 0");
        }
        if (!std::isfinite(options.gain_db) || std::abs(options.gain_db) > 200.0) {
            fail("gain-db 超出支持范围");
        }
        // Options this build cannot honour: fail loudly instead of ignoring them.
        if (options.backend == "python") {
            fail("--backend python 在本构建中不可用（已无 Python 后端）；请使用 auto 或 native");
        }
        if (options.metadata_backend == "sidecar" || !options.metadata_dir.empty() ||
            !options.metadata_cache.empty()) {
            fail("metadata sidecar 在本构建中不可用（始终直接扫描 EMDF）");
        }
        if (!options.native_library.empty()) {
            std::fprintf(stderr, "[info] --native-library 在本构建中忽略（单一 joc_core.dll）\n");
        }
        if (!fs_utf8::exists(options.input)) {
            fail("输入文件不存在: " + options.input);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "joc_cli: error: %s\n", error.what());
        return 2;
    }

    const bool speaker_mode = !options.speaker_layout.empty();
    const bool binaural_mode = options.binaural;
    const std::string project_directory =
        executable_dir(arguments.empty() ? std::string() : arguments.front());
    const std::string output_path = resolve_output(options, options.input, project_directory);
    std::error_code directory_error;
    fs::create_directories(fs_utf8::to_path(output_path).parent_path(), directory_error);

    joc_task_config config{};
    config.struct_size = sizeof(config);
    config.struct_version = JOC_TASK_CONFIG_VERSION;
    config.input_path = options.input.c_str();
    config.output_path = output_path.c_str();
    config.ffmpeg_path = options.ffmpeg.empty() ? nullptr : options.ffmpeg.c_str();
    config.bed_path = options.bed.empty() ? nullptr : options.bed.c_str();
    config.work_dir = options.work_dir.empty() ? nullptr : options.work_dir.c_str();
    config.eac3_drc_scale = options.eac3_drc_scale;
    config.eac3_target_level = options.eac3_target_level;
    config.operation =
        binaural_mode ? JOC_OP_BINAURAL : speaker_mode ? JOC_OP_SPEAKER : JOC_OP_ADM_BWF;
    const std::string& requested_format =
        binaural_mode ? options.binaural_format : options.speaker_format;
    config.output_format =
        requested_format == "int24" ? JOC_FORMAT_PCM24 : JOC_FORMAT_FLOAT32;
    config.clip_action = clip_action_value(options.clip_action);
    config.speaker_layout_name = speaker_mode ? options.speaker_layout.c_str() : nullptr;
    config.speaker_metadata_offset = static_cast<std::uint32_t>(options.speaker_metadata_offset);
    config.binaural_mode = binaural_mode_value(options.binaural_mode);
    config.adm_binaural_mode = binaural_mode_value(options.binaural_mode);
    config.binaural_tail_seconds = options.binaural_tail_seconds;
    config.binaural_tail_threshold = options.binaural_tail_threshold;
    config.binaural_chunk_frames = static_cast<std::uint32_t>(options.binaural_chunk_frames);
    config.object_delay_samples = static_cast<std::uint32_t>(options.object_delay_samples);
    config.trajectory_mode =
        options.trajectory_mode == "dense64" ? JOC_TRAJECTORY_DENSE64 : JOC_TRAJECTORY_COMPACT;
    config.gain_db = options.gain_db;
    config.progress_interval_frames = static_cast<std::uint32_t>(options.progress_every);
    config.native_threads =
        options.native_threads_set ? static_cast<std::uint32_t>(options.native_threads) : 0u;
    config.print_metadata = options.print_metadata == "frames"     ? 2u
                            : options.print_metadata == "summary"  ? 1u
                                                                   : 0u;
    config.metadata_json_path =
        options.metadata_json.empty() ? nullptr : options.metadata_json.c_str();
    config.duration_frames =
        options.duration_set
            ? static_cast<std::uint64_t>(
                  std::ceil(options.duration * kRate / static_cast<double>(kFrameSamples)))
            : 0u;
    config.flags = 0u;
    if (options.skip_sha256) { config.flags |= JOC_TASK_F_SKIP_SHA256; }
    if (options.keep_raw) { config.flags |= JOC_TASK_F_KEEP_INTERMEDIATE; }
    if (options.metadata_only) { config.flags |= JOC_TASK_F_METADATA_ONLY; }
    if (options.quiet) { config.flags |= JOC_TASK_F_QUIET; }

    std::string hrtf_path;
    std::string hrtf_sofa_path;
    std::string hrtf_cache_dir;
    std::string personalized_path;
    std::string kernels_path;
    if (binaural_mode) {
        try {
            const HrtfInput input = resolve_hrtf_input(options, project_directory);
            hrtf_path = input.compiled_path;
            hrtf_sofa_path = input.sofa_path;
            hrtf_cache_dir = input.cache_dir;
            personalized_path = input.personalized_path;
            kernels_path = find_kernels(options, arguments.empty() ? std::string()
                                                                   : arguments.front());
        } catch (const std::exception& error) {
            std::fprintf(stderr, "joc_cli: error: %s\n", error.what());
            return 2;
        }
    }
    config.hrtf_path = hrtf_path.empty() ? nullptr : hrtf_path.c_str();
    config.hrtf_sofa_path = hrtf_sofa_path.empty() ? nullptr : hrtf_sofa_path.c_str();
    config.hrtf_cache_dir = hrtf_cache_dir.empty() ? nullptr : hrtf_cache_dir.c_str();
    config.personalized_headphone_path =
        personalized_path.empty() ? nullptr : personalized_path.c_str();
    config.hrtf_cache_policy = options.hrtf_cache_policy == "disk"    ? JOC_HRTF_CACHE_DISK
                               : options.hrtf_cache_policy == "none" ? JOC_HRTF_CACHE_NONE
                                                                    : JOC_HRTF_CACHE_MEMORY;
    config.hrtf_radius_m = options.hrtf_radius_m;
    config.kernels_path = kernels_path.empty() ? nullptr : kernels_path.c_str();

    joc_validation_issue issues[32];
    std::uint32_t issue_count = 0;
    const joc_error validated = joc_task_validate(&config, issues, 32u, &issue_count);
    for (std::uint32_t index = 0; index < std::min(issue_count, 32u); ++index) {
        if (issues[index].severity >= 2u) {
            std::fprintf(stderr, "[error] %s: %s\n", issues[index].field, issues[index].message);
        } else if (!options.quiet) {
            std::fprintf(stderr, "[warning] %s: %s\n", issues[index].field,
                         issues[index].message);
        }
    }
    if (validated != JOC_OK) {
        std::fprintf(stderr, "joc_cli: error: configuration rejected (%u issue(s))\n", issue_count);
        return 2;
    }
    if (options.dry_run) {
        std::printf("configuration accepted (%u issue(s))\n", issue_count);
        return 0;
    }

    joc_event_sink sink{};
    sink.struct_size = sizeof(sink);
    sink.callback = &on_event;
    sink.user = &options;

    if (!options.quiet) {
        const char* mode_name = binaural_mode ? "binaural" : speaker_mode ? "speaker" : "adm";
        std::printf("[cli] %s -> %s (%s)\n", options.input.c_str(), output_path.c_str(),
                    mode_name);
        std::fflush(stdout);
    }

    joc_task_result result{};
    const joc_error status = joc_task_execute(&config, &sink, &result);

    const std::string report_path =
        options.report_json_set ? options.report_json : (output_path + ".report.json");
    {
        std::size_t needed = 0;
        joc_task_result_to_json(&result, nullptr, 0u, &needed);
        std::vector<char> buffer(needed + 1u);
        if (joc_task_result_to_json(&result, buffer.data(), buffer.size(), &needed) == JOC_OK) {
            if (std::FILE* file = fs_utf8::fopen(report_path, "wb")) {
                std::fwrite(buffer.data(), 1, std::strlen(buffer.data()), file);
                std::fputc('\n', file);
                std::fclose(file);
            }
        }
    }

    std::printf("\nresult: %s\n", status == JOC_OK ? "ok" : joc_error_name(status));
    std::printf("  output        : %s\n", output_path.c_str());
    std::printf("  frames        : %llu (%.2f s)\n",
                static_cast<unsigned long long>(result.input_frames), result.duration_sec);
    std::printf("  output samples: %llu\n",
                static_cast<unsigned long long>(result.output_samples));
    std::printf("  output bytes  : %llu\n",
                static_cast<unsigned long long>(result.output_file_bytes));
    std::printf("  format        : %s\n",
                result.output_format_actual == JOC_FORMAT_PCM24 ? "int24" : "float32");
    std::printf("  peak          : %.9g (%llu sample(s) above full scale)\n", result.output_peak,
                static_cast<unsigned long long>(result.output_over_unity_values));
    std::printf("  sha256        : %s\n",
                result.output_sha256[0] != '\0' ? result.output_sha256 : "(skipped)");
    std::printf("  report        : %s\n", report_path.c_str());
    // Each stage time is measured where that stage actually runs, and the three
    // stages now overlap (see the pipeline in src/task/task.cpp), so the stage
    // times deliberately do not add up to the wall-clock total.
    std::printf("  timings       : decode %.2fs, joc %.2fs, dsp %.2fs, write %.2fs"
                " (stage times, concurrent), total %.2fs\n",
                result.t_decode_bed, result.t_render, result.t_render_dsp, result.t_write_file,
                result.t_total);
    if (status != JOC_OK) {
        std::fprintf(stderr, "joc_cli: error: %s: %s\n", result.error_stage, result.error_message);
    }
    return status == JOC_OK ? 0 : 1;
}
