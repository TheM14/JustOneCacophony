// Multichannel PCM WAV -> binaural WAV frontend.
//
// Ordinary 2.0/5.1/7.1 PCM has no OAMD.  Each non-LFE channel is assigned a
// fixed ADM direction, the LFE channel goes to source 0, and the resulting
// sixteen JOC sources feed the Rosella binaural runtime.  The runtime is
// native 48 kHz; other input rates are resampled on the way in and the stereo
// result is resampled to the requested output rate on the way out.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "binaural/multichannel_layout.h"
#include "binaural/pcm_resampler.h"
#include "foundation/status.h"
#include "hrtf/rosella_model.h"
#include "hrtf/rosella_renderer.h"
#include "io/wav_pcm.h"
#include "oamd/oamd_parser.h"

namespace {

constexpr std::uint32_t kFrameSamples = 1536;
constexpr std::uint32_t kSourceChannels = 16;
constexpr std::uint32_t kTargetRate = 48000;

void usage() {
    std::printf(
        "usage: joc_multichannel INPUT.wav -o OUTPUT.wav [options]\n"
        "  --layout auto|2.0|5.1|7.1   WAV channel layout (auto by channel count)\n"
        "  --model PATH                 Rosella .personalized_headphone model\n"
        "  --mode near|mid|far          Rosella binaural profile (default mid)\n"
        "  --filterbank 1|2             Rosella filterbank (default 1)\n"
        "  --tail-seconds SECONDS       room tail (default 2.0)\n"
        "  --output-rate 48000|input|N  output rate (default 48000; non-48k adds\n"
        "                               a second resampling pass)\n"
        "  --list-layouts\n"
        "\n"
        "Prefer a 48 kHz float32 WAV; let ffmpeg/libsoxr do the one conversion:\n"
        "  ffmpeg -i INPUT -af \"aresample=resampler=soxr:precision=28:osr=48000\" "
        "-c:a pcm_f32le OUT.wav\n"
        "44.1/88.2/176.4 or 48/96/192 kHz accepted; non-48k is converted once in.\n");
}

std::int16_t q_for_position(double value) {
    if (value <= -0.999) {
        return 0;
    }
    if (value >= 0.999) {
        return 32767;
    }
    return 16384;
}

void fill_initial_update(const joc::multichannel::Layout& layout, joc::oamd::OamdUpdate* update) {
    update->has_frame_state = true;
    update->object_count = kSourceChannels;
    update->frame_state.object_count = kSourceChannels;
    update->frame_state.block_count = 1;
    update->frame_state.blocks.resize(1);
    update->frame_state.objects.resize(kSourceChannels);
    double positions[15][3];
    const joc::Status status = joc::multichannel::object_positions(layout, positions);
    if (!status.ok()) {
        return;
    }
    for (int object = 0; object < static_cast<int>(kSourceChannels); ++object) {
        joc::oamd::ObjectElementState& element = update->frame_state.objects[object];
        element.kind = object == 0 ? joc::oamd::ObjectKind::kBed
                                   : joc::oamd::ObjectKind::kDynamic;
        element.blocks.resize(1);
        joc::oamd::ObjectRenderState& render = element.blocks[0].render;
        render.not_active = false;
        render.gain_q = 4096;
        render.priority_q = 32767;
        render.enable_elevation = true;
        double x = 0.0;
        double y = 1.0;
        double z = 0.0;
        if (object > 0) {
            x = positions[object - 1][0];
            y = positions[object - 1][1];
            z = positions[object - 1][2];
        }
        render.qx = q_for_position(x);
        render.qy = q_for_position(-y);
        render.qz = static_cast<std::int32_t>(std::llround(z * 32767.0));
    }
}

std::string model_default() {
    return "HRTF/binaural.personalized_headphone";
}

}  // namespace

int main(int argc, char** argv) {
    std::string input;
    std::string output;
    std::string layout_name = "auto";
    std::string model_path = model_default();
    std::string mode = "mid";
    int filterbank = 1;
    double tail_seconds = 2.0;
    std::string output_rate = "48000";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", option);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--layout") {
            layout_name = next("--layout");
        } else if (arg == "--model") {
            model_path = next("--model");
        } else if (arg == "--mode") {
            mode = next("--mode");
        } else if (arg == "--filterbank") {
            filterbank = std::stoi(next("--filterbank"));
        } else if (arg == "--tail-seconds") {
            tail_seconds = std::stod(next("--tail-seconds"));
        } else if (arg == "--output-rate") {
            output_rate = next("--output-rate");
        } else if (arg == "--list-layouts") {
            std::printf("2.0: 2 channels\n5.1: 6 channels\n7.1: 8 channels\n");
            return 0;
        } else if (arg == "-o" || arg == "--output") {
            output = next("-o");
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return 2;
        } else if (input.empty()) {
            input = arg;
        } else {
            std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (input.empty()) {
        usage();
        return 2;
    }
    if (output.empty()) {
        output = input + ".binaural.wav";
    }

    joc::io::WavAudio wav;
    joc::Status status = joc::io::read_wav_file(input, &wav);
    if (!status.ok()) {
        std::fprintf(stderr, "read: %s\n", status.message().c_str());
        return 1;
    }
    const joc::multichannel::Layout* layout =
        (layout_name == "auto") ? joc::multichannel::detect_layout(wav.channels)
                                : joc::multichannel::find_layout(layout_name);
    if (layout == nullptr) {
        std::fprintf(stderr, "unsupported layout: %s (%u channels)\n", layout_name.c_str(),
                     static_cast<unsigned>(wav.channels));
        return 2;
    }
    if (wav.channels != layout->channels) {
        std::fprintf(stderr, "layout %s expects %d channels, WAV has %u\n", layout->name,
                     layout->channels, static_cast<unsigned>(wav.channels));
        return 2;
    }

    std::vector<float> pcm48;
    if (wav.sample_rate == kTargetRate) {
        pcm48 = wav.samples;
    } else {
        std::fprintf(
            stderr,
            "warning: input is %u Hz and was converted once to 48000 Hz; prefer "
            "ffmpeg/libsoxr to produce 48 kHz float32 first:\n"
            "  ffmpeg -i INPUT -af \"aresample=resampler=soxr:precision=28:osr=48000\" "
            "-c:a pcm_f32le OUT.wav\n",
            static_cast<unsigned>(wav.sample_rate));
        status = joc::multichannel::resample_interleaved(
            wav.samples.data(), wav.samples.size() / wav.channels, wav.channels,
            wav.sample_rate, kTargetRate, &pcm48);
        if (!status.ok()) {
            std::fprintf(stderr, "resample: %s\n", status.message().c_str());
            return 1;
        }
    }
    const std::size_t frames = pcm48.size() / wav.channels;

    joc::hrtf::RosellaModel model;
    status = joc::hrtf::load_personalized_headphone(model_path, &model);
    if (!status.ok()) {
        std::fprintf(stderr, "model: %s\n", status.message().c_str());
        return 1;
    }
    joc::hrtf::RosellaRenderOptions options;
    options.profile = (mode == "near") ? joc::hrtf::RosellaProfile::Near
                     : (mode == "far") ? joc::hrtf::RosellaProfile::Far
                                       : joc::hrtf::RosellaProfile::Mid;
    options.bank = (filterbank == 2) ? joc::hrtf::RosellaBankKind::Captured
                                     : joc::hrtf::RosellaBankKind::Public;
    options.tail_seconds = tail_seconds;
    joc::hrtf::RosellaRuntime runtime;
    status = runtime.open(model, options);
    if (!status.ok()) {
        std::fprintf(stderr, "open: %s\n", status.message().c_str());
        return 1;
    }

    joc::oamd::OamdUpdate initial{};
    fill_initial_update(*layout, &initial);
    std::vector<float> objects16(kSourceChannels * kFrameSamples, 0.0f);
    std::vector<double> rendered;
    for (std::size_t start = 0, frame = 0; start < frames; start += kFrameSamples, ++frame) {
        const std::size_t count = std::min<std::size_t>(kFrameSamples, frames - start);
        std::fill(objects16.begin(), objects16.end(), 0.0f);
        for (int channel = 0; channel < layout->channels; ++channel) {
            const joc::multichannel::ChannelSpec& spec = layout->specs[channel];
            int destination = 0;
            if (!spec.lfe) {
                destination = 1;
                for (int previous = 0; previous < channel; ++previous) {
                    if (!layout->specs[previous].lfe) {
                        ++destination;
                    }
                }
            }
            for (std::size_t sample = 0; sample < count; ++sample) {
                objects16[static_cast<std::size_t>(destination) * kFrameSamples + sample] =
                    pcm48[(start + sample) * layout->channels + channel];
            }
        }
        status = runtime.submit_frame(objects16.data(), frame == 0 ? &initial : nullptr,
                                      static_cast<std::int64_t>(frame), 0, 0);
        if (!status.ok()) {
            std::fprintf(stderr, "render: %s\n", status.message().c_str());
            return 1;
        }
        std::vector<double> chunk;
        runtime.take_output(&chunk);
        rendered.insert(rendered.end(), chunk.begin(), chunk.end());
    }
    std::vector<double> tail;
    status = runtime.finish(runtime.finish_capacity(tail_seconds), &tail);
    if (!status.ok()) {
        std::fprintf(stderr, "finish: %s\n", status.message().c_str());
        return 1;
    }
    rendered.insert(rendered.end(), tail.begin(), tail.end());

    std::vector<float> stereo(rendered.size());
    for (std::size_t index = 0; index < rendered.size(); ++index) {
        stereo[index] = static_cast<float>(rendered[index]);
    }
    std::uint32_t target_rate = wav.sample_rate;
    if (output_rate == "48k" || output_rate == "48000") {
        target_rate = kTargetRate;
    } else if (output_rate != "input") {
        target_rate = static_cast<std::uint32_t>(std::stoul(output_rate));
    }
    if (target_rate != kTargetRate) {
        std::fprintf(stderr,
                     "warning: non-48k output adds a second resampling pass; the "
                     "native binaural output rate is 48000 Hz\n");
        std::vector<float> converted;
        status = joc::multichannel::resample_interleaved(
            stereo.data(), stereo.size() / 2, 2, kTargetRate, target_rate, &converted);
        if (!status.ok()) {
            std::fprintf(stderr, "output resample: %s\n", status.message().c_str());
            return 1;
        }
        stereo.swap(converted);
    }
    status = joc::io::write_wav_float32(output, target_rate, 2, stereo.data(),
                                        stereo.size() / 2);
    if (!status.ok()) {
        std::fprintf(stderr, "write: %s\n", status.message().c_str());
        return 1;
    }
    std::printf("input: %s (%u ch, %u Hz, %zu frames)\n", input.c_str(),
                static_cast<unsigned>(wav.channels), static_cast<unsigned>(wav.sample_rate),
                frames);
    std::printf("layout: %s\n", layout->name);
    std::printf("output: %s (%u Hz, %zu frames)\n", output.c_str(),
                static_cast<unsigned>(target_rate), stereo.size() / 2);
    return 0;
}
