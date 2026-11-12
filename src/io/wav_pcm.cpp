#include "io/wav_pcm.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace joc::io {

namespace {

std::uint16_t read_u16(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0] | (data[1] << 8));
}

std::uint32_t read_u32(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

void write_u16(std::vector<std::uint8_t>* out, std::uint16_t value) {
    out->push_back(static_cast<std::uint8_t>(value & 0xFFu));
    out->push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFu));
}

void write_u32(std::vector<std::uint8_t>* out, std::uint32_t value) {
    out->push_back(static_cast<std::uint8_t>(value & 0xFFu));
    out->push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFu));
    out->push_back(static_cast<std::uint8_t>((value >> 16) & 0xFFu));
    out->push_back(static_cast<std::uint8_t>((value >> 24) & 0xFFu));
}

std::string fourcc(const std::uint8_t* data) {
    return std::string(reinterpret_cast<const char*>(data), 4);
}

float sample_to_float(const std::uint8_t* data, std::uint16_t format, std::uint16_t bits) {
    if (format == 3u) {
        if (bits == 32u) {
            float value = 0.0f;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        double value = 0.0;
        std::memcpy(&value, data, sizeof(value));
        return static_cast<float>(value);
    }
    if (bits == 16u) {
        const std::int16_t value = static_cast<std::int16_t>(read_u16(data));
        return static_cast<float>(value) / 32768.0f;
    }
    if (bits == 24u) {
        std::int32_t value = static_cast<std::int32_t>(data[0]) |
                             (static_cast<std::int32_t>(data[1]) << 8) |
                             (static_cast<std::int32_t>(data[2]) << 16);
        if ((value & 0x00800000) != 0) {
            value |= static_cast<std::int32_t>(0xFF000000u);
        }
        return static_cast<float>(value) / 8388608.0f;
    }
    if (bits == 32u) {
        const std::int32_t value = static_cast<std::int32_t>(read_u32(data));
        return static_cast<float>(static_cast<double>(value) / 2147483648.0);
    }
    return 0.0f;
}

}  // namespace

Status read_wav_file(const std::string& path, WavAudio* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kFoundation,
                            "WAV output pointer is null");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Status::fail(JOC_ERR_IO, stage::kFoundation,
                            "cannot open WAV file: " + path);
    }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),
                                    std::istreambuf_iterator<char>());
    if (bytes.size() < 12u || fourcc(bytes.data()) != "RIFF" ||
        fourcc(bytes.data() + 8) != "WAVE") {
        return Status::fail(JOC_ERR_INPUT_FORMAT, stage::kFoundation,
                            "not a RIFF/WAVE file: " + path);
    }

    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t block_align = 0;
    std::uint16_t bits = 0;
    bool have_format = false;
    std::size_t data_offset = 0;
    std::size_t data_size = 0;
    bool have_data = false;

    std::size_t offset = 12;
    while (offset + 8u <= bytes.size()) {
        const std::string id = fourcc(bytes.data() + offset);
        const std::uint32_t size = read_u32(bytes.data() + offset + 4);
        const std::size_t payload = offset + 8u;
        if (payload + size > bytes.size()) {
            return Status::fail(JOC_ERR_INPUT_FORMAT, stage::kFoundation,
                                "truncated WAV chunk: " + id);
        }
        if (id == "fmt " && size >= 16u) {
            format = read_u16(bytes.data() + payload);
            channels = read_u16(bytes.data() + payload + 2);
            sample_rate = read_u32(bytes.data() + payload + 4);
            block_align = read_u16(bytes.data() + payload + 12);
            bits = read_u16(bytes.data() + payload + 14);
            if (format == 0xFFFEu && size >= 40u) {
                // WAVEFORMATEXTENSIBLE: the sub-format GUID starts at +24.
                format = read_u16(bytes.data() + payload + 24);
            }
            have_format = true;
        } else if (id == "data") {
            data_offset = payload;
            data_size = size;
            have_data = true;
        }
        offset = payload + size + (size & 1u);
    }

    if (!have_format || !have_data || channels == 0 || sample_rate == 0) {
        return Status::fail(JOC_ERR_INPUT_FORMAT, stage::kFoundation,
                            "WAV file has no usable fmt/data chunk: " + path);
    }
    if (format != 1u && format != 3u) {
        return Status::fail(JOC_ERR_INPUT_FORMAT, stage::kFoundation,
                            "unsupported WAV sample format");
    }
    const std::uint16_t bytes_per_sample = static_cast<std::uint16_t>(bits / 8u);
    if (bytes_per_sample == 0u) {
        return Status::fail(JOC_ERR_INPUT_FORMAT, stage::kFoundation,
                            "unsupported WAV bit depth");
    }
    if (block_align == 0u) {
        block_align = static_cast<std::uint16_t>(channels * bytes_per_sample);
    }
    if (block_align < channels * bytes_per_sample) {
        return Status::fail(JOC_ERR_INPUT_FORMAT, stage::kFoundation,
                            "invalid WAV block alignment");
    }
    const std::size_t frames = data_size / block_align;
    out->sample_rate = sample_rate;
    out->channels = channels;
    out->samples.resize(frames * channels);
    const std::uint8_t* source = bytes.data() + data_offset;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
            out->samples[frame * channels + channel] = sample_to_float(
                source + frame * block_align + channel * bytes_per_sample, format, bits);
        }
    }
    return Status::success();
}

Status write_wav_float32(const std::string& path, std::uint32_t sample_rate,
                         std::uint16_t channels, const float* interleaved, std::size_t frames) {
    if (channels == 0u || sample_rate == 0u || interleaved == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput,
                            "WAV writer needs channels, a sample rate and samples");
    }
    if (frames > (std::numeric_limits<std::uint32_t>::max() - 36u) /
                     (channels * sizeof(float))) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput,
                            "WAV output is too large for RIFF");
    }
    const std::uint32_t data_bytes =
        static_cast<std::uint32_t>(frames * channels * sizeof(float));
    std::vector<std::uint8_t> header;
    header.reserve(44);
    header.insert(header.end(), {'R', 'I', 'F', 'F'});
    write_u32(&header, 36u + data_bytes);
    header.insert(header.end(), {'W', 'A', 'V', 'E'});
    header.insert(header.end(), {'f', 'm', 't', ' '});
    write_u32(&header, 16u);
    write_u16(&header, 3u);  // IEEE float
    write_u16(&header, channels);
    write_u32(&header, sample_rate);
    write_u32(&header, sample_rate * static_cast<std::uint32_t>(channels * sizeof(float)));
    write_u16(&header, static_cast<std::uint16_t>(channels * sizeof(float)));
    write_u16(&header, 32u);
    header.insert(header.end(), {'d', 'a', 't', 'a'});
    write_u32(&header, data_bytes);

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        return Status::fail(JOC_ERR_IO, stage::kOutput,
                            "cannot open WAV output: " + path);
    }
    stream.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(header.size()));
    stream.write(reinterpret_cast<const char*>(interleaved),
                 static_cast<std::streamsize>(data_bytes));
    if (!stream) {
        return Status::fail(JOC_ERR_IO, stage::kOutput,
                            "cannot write WAV output: " + path);
    }
    return Status::success();
}

}  // namespace joc::io
