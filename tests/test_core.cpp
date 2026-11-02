// Unit tests for the engine's self-contained parts: no test data files, no
// reference implementation, no external framework.  Run with `ctest` or directly.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "adm/adm_metadata.h"
#include "eac3_transport/eac3_reader.h"
#include "eac3joc_core.h"
#include "foundation/bit_reader.h"
#include "foundation/fs_utf8.h"
#include "foundation/geometry.h"
#include "hrtf/jochrtf.h"
#include "hrtf/rosella_model.h"
#include "hrtf/rosella_renderer.h"
#include "foundation/mini_json.h"
#include "foundation/sha256.h"
#include "io/inflate.h"
#include "io/hdf5.h"
#include "io/npy_writer.h"
#include "io/npy.h"
#include "io/process.h"
#include "io/wav_writer.h"
#include "io/zip_reader.h"
#include "oamd/oamd_parser.h"
#include "speaker/speaker_layout_lookup.h"
#include "speaker/speaker_trim.h"
#include "stream/stream.h"
#include "task/task.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expression, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL %s:%d  %s\n", file, line, expression);
    }
}

#define CHECK(expression) check((expression), #expression, __FILE__, __LINE__)

std::vector<std::uint8_t> from_hex(const char* text) {
    auto digit = [](char c) {
        return static_cast<std::uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
    };
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; text[i] != '\0' && text[i + 1] != '\0'; i += 2) {
        out.push_back(static_cast<std::uint8_t>((digit(text[i]) << 4) | digit(text[i + 1])));
    }
    return out;
}

std::string bytes_to_string(const std::vector<std::uint8_t>& bytes) {
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

ejoc_speaker_controls speaker_controls(std::uint32_t warp, std::uint32_t trim_mode) {
    ejoc_speaker_controls controls{};
    controls.struct_size = sizeof(controls);
    controls.struct_version = EJOC_SPEAKER_CONTROLS_VERSION;
    controls.warp_mode = warp;
    controls.trim_mode = trim_mode;
    return controls;
}

// Gain vector of one object, in standard output-channel order.  The object sits
// at (u, v, z) Q15 and contributes a single unit sample so the value read back
// from sample 0 is the target gain.
std::vector<double> speaker_gains(const joc::speaker::LayoutInfo& layout,
                                  const ejoc_speaker_controls& controls, std::uint16_t u,
                                  std::uint16_t v, std::uint16_t z,
                                  bool bypass_object = false) {
    std::vector<double> gains(layout.channel_count, 0.0);
    ejoc_speaker_renderer_handle handle =
        ejoc_speaker_renderer_create(layout.speaker_bitfield);
    CHECK(handle != nullptr);
    if (handle == nullptr) {
        return gains;
    }
    std::vector<float> input(static_cast<std::size_t>(EJOC_SPEAKER_BLOCK_SAMPLES) *
                                 JOC_OUTPUT_CHANNELS,
                             0.0f);
    input[1] = 1.0f;  // object slot 1, sample 0
    std::uint16_t positions[joc::oamd::kObjects][3] = {};
    positions[0][0] = u;
    positions[0][1] = v;
    positions[0][2] = z;
    const std::uint32_t offset = 0;
    const std::uint32_t ramp = 0;
    ejoc_speaker_controls applied = controls;
    if (bypass_object) {
        applied.trim_bypass[1] = 1;
    }
    std::vector<double> output(static_cast<std::size_t>(EJOC_SPEAKER_BLOCK_SAMPLES) *
                                   layout.channel_count,
                               0.0);
    const int result = ejoc_speaker_renderer_process(
        handle, input.data(), EJOC_SPEAKER_BLOCK_SAMPLES, 1, &offset, &ramp,
        &positions[0][0], nullptr, nullptr, nullptr, &applied, output.data());
    CHECK(result == 0);
    for (std::uint32_t channel = 0; channel < layout.channel_count; ++channel) {
        gains[channel] = output[channel];
    }
    ejoc_speaker_renderer_destroy(handle);
    return gains;
}

void check_gains(const std::vector<double>& actual, const std::vector<double>& expected) {
    CHECK(actual.size() == expected.size());
    if (actual.size() != expected.size()) {
        return;
    }
    for (std::size_t channel = 0; channel < actual.size(); ++channel) {
        if (std::fabs(actual[channel] - expected[channel]) > 1e-13) {
            std::printf("  channel %zu actual %.17g expected %.17g\n", channel,
                        actual[channel], expected[channel]);
        }
        CHECK(std::fabs(actual[channel] - expected[channel]) <= 1e-13);
    }
}

void test_sha256() {
    CHECK(joc::crypto::sha256_hex("", 0) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(joc::crypto::sha256_hex("abc", 3) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char* two_block = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(joc::crypto::sha256_hex(two_block, std::strlen(two_block)) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    std::vector<char> million(1000000, 'a');
    CHECK(joc::crypto::sha256_hex(million.data(), million.size()) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // Incremental updates must equal a single-shot hash.
    joc::crypto::Sha256 incremental;
    incremental.update("ab", 2);
    incremental.update("c", 1);
    CHECK(incremental.finish_hex() == joc::crypto::sha256_hex("abc", 3));
}

void test_crc32_and_inflate() {
    CHECK(joc::io::crc32_of(reinterpret_cast<const std::uint8_t*>("123456789"), 9) == 0xCBF43926u);
    // One stored DEFLATE block: BFINAL=1, BTYPE=00, LEN=3, NLEN=~LEN, "abc".
    const std::uint8_t stored[] = {0x01, 0x03, 0x00, 0xFC, 0xFF, 'a', 'b', 'c'};
    std::vector<std::uint8_t> out;
    CHECK(joc::io::inflate_raw(stored, sizeof(stored), &out));
    CHECK(out.size() == 3 && std::memcmp(out.data(), "abc", 3) == 0);
    const std::uint8_t truncated[] = {0x01, 0x10, 0x00, 0xEF, 0xFF, 'a'};
    CHECK(!joc::io::inflate_raw(truncated, sizeof(truncated), &out));
}

void test_bit_reader() {
    const std::uint8_t data[] = {0xB2, 0x5D};  // 1011 0010 0101 1101
    joc::bits::BitReader reader(data, sizeof(data));
    CHECK(reader.read(3) == 0x5);
    CHECK(reader.read(5) == 0x12);
    CHECK(reader.read(8) == 0x5D);
    CHECK(reader.position() == 16);
    CHECK(!reader.failed());

    joc::bits::BitReader short_read(data, 1);
    CHECK(short_read.read(9) == 0);
    CHECK(short_read.failed());
    CHECK(short_read.error() == JOC_ERR_BITSTREAM_TRUNCATED);
    CHECK(short_read.position() == 0);

    joc::bits::BitReader limited(data, sizeof(data));
    limited.set_limit_bits(4);
    CHECK(limited.read(4) == 0xB);
    CHECK(limited.read(1) == 0 && limited.failed());
}

// Writes MSB-first bits; used only to build test vectors.
class BitWriter {
public:
    void write(std::uint32_t value, unsigned count) {
        for (unsigned i = count; i-- > 0;) {
            const std::uint8_t bit = static_cast<std::uint8_t>((value >> i) & 1u);
            if (used_ == 0) {
                bytes_.push_back(0);
            }
            bytes_.back() |= static_cast<std::uint8_t>(bit << (7u - used_));
            used_ = (used_ + 1u) % 8u;
        }
    }
    const std::vector<std::uint8_t>& bytes() const { return bytes_; }

private:
    std::vector<std::uint8_t> bytes_;
    unsigned used_ = 0;
};

void test_variable_bits() {
    // 300 with width 8 is two groups: 0 (continue) then 44 (stop).
    BitWriter writer;
    writer.write(0, 8);
    writer.write(1, 1);
    writer.write(44, 8);
    writer.write(0, 1);
    joc::bits::BitReader reader(writer.bytes().data(), writer.bytes().size());
    std::uint32_t value = 0;
    CHECK(joc::bits::variable_bits(reader, 8, 8, &value));
    CHECK(value == 300);

    // Eight continuation groups exceed the limit and must fail, not loop.
    BitWriter endless;
    for (int group = 0; group < 8; ++group) {
        endless.write(0, 8);
        endless.write(1, 1);
    }
    joc::bits::BitReader endless_reader(endless.bytes().data(), endless.bytes().size());
    CHECK(!joc::bits::variable_bits(endless_reader, 8, 8, &value));
    CHECK(endless_reader.error() == JOC_ERR_EMDF_SYNTAX);
}

void test_eac3_frame_bytes() {
    std::vector<std::uint8_t> frame(3072, 0);
    frame[0] = 0x0B;
    frame[1] = 0x77;
    frame[2] = 0x05;
    frame[3] = 0xFF;  // frmsiz words = 1536 -> 3072 bytes
    frame[5] = 0x80;  // bsid (bits 40..44) = 16 marks E-AC-3
    std::size_t size = 0;
    CHECK(joc::eac3::FrameReader::frame_bytes(frame.data(), frame.size(), 0, &size) == JOC_OK);
    CHECK(size == 3072);
    CHECK(joc::eac3::FrameReader::frame_bytes(frame.data(), 100, 0, &size) ==
          JOC_ERR_BITSTREAM_TRUNCATED);
    frame[1] = 0x78;
    CHECK(joc::eac3::FrameReader::frame_bytes(frame.data(), frame.size(), 0, &size) ==
          JOC_ERR_EAC3_SYNCFRAME);

    // bsid <= 8 is AC-3: the frame length comes from fscod/frmsizecod instead.
    std::vector<std::uint8_t> ac3(2560, 0);
    ac3[0] = 0x0B;
    ac3[1] = 0x77;
    ac3[4] = 0x24;  // fscod = 0, frmsizecod = 36 -> 1280 words -> 2560 bytes
    ac3[5] = 0x30;  // bsid = 6
    CHECK(joc::eac3::FrameReader::frame_bytes(ac3.data(), ac3.size(), 0, &size) == JOC_OK);
    CHECK(size == 2560);
    CHECK(joc::eac3::FrameReader::frame_bytes(ac3.data(), ac3.size() - 1, 0, &size) ==
          JOC_ERR_BITSTREAM_TRUNCATED);
}

void test_mini_json() {
    const std::string text = R"({"a":1,"b":"x\ny","c":{"d":true},"e":-2.5})";
    std::vector<joc::json::Member> members;
    std::string error;
    CHECK(joc::json::parse_object(text, &members, &error));
    double number = 0.0;
    CHECK(joc::json::as_number(*joc::json::find(members, "a"), &number) && number == 1.0);
    CHECK(joc::json::as_number(*joc::json::find(members, "e"), &number) && number == -2.5);
    std::string value;
    CHECK(joc::json::as_string(*joc::json::find(members, "b"), &value) && value == "x\ny");
    CHECK(joc::json::find(members, "c")->raw == R"({"d":true})");
    std::vector<joc::json::Member> duplicate;
    CHECK(!joc::json::parse_object(R"({"a":1,"a":2})", &duplicate, &error));
    CHECK(!joc::json::parse_object("[1,2]", &duplicate, &error));
}

void test_json_text() {
    // Escaping matches json.dumps(..., ensure_ascii=False): the named escapes,
    // \u00XX for the other control characters, UTF-8 bytes passed through.
    CHECK(joc::json::quote("") == "\"\"");
    CHECK(joc::json::quote("a\"b\\c") == "\"a\\\"b\\\\c\"");
    CHECK(joc::json::quote("x\ny\tz") == "\"x\\ny\\tz\"");
    CHECK(joc::json::quote(std::string("\b\f", 2u)) == "\"\\b\\f\"");
    CHECK(joc::json::quote("\x01") == "\"\\u0001\"");
    CHECK(joc::json::quote("月と私") == "\"月と私\"");

    // Two-space indentation, one member per line, no trailing newline.
    CHECK(joc::json::pretty_object({}) == "{}");
    const std::string document = joc::json::pretty_object({
        {"status", "0"},
        {"message", joc::json::quote("cannot open C:\\out\\a.wav")},
    });
    CHECK(document == "{\n  \"status\": 0,\n  \"message\": "
                      "\"cannot open C:\\\\out\\\\a.wav\"\n}");
    std::vector<joc::json::Member> members;
    std::string error;
    CHECK(joc::json::parse_object(document, &members, &error));
    std::string message;
    CHECK(joc::json::as_string(*joc::json::find(members, "message"), &message) &&
          message == "cannot open C:\\out\\a.wav");
}

void test_task_result_json() {
    joc_task_result result{};
    result.struct_size = sizeof(result);
    result.status = JOC_TASK_FAILED;
    result.error_code = JOC_ERR_OUTPUT_OPEN;
    std::snprintf(result.error_stage, sizeof(result.error_stage), "%s", "output");
    // A Windows path and a quote in the message: the document has to stay valid.
    std::snprintf(result.error_message, sizeof(result.error_message), "%s",
                  "cannot write \"C:\\out\\a.wav\"");
    result.output_peak = 1.5;
    result.t_total = 6.355818;
    std::string text;
    CHECK(joc::task::result_to_json(result, &text).ok());
    CHECK(text.find("\n  \"error_message\": ") != std::string::npos);
    std::vector<joc::json::Member> members;
    std::string error;
    CHECK(joc::json::parse_object(text, &members, &error));
    std::string message;
    CHECK(joc::json::as_string(*joc::json::find(members, "error_message"), &message) &&
          message == "cannot write \"C:\\out\\a.wav\"");
    double number = 0.0;
    CHECK(joc::json::as_number(*joc::json::find(members, "output_peak"), &number) && number == 1.5);
    CHECK(joc::json::as_number(*joc::json::find(members, "t_total"), &number) &&
          number == 6.355818);
}

void test_npy() {
    // Build a minimal <f8 (2,) image in memory, padded the way NumPy pads it.
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
    const std::size_t prefix = 10;  // magic + version + uint16 header length
    while ((prefix + header.size() + 1) % 64 != 0) {
        header.push_back(' ');
    }
    header.push_back('\n');
    std::vector<std::uint8_t> image;
    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    image.insert(image.end(), magic, magic + 6);
    image.push_back(1);
    image.push_back(0);
    image.push_back(static_cast<std::uint8_t>(header.size() & 0xFFu));
    image.push_back(static_cast<std::uint8_t>(header.size() >> 8));
    image.insert(image.end(), header.begin(), header.end());
    const double values[2] = {1.5, -0.25};
    const std::uint8_t* raw = reinterpret_cast<const std::uint8_t*>(values);
    image.insert(image.end(), raw, raw + sizeof(values));

    joc::io::NpyArray array;
    std::string error;
    CHECK(joc::io::parse_npy(image.data(), image.size(), &array, &error));
    CHECK(array.descr == "<f8");
    CHECK(array.shape.size() == 1 && array.shape[0] == 2);
    CHECK(!array.fortran_order);
    std::vector<double> loaded;
    CHECK(joc::io::npy_to_double(array, &loaded, &error));
    CHECK(loaded.size() == 2 && loaded[0] == 1.5 && loaded[1] == -0.25);
}

void test_oamd_and_adm_helpers() {
    CHECK(joc::oamd::q_of(0, 62) == 0);
    CHECK(joc::oamd::q_of(62, 62) == 32767);
    CHECK(joc::oamd::q_of(31, 62) == 16384);
    CHECK(joc::oamd::q_of(15, 15) == 32767);

    CHECK(joc::adm::ts(0.0) == "00:00:00.00000");
    CHECK(joc::adm::ts(1.5) == "00:00:01.50000");
    CHECK(joc::adm::ts(3661.25) == "01:01:01.25000");
    // 0.999999 s rounds to the next second, not to 100000 microseconds.
    CHECK(joc::adm::ts(0.999999) == "00:00:01.00000");

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    joc::geometry::q_to_adm_xyz(0, 32767, 0, &x, &y, &z);
    CHECK(x == -1.0 && y == -1.0 && z == 0.0);
    joc::geometry::q_to_adm_xyz(32767, 0, 0, &x, &y, &z);
    CHECK(x == 1.0 && y == 1.0 && z == 0.0);
}

// DBMD segment 10: the nine configuration trims must default to automatic (bit 0
// of each trim's first byte), and every other field of the block must keep the
// layout the writer has always produced.
void test_dbmd_auto_trim() {
    std::string dbmd;
    CHECK(joc::adm::build_dbmd(25, joc::adm::BinauralMode::Mid, &dbmd).ok());

    std::string segment10;
    std::size_t offset = 4;
    while (offset + 2 < dbmd.size() && static_cast<unsigned char>(dbmd[offset]) != 0) {
        const std::size_t size =
            static_cast<std::size_t>(static_cast<unsigned char>(dbmd[offset + 1])) |
            (static_cast<std::size_t>(static_cast<unsigned char>(dbmd[offset + 2])) << 8);
        const std::size_t start = offset + 3;
        CHECK(start + size + 1 <= dbmd.size());
        unsigned sum = static_cast<unsigned>(size);
        for (std::size_t i = 0; i < size; ++i) {
            sum += static_cast<unsigned char>(dbmd[start + i]);
        }
        sum += static_cast<unsigned char>(dbmd[start + size]);
        CHECK((sum & 0xFFu) == 0u);
        if (static_cast<unsigned char>(dbmd[offset]) == 10) {
            segment10 = dbmd.substr(start, size);
        }
        offset = start + size + 1;
    }

    CHECK(segment10.size() == 292);
    CHECK(static_cast<unsigned char>(segment10[0]) == 0xBD);
    CHECK(static_cast<unsigned char>(segment10[1]) == 0x6F);
    CHECK(static_cast<unsigned char>(segment10[2]) == 0x72);
    CHECK(static_cast<unsigned char>(segment10[3]) == 0xF8);
    CHECK(static_cast<unsigned char>(segment10[4]) == 25);
    CHECK(static_cast<unsigned char>(segment10[5]) == 0);
    CHECK(static_cast<unsigned char>(segment10[6]) == 0);

    // Nine 15-byte configuration trims: automatic, payload untouched.
    for (std::size_t layout = 0; layout < 9; ++layout) {
        const std::size_t flag = 7 + 15 * layout;
        CHECK(static_cast<unsigned char>(segment10[flag]) == 0x01);
        for (std::size_t i = 1; i < 15; ++i) {
            CHECK(static_cast<unsigned char>(segment10[flag + i]) == 0);
        }
    }
    // One trim-bypass byte per track stays clear.
    for (std::size_t i = 7 + 9 * 15; i < 7 + 9 * 15 + 25; ++i) {
        CHECK(static_cast<unsigned char>(segment10[i]) == 0);
    }
    // Headphone modes: bed tracks clear, the object slots carry the mode value.
    for (std::size_t i = 0; i < 25; ++i) {
        const unsigned char expected = i < 10 ? 0x00 : 0x03;
        CHECK(static_cast<unsigned char>(segment10[7 + 9 * 15 + 25 + i]) == expected);
    }
    // Nothing between the mode array and the trailing template fill changed.
    for (std::size_t i = 7 + 9 * 15 + 2 * 25; i < 5 + 262; ++i) {
        CHECK(static_cast<unsigned char>(segment10[i]) == 0);
    }
    for (std::size_t i = 5 + 262; i < segment10.size(); ++i) {
        CHECK(static_cast<unsigned char>(segment10[i]) == 0x84);
    }
}

// OAMD trim element parsing: warp mode, global trim mode, the nine
// configuration records and the per-object bypass flags.  The payloads are
// generated by the reference parser's own bit layout.
void test_oamd_trim_parse() {
    // warp 2, global trim mode 2; config 0 default, 1 disabled, 2 explicit
    // (centre 6 dB, surround -15 dB, height -36 dB, balance +1 / -0.5), the
    // rest default; objects 1 and 5 bypassed.
    const std::vector<std::uint8_t> explicit_payload = from_hex("1f049c8a9f0eff9ffa2000");
    joc::oamd::OamdTrim trim;
    bool present = false;
    CHECK(joc::oamd::parse_trim(explicit_payload.data(), explicit_payload.size(), &trim,
                                &present)
              .ok());
    CHECK(present);
    CHECK(trim.warp_mode == 2);
    CHECK(trim.global_trim_mode == 2);
    CHECK(trim.object_count == 16);
    CHECK(trim.configs[0].kind == joc::oamd::TrimConfigKind::kDefault);
    CHECK(trim.configs[1].kind == joc::oamd::TrimConfigKind::kDisabled);
    CHECK(trim.configs[2].kind == joc::oamd::TrimConfigKind::kExplicit);
    CHECK(trim.configs[2].presence == 0x1F);
    CHECK(trim.configs[2].centre == 6.0);
    CHECK(trim.configs[2].surround == -15.0);
    CHECK(trim.configs[2].height == -36.0);
    CHECK(trim.configs[2].balance_overhead == 1.0);
    CHECK(trim.configs[2].balance_listener == -0.5);
    CHECK(trim.configs[3].kind == joc::oamd::TrimConfigKind::kDefault);
    CHECK(trim.configs[8].kind == joc::oamd::TrimConfigKind::kDefault);
    CHECK(trim.disable_per_object[1]);
    CHECK(trim.disable_per_object[5]);
    CHECK(!trim.disable_per_object[0]);
    CHECK(!trim.disable_per_object[2]);

    // warp 1, automatic trim, no configuration records and no bypass table.
    const std::vector<std::uint8_t> auto_payload = from_hex("1f048040");
    CHECK(joc::oamd::parse_trim(auto_payload.data(), auto_payload.size(), &trim, &present)
              .ok());
    CHECK(present);
    CHECK(trim.warp_mode == 1);
    CHECK(trim.global_trim_mode == 0);
    for (int index = 0; index < joc::oamd::kTrimConfigs; ++index) {
        CHECK(trim.configs[index].kind == joc::oamd::TrimConfigKind::kAbsent);
    }
    for (int slot = 0; slot < joc::oamd::kMaxSlots; ++slot) {
        CHECK(!trim.disable_per_object[slot]);
    }
}

void test_speaker_trim_parameters() {
    joc::speaker::LayoutInfo layout51{};
    joc::speaker::LayoutInfo layout71{};
    joc::speaker::LayoutInfo layout40{};
    joc::speaker::LayoutInfo layout20{};
    joc::speaker::LayoutInfo layout512{};
    joc::speaker::LayoutInfo layout514{};
    joc::speaker::LayoutInfo layout714{};
    CHECK(joc::speaker::layout_by_name("5.1", &layout51));
    CHECK(joc::speaker::layout_by_name("7.1", &layout71));
    CHECK(joc::speaker::layout_by_name("4.0", &layout40));
    CHECK(joc::speaker::layout_by_name("2.0", &layout20));
    CHECK(joc::speaker::layout_by_name("5.1.2", &layout512));
    CHECK(joc::speaker::layout_by_name("5.1.4", &layout514));
    CHECK(joc::speaker::layout_by_name("7.1.4", &layout714));
    CHECK(joc::speaker::trim_config_index(layout20) == 0);
    CHECK(joc::speaker::trim_config_index(layout40) == 1);
    CHECK(joc::speaker::trim_config_index(layout51) == 1);
    CHECK(joc::speaker::trim_config_index(layout71) == 2);
    CHECK(joc::speaker::trim_config_index(layout512) == 4);
    CHECK(joc::speaker::trim_config_index(layout514) == 7);

    ejoc_speaker_controls controls{};
    CHECK(joc::speaker::parameters_from_oamd(nullptr, layout51, &controls).ok());
    CHECK(controls.warp_mode == 0);
    CHECK(controls.trim_mode == EJOC_SPEAKER_TRIM_AUTOMATIC);
    CHECK(joc::speaker::parameters_from_oamd(nullptr, layout714, &controls).ok());
    CHECK(controls.trim_mode == EJOC_SPEAKER_TRIM_DISABLED);

    joc::oamd::OamdTrim trim{};
    trim.warp_mode = 1;
    trim.global_trim_mode = 1;
    CHECK(joc::speaker::parameters_from_oamd(&trim, layout51, &controls).ok());
    CHECK(controls.warp_mode == 1);
    CHECK(controls.trim_mode == EJOC_SPEAKER_TRIM_DISABLED);

    trim = joc::oamd::OamdTrim{};
    trim.global_trim_mode = 2;
    trim.configs[1].kind = joc::oamd::TrimConfigKind::kDefault;
    CHECK(joc::speaker::parameters_from_oamd(&trim, layout51, &controls).ok());
    CHECK(controls.trim_mode == EJOC_SPEAKER_TRIM_AUTOMATIC);
    trim.configs[1].kind = joc::oamd::TrimConfigKind::kDisabled;
    CHECK(joc::speaker::parameters_from_oamd(&trim, layout51, &controls).ok());
    CHECK(controls.trim_mode == EJOC_SPEAKER_TRIM_DISABLED);

    trim.configs[1].kind = joc::oamd::TrimConfigKind::kExplicit;
    trim.configs[1].centre = 1.5;
    trim.configs[1].surround = 3.0;
    trim.configs[1].height = -6.0;
    trim.configs[1].balance_listener = -0.5;
    trim.configs[1].balance_overhead = 0.25;
    trim.disable_per_object[1] = true;
    CHECK(joc::speaker::parameters_from_oamd(&trim, layout51, &controls).ok());
    CHECK(controls.trim_mode == EJOC_SPEAKER_TRIM_EXPLICIT);
    CHECK(controls.centre_db == 1.5);
    CHECK(controls.surround_db == 0.0);
    CHECK(controls.height_db == -6.0);
    CHECK(controls.listener_balance == -0.5);
    CHECK(controls.overhead_balance == 0.25);
    CHECK(controls.trim_bypass[1] == 1);
    CHECK(controls.trim_bypass[2] == 0);

    trim.configs[1].balance_listener = 1.5;
    CHECK(!joc::speaker::parameters_from_oamd(&trim, layout51, &controls).ok());
    trim.configs[1].balance_listener = -0.5;
    trim.configs[1].centre = std::numeric_limits<double>::quiet_NaN();
    CHECK(!joc::speaker::parameters_from_oamd(&trim, layout51, &controls).ok());
    trim.configs[1].centre = 1.5;

    trim.warp_mode = 2;
    const joc::Status warp2 = joc::speaker::parameters_from_oamd(&trim, layout51, &controls);
    CHECK(!warp2.ok());
    CHECK(warp2.code() == JOC_ERR_NOT_SUPPORTED);
    trim.warp_mode = 3;
    CHECK(!joc::speaker::parameters_from_oamd(&trim, layout714, &controls).ok());
    trim.warp_mode = 0;
    trim.global_trim_mode = 3;
    CHECK(!joc::speaker::parameters_from_oamd(&trim, layout51, &controls).ok());
    // A layout without trim semantics settles on disabled before the reserved
    // global mode is looked at.
    CHECK(joc::speaker::parameters_from_oamd(&trim, layout714, &controls).ok());
    CHECK(controls.trim_mode == EJOC_SPEAKER_TRIM_DISABLED);
}

void test_speaker_warp_and_trim() {
    joc::speaker::LayoutInfo layout51{};
    joc::speaker::LayoutInfo layout514{};
    CHECK(joc::speaker::layout_by_name("5.1", &layout51));
    CHECK(joc::speaker::layout_by_name("5.1.4", &layout514));

    ejoc_speaker_controls automatic51{};
    CHECK(joc::speaker::parameters_from_oamd(nullptr, layout51, &automatic51).ok());
    ejoc_speaker_controls automatic514{};
    CHECK(joc::speaker::parameters_from_oamd(nullptr, layout514, &automatic514).ok());

    // Warp 0 leaves the floor Y alone; warp 1 doubles it before the clamp.
    const std::vector<double> warp0 = {0.9126902722074105, 0, 0, 0, 0.18155106823708969, 0};
    const std::vector<double> warp1 = {0.85973219419392444, 0, 0, 0, 0.35612480624995158, 0};
    const std::vector<double> clamped = {0, 0, 0, 0, 0.70794578438413791, 0};
    check_gains(speaker_gains(layout51, automatic51, 0, 4096, 0), warp0);
    ejoc_speaker_controls warped = automatic51;
    warped.warp_mode = 1;
    check_gains(speaker_gains(layout51, warped, 0, 4096, 0), warp1);
    check_gains(speaker_gains(layout51, warped, 0, 24576, 0), clamped);

    // The height layer is not warped.
    const std::vector<double> height_top = {0, 0, 0, 0, 0, 0, 0.84139514164519513, 0, 0, 0};
    const std::vector<double> height_warp0 =
        speaker_gains(layout514, automatic514, 0, 4096, 32767);
    ejoc_speaker_controls warped514 = automatic514;
    warped514.warp_mode = 1;
    check_gains(height_warp0, height_top);
    check_gains(speaker_gains(layout514, warped514, 0, 4096, 32767), height_top);

    // Explicit trim with balance: warp then balance for the floor, balance on
    // the original Y for the trim evaluation.
    ejoc_speaker_controls explicit_controls = speaker_controls(
        1, EJOC_SPEAKER_TRIM_EXPLICIT);
    explicit_controls.surround_db = -6.0;
    explicit_controls.listener_balance = 9.0 / 16.0;
    explicit_controls.overhead_balance = -9.0 / 16.0;
    const std::vector<double> trimmed = {0.68117911059016056, 0, 0, 0, 0.41744281702545366, 0};
    check_gains(speaker_gains(layout51, explicit_controls, 0, 4096, 0), trimmed);
    // Per-object bypass keeps warp but drops balance and trim.
    const std::vector<double> bypassed = {0.92387494614065435, 0, 0, 0, 0.38269450465561039, 0};
    check_gains(speaker_gains(layout51, explicit_controls, 0, 4096, 0, true), bypassed);

    // Centre trim is local to the front-centre region; height trim scales with Z.
    ejoc_speaker_controls centre_controls = speaker_controls(0, EJOC_SPEAKER_TRIM_EXPLICIT);
    centre_controls.centre_db = -6.0;
    const std::vector<double> centre = {3.0688867071757785e-17, 0, 0.50118723362727224, 0, 0, 0};
    check_gains(speaker_gains(layout51, centre_controls, 16384, 0, 0), centre);
    ejoc_speaker_controls height_controls = speaker_controls(0, EJOC_SPEAKER_TRIM_EXPLICIT);
    height_controls.height_db = -6.0;
    const std::vector<double> height_half = {0.50059326485045341, 0, 0, 0, 0,
                                             0, 0.50059326485045341, 0, 0, 0};
    check_gains(speaker_gains(layout514, height_controls, 0, 0, 16384), height_half);

    // LFE passes straight through, untouched by warp, trim or balance.
    ejoc_speaker_renderer_handle handle =
        ejoc_speaker_renderer_create(layout51.speaker_bitfield);
    CHECK(handle != nullptr);
    if (handle != nullptr) {
        std::vector<float> input(static_cast<std::size_t>(EJOC_SPEAKER_BLOCK_SAMPLES) *
                                     JOC_OUTPUT_CHANNELS,
                                 0.0f);
        for (std::uint32_t sample = 0; sample < EJOC_SPEAKER_BLOCK_SAMPLES; ++sample) {
            input[sample * JOC_OUTPUT_CHANNELS] = 0.25f;
        }
        std::vector<double> output(static_cast<std::size_t>(EJOC_SPEAKER_BLOCK_SAMPLES) *
                                       layout51.channel_count,
                                   0.0);
        CHECK(ejoc_speaker_renderer_process(handle, input.data(), EJOC_SPEAKER_BLOCK_SAMPLES,
                                            0, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &explicit_controls, output.data()) == 0);
        for (std::uint32_t sample = 0; sample < EJOC_SPEAKER_BLOCK_SAMPLES; ++sample) {
            for (std::uint32_t channel = 0; channel < layout51.channel_count; ++channel) {
                const double value = output[sample * layout51.channel_count + channel];
                CHECK(channel == 3 ? std::fabs(value - 0.25) <= 1e-15 : value == 0.0);
            }
        }
        // Reserved warp modes are rejected before any rendering happens.
        ejoc_speaker_controls reserved = speaker_controls(2, EJOC_SPEAKER_TRIM_DISABLED);
        CHECK(ejoc_speaker_renderer_process(handle, input.data(), EJOC_SPEAKER_BLOCK_SAMPLES,
                                            0, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &reserved, output.data()) != 0);
        ejoc_speaker_renderer_destroy(handle);
    }
}

// The speaker step takes the OAMD trim element from each payload; a payload
// without it keeps the previous controls.
void test_speaker_step_trim_events() {
    joc::speaker::SpeakerStep step;
    CHECK(joc::speaker::layout_by_name("5.1", &step.layout));
    step.handle = ejoc_speaker_renderer_create(step.layout.speaker_bitfield);
    CHECK(step.handle != nullptr);
    if (step.handle == nullptr) {
        return;
    }
    std::vector<float> frame(static_cast<std::size_t>(JOC_FRAME_SAMPLES) * JOC_OUTPUT_CHANNELS,
                             0.0f);
    for (std::uint32_t sample = 0; sample < JOC_FRAME_SAMPLES; ++sample) {
        frame[1u * JOC_FRAME_SAMPLES + sample] = 1.0f;
    }
    std::string error;
    const std::size_t read = 1472u * step.layout.channel_count;

    joc::oamd::OamdUpdate update;
    update.slots[1].q2 = 4096;
    update.has_trim = true;
    update.trim.warp_mode = 1;
    update.trim.global_trim_mode = 0;
    update.trim.object_count = 16;
    CHECK(joc::speaker::step(&step, frame, &update, &error).ok());
    const std::vector<double> warp1 = {0.85973219419392444, 0, 0, 0, 0.35612480624995158, 0};
    std::vector<double> gains(step.layout.channel_count);
    for (std::uint32_t channel = 0; channel < step.layout.channel_count; ++channel) {
        gains[channel] = step.output[read + channel];
    }
    check_gains(gains, warp1);

    // No trim element: warp 1 and automatic trim stay in effect.
    update = joc::oamd::OamdUpdate{};
    update.slots[1].q2 = 2048;
    CHECK(joc::speaker::step(&step, frame, &update, &error).ok());
    for (std::uint32_t channel = 0; channel < step.layout.channel_count; ++channel) {
        gains[channel] = step.output[read + channel];
    }
    const std::vector<double> half_warp1 = {0.9461247896349676, 0, 0, 0,
                                            0.18820181552760662, 0};
    check_gains(gains, half_warp1);

    // A new trim element switches the controls back to warp 0.
    update = joc::oamd::OamdUpdate{};
    update.slots[1].q2 = 2048;
    update.has_trim = true;
    update.trim.warp_mode = 0;
    update.trim.global_trim_mode = 0;
    update.trim.object_count = 16;
    CHECK(joc::speaker::step(&step, frame, &update, &error).ok());
    for (std::uint32_t channel = 0; channel < step.layout.channel_count; ++channel) {
        gains[channel] = step.output[read + channel];
    }
    const std::vector<double> half_warp0 = {0.96001622724230795, 0, 0, 0,
                                            0.094556249720175942, 0};
    check_gains(gains, half_warp0);

    ejoc_speaker_renderer_destroy(step.handle);
}


void test_dbmd_and_axml_trim() {
    joc::oamd::OamdTrim trim{};
    trim.warp_mode = 2;
    trim.global_trim_mode = 2;
    trim.object_count = 16;
    trim.configs[0].kind = joc::oamd::TrimConfigKind::kDefault;
    trim.configs[1].kind = joc::oamd::TrimConfigKind::kDisabled;
    trim.configs[2].kind = joc::oamd::TrimConfigKind::kExplicit;
    trim.configs[2].presence = 0x1F;
    trim.configs[2].centre = 6.0;
    trim.configs[2].surround = -15.0;
    trim.configs[2].height = -36.0;
    trim.configs[2].balance_overhead = 1.0;
    trim.configs[2].balance_listener = -0.5;
    for (int index = 3; index < joc::oamd::kTrimConfigs; ++index) {
        trim.configs[index].kind = joc::oamd::TrimConfigKind::kDefault;
    }

    std::string dbmd;
    CHECK(joc::adm::build_dbmd(25, joc::adm::BinauralMode::Mid, &dbmd, &trim).ok());
    std::string segment9;
    std::string segment10;
    std::size_t offset = 4;
    while (offset + 2 < dbmd.size() && static_cast<unsigned char>(dbmd[offset]) != 0) {
        const std::size_t size =
            static_cast<std::size_t>(static_cast<unsigned char>(dbmd[offset + 1])) |
            (static_cast<std::size_t>(static_cast<unsigned char>(dbmd[offset + 2])) << 8);
        const std::size_t start = offset + 3;
        unsigned sum = static_cast<unsigned>(size);
        for (std::size_t i = 0; i < size; ++i) {
            sum += static_cast<unsigned char>(dbmd[start + i]);
        }
        sum += static_cast<unsigned char>(dbmd[start + size]);
        CHECK((sum & 0xFFu) == 0u);
        const unsigned id = static_cast<unsigned char>(dbmd[offset]);
        if (id == 9u) {
            segment9 = dbmd.substr(start, size);
        } else if (id == 10u) {
            segment10 = dbmd.substr(start, size);
        }
        offset = start + size + 1;
    }
    CHECK(segment9.size() == 248);
    CHECK(static_cast<unsigned char>(segment9[152]) == 2);
    CHECK(segment10.size() == 292);
    const std::string expected_configs = bytes_to_string(from_hex(
        "010000000000000000000000000000"
        "000000000000000000000000000000"
        "000000c040000070c1000010c27fc0"
        "010000000000000000000000000000"
        "010000000000000000000000000000"
        "010000000000000000000000000000"
        "010000000000000000000000000000"
        "010000000000000000000000000000"
        "010000000000000000000000000000"));
    CHECK(segment10.substr(7, 135) == expected_configs);
    // The bypass table and headphone modes stay untouched.
    for (std::size_t i = 7 + 9 * 15; i < 7 + 9 * 15 + 2 * 25; ++i) {
        const unsigned char expected =
            (i >= 7 + 9 * 15 + 25 + 10) ? 0x03u : 0x00u;
        CHECK(static_cast<unsigned char>(segment10[i]) == expected);
    }

    // Without a trim element the warp field keeps the template's zero.
    std::string plain;
    CHECK(joc::adm::build_dbmd(25, joc::adm::BinauralMode::Mid, &plain).ok());
    CHECK(plain.size() > 4);
    std::string plain_segment9;
    offset = 4;
    while (offset + 2 < plain.size() && static_cast<unsigned char>(plain[offset]) != 0) {
        const std::size_t size =
            static_cast<std::size_t>(static_cast<unsigned char>(plain[offset + 1])) |
            (static_cast<std::size_t>(static_cast<unsigned char>(plain[offset + 2])) << 8);
        const std::size_t start = offset + 3;
        if (static_cast<unsigned char>(plain[offset]) == 9u) {
            plain_segment9 = plain.substr(start, size);
        }
        offset = start + size + 1;
    }
    CHECK(plain_segment9.size() == 248);
    CHECK(static_cast<unsigned char>(plain_segment9[152]) == 0u);

    std::vector<joc::adm::Track> tracks(15);
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        tracks[i].name = "obj" + std::to_string(i);
    }
    const std::string axml = joc::adm::build_axml(tracks, 1.0, 48000, &trim);
    const std::string expected_block =
        "<DolbyAtmosMasterFile><DolbyMasterContentInfo><render_warp_mode>2</render_warp_mode>"
        "</DolbyMasterContentInfo><DolbyTrimModeSetup>"
        "<records><type>0</type></records>"
        "<records><type>1</type><centerTrim>0.000000</centerTrim>"
        "<surroundTrim>0.000000</surroundTrim><heightTrim>0.000000</heightTrim>"
        "<balanceOverhead>0.000000</balanceOverhead>"
        "<balanceListener>0.000000</balanceListener></records>"
        "<records><type>2</type><centerTrim>6.000000</centerTrim>"
        "<surroundTrim>-15.000000</surroundTrim><heightTrim>-36.000000</heightTrim>"
        "<balanceOverhead>1.000000</balanceOverhead>"
        "<balanceListener>-0.500000</balanceListener></records>"
        "<records><type>3</type></records><records><type>4</type></records>"
        "<records><type>5</type></records><records><type>6</type></records>"
        "<records><type>7</type></records><records><type>8</type></records>"
        "</DolbyTrimModeSetup><DolbyWarpMode>2</DolbyWarpMode></DolbyAtmosMasterFile>";
    const std::size_t found = axml.find(expected_block);
    CHECK(found != std::string::npos);
    CHECK(found < axml.find("</audioFormatExtended>"));
    const std::string plain_axml = joc::adm::build_axml(tracks, 1.0, 48000);
    CHECK(plain_axml.find("<DolbyAtmosMasterFile>") == std::string::npos);
}

void test_int24_packing() {
    // Interleaved (L,R) for two frames: 0.0, +1.0, -1.0, +2.0 (clips).
    const float values[4] = {0.0f, 1.0f, -1.0f, 2.0f};
    std::string packed;
    joc::io::pack_int24(values, 2, 2, &packed);
    CHECK(packed.size() == 12);
    auto byte = [&](std::size_t index) {
        return static_cast<std::uint8_t>(packed[index]);
    };
    CHECK(byte(0) == 0x00 && byte(1) == 0x00 && byte(2) == 0x00);          // 0.0
    CHECK(byte(3) == 0xFF && byte(4) == 0xFF && byte(5) == 0x7F);          // +1.0
    CHECK(byte(6) == 0x01 && byte(7) == 0x00 && byte(8) == 0x80);          // -1.0
    CHECK(byte(9) == 0xFF && byte(10) == 0xFF && byte(11) == 0x7F);        // +2.0 clips
}

void test_utf8_paths() {
    // Non-ASCII paths must survive the OS boundary on every platform: this is the
    // regression guard for the ANSI-code-page bug class (a Japanese path used to
    // reach ffmpeg as mojibake on Windows).
    const std::string directory = joc::fs_utf8::temp_directory();
    CHECK(!directory.empty());
    const std::string path = joc::fs_utf8::from_path(
        joc::fs_utf8::to_path(directory) / joc::fs_utf8::to_path("joc_テスト_須田景凪_测试.bin"));
    CHECK(joc::fs_utf8::from_path(joc::fs_utf8::to_path(path)) == path);

    std::FILE* file = joc::fs_utf8::fopen(path, "wb");
    CHECK(file != nullptr);
    if (file != nullptr) {
        const char payload[] = "joc-utf8";
        CHECK(std::fwrite(payload, 1, sizeof(payload) - 1u, file) == sizeof(payload) - 1u);
        std::fclose(file);
    }
    CHECK(joc::fs_utf8::exists(path));
    std::error_code error;
    CHECK(joc::fs_utf8::file_size(path, error) == sizeof("joc-utf8") - 1u);

    std::ifstream input = joc::fs_utf8::open_input(path);
    CHECK(input.good());
    std::string content(sizeof("joc-utf8") - 1u, '\0');
    input.read(content.data(), static_cast<std::streamsize>(content.size()));
    CHECK(content == "joc-utf8");
    input.close();

    std::ofstream output = joc::fs_utf8::open_output(path + ".copy");
    CHECK(output.good());
    output << content;
    output.close();
    CHECK(joc::fs_utf8::exists(path + ".copy"));

    CHECK(joc::fs_utf8::remove(path) == 0);
    CHECK(joc::fs_utf8::remove(path + ".copy") == 0);
    CHECK(!joc::fs_utf8::exists(path));
}

void test_process_runner() {
    // The ffmpeg boundary must surface both the exit code and the child's stderr.
#if defined(_WIN32)
    const std::vector<std::string> command = {"cmd.exe", "/c", "echo boom 1>&2 & exit 3"};
#else
    const std::vector<std::string> command = {"/bin/sh", "-c", "echo boom 1>&2; exit 3"};
#endif
    joc::io::ProcessResult result;
    const joc::Status status = joc::io::run_process(command, &result);
    CHECK(!status.ok());
    CHECK(result.exit_code == 3u);
    CHECK(result.output.find("boom") != std::string::npos);
}

void test_find_executable() {
    // An executable reference has to become a concrete path: that is what turns the
    // default "ffmpeg" into something a child process starts without a PATH search of
    // its own, on both platforms.
    const std::string bare =
#if defined(_WIN32)
        "cmd";
#else
        "sh";
#endif
    const std::string resolved = joc::fs_utf8::find_executable(bare);
    CHECK(!resolved.empty());
    CHECK(resolved.find_first_of("/\\") != std::string::npos);
    CHECK(joc::fs_utf8::exists(resolved));
    // An already concrete path is kept verbatim instead of being searched for again.
    CHECK(joc::fs_utf8::find_executable(resolved) == resolved);
    CHECK(joc::fs_utf8::find_executable("").empty());
    CHECK(joc::fs_utf8::find_executable("joc_no_such_program_9f2c").empty());
    // A reference with a directory separator names a location, so it is never answered
    // by a same-named program found somewhere else.
    CHECK(joc::fs_utf8::find_executable("joc_no_such_dir/joc_no_such_program_9f2c").empty());
#if !defined(_WIN32)
    // Without the execute bit the kernel refuses the file, so it is not a match.
    const std::string plain = joc::fs_utf8::from_path(
        joc::fs_utf8::to_path(joc::fs_utf8::temp_directory()) /
        joc::fs_utf8::to_path("joc_not_executable_9f2c"));
    std::FILE* file = joc::fs_utf8::fopen(plain, "wb");
    CHECK(file != nullptr);
    if (file != nullptr) {
        std::fclose(file);
    }
    CHECK(joc::fs_utf8::find_executable(plain).empty());
    CHECK(joc::fs_utf8::remove(plain) == 0);
#endif
}

void test_npy_and_zip_writers() {
    using joc::io::NpyMember;
    // Python's repr, which is what json.dumps writes into the cache metadata.
    CHECK(joc::io::python_float_repr(48000.0) == "48000.0");
    CHECK(joc::io::python_float_repr(1.0) == "1.0");
    CHECK(joc::io::python_float_repr(0.001) == "0.001");
    CHECK(joc::io::python_float_repr(1.0e-5) == "1e-05");
    CHECK(joc::io::python_float_repr(1.5) == "1.5");
    CHECK(joc::io::python_float_repr(-2.25) == "-2.25");
    CHECK(joc::io::python_float_repr(1.0e16) == "1e+16");
    CHECK(joc::io::python_float_repr(1234567890123456.0) == "1234567890123456.0");

    // An NPY image must round-trip through the reader that loads the cache.
    const std::vector<double> values = {1.5, -2.25, 3.75, 4.0, 0.5, 0.25};
    std::vector<std::uint8_t> payload(values.size() * sizeof(double));
    std::memcpy(payload.data(), values.data(), payload.size());
    const std::vector<std::uint8_t> image = joc::io::npy_image("<f8", {2u, 3u}, payload);
    joc::io::NpyArray array;
    std::string error;
    CHECK(joc::io::parse_npy(image.data(), image.size(), &array, &error));
    CHECK(array.descr == "<f8");
    CHECK(joc::io::npy_shape_is(array, {2, 3}));
    std::vector<double> restored;
    CHECK(joc::io::npy_to_double(array, &restored, &error));
    CHECK(restored.size() == values.size());
    bool same = restored.size() == values.size();
    for (std::size_t index = 0; index < restored.size() && same; ++index) {
        same = restored[index] == values[index];
    }
    CHECK(same);

    // A Unicode scalar string member, as the cache stores its metadata.
    const std::string text = "{\"k\":\"v\"}";
    const std::vector<std::uint8_t> unicode = joc::io::utf8_to_utf32le(text);
    CHECK(unicode.size() == text.size() * 4u);
    CHECK(unicode[0] == static_cast<std::uint8_t>('{') && unicode[1] == 0u &&
          unicode[3] == 0u);
    const std::vector<std::uint8_t> text_image =
        joc::io::npy_image("<U" + std::to_string(text.size()), {}, unicode);
    CHECK(joc::io::parse_npy(text_image.data(), text_image.size(), &array, &error));
    std::string decoded;
    CHECK(joc::io::npy_unicode_to_utf8(array, &decoded, &error));
    CHECK(decoded == text);

    // The ZIP container round-trips through the archive reader.
    const std::string path = joc::fs_utf8::temp_directory() + "/joc_writer_test.jochrtf";
    std::vector<NpyMember> members;
    NpyMember member;
    member.name = "band_center_frequencies_hz";
    member.descr = "<f8";
    member.shape = {2u, 3u};
    member.data = payload;
    members.push_back(member);
    CHECK(joc::io::write_zip(path, members, &error));
    joc::io::ZipArchive archive;
    CHECK(archive.open(path, &error));
    CHECK(archive.entries().size() == 1u);
    std::vector<std::uint8_t> read_back;
    CHECK(archive.read_member("band_center_frequencies_hz.npy", &read_back, &error));
    CHECK(read_back.size() == image.size());
    CHECK(read_back.size() == image.size() &&
          std::memcmp(read_back.data(), image.data(), image.size()) == 0);
    CHECK(joc::fs_utf8::remove(path) == 0);
}

// ---------------------------------------------------------------------------
// HRTF import and compile path.
//
// Everything below builds its input in memory, so the suite reads no external
// files; the parts that need one are exercised through their error paths.
// ---------------------------------------------------------------------------

constexpr double kPi = 3.14159265358979323846;

// A SimpleFreeFieldHRIR set with `measurements` directions on one 1 m shell and
// `taps` decaying taps per ear.  The values are closed-form placeholders; what
// the canonicaliser and the spherical-harmonic fit actually read is the
// geometry and the shape.





void test_rosella_model_errors() {
    // A real model is proprietary user data and is not part of this repository.
    // What can be checked without one is that every malformed input is reported
    // instead of being half-loaded.
    joc::hrtf::RosellaModel model;
    const std::string missing = joc::fs_utf8::temp_directory() + "/joc_absent.headphone";
    CHECK(!joc::hrtf::load_personalized_headphone(missing, &model).ok());
    CHECK(!joc::hrtf::load_personalized_headphone(missing, nullptr).ok());

    const std::string raw = joc::fs_utf8::temp_directory() + "/joc_raw.headphone";
    {
        std::ofstream stream = joc::fs_utf8::open_output(raw);
        stream << "rp binary model, not JSON";
    }
    const joc::Status raw_status = joc::hrtf::load_personalized_headphone(raw, &model);
    CHECK(!raw_status.ok());
    CHECK(raw_status.message().find("JSON") != std::string::npos);
    CHECK(joc::fs_utf8::remove(raw) == 0);

    const std::string empty = joc::fs_utf8::temp_directory() + "/joc_empty.headphone";
    {
        std::ofstream stream = joc::fs_utf8::open_output(empty);
        stream << "{}";
    }
    CHECK(!joc::hrtf::load_personalized_headphone(empty, &model).ok());
    CHECK(joc::fs_utf8::remove(empty) == 0);

    const std::string no_coefficients = joc::fs_utf8::temp_directory() + "/joc_no_coeff.headphone";
    {
        std::ofstream stream = joc::fs_utf8::open_output(no_coefficients);
        stream << "{\"personalized_hrtf\":{\"virtualizer_parameters\":{}}}";
    }
    const joc::Status status =
        joc::hrtf::load_personalized_headphone(no_coefficients, &model);
    CHECK(!status.ok());
    CHECK(status.message().find("rosella_coefficients") != std::string::npos);
    CHECK(joc::fs_utf8::remove(no_coefficients) == 0);
}

void test_builtin_kernel_tables() {
    // The filterbank tables are compiled in, and their content must stay identical
    // to the standard-defined archive the file loader accepts.  The hashes below are
    // the SHA-256 of each member's C-order bytes.
    const joc::hrtf::Kernels& kernels = joc::hrtf::builtin_kernels();
    CHECK(kernels.qmf_analysis.size() == 640);
    CHECK(kernels.hybrid_low.size() == 2496);
    CHECK(kernels.hybrid_indices.size() == 616);
    CHECK(kernels.hybrid_values.size() == 154);
    CHECK(kernels.qmf_basis.size() == 32768);
    CHECK(kernels.qmf_taps.size() == 2560);
    CHECK(kernels.hybrid_count == 154);

    auto hash_floats = [](const std::vector<double>& values, bool narrow) {
        joc::crypto::Sha256 hash;
        if (narrow) {
            std::vector<float> raw(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                raw[i] = static_cast<float>(values[i]);
            }
            hash.update(raw.data(), raw.size() * sizeof(float));
        } else {
            hash.update(values.data(), values.size() * sizeof(double));
        }
        return hash.finish_hex();
    };
    CHECK(hash_floats(kernels.qmf_analysis, true) ==
          "aeff6c7117d41664b9c4bf03bbf563f5319ec1b8c551f171adbb90cf19d9d306");
    CHECK(hash_floats(kernels.hybrid_low, true) ==
          "d00d36133b81ba699a7630c4df1be203fa1b7e371e595eaaebbe8957db322627");
    CHECK(hash_floats(kernels.hybrid_values, true) ==
          "99409fdd9d20d1d7c2be16bbc1e2159c8042487227c72160850745164c9cee7f");
    CHECK(hash_floats(kernels.qmf_basis, false) ==
          "a0c4a55385f6d6c7c92d7615c83ad5fbda51046d9ef785cac0b9ac9a760dc527");
    CHECK(hash_floats(kernels.qmf_taps, false) ==
          "cd7756d060d51fbf02f44c1ce53cb6225b221099505c94c3f58d3bee6f428150");
    joc::crypto::Sha256 indices;
    indices.update(kernels.hybrid_indices.data(),
                   kernels.hybrid_indices.size() * sizeof(std::int16_t));
    CHECK(indices.finish_hex() ==
          "f5beb3220e4530fcf28e7f4da7f07e821074265d118c911d61a590e00753a573");
}

// Drives one objects16 stream over `block` and returns what it renders.  With
// `frame_at_a_time` each push carries a single frame; otherwise the whole batch
// goes in at once.  `peak` reports the largest backlog a push left behind.
std::vector<float> drive_objects16_stream(const std::vector<float>& block, std::size_t frames,
                                          bool frame_at_a_time, std::size_t* peak) {
    constexpr std::size_t kFrameValues = JOC_OUTPUT_CHANNELS * JOC_FRAME_SAMPLES;
    joc::stream::Stream stream;
    joc::stream::Config config;
    config.input = JOC_STREAM_IN_PCM_OBJECTS16;
    config.output = JOC_STREAM_OUT_SPEAKER;
    config.layout = "5.1";
    const joc::Status created = stream.create(config);
    CHECK(created.ok());
    if (!created.ok()) {
        return {};
    }
    const std::size_t channels = stream.info().output_channels;
    CHECK(channels != 0u);
    std::vector<float> pulled(4096u * channels);
    std::vector<float> rendered;
    const auto drain = [&] {
        for (;;) {
            std::size_t produced = 0;
            const joc::Status status = stream.pull(pulled.data(), 4096u, &produced);
            CHECK(status.ok());
            if (!status.ok() || produced == 0u) {
                return;
            }
            rendered.insert(rendered.end(), pulled.begin(),
                            pulled.begin() + static_cast<std::ptrdiff_t>(produced * channels));
        }
    };

    std::size_t consumed = 0;
    while (consumed < frames) {
        const std::size_t count = frame_at_a_time ? 1u : frames - consumed;
        std::size_t taken = 0;
        const joc::Status pushed = stream.push_objects16(
            block.data() + consumed * kFrameValues, count * JOC_FRAME_SAMPLES, &taken);
        CHECK(pushed.ok());
        if (!pushed.ok()) {
            return rendered;
        }
        // A push takes everything it is given; only the rendering is bounded.
        CHECK(taken == count * JOC_FRAME_SAMPLES);
        *peak = std::max(*peak, stream.buffered_samples());
        consumed += taken / JOC_FRAME_SAMPLES;
        drain();
    }
    CHECK(stream.flush().ok());
    drain();
    return rendered;
}

void test_rosella_filterbank_config() {
    // validate() reports a selector it cannot honour instead of ignoring it: the
    // captured bank needs the Rosella model, and any other value is out of range.
    const auto reported = [](std::uint32_t selector, const char* personalized) {
        joc_task_config config{};
        config.struct_size = sizeof(config);
        config.struct_version = JOC_TASK_CONFIG_VERSION;
        config.input_path = "input.eac3";
        config.output_path = "output.wav";
        config.operation = JOC_OP_BINAURAL;
        config.binaural_mode = JOC_BINAURAL_MID;
        config.personalized_headphone_path = personalized;
        config.hrtf_path = personalized == nullptr ? "cache.jochrtf" : nullptr;
        config.rosella_filterbank = selector;
        std::vector<joc_validation_issue> issues;
        std::uint32_t errors = 0;
        joc::task::validate(config, &issues, &errors);
        for (const joc_validation_issue& issue : issues) {
            if (std::string(issue.field) == "rosella_filterbank") {
                return true;
            }
        }
        return false;
    };
    CHECK(!reported(JOC_ROSELLA_FILTERBANK_PUBLIC, "model.personalized_headphone"));
    CHECK(!reported(JOC_ROSELLA_FILTERBANK_CAPTURED, "model.personalized_headphone"));
    CHECK(reported(3u, "model.personalized_headphone"));
    CHECK(reported(JOC_ROSELLA_FILTERBANK_CAPTURED, nullptr));
}

// A push renders every frame it makes ready, so the stream bounds how far ahead of
// the caller it renders.  A batch pushed at once must therefore leave a bounded
// backlog, and it must render the same samples as pushing one frame at a time.
void test_stream_render_ahead() {
    constexpr std::size_t kFrames = 32;
    constexpr std::size_t kFrameValues = JOC_OUTPUT_CHANNELS * JOC_FRAME_SAMPLES;
    std::vector<float> block(kFrames * kFrameValues);
    for (std::size_t index = 0; index < block.size(); ++index) {
        block[index] = static_cast<float>(std::sin(static_cast<double>(index) * 0.0007) * 0.25);
    }

    std::size_t frame_peak = 0;
    const std::vector<float> reference =
        drive_objects16_stream(block, kFrames, true, &frame_peak);
    std::size_t batch_peak = 0;
    const std::vector<float> batched = drive_objects16_stream(block, kFrames, false, &batch_peak);

    CHECK(!reference.empty());
    CHECK(batched == reference);
    // The batch is larger than the bound, so it cannot all be rendered at once.
    CHECK(batch_peak < kFrames * JOC_FRAME_SAMPLES / 2u);
}

}  // namespace

// The JOC clipgain is the analysis-side scalar's per-frame factor: it scales the
// frame that carries it, ramps over that frame's first 256 samples, starts from
// the frame's own value after a reset, and never reaches LFE.
struct ClipGainInputs {
    std::vector<float> bed;
    std::vector<float> lfe;
    std::vector<std::uint8_t> n_bands;
    std::vector<std::uint8_t> n_dpoints;
    std::vector<std::uint8_t> slope;
    std::vector<std::uint8_t> offsets;
    std::vector<double> dq;

    ClipGainInputs()
        : bed(EJOC_CORE_CHANNELS * EJOC_FRAME_SAMPLES, 0.0f),
          lfe(EJOC_FRAME_SAMPLES, 0.0f),
          n_bands(EJOC_MAX_OBJECTS, 0),
          n_dpoints(EJOC_MAX_OBJECTS, 0),
          slope(EJOC_MAX_OBJECTS, 0),
          offsets(EJOC_MAX_OBJECTS * EJOC_MAX_DPOINTS, 0),
          dq(static_cast<std::size_t>(EJOC_MAX_OBJECTS) * EJOC_MAX_DPOINTS *
                 EJOC_MAX_CORE_CHANNELS * EJOC_MAX_PARAMETER_BANDS,
             0.0) {
        const double two_pi = 6.283185307179586476925286766559;
        for (int sample = 0; sample < EJOC_FRAME_SAMPLES; ++sample) {
            const double t = static_cast<double>(sample) / 48000.0;
            bed[sample] = static_cast<float>(0.05 * std::sin(two_pi * 440.0 * t));
            lfe[sample] = static_cast<float>(0.02 * std::sin(two_pi * 55.0 * t));
        }
        // Object 0 passes the first core channel through one parameter band.
        n_bands[0] = 1;
        n_dpoints[0] = 1;
        dq[0] = 1.0;
    }
};

bool clipgain_process(ejoc_renderer_handle handle, const ClipGainInputs& in, double clipgain,
                      std::vector<float>* out) {
    out->assign(static_cast<std::size_t>(EJOC_OUTPUT_CHANNELS) * EJOC_FRAME_SAMPLES, 0.0f);
    return ejoc_renderer_process(handle, in.bed.data(), in.lfe.data(), EJOC_CORE_CHANNELS, 0u, 1u,
                                 in.n_bands.data(), in.n_dpoints.data(), in.slope.data(),
                                 in.offsets.data(), in.dq.data(), clipgain, 0.0625f, 1.0f,
                                 out->data()) == 0;
}

void test_clipgain_first_frame_uses_its_own_gain() {
    const ClipGainInputs in;
    std::vector<float> unity;
    std::vector<float> doubled;
    auto render = [&in](double clipgain, std::vector<float>* out) {
        ejoc_renderer_handle handle = ejoc_renderer_create();
        ejoc_renderer_set_threads(handle, 1);
        const bool ok = clipgain_process(handle, in, clipgain, out);
        ejoc_renderer_destroy(handle);
        return ok;
    };
    CHECK(render(1.0, &unity));
    CHECK(render(2.0, &doubled));

    float peak = 0.0f;
    bool scaled = true;
    for (int sample = 0; sample < EJOC_FRAME_SAMPLES; ++sample) {
        const float value = unity[EJOC_FRAME_SAMPLES + sample];  // object 0 is channel 1
        peak = std::max(peak, std::fabs(value));
        if (doubled[EJOC_FRAME_SAMPLES + sample] != 2.0f * value) {
            scaled = false;
        }
    }
    // A fresh handle takes the current gain with no transition.
    CHECK(peak > 1.0e-3f);
    CHECK(scaled);
    // LFE bypasses the scalar, so its samples are bit-identical.
    CHECK(std::memcmp(unity.data(), doubled.data(), sizeof(float) * EJOC_FRAME_SAMPLES) == 0);
}

void test_clipgain_change_applies_to_the_current_frame() {
    const ClipGainInputs in;
    std::vector<float> changed;
    std::vector<float> held;
    auto run = [&in](double second_gain, std::vector<float>* out) {
        ejoc_renderer_handle handle = ejoc_renderer_create();
        ejoc_renderer_set_threads(handle, 1);
        std::vector<float> first;
        const bool ok = clipgain_process(handle, in, 1.0, &first) &&
                        clipgain_process(handle, in, second_gain, out);
        ejoc_renderer_destroy(handle);
        return ok;
    };
    CHECK(run(2.0, &changed));
    CHECK(run(1.0, &held));

    // The change has to land in the frame carrying it; a one-frame-late gain
    // would make these two frames identical.
    double difference = 0.0;
    double energy = 0.0;
    for (int object = 1; object < EJOC_OUTPUT_CHANNELS; ++object) {
        for (int sample = 0; sample < EJOC_FRAME_SAMPLES; ++sample) {
            const std::size_t index =
                static_cast<std::size_t>(object) * EJOC_FRAME_SAMPLES + sample;
            difference += std::fabs(static_cast<double>(changed[index]) - held[index]);
            energy += std::fabs(static_cast<double>(held[index]));
        }
    }
    CHECK(difference > 0.05 * energy);

    // The analysis delay lines and the QMF windows hold the new gain back by
    // about 1400 samples, so the start of the frame still carries the old value
    // and the end is scaled by the new one.
    bool head_intact = true;
    for (int sample = 0; sample < 256; ++sample) {
        const std::size_t index = static_cast<std::size_t>(EJOC_FRAME_SAMPLES) + sample;
        if (changed[index] != held[index]) {
            head_intact = false;
        }
    }
    CHECK(head_intact);

    double tail = 0.0;
    for (int sample = 1504; sample < EJOC_FRAME_SAMPLES; ++sample) {
        const std::size_t index = static_cast<std::size_t>(EJOC_FRAME_SAMPLES) + sample;
        tail = std::max(tail, std::fabs(static_cast<double>(changed[index]) -
                                       2.0 * static_cast<double>(held[index])));
    }
    CHECK(tail < 1.0e-4);
}

void test_clipgain_reset_clears_the_previous_gain() {
    const ClipGainInputs in;
    std::vector<float> scratch;
    std::vector<float> after_reset;
    std::vector<float> fresh;

    ejoc_renderer_handle handle = ejoc_renderer_create();
    ejoc_renderer_set_threads(handle, 1);
    CHECK(clipgain_process(handle, in, 3.0, &scratch));
    CHECK(clipgain_process(handle, in, 3.0, &scratch));
    CHECK(ejoc_renderer_reset(handle) == 0);
    CHECK(clipgain_process(handle, in, 1.0, &after_reset));
    ejoc_renderer_destroy(handle);

    ejoc_renderer_handle pristine = ejoc_renderer_create();
    ejoc_renderer_set_threads(pristine, 1);
    CHECK(clipgain_process(pristine, in, 1.0, &fresh));
    ejoc_renderer_destroy(pristine);

    // No gain state may survive the reset: the first frame must match a fresh
    // renderer instead of ramping down from the previous 3x.
    CHECK(after_reset.size() == fresh.size());
    CHECK(std::memcmp(after_reset.data(), fresh.data(),
                      sizeof(float) * after_reset.size()) == 0);
}

int main() {
    test_sha256();
    test_crc32_and_inflate();
    test_bit_reader();
    test_variable_bits();
    test_eac3_frame_bytes();
    test_mini_json();
    test_json_text();
    test_task_result_json();
    test_npy();
    test_oamd_and_adm_helpers();
    test_dbmd_auto_trim();
    test_oamd_trim_parse();
    test_speaker_trim_parameters();
    test_speaker_warp_and_trim();
    test_speaker_step_trim_events();
    test_dbmd_and_axml_trim();
    test_int24_packing();
    test_utf8_paths();
    test_process_runner();
    test_find_executable();
    test_builtin_kernel_tables();
    test_npy_and_zip_writers();
    test_rosella_model_errors();
    test_rosella_filterbank_config();
    test_stream_render_ahead();
    test_clipgain_first_frame_uses_its_own_gain();
    test_clipgain_change_applies_to_the_current_frame();
    test_clipgain_reset_clears_the_previous_gain();
    std::printf("%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
