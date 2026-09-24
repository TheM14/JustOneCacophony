// Unit tests for the engine's self-contained parts: no test data files, no
// reference implementation, no external framework.  Run with `ctest` or directly.
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
#include "foundation/bit_reader.h"
#include "foundation/fs_utf8.h"
#include "foundation/geometry.h"
#include "hrtf/jochrtf.h"
#include "hrtf/sofa.h"
#include "hrtf/sofa_cache.h"
#include "hrtf/rosella_model.h"
#include "hrtf/rosella_renderer.h"
#include "hrtf/sofa_field.h"
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
    std::size_t size = 0;
    CHECK(joc::eac3::FrameReader::frame_bytes(frame.data(), frame.size(), 0, &size) == JOC_OK);
    CHECK(size == 3072);
    CHECK(joc::eac3::FrameReader::frame_bytes(frame.data(), 100, 0, &size) ==
          JOC_ERR_BITSTREAM_TRUNCATED);
    frame[1] = 0x78;
    CHECK(joc::eac3::FrameReader::frame_bytes(frame.data(), frame.size(), 0, &size) ==
          JOC_ERR_EAC3_SYNCFRAME);
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
joc::hrtf::SofaHrir synthetic_sofa(std::uint32_t measurements, std::uint32_t taps) {
    joc::hrtf::SofaHrir sofa;
    sofa.sample_rate = 48000.0;
    sofa.sampling_rate_units = "hertz";
    sofa.ir_count = measurements;
    sofa.ir_length = taps;
    sofa.conventions = "SOFA";
    sofa.sofa_conventions = "SimpleFreeFieldHRIR";
    sofa.convention_version = "1.0";
    sofa.version = "1.0";
    sofa.data_type = "FIR";
    sofa.room_type = "free field";
    sofa.receiver_position[0] = 0.0;
    sofa.receiver_position[1] = 0.09;
    sofa.receiver_position[2] = 0.0;
    sofa.receiver_position[3] = 0.0;
    sofa.receiver_position[4] = -0.09;
    sofa.receiver_position[5] = 0.0;
    sofa.source_position_coordinates.type = "spherical";
    sofa.source_position_coordinates.units = "degree, degree, metre";
    sofa.emitter_position_coordinates.type = "cartesian";
    sofa.emitter_position_coordinates.units = "metre";
    sofa.listener_position_coordinates.type = "cartesian";
    sofa.listener_position_coordinates.units = "metre";
    sofa.listener_view_coordinates.type = "cartesian";
    sofa.listener_view_coordinates.units = "metre";
    sofa.receiver_position_coordinates.type = "cartesian";
    sofa.receiver_position_coordinates.units = "metre";
    sofa.source_sha256 = std::string(64u, '0');
    sofa.source_position.assign(static_cast<std::size_t>(measurements) * 3u, 0.0);
    sofa.ir.assign(static_cast<std::size_t>(measurements) * 2u * taps, 0.0);
    for (std::uint32_t index = 0; index < measurements; ++index) {
        // Fibonacci sphere: an even spread over the whole sphere, every point at
        // one metre, so the set is a single shell.
        const double unit = (static_cast<double>(index) + 0.5) / static_cast<double>(measurements);
        const double elevation = std::asin(2.0 * unit - 1.0);
        const double azimuth =
            2.0 * kPi * std::fmod(static_cast<double>(index) * 0.6180339887498949, 1.0);
        sofa.source_position[index * 3u + 0u] = azimuth * 180.0 / kPi;
        sofa.source_position[index * 3u + 1u] = elevation * 180.0 / kPi;
        sofa.source_position[index * 3u + 2u] = 1.0;
        for (std::uint32_t ear = 0; ear < 2u; ++ear) {
            for (std::uint32_t tap = 0; tap < taps; ++tap) {
                const double decay =
                    std::exp(-4.0 * static_cast<double>(tap) / static_cast<double>(taps));
                const double phase = 0.2 * static_cast<double>(tap) +
                                     0.05 * static_cast<double>(index) +
                                     1.5 * static_cast<double>(ear) + elevation;
                sofa.ir[(static_cast<std::size_t>(index) * 2u + ear) * taps + tap] =
                    decay * std::sin(phase);
            }
        }
    }
    return sofa;
}

void test_sofa_canonicalize() {
    const joc::hrtf::SofaHrir sofa = synthetic_sofa(64u, 64u);

    joc::hrtf::CanonicalHrtf canonical;
    const joc::Status status = joc::hrtf::canonicalize_sofa(sofa, &canonical);
    CHECK(status.ok());
    if (!status.ok()) {
        std::printf("      canonicalize_sofa failed: %s\n", status.message().c_str());
        return;
    }
    CHECK(canonical.measurements == 64u && canonical.taps == 64u);
    CHECK(canonical.sample_rate_hz == 48000.0);
    CHECK(canonical.hrir.size() == 64u * 2u * 64u);
    CHECK(canonical.delay_samples.size() == 64u * 2u);
    CHECK(canonical.unit_directions.size() == 64u * 3u);
    CHECK(canonical.source_position_cartesian_m.size() == 64u * 3u);
    CHECK(canonical.left_receiver_index != canonical.right_receiver_index);
    CHECK(canonical.left_receiver_index >= 0 && canonical.left_receiver_index < 2);
    CHECK(canonical.right_receiver_index >= 0 && canonical.right_receiver_index < 2);
    bool unit_length = canonical.unit_directions.size() == 64u * 3u;
    bool one_shell = canonical.measurement_radius_m.size() == 64u;
    for (std::size_t index = 0u; index < 64u; ++index) {
        const double x = canonical.unit_directions[index * 3u + 0u];
        const double y = canonical.unit_directions[index * 3u + 1u];
        const double z = canonical.unit_directions[index * 3u + 2u];
        unit_length = unit_length && std::abs(x * x + y * y + z * z - 1.0) < 1.0e-9;
        one_shell = one_shell && std::abs(canonical.measurement_radius_m[index] - 1.0) < 1.0e-9;
    }
    CHECK(unit_length);
    CHECK(one_shell);

    // The shell the compile step selects, and the radius it reports for it.
    double actual_radius = 0.0;
    const std::vector<std::size_t> shell =
        joc::hrtf::canonical_shell_indices(canonical, 0.5, &actual_radius);
    CHECK(shell.size() == 64u);
    CHECK(std::abs(actual_radius - 1.0) < 1.0e-9);

    // The import is strict: each of these is a rejection, never a silent repair.
    joc::hrtf::CanonicalHrtf rejected;
    joc::hrtf::SofaHrir broken = sofa;
    broken.conventions = "not-sofa";
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.sofa_conventions = "SimpleFreeFieldHRTF";
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.convention_version = "9.9";
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.data_type = "FLOAT";
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.room_type = "";
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.sampling_rate_units = "furlongs";
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.sample_rate = 0.0;
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.ir[0] = std::numeric_limits<double>::quiet_NaN();
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.delay[1] = -1.0;
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.emitter_position[0] = 0.5;
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.listener_view[0] = broken.listener_view[1] = broken.listener_view[2] = 0.0;
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.listener_up[0] = 1.0;  // parallel to ListenerView
    broken.listener_up[1] = 0.0;
    broken.listener_up[2] = 0.0;
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.source_position[2] = -1.0;  // a negative spherical radius
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
    broken = sofa;
    broken.ir.resize(7u);
    CHECK(!joc::hrtf::canonicalize_sofa(broken, &rejected).ok());
}

void test_sofa_compile_and_cache() {
    const joc::hrtf::SofaHrir sofa = synthetic_sofa(64u, 64u);
    const joc::hrtf::CompileOptions options = joc::hrtf::CompileOptions();
    joc::hrtf::Field field;

    // The runtime field is fixed at fifth order and at 48 kHz.
    joc::hrtf::CompileOptions wrong_order = options;
    wrong_order.order = 3;
    CHECK(!joc::hrtf::compile_sofa_field(sofa, wrong_order, &field).ok());
    joc::hrtf::SofaHrir wrong_rate = sofa;
    wrong_rate.sample_rate = 44100.0;
    CHECK(!joc::hrtf::compile_sofa_field(wrong_rate, options, &field).ok());

    const joc::Status status = joc::hrtf::compile_sofa_field(sofa, options, &field);
    CHECK(status.ok());
    if (!status.ok()) {
        std::printf("      compile_sofa_field failed: %s\n", status.message().c_str());
        return;
    }
    CHECK(field.order == 5);
    CHECK(field.measurement_radius_m == 1.0);
    CHECK(field.band_centers_hz.size() == 77u);
    CHECK(field.coefficients.size() == 36u * 2u * 77u * 2u);
    CHECK(field.delay_coefficients.size() == 36u * 2u);
    CHECK(field.delay_bounds.size() == 4u);
    CHECK(!field.cache_key.empty() && !field.payload_sha256.empty());
    CHECK(!field.metadata_json.empty() && !field.delay_source.empty());
    bool finite = true;
    for (const double value : field.coefficients) {
        finite = finite && std::isfinite(value);
    }
    for (const double value : field.delay_coefficients) {
        finite = finite && std::isfinite(value);
    }
    CHECK(finite);
    // The payload digest is a pure function of the compiled arrays.
    CHECK(joc::hrtf::field_payload_sha256(field) == field.payload_sha256);

    // The cache round-trips through the shipped writer and loader.
    const std::string cache_path = joc::fs_utf8::temp_directory() + "/joc_test_cache.jochrtf";
    CHECK(joc::hrtf::write_jochrtf(field, cache_path).ok());
    joc::hrtf::Field reloaded;
    const joc::Status loaded = joc::hrtf::load_jochrtf(cache_path, &reloaded);
    CHECK(loaded.ok());
    if (loaded.ok()) {
        CHECK(reloaded.cache_key == field.cache_key);
        CHECK(reloaded.source_sha256 == field.source_sha256);
        CHECK(reloaded.order == field.order);
        CHECK(reloaded.measurement_radius_m == field.measurement_radius_m);
        CHECK(reloaded.band_centers_hz == field.band_centers_hz);
        CHECK(reloaded.coefficients == field.coefficients);
        CHECK(reloaded.delay_coefficients == field.delay_coefficients);
        CHECK(reloaded.delay_bounds == field.delay_bounds);
        CHECK(reloaded.payload_sha256 == field.payload_sha256);
    }
    // A cache is trusted only after its key and payload verify; a damaged one is
    // rejected instead of being used.
    CHECK(joc::hrtf::validate_jochrtf(cache_path, field.source_sha256, field.cache_key, nullptr)
              .ok());
    CHECK(!joc::hrtf::validate_jochrtf(cache_path, field.source_sha256, "00", nullptr).ok());
    CHECK(joc::fs_utf8::remove(cache_path) == 0);

    const std::string poisoned = joc::fs_utf8::temp_directory() + "/joc_test_poisoned.jochrtf";
    CHECK(joc::io::write_zip(poisoned, {}, nullptr));
    CHECK(!joc::hrtf::validate_jochrtf(poisoned, field.source_sha256, field.cache_key, nullptr)
               .ok());
    CHECK(!joc::hrtf::load_jochrtf(poisoned, &reloaded).ok());
    CHECK(joc::fs_utf8::remove(poisoned) == 0);
}

void test_sofa_reader_errors() {
    // No SOFA file is needed to check that the reader answers correctly: a path
    // that does not exist and a file that is not HDF5 are distinct, reported
    // errors rather than a partially filled structure.
    joc::hrtf::SofaHrir sofa;
    const std::string missing = joc::fs_utf8::temp_directory() + "/joc_absent.sofa";
    const joc::Status absent = joc::hrtf::load_sofa(missing, &sofa);
    CHECK(!absent.ok());
    CHECK(!absent.message().empty());
    CHECK(!joc::hrtf::load_sofa(std::string(), &sofa).ok());
    CHECK(!joc::hrtf::load_sofa(missing, nullptr).ok());

    const std::string garbage = joc::fs_utf8::temp_directory() + "/joc_garbage.sofa";
    {
        std::ofstream stream = joc::fs_utf8::open_output(garbage);
        stream << "this is not an HDF5 container";
    }
    const joc::Status malformed = joc::hrtf::load_sofa(garbage, &sofa);
    CHECK(!malformed.ok());
    CHECK(!malformed.message().empty());
    CHECK(joc::fs_utf8::remove(garbage) == 0);
}

void test_hrtf_cache_policy() {
    // The policy parser and the miss path of the cache layer, neither of which
    // needs a SOFA file.
    joc::hrtf::CachePolicy policy = joc::hrtf::CachePolicy::None;
    CHECK(joc::hrtf::parse_cache_policy("none", &policy).ok() &&
          policy == joc::hrtf::CachePolicy::None);
    CHECK(joc::hrtf::parse_cache_policy("memory", &policy).ok() &&
          policy == joc::hrtf::CachePolicy::Memory);
    CHECK(joc::hrtf::parse_cache_policy("disk", &policy).ok() &&
          policy == joc::hrtf::CachePolicy::Disk);
    CHECK(!joc::hrtf::parse_cache_policy("sometimes", &policy).ok());

    joc::hrtf::SofaFieldRequest request;
    request.sofa_path = joc::fs_utf8::temp_directory() + "/joc_absent.sofa";
    request.policy = joc::hrtf::CachePolicy::Disk;
    request.cache_dir = joc::fs_utf8::temp_directory() + "/joc_absent_cache";
    joc::hrtf::Field field;
    std::string written = "sentinel";
    CHECK(!joc::hrtf::load_or_compile_sofa_field(request, &field, &written).ok());
}

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

}  // namespace

int main() {
    test_sha256();
    test_crc32_and_inflate();
    test_bit_reader();
    test_variable_bits();
    test_eac3_frame_bytes();
    test_mini_json();
    test_npy();
    test_oamd_and_adm_helpers();
    test_int24_packing();
    test_utf8_paths();
    test_process_runner();
    test_builtin_kernel_tables();
    test_npy_and_zip_writers();
    test_sofa_canonicalize();
    test_sofa_compile_and_cache();
    test_sofa_reader_errors();
    test_hrtf_cache_policy();
    test_rosella_model_errors();
    std::printf("%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
