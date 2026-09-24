#include "hrtf/sofa.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <utility>

#include "foundation/fs_utf8.h"
#include "foundation/sha256.h"
#include "io/hdf5.h"

namespace joc::hrtf {

namespace {

Status sofa_fail(joc_error code, const std::string& message) {
    return Status::fail(code, stage::kRender, message);
}

std::string format_number(double value) {
    if (std::isfinite(value) && value == std::floor(value) && std::fabs(value) < 1.0e15) {
        return std::to_string(static_cast<long long>(value));
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return std::string(buffer);
}

// Every array is checked against the element count the convention prescribes, so
// a file whose shape disagrees with its metadata is rejected instead of silently
// producing a shifted impulse response.
Status read_doubles(const io::Hdf5File& file, const std::string& path, std::uint64_t expected,
                    std::vector<double>* out) {
    if (!file.has_dataset(path)) {
        return sofa_fail(JOC_ERR_HRTF_FORMAT, "SOFA file has no " + path + " dataset");
    }
    const Status status = file.read_dataset_double(path, out);
    if (!status.ok()) {
        return sofa_fail(JOC_ERR_HRTF_FORMAT, "SOFA dataset " + path + ": " + status.message());
    }
    if (out->size() != expected) {
        return sofa_fail(JOC_ERR_HRTF_FORMAT,
                         "SOFA dataset " + path + " holds " + std::to_string(out->size()) +
                             " values, expected " + std::to_string(expected));
    }
    return Status::success();
}

Status read_text(const io::Hdf5File& file, const std::string& name, bool required,
                 std::string* out) {
    io::Hdf5Attribute attribute;
    const Status status = file.attribute("", name, &attribute);
    if (!status.ok()) {
        if (required) {
            return sofa_fail(JOC_ERR_HRTF_FORMAT,
                             "SOFA file has no root attribute " + name + ": " + status.message());
        }
        return Status::success();
    }
    *out = attribute.text;
    return Status::success();
}

// SHA-256 of the whole file: the compiled-cache key is derived from it, so the
// digest is taken over the exact bytes the parse consumed.
std::string file_digest(const std::string& path) {
    std::ifstream stream = fs_utf8::open_input(path);
    if (!stream.good()) {
        return std::string();
    }
    crypto::Sha256 hash;
    std::vector<char> buffer(1u << 20);
    while (stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            hash.update(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    return hash.finish_hex();
}

// The coordinate declaration of one dataset, when the file carries it.
void read_coordinates(const io::Hdf5File& file, const std::string& dataset,
                      SofaCoordinate* out) {
    io::Hdf5Attribute attribute;
    if (file.attribute(dataset, "Type", &attribute).ok()) {
        out->type = attribute.text;
    }
    if (file.attribute(dataset, "Units", &attribute).ok()) {
        out->units = attribute.text;
    }
}

}  // namespace

std::string SofaHrir::summary() const {
    return "SOFA " + sofa_conventions + ", " + std::to_string(ir_count) + " IRs x " +
           std::to_string(ir_length) + " taps @ " + format_number(sample_rate) + " Hz";
}

Status load_sofa(const std::string& path, SofaHrir* out) {
    if (out == nullptr) {
        return sofa_fail(JOC_ERR_INVALID_ARGUMENT, "null SOFA destination");
    }
    if (!fs_utf8::exists(path)) {
        return sofa_fail(JOC_ERR_HRTF_NOT_FOUND, "SOFA file not found: " + path);
    }
    io::Hdf5File file;
    Status status = file.open(path);
    if (!status.ok()) {
        return sofa_fail(JOC_ERR_HRTF_FORMAT, "SOFA file " + path + ": " + status.message());
    }

    SofaHrir sofa;
    sofa.source_path = path;
    sofa.source_sha256 = file_digest(path);
    for (char& character : sofa.source_sha256) {
        character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    }
    status = read_text(file, "Conventions", true, &sofa.conventions);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "SOFAConventions", true, &sofa.sofa_conventions);
    if (!status.ok()) {
        return status;
    }
    if (sofa.conventions != "SOFA" || sofa.sofa_conventions != "SimpleFreeFieldHRIR") {
        return sofa_fail(JOC_ERR_HRTF_UNSUPPORTED_CONVENTION,
                         "SOFA conventions " + sofa.conventions + "/" + sofa.sofa_conventions +
                             " are not SimpleFreeFieldHRIR");
    }
    status = read_text(file, "SOFAConventionsVersion", false, &sofa.convention_version);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "Version", false, &sofa.version);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "DataType", false, &sofa.data_type);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "RoomType", false, &sofa.room_type);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "Title", false, &sofa.title);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "DatabaseName", false, &sofa.database_name);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "ListenerShortName", false, &sofa.listener_short_name);
    if (!status.ok()) {
        return status;
    }
    status = read_text(file, "Comment", false, &sofa.comment);
    if (!status.ok()) {
        return status;
    }

    io::Hdf5DatasetInfo info;
    status = file.dataset_info("Data.IR", &info);
    if (!status.ok()) {
        return sofa_fail(JOC_ERR_HRTF_FORMAT,
                         "SOFA file has no usable Data.IR dataset: " + status.message());
    }
    if (info.shape.size() != 3u || info.shape[1] != 2u || info.shape[0] == 0u ||
        info.shape[2] == 0u) {
        return sofa_fail(JOC_ERR_HRTF_FORMAT, "SOFA Data.IR is not shaped (M, 2, N)");
    }
    if (info.shape[0] > 0xFFFFFFFFull || info.shape[2] > 0xFFFFFFFFull) {
        return sofa_fail(JOC_ERR_HRTF_FORMAT, "SOFA Data.IR is larger than this reader accepts");
    }
    sofa.ir_count = static_cast<std::uint32_t>(info.shape[0]);
    sofa.ir_length = static_cast<std::uint32_t>(info.shape[2]);
    const std::uint64_t taps = static_cast<std::uint64_t>(sofa.ir_count) * 2u * sofa.ir_length;
    status = read_doubles(file, "Data.IR", taps, &sofa.ir);
    if (!status.ok()) {
        return status;
    }

    std::vector<double> scalar;
    status = read_doubles(file, "Data.SamplingRate", 1u, &scalar);
    if (!status.ok()) {
        return status;
    }
    sofa.sample_rate = scalar[0];
    io::Hdf5Attribute attribute;
    if (file.attribute("Data.SamplingRate", "Units", &attribute).ok()) {
        sofa.sampling_rate_units = attribute.text;
    }

    // Data.Delay is optional in the wild; absent means "no delay was measured".
    if (file.has_dataset("Data.Delay")) {
        std::vector<double> delay;
        status = read_doubles(file, "Data.Delay", 2u, &delay);
        if (!status.ok()) {
            return status;
        }
        sofa.delay[0] = delay[0];
        sofa.delay[1] = delay[1];
    }

    const std::uint64_t measurements = sofa.ir_count;
    status = read_doubles(file, "SourcePosition", measurements * 3u, &sofa.source_position);
    if (!status.ok()) {
        return status;
    }
    read_coordinates(file, "SourcePosition", &sofa.source_position_coordinates);

    std::vector<double> vector;
    status = read_doubles(file, "ListenerPosition", 3u, &vector);
    if (!status.ok()) {
        return status;
    }
    std::copy(vector.begin(), vector.end(), sofa.listener_position);
    read_coordinates(file, "ListenerPosition", &sofa.listener_position_coordinates);
    status = read_doubles(file, "ListenerView", 3u, &vector);
    if (!status.ok()) {
        return status;
    }
    std::copy(vector.begin(), vector.end(), sofa.listener_view);
    read_coordinates(file, "ListenerView", &sofa.listener_view_coordinates);
    status = read_doubles(file, "ListenerUp", 3u, &vector);
    if (!status.ok()) {
        return status;
    }
    std::copy(vector.begin(), vector.end(), sofa.listener_up);
    read_coordinates(file, "ListenerUp", &sofa.listener_up_coordinates);
    status = read_doubles(file, "EmitterPosition", 3u, &vector);
    if (!status.ok()) {
        return status;
    }
    std::copy(vector.begin(), vector.end(), sofa.emitter_position);
    read_coordinates(file, "EmitterPosition", &sofa.emitter_position_coordinates);
    status = read_doubles(file, "ReceiverPosition", 6u, &vector);
    if (!status.ok()) {
        return status;
    }
    std::copy(vector.begin(), vector.end(), sofa.receiver_position);
    read_coordinates(file, "ReceiverPosition", &sofa.receiver_position_coordinates);

    *out = std::move(sofa);
    return Status::success();
}

}  // namespace joc::hrtf
