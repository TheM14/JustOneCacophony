#include "adm/adm_metadata.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "foundation/py_num.h"

namespace joc::adm {

namespace {

constexpr const char* kBedNames[10] = {
    "RoomCentricLeft", "RoomCentricRight", "RoomCentricCenter", "RoomCentricLFE",
    "RoomCentricLeftSideSurround", "RoomCentricRightSideSurround",
    "RoomCentricLeftRearSurround", "RoomCentricRightRearSurround",
    "RoomCentricLeftTopSurround", "RoomCentricRightTopSurround"};
constexpr const char* kBedLabels[10] = {"RC_L", "RC_R",  "RC_C",  "RC_LFE", "RC_Lss",
                                        "RC_Rss", "RC_Lrs", "RC_Rrs", "RC_Lts", "RC_Rts"};
constexpr double kBedPos[10][3] = {{-1.0, 1.0, 0.0},   {1.0, 1.0, 0.0},   {0.0, 1.0, 0.0},
                                   {-1.0, 1.0, -1.0},  {-1.0, 0.0, 0.0},  {1.0, 0.0, 0.0},
                                   {-1.0, -1.0, 0.0},  {1.0, -1.0, 0.0},  {-1.0, 0.0, 1.0},
                                   {1.0, 0.0, 1.0}};

std::string hex4(std::uint32_t value) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%04x", value);
    return std::string(buffer);
}

std::string hex8(std::uint32_t value) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%08x", value);
    return std::string(buffer);
}

void put_u16(std::string* out, std::uint16_t value) {
    char buffer[2];
    std::memcpy(buffer, &value, 2);
    out->append(buffer, 2);
}

void put_u32(std::string* out, std::uint32_t value) {
    char buffer[4];
    std::memcpy(buffer, &value, 4);
    out->append(buffer, 4);
}

std::uint8_t checksum(const std::string& segment) {
    int sum = static_cast<int>(segment.size());
    for (const char raw : segment) {
        sum += static_cast<unsigned char>(raw);
    }
    return static_cast<std::uint8_t>((~sum + 1) & 0xFF);
}

}  // namespace

std::string ts(double seconds) {
    long long whole = static_cast<long long>(seconds);
    long long fraction = pynum::py_round((seconds - static_cast<double>(whole)) * 100000.0);
    if (fraction >= 100000) {
        whole += 1;
        fraction = 0;
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%02lld:%02lld:%02lld.%05lld", whole / 3600,
                  (whole % 3600) / 60, whole % 60, fraction);
    return std::string(buffer);
}

bool binaural_mode_from_name(const char* name, BinauralMode* out) {
    if (name == nullptr || out == nullptr) {
        return false;
    }
    if (std::strcmp(name, "off") == 0) { *out = BinauralMode::Off; return true; }
    if (std::strcmp(name, "near") == 0) { *out = BinauralMode::Near; return true; }
    if (std::strcmp(name, "far") == 0) { *out = BinauralMode::Far; return true; }
    if (std::strcmp(name, "mid") == 0) { *out = BinauralMode::Mid; return true; }
    if (std::strcmp(name, "unspecified") == 0) { *out = BinauralMode::Unspecified; return true; }
    return false;
}

std::string build_chna() {
    std::string out;
    put_u16(&out, static_cast<std::uint16_t>(kTrackCount));
    put_u16(&out, static_cast<std::uint16_t>(kTrackCount));
    for (std::uint32_t i = 0; i < 10; ++i) {
        put_u16(&out, static_cast<std::uint16_t>(i + 1));
        out += "ATU_" + hex8(i + 1);
        out += "AT_0001" + hex4(0x1001 + i) + "_01";
        out += "AP_00011001";
        out.push_back('\0');
    }
    for (std::uint32_t i = 0; i < kObjectCount; ++i) {
        put_u16(&out, static_cast<std::uint16_t>(i + 11));
        out += "ATU_" + hex8(i + 11);
        out += "AT_0003" + hex4(0x1001 + i) + "_01";
        out += "AP_0003" + hex4(0x1001 + i);
        out.push_back('\0');
    }
    return out;
}

Status build_dbmd(std::uint32_t object_count, BinauralMode mode, std::string* out) {
    if (out == nullptr) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput, "null output");
    }
    const std::uint32_t mode_value = static_cast<std::uint32_t>(mode);
    if (mode_value > 4u) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput,
                            "invalid JOC binaural render mode");
    }
    out->clear();
    put_u32(out, 0x01000006u);

    std::string segment7(96, '\0');
    segment7[1] = static_cast<char>(0x47);
    segment7[5] = static_cast<char>(0x60);
    segment7[8] = static_cast<char>(0x24);
    segment7[9] = static_cast<char>(0x24);
    out->push_back(7);
    put_u16(out, 96);
    out->append(segment7);
    out->push_back(static_cast<char>(checksum(segment7)));

    std::string segment9(248, '\0');
    const std::string creator = "Created with EAC3JOC";
    const std::string renderer = "EAC3JOC Python Renderer";
    std::memcpy(&segment9[0], creator.data(), creator.size());
    std::memcpy(&segment9[32], renderer.data(), renderer.size());
    segment9[96] = 2;
    segment9[97] = 1;
    segment9[98] = 0;
    segment9[103] = 0x03;
    segment9[106] = 0x01;
    segment9[111] = 0x22;
    segment9[112] = static_cast<char>(0xFF);
    out->push_back(9);
    put_u16(out, 248);
    out->append(segment9);
    out->push_back(static_cast<char>(checksum(segment9)));

    // The reference allocates the body zeroed and then fills only the trailing
    // `object_count` bytes with 0x84, so the template region stays zero.
    const std::size_t object_body = 5u + 262u + object_count;
    std::string segment10(object_body, '\0');
    const std::uint32_t sync = 0xF8726FBDu;
    std::memcpy(&segment10[0], &sync, 4);
    segment10[4] = static_cast<char>(object_count);
    for (std::size_t i = 5u + 262u; i < segment10.size(); ++i) {
        segment10[i] = static_cast<char>(0x84);
    }
    const std::size_t object_modes = 4u + 2u + 1u + 9u * 15u + object_count;
    for (std::uint32_t i = 10; i < std::min<std::uint32_t>(object_count, 10u + kObjectCount); ++i) {
        const std::size_t index = object_modes + i;
        if (index >= segment10.size()) {
            return Status::fail(JOC_ERR_INTERNAL, stage::kOutput, "dbmd object slot out of range");
        }
        segment10[index] = static_cast<char>((static_cast<unsigned char>(segment10[index]) & 0xF8u) |
                                             mode_value);
    }
    out->push_back(10);
    put_u16(out, static_cast<std::uint16_t>(segment10.size()));
    out->append(segment10);
    out->push_back(static_cast<char>(checksum(segment10)));
    out->append("\0\0", 2);
    return Status::success();
}

std::string build_axml(const std::vector<Track>& tracks, double duration_sec, std::uint32_t rate) {
    const double scale = static_cast<double>(rate);
    std::string out;
    out.reserve(64u * 1024u);
    const std::string duration_ts = ts(duration_sec);

    out += "<?xml version=\"1.0\" encoding=\"utf-8\"?>";
    out += "<ebuCoreMain xsi:schemaLocation=\"urn:ebu:metadata-schema:ebuCore_2016 ebucore.xsd\" "
           "lang=\"en\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
           "xmlns=\"urn:ebu:metadata-schema:ebuCore_2016\">";
    out += "<coreMetadata><format><audioFormatExtended>";
    out += "<audioProgramme audioProgrammeID=\"APR_1001\" audioProgrammeName=\"EAC3JOC_Export\" "
           "start=\"" +
           ts(0.0) + "\" end=\"" + duration_ts + "\">";
    out += "<audioContentIDRef>ACO_1001</audioContentIDRef>";
    out += "<audioContentIDRef>ACO_1002</audioContentIDRef>";
    out += "</audioProgramme>";
    out += "<audioContent audioContentID=\"ACO_1001\" "
           "audioContentName=\"EAC3JOC_Master_Content\">";
    out += "<audioObjectIDRef>AO_1001</audioObjectIDRef>";
    out += "<dialogue mixedContentKind=\"0\">2</dialogue>";
    out += "</audioContent>";
    out += "<audioContent audioContentID=\"ACO_1002\" audioContentName=\"Objects\">";
    for (std::uint32_t i = 0; i < kObjectCount; ++i) {
        out += "<audioObjectIDRef>AO_" + hex4(0x100b + i) + "</audioObjectIDRef>";
    }
    out += "<dialogue mixedContentKind=\"0\">2</dialogue>";
    out += "</audioContent>";
    out += "<audioObject audioObjectID=\"AO_1001\" audioObjectName=\"Bed\" start=\"" + ts(0.0) +
           "\" duration=\"" + duration_ts + "\">";
    out += "<audioPackFormatIDRef>AP_00011001</audioPackFormatIDRef>";
    for (std::uint32_t i = 0; i < 10; ++i) {
        out += "<audioTrackUIDRef>ATU_" + hex8(i + 1) + "</audioTrackUIDRef>";
    }
    out += "</audioObject>";
    for (std::uint32_t i = 0; i < kObjectCount; ++i) {
        out += "<audioObject audioObjectID=\"AO_" + hex4(0x100b + i) +
               "\" audioObjectName=\"Audio Object " + std::to_string(i + 1) + "\" start=\"" +
               ts(0.0) + "\" duration=\"" + duration_ts + "\">";
        out += "<audioPackFormatIDRef>AP_0003" + hex4(0x1001 + i) + "</audioPackFormatIDRef>";
        out += "<audioTrackUIDRef>ATU_" + hex8(11 + i) + "</audioTrackUIDRef>";
        out += "</audioObject>";
    }
    out += "<audioPackFormat audioPackFormatID=\"AP_00011001\" "
           "audioPackFormatName=\"EAC3JOCBedPack\" typeDefinition=\"DirectSpeakers\" "
           "typeLabel=\"0001\">";
    for (std::uint32_t i = 0; i < 10; ++i) {
        out += "<audioChannelFormatIDRef>AC_0001" + hex4(0x1001 + i) +
               "</audioChannelFormatIDRef>";
    }
    out += "</audioPackFormat>";
    for (std::uint32_t i = 0; i < kObjectCount; ++i) {
        out += "<audioPackFormat audioPackFormatID=\"AP_0003" + hex4(0x1001 + i) +
               "\" audioPackFormatName=\"JOC_Object_" + std::to_string(i + 1) +
               "\" typeDefinition=\"Objects\" typeLabel=\"0003\">";
        out += "<audioChannelFormatIDRef>AC_0003" + hex4(0x1001 + i) +
               "</audioChannelFormatIDRef>";
        out += "</audioPackFormat>";
    }
    for (std::uint32_t i = 0; i < 10; ++i) {
        out += "<audioChannelFormat audioChannelFormatID=\"AC_0001" + hex4(0x1001 + i) +
               "\" audioChannelFormatName=\"" + kBedNames[i] +
               "\" typeDefinition=\"DirectSpeakers\" typeLabel=\"0001\">";
        out += "<audioBlockFormat audioBlockFormatID=\"AB_0001" + hex4(0x1001 + i) +
               "_00000001\">";
        out += "<cartesian>1</cartesian>";
        out += "<position coordinate=\"X\">" + pynum::format_fixed(kBedPos[i][0], 10) +
               "</position>";
        out += "<position coordinate=\"Y\">" + pynum::format_fixed(kBedPos[i][1], 10) +
               "</position>";
        if (kBedPos[i][2] != 0.0) {
            out += "<position coordinate=\"Z\">" + pynum::format_fixed(kBedPos[i][2], 10) +
                   "</position>";
        }
        out += std::string("<speakerLabel>") + kBedLabels[i] + "</speakerLabel>";
        out += "</audioBlockFormat>";
        out += "</audioChannelFormat>";
    }
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        const Track& track = tracks[i];
        out += "<audioChannelFormat audioChannelFormatID=\"AC_0003" +
               hex4(0x1001 + static_cast<std::uint32_t>(i)) + "\" audioChannelFormatName=\"" +
               track.name + "\" typeDefinition=\"Objects\" typeLabel=\"0003\">";
        for (std::size_t k = 0; k < track.blocks.size(); ++k) {
            const Keyframe& block = track.blocks[k];
            out += "<audioBlockFormat audioBlockFormatID=\"AB_0003" +
                   hex4(0x1001 + static_cast<std::uint32_t>(i)) + "_" +
                   hex8(static_cast<std::uint32_t>(k + 1)) + "\" rtime=\"" +
                   ts(static_cast<double>(block.rtime_samples) / scale) + "\" duration=\"" +
                   ts(static_cast<double>(block.duration_samples) / scale) + "\">";
            out += "<cartesian>1</cartesian>";
            out += "<position coordinate=\"X\">" + pynum::format_fixed(block.x, 10) + "</position>";
            out += "<position coordinate=\"Y\">" + pynum::format_fixed(block.y, 10) + "</position>";
            if (block.z != 0.0) {
                out += "<position coordinate=\"Z\">" + pynum::format_fixed(block.z, 10) +
                       "</position>";
            }
            out += "<jumpPosition interpolationLength=\"" +
                   pynum::format_fixed(static_cast<double>(block.interpolation_samples) / scale,
                                       5) +
                   "\">1</jumpPosition>";
            out += "</audioBlockFormat>";
        }
        out += "</audioChannelFormat>";
    }
    for (std::uint32_t i = 0; i < 10; ++i) {
        out += "<audioTrackUID UID=\"ATU_" + hex8(i + 1) +
               "\" bitDepth=\"24\" sampleRate=\"48000\">";
        out += "<audioTrackFormatIDRef>AT_0001" + hex4(0x1001 + i) + "_01</audioTrackFormatIDRef>";
        out += "<audioPackFormatIDRef>AP_00011001</audioPackFormatIDRef>";
        out += "</audioTrackUID>";
    }
    for (std::uint32_t i = 0; i < kObjectCount; ++i) {
        out += "<audioTrackUID UID=\"ATU_" + hex8(11 + i) +
               "\" bitDepth=\"24\" sampleRate=\"48000\">";
        out += "<audioTrackFormatIDRef>AT_0003" + hex4(0x1001 + i) + "_01</audioTrackFormatIDRef>";
        out += "<audioPackFormatIDRef>AP_0003" + hex4(0x1001 + i) + "</audioPackFormatIDRef>";
        out += "</audioTrackUID>";
    }
    for (std::uint32_t i = 0; i < 10; ++i) {
        out += "<audioTrackFormat audioTrackFormatID=\"AT_0001" + hex4(0x1001 + i) +
               "_01\" audioTrackFormatName=\"PCM_" + kBedNames[i] +
               "\" formatDefinition=\"PCM\" formatLabel=\"0001\">";
        out += "<audioStreamFormatIDRef>AS_0001" + hex4(0x1001 + i) +
               "</audioStreamFormatIDRef>";
        out += "</audioTrackFormat>";
    }
    for (std::uint32_t i = 0; i < kObjectCount; ++i) {
        out += "<audioTrackFormat audioTrackFormatID=\"AT_0003" + hex4(0x1001 + i) +
               "_01\" audioTrackFormatName=\"PCM_JOC_Object_" + std::to_string(i + 1) +
               "\" formatDefinition=\"PCM\" formatLabel=\"0001\">";
        out += "<audioStreamFormatIDRef>AS_0003" + hex4(0x1001 + i) +
               "</audioStreamFormatIDRef>";
        out += "</audioTrackFormat>";
    }
    for (std::uint32_t i = 0; i < 10; ++i) {
        out += "<audioStreamFormat audioStreamFormatID=\"AS_0001" + hex4(0x1001 + i) +
               "\" audioStreamFormatName=\"PCM_" + kBedNames[i] +
               "\" formatDefinition=\"PCM\" formatLabel=\"0001\">";
        out += "<audioChannelFormatIDRef>AC_0001" + hex4(0x1001 + i) +
               "</audioChannelFormatIDRef>";
        out += "<audioPackFormatIDRef>AP_00011001</audioPackFormatIDRef>";
        out += "<audioTrackFormatIDRef>AT_0001" + hex4(0x1001 + i) +
               "_01</audioTrackFormatIDRef>";
        out += "</audioStreamFormat>";
    }
    for (std::uint32_t i = 0; i < kObjectCount; ++i) {
        out += "<audioStreamFormat audioStreamFormatID=\"AS_0003" + hex4(0x1001 + i) +
               "\" audioStreamFormatName=\"PCM_JOC_Object_" + std::to_string(i + 1) +
               "\" formatDefinition=\"PCM\" formatLabel=\"0001\">";
        out += "<audioChannelFormatIDRef>AC_0003" + hex4(0x1001 + i) +
               "</audioChannelFormatIDRef>";
        out += "<audioPackFormatIDRef>AP_0003" + hex4(0x1001 + i) + "</audioPackFormatIDRef>";
        out += "<audioTrackFormatIDRef>AT_0003" + hex4(0x1001 + i) +
               "_01</audioTrackFormatIDRef>";
        out += "</audioStreamFormat>";
    }
    out += "</audioFormatExtended></format></coreMetadata>";
    out += "</ebuCoreMain>";
    return out;
}

}  // namespace joc::adm
