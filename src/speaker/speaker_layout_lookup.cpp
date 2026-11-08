#include "speaker/speaker_layout_lookup.h"

#include <cstring>

namespace joc::speaker {

namespace {

struct FrozenLayout {
    const char* name;
    const char* compact_name;
    std::uint32_t out_ch_config;
    std::uint32_t speaker_bitfield;
    std::uint32_t surround_channels;
    std::uint32_t height_channels;
    bool carries_trim;
    const char* const* channel_names;
};

const char* const kChannels20[] = {"L", "R"};
const char* const kChannels31[] = {"L", "R", "C", "LFE"};
const char* const kChannels40[] = {"L", "R", "Lrs", "Rrs"};
const char* const kChannels51[] = {"L", "R", "C", "LFE", "Ls", "Rs"};
const char* const kChannels71[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs"};
const char* const kChannels512[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Ltm", "Rtm"};
const char* const kChannels514[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Ltf", "Rtf",
                                    "Ltr", "Rtr"};
const char* const kChannels712[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
                                    "Ltm", "Rtm"};
const char* const kChannels714[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
                                    "Ltf", "Rtf", "Ltr", "Rtr"};
const char* const kChannels914[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
                                    "Lw", "Rw", "Ltf", "Rtf", "Ltr", "Rtr"};
const char* const kChannels916[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
                                    "Lw", "Rw", "Ltf", "Rtf", "Ltm", "Rtm", "Ltr", "Rtr"};
const char* const kChannels60[] = {"L", "R", "Ls", "Rs", "Lrs", "Rrs"};

constexpr FrozenLayout kLayouts[kLayoutCount] = {
    {"2.0", "20", 0, 1, 0, 0, true, kChannels20},
    {"3.1", "31", 3, 7, 0, 0, true, kChannels31},
    {"4.0", "40", 0xFF, 9, 2, 0, true, kChannels40},
    {"5.1", "51", 7, 15, 2, 0, true, kChannels51},
    {"7.1", "71", 11, 31, 4, 0, true, kChannels71},
    {"5.1.2", "512", 13, 1039, 2, 2, true, kChannels512},
    {"5.1.4", "514", 14, 2575, 2, 4, true, kChannels514},
    {"7.1.2", "712", 15, 1055, 4, 2, false, kChannels712},
    {"7.1.4", "714", 16, 2591, 4, 4, false, kChannels714},
    {"9.1.4", "914", 19, 2719, 4, 4, false, kChannels914},
    {"9.1.6", "916", 20, 3743, 4, 6, false, kChannels916},
    {"6.0", "60", 0xFF, 25, 4, 0, false, kChannels60},
};

void fill(const FrozenLayout& source, LayoutInfo* out) {
    out->name = source.name;
    out->compact_name = source.compact_name;
    out->out_ch_config = source.out_ch_config;
    out->speaker_bitfield = source.speaker_bitfield;
    out->channel_count = ejoc_speaker_layout_channel_count(source.speaker_bitfield);
    out->channel_names = source.channel_names;
    out->surround_channels = source.surround_channels;
    out->height_channels = source.height_channels;
    out->carries_trim = source.carries_trim;
}

}  // namespace

bool layout_at(int index, LayoutInfo* out) {
    if (out == nullptr || index < 0 || index >= kLayoutCount) {
        return false;
    }
    fill(kLayouts[index], out);
    return true;
}

bool layout_by_name(const char* name, LayoutInfo* out) {
    if (name == nullptr || out == nullptr) {
        return false;
    }
    for (int index = 0; index < kLayoutCount; ++index) {
        if (std::strcmp(name, kLayouts[index].name) == 0) {
            fill(kLayouts[index], out);
            return true;
        }
    }
    return false;
}

bool layout_by_bitfield(std::uint32_t bitfield, LayoutInfo* out) {
    if (out == nullptr) {
        return false;
    }
    for (int index = 0; index < kLayoutCount; ++index) {
        if (kLayouts[index].speaker_bitfield == bitfield) {
            fill(kLayouts[index], out);
            return true;
        }
    }
    return false;
}

}  // namespace joc::speaker
