#include "io/inflate.h"

#include <cstring>

namespace joc::io {

namespace {

class LsbBitReader {
public:
    LsbBitReader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    bool ok() const { return ok_; }
    std::size_t byte_position() const { return position_ >> 3; }

    std::uint32_t bits(unsigned count) {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < count; ++i) {
            if ((position_ >> 3) >= size_) {
                ok_ = false;
                return value;
            }
            const std::uint32_t bit = (data_[position_ >> 3] >> (position_ & 7u)) & 1u;
            value |= bit << i;
            ++position_;
        }
        return value;
    }

    void align_to_byte() { position_ = (position_ + 7u) & ~static_cast<std::size_t>(7u); }

    void skip_bytes(std::size_t count) { position_ += count * 8u; }

private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t position_ = 0;
    bool ok_ = true;
};

struct Huffman {
    std::uint16_t counts[16] = {};
    std::uint16_t symbols[288] = {};
    int max_length = 0;

    bool build(const std::uint8_t* lengths, int count) {
        for (int i = 0; i < 16; ++i) {
            counts[i] = 0;
        }
        for (int i = 0; i < count; ++i) {
            counts[lengths[i]]++;
        }
        counts[0] = 0;
        std::uint16_t offsets[16] = {};
        std::uint16_t total = 0;
        for (int length = 1; length < 16; ++length) {
            offsets[length] = total;
            total = static_cast<std::uint16_t>(total + counts[length]);
        }
        if (total == 0) {
            return false;
        }
        for (int symbol = 0; symbol < count; ++symbol) {
            const std::uint8_t length = lengths[symbol];
            if (length != 0) {
                symbols[offsets[length]++] = static_cast<std::uint16_t>(symbol);
            }
        }
        max_length = 15;
        while (max_length > 0 && counts[max_length] == 0) {
            --max_length;
        }
        return max_length != 0;
    }

    int decode(LsbBitReader* reader) const {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int length = 1; length <= max_length; ++length) {
            code |= static_cast<int>(reader->bits(1));
            if (!reader->ok()) {
                return -1;
            }
            const int count = counts[length];
            if (code - first < count) {
                return symbols[index + (code - first)];
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        return -1;
    }
};

constexpr std::uint16_t kLengthBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,
                                      23, 27, 31, 35, 43, 51, 59, 67,  83,  99,  115, 131, 163,
                                      195, 227, 258};
constexpr std::uint8_t kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                      2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::uint16_t kDistanceBase[30] = {1,    2,    3,    4,    5,    7,     9,     13,
                                        17,   25,   33,   49,   65,   97,    129,   193,
                                        257,  385,  513,  769,  1025, 1537,  2049,  3073,
                                        4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::uint8_t kDistanceExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
constexpr std::uint8_t kCodeLengthOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2,
                                          14, 1, 15};

bool inflate_block_data(LsbBitReader* reader, const Huffman& literal, const Huffman& distance,
                        std::vector<std::uint8_t>* out) {
    for (;;) {
        const int symbol = literal.decode(reader);
        if (symbol < 0) {
            return false;
        }
        if (symbol < 256) {
            out->push_back(static_cast<std::uint8_t>(symbol));
            continue;
        }
        if (symbol == 256) {
            return true;
        }
        const int length_index = symbol - 257;
        if (length_index >= 29) {
            return false;
        }
        const std::uint32_t length =
            kLengthBase[length_index] + reader->bits(kLengthExtra[length_index]);
        const int distance_symbol = distance.decode(reader);
        if (distance_symbol < 0 || distance_symbol >= 30) {
            return false;
        }
        const std::uint32_t distance_value =
            kDistanceBase[distance_symbol] + reader->bits(kDistanceExtra[distance_symbol]);
        if (!reader->ok() || distance_value == 0 || distance_value > out->size()) {
            return false;
        }
        const std::size_t start = out->size() - distance_value;
        for (std::uint32_t i = 0; i < length; ++i) {
            out->push_back((*out)[start + i]);
        }
    }
}

bool inflate_fixed(LsbBitReader* reader, std::vector<std::uint8_t>* out) {
    std::uint8_t lengths[288];
    for (int i = 0; i < 144; ++i) { lengths[i] = 8; }
    for (int i = 144; i < 256; ++i) { lengths[i] = 9; }
    for (int i = 256; i < 280; ++i) { lengths[i] = 7; }
    for (int i = 280; i < 288; ++i) { lengths[i] = 8; }
    Huffman literal;
    if (!literal.build(lengths, 288)) {
        return false;
    }
    std::uint8_t distance_lengths[30];
    for (int i = 0; i < 30; ++i) { distance_lengths[i] = 5; }
    Huffman distance;
    if (!distance.build(distance_lengths, 30)) {
        return false;
    }
    return inflate_block_data(reader, literal, distance, out);
}

bool inflate_dynamic(LsbBitReader* reader, std::vector<std::uint8_t>* out) {
    const int literal_count = static_cast<int>(reader->bits(5)) + 257;
    const int distance_count = static_cast<int>(reader->bits(5)) + 1;
    const int code_length_count = static_cast<int>(reader->bits(4)) + 4;
    if (!reader->ok() || literal_count > 286 || distance_count > 30) {
        return false;
    }
    std::uint8_t code_lengths[19] = {};
    for (int i = 0; i < code_length_count; ++i) {
        code_lengths[kCodeLengthOrder[i]] = static_cast<std::uint8_t>(reader->bits(3));
    }
    if (!reader->ok()) {
        return false;
    }
    Huffman code_length_tree;
    if (!code_length_tree.build(code_lengths, 19)) {
        return false;
    }
    std::uint8_t lengths[288 + 30] = {};
    const int total = literal_count + distance_count;
    int index = 0;
    while (index < total) {
        const int symbol = code_length_tree.decode(reader);
        if (symbol < 0) {
            return false;
        }
        if (symbol < 16) {
            lengths[index++] = static_cast<std::uint8_t>(symbol);
            continue;
        }
        int repeat = 0;
        std::uint8_t value = 0;
        if (symbol == 16) {
            if (index == 0) {
                return false;
            }
            value = lengths[index - 1];
            repeat = 3 + static_cast<int>(reader->bits(2));
        } else if (symbol == 17) {
            repeat = 3 + static_cast<int>(reader->bits(3));
        } else {
            repeat = 11 + static_cast<int>(reader->bits(7));
        }
        if (!reader->ok() || index + repeat > total) {
            return false;
        }
        for (int i = 0; i < repeat; ++i) {
            lengths[index++] = value;
        }
    }
    Huffman literal;
    if (!literal.build(lengths, literal_count)) {
        return false;
    }
    Huffman distance;
    if (!distance.build(lengths + literal_count, distance_count)) {
        return false;
    }
    return inflate_block_data(reader, literal, distance, out);
}

}  // namespace

bool inflate_raw(const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>* out) {
    if (data == nullptr || out == nullptr) {
        return false;
    }
    out->clear();
    LsbBitReader reader(data, size);
    for (;;) {
        const std::uint32_t final_block = reader.bits(1);
        const std::uint32_t type = reader.bits(2);
        if (!reader.ok()) {
            return false;
        }
        if (type == 0) {
            reader.align_to_byte();
            const std::size_t position = reader.byte_position();
            if (position + 4 > size) {
                return false;
            }
            const std::uint16_t length = static_cast<std::uint16_t>(data[position] | (data[position + 1] << 8));
            const std::uint16_t complement =
                static_cast<std::uint16_t>(data[position + 2] | (data[position + 3] << 8));
            if (static_cast<std::uint16_t>(length ^ 0xFFFFu) != complement) {
                return false;
            }
            if (position + 4 + length > size) {
                return false;
            }
            out->insert(out->end(), data + position + 4, data + position + 4 + length);
            reader.skip_bytes(4u + length);
        } else if (type == 1) {
            if (!inflate_fixed(&reader, out)) {
                return false;
            }
        } else if (type == 2) {
            if (!inflate_dynamic(&reader, out)) {
                return false;
            }
        } else {
            return false;
        }
        if (final_block != 0u) {
            break;
        }
    }
    return true;
}

}  // namespace joc::io
