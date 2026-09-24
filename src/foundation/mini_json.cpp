#include "foundation/mini_json.h"

#include <cmath>
#include <cstdlib>

namespace joc::json {

namespace {

void skip_space(const std::string& text, std::size_t* index) {
    while (*index < text.size() &&
           (text[*index] == ' ' || text[*index] == '\t' || text[*index] == '\n' ||
            text[*index] == '\r')) {
        ++(*index);
    }
}

bool read_string(const std::string& text, std::size_t* index, std::string* out) {
    if (*index >= text.size() || text[*index] != '"') {
        return false;
    }
    ++(*index);
    out->clear();
    while (*index < text.size()) {
        const char c = text[*index];
        if (c == '\\') {
            if (*index + 1 >= text.size()) {
                return false;
            }
            const char escape = text[*index + 1];
            *index += 2;
            switch (escape) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    if (*index + 4 > text.size()) {
                        return false;
                    }
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char digit = text[*index + static_cast<std::size_t>(i)];
                        code <<= 4;
                        if (digit >= '0' && digit <= '9') { code |= static_cast<unsigned>(digit - '0'); }
                        else if (digit >= 'a' && digit <= 'f') { code |= static_cast<unsigned>(digit - 'a' + 10); }
                        else if (digit >= 'A' && digit <= 'F') { code |= static_cast<unsigned>(digit - 'A' + 10); }
                        else { return false; }
                    }
                    *index += 4;
                    if (code < 0x80u) {
                        out->push_back(static_cast<char>(code));
                    } else if (code < 0x800u) {
                        out->push_back(static_cast<char>(0xC0u | (code >> 6)));
                        out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                    } else {
                        out->push_back(static_cast<char>(0xE0u | (code >> 12)));
                        out->push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
                        out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                    }
                    break;
                }
                default: return false;
            }
            continue;
        }
        if (c == '"') {
            ++(*index);
            return true;
        }
        out->push_back(c);
        ++(*index);
    }
    return false;
}

bool read_compound(const std::string& text, std::size_t* index, std::string* out) {
    const char open = text[*index];
    const char close = open == '{' ? '}' : ']';
    int depth = 0;
    const std::size_t start = *index;
    while (*index < text.size()) {
        const char c = text[*index];
        if (c == '"') {
            std::string ignored;
            if (!read_string(text, index, &ignored)) {
                return false;
            }
            continue;
        }
        if (c == open) {
            ++depth;
        } else if (c == close) {
            --depth;
            if (depth == 0) {
                ++(*index);
                *out = text.substr(start, *index - start);
                return true;
            }
        }
        ++(*index);
    }
    return false;
}

}  // namespace

bool parse_object(const std::string& text, std::vector<Member>* out, std::string* error) {
    out->clear();
    std::size_t index = 0;
    skip_space(text, &index);
    if (index >= text.size() || text[index] != '{') {
        if (error != nullptr) { *error = "metadata is not a JSON object"; }
        return false;
    }
    ++index;
    for (;;) {
        skip_space(text, &index);
        if (index < text.size() && text[index] == '}') {
            ++index;
            break;
        }
        if (index >= text.size() || text[index] == ',') {
            if (index >= text.size()) {
                if (error != nullptr) { *error = "unterminated JSON object"; }
                return false;
            }
            ++index;
            continue;
        }
        Member member;
        if (!read_string(text, &index, &member.key)) {
            if (error != nullptr) { *error = "expected a JSON key"; }
            return false;
        }
        skip_space(text, &index);
        if (index >= text.size() || text[index] != ':') {
            if (error != nullptr) { *error = "expected ':' after JSON key " + member.key; }
            return false;
        }
        ++index;
        skip_space(text, &index);
        if (index >= text.size()) {
            if (error != nullptr) { *error = "missing JSON value for " + member.key; }
            return false;
        }
        if (text[index] == '"') {
            member.is_string = true;
            if (!read_string(text, &index, &member.raw)) {
                if (error != nullptr) { *error = "bad JSON string for " + member.key; }
                return false;
            }
        } else if (text[index] == '{' || text[index] == '[') {
            if (!read_compound(text, &index, &member.raw)) {
                if (error != nullptr) { *error = "bad JSON container for " + member.key; }
                return false;
            }
        } else {
            const std::size_t start = index;
            while (index < text.size() && text[index] != ',' && text[index] != '}') {
                ++index;
            }
            member.raw = text.substr(start, index - start);
            while (!member.raw.empty() &&
                   (member.raw.back() == ' ' || member.raw.back() == '\n' ||
                    member.raw.back() == '\r' || member.raw.back() == '\t')) {
                member.raw.pop_back();
            }
        }
        for (const Member& existing : *out) {
            if (existing.key == member.key) {
                if (error != nullptr) { *error = "duplicate JSON key " + member.key; }
                return false;
            }
        }
        out->push_back(std::move(member));
    }
    return true;
}

const Member* find(const std::vector<Member>& members, const std::string& key) {
    for (const Member& member : members) {
        if (member.key == key) {
            return &member;
        }
    }
    return nullptr;
}

bool as_string(const Member& member, std::string* out) {
    if (!member.is_string || out == nullptr) {
        return false;
    }
    *out = member.raw;
    return true;
}

bool as_number(const Member& member, double* out) {
    if (member.is_string || out == nullptr) {
        return false;
    }
    char* end = nullptr;
    const double value = std::strtod(member.raw.c_str(), &end);
    if (end == member.raw.c_str() || !std::isfinite(value)) {
        return false;
    }
    *out = value;
    return true;
}

bool as_integer(const Member& member, long long* out) {
    double value = 0.0;
    if (!as_number(member, &value) || out == nullptr) {
        return false;
    }
    if (value != std::floor(value)) {
        return false;
    }
    *out = static_cast<long long>(value);
    return true;
}

}  // namespace joc::json
