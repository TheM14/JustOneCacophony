
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace joc::json {

struct Member {
    std::string key;
    std::string raw;
    bool is_string = false;
};

// Parses a top-level JSON object.  Rejects non-objects and duplicate keys.
bool parse_object(const std::string& text, std::vector<Member>* out, std::string* error);

// One member of an object being written: `key` is escaped here, `value` is
// already JSON text (a quoted string, a number, a literal).
struct Field {
    std::string key;
    std::string value;
};

// Escapes `text` as a JSON string literal, quotes included.  Matches
// json.dumps(..., ensure_ascii=False): the named escapes for `"` and `\` and
// for U+0008/U+0009/U+000A/U+000C/U+000D, `\u00XX` for the other control
// characters, and every other byte - UTF-8 included - passed through.
std::string quote(const std::string& text);

// Renders a flat object the way json.dumps(members, ensure_ascii=False,
// indent=2) does: two-space indentation, one member per line, the members in
// the order given, and no trailing newline.
std::string pretty_object(const std::vector<Field>& fields);

const Member* find(const std::vector<Member>& members, const std::string& key);

bool as_string(const Member& member, std::string* out);
bool as_number(const Member& member, double* out);
bool as_integer(const Member& member, long long* out);

}  // namespace joc::json
