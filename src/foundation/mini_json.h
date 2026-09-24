
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

const Member* find(const std::vector<Member>& members, const std::string& key);

bool as_string(const Member& member, std::string* out);
bool as_number(const Member& member, double* out);
bool as_integer(const Member& member, long long* out);

}  // namespace joc::json
