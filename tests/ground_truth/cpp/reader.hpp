#pragma once
#include <string>

class Reader {
public:
    int parse(const std::string &text) { return static_cast<int>(text.size()); }
};

inline int load(Reader &reader, const std::string &text) { return reader.parse(text); }
