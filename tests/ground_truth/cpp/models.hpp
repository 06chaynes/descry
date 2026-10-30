#pragma once
#include <string>

class Base {
public:
    bool save() { return validate(); }
    virtual bool validate() { return true; }
};

class User : public Base {
public:
    bool validate() override { return !name.empty(); }
    bool rename(const std::string &newName) {
        name = newName;
        return save();
    }

private:
    std::string name;
};

class Order {
public:
    int parse(const std::string &text) { return text.empty() ? 0 : text[0]; }
};
