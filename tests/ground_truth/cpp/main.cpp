#include "models.hpp"
#include "reader.hpp"

int main() {
    User user;
    user.rename("ann");
    Reader reader;
    return load(reader, "[1]");
}
