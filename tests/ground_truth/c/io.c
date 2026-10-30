#include <string.h>

#include "io.h"

static int parse(const char *text) { return (int)strlen(text); }

int load(const char *text) { return parse(text); }
