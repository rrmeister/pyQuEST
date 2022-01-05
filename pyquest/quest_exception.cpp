#include <string>
#include "quest_exception.h"

// Override weak symbol of the generic QuEST error handler (which just
// exits the executable) with one that throws an exception.
extern "C" void invalidQuESTInputError(const char* errMsg, const char* errFunc) {
    throw quest_exception("Error in QuEST function "
                          + std::string(errFunc) + ": " + std::string(errMsg));
}
