#ifndef QUEST_EXCEPT
#define QUEST_EXCEPT

#include <stdexcept>

// Our custom exception for any QuEST errors.
class quest_exception : public std::runtime_error {
public:
    quest_exception(std::string message) : std::runtime_error(message) {}
};

#endif //QUEST_EXCEPT
