#ifndef NOTFOUNDEXCEPTION_H
#define NOTFOUNDEXCEPTION_H
#include <stdexcept>
#include <string>
class NotFoundException : public std::runtime_error
{
public:
	NotFoundException(std::string msg) : std::runtime_error(msg) {}
};

#endif