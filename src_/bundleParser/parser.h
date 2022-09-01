#ifndef PARSER_H
#define PARSER_H
#include "elf.hpp"
void processBundle(std::string filename);
elfio::File extractFromBundle(std::string filename);
#endif