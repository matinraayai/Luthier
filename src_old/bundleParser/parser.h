#ifndef PARSER_H
#define PARSER_H
#include "elf.hpp"
void processBundle(std::string filename);
uint64_t processBundle(char *data);
void registerFatBinary(std::string filename, std::string target);
elfio::File extractFromBundle(std::string filename);
#endif
