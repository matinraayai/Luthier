#ifndef ENCODETABLE_H
#define ENCODETABLE_H
#include "inst.h"
#include <map>
#include <memory>
#include <string>

extern std::map<FormatType, Format> FormatTable;
extern std::map<std::string, InstType> EncodeTable;

void initEncodeTable();

#endif