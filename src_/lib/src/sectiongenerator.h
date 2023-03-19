#ifndef SECTIONGENERATOR_H
#define SECTIONGENERATOR_H
#include "elf.hpp"
void getDynsymSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec);
#endif