#ifndef SECTIONGENERATOR_H
#define SECTIONGENERATOR_H
#include "elf.hpp"
void getDynsymSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec);
void getSymtabSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec);
void getShstrtabSecBinary(char *newBinary, elfio::Section *pSec);
void getStrtabSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec);
void getDynstrSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec);
void getHashSecBinary(char *newBinary, char *dynstr);
unsigned int elf_Hash(const char *name);

#endif