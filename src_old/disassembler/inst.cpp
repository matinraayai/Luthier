#include "inst.h"

std::map<FormatType, Format> FormatTable{
    {SOP1, {"sop1", 0xBE800000, 0xFF800000, 4, 8, 15}},
    {SOPC, {"sopc", 0xBF000000, 0xFF800000, 4, 16, 22}},
    {SOPP, {"sopp", 0xBF800000, 0xFF800000, 4, 16, 22}},
    {VOP1, {"vop1", 0x7E000000, 0xFE000000, 4, 9, 16}},
    {VOPC, {"vopc", 0x7C000000, 0xFE000000, 4, 17, 24}},
    {SMEM, {"smem", 0xC0000000, 0xFC000000, 8, 18, 25}},
    {VOP3a, {"vop3a", 0xD0000000, 0xFC000000, 8, 16, 25}},
    {VOP3b, {"vop3b", 0xD0000000, 0xFC000000, 8, 16, 25}},
    {VINTRP, {"vintrp", 0xC8000000, 0xFC000000, 4, 16, 17}},
    {DS, {"ds", 0xD8000000, 0xFC000000, 8, 17, 24}},
    {MUBUF, {"mubuf", 0xE0000000, 0xFC000000, 8, 18, 24}},
    {MTBUF, {"mtbuf", 0xE8000000, 0xFC000000, 8, 15, 18}},
    {MIMG, {"mimg", 0xF0000000, 0xFC000000, 8, 18, 24}},
    {FLAT, {"flat", 0xDC000000, 0xFC000000, 8, 18, 24}},
    {SOPK, {"sopk", 0xB0000000, 0xF0000000, 4, 23, 27}},
    {SOP2, {"sop2", 0x80000000, 0xC0000000, 4, 23, 29}},
    {VOP2, {"vop2", 0x00000000, 0x80000000, 4, 25, 30}},
    {VOP3P, {"vop3p", 0xCC000000, 0xFC000000, 8, 16, 22}},
};