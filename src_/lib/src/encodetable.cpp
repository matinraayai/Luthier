// #include "inst.h"
// #include <map>
// #include <memory>
// #include <string>

#include "assembler.h"
#include <memory>
#include <string>

std::map<std::string, InstType> EncodeTable;

void Assembler::initEncodeTable()
{
    // SOP1 instrucions
    EncodeTable["s_getpc_b64"] = {
        "s_getpc_b64", 28, FormatTable[SOP1], 0, ExeUnitScalar, 64, 0, 0, 0, 0
        };
    EncodeTable["s_setpc_b64"] = {
        "s_setpc_b64", 29, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0
        };
    EncodeTable["s_swappc_b64"] = {
        "s_swappc_b64", 30, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0
        };

    // VOP1 instructions
    EncodeTable["v_mov_b32"] = {
        "v_mov_b32", 1, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 0, 0, 0
        };
        
    // SMEM instructions
    EncodeTable["s_load_dword"] = {
        "s_load_dword", 0, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32, 32, 0, 0
        };
    EncodeTable["s_load_dwordx2"] = {
        "s_load_dwordx2", 1, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32, 32, 32, 0
        };
    EncodeTable["s_load_dwordx4"] = {
        "s_load_dwordx4", 2, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32, 32, 0, 0
                };
    EncodeTable["s_load_dwordx8"] = {
        "s_load_dwordx8", 3, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32, 32, 0, 0
        };
    EncodeTable["s_load_dwordx16"] = {
        "s_load_dwordx16", 4, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32, 32, 0, 0
        };


    // SOP2 instructions
    EncodeTable["s_add_u32"] = {
        "s_add_u32", 0, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0
        };
    EncodeTable["s_addc_u32"] = {
        "s_addc_u32", 4, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0
        };
}

