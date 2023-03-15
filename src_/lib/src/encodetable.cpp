#include "assembler.h"
#include <memory>
#include <string>

std::map<std::string, InstType> EncodeTable;

void Assembler::initEncodeTable()
{
  // SOP2 instructions
  EncodeTable["s_add_u32"] = {
    "s_add_u32", 0, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0
    };
  EncodeTable["s_addc_u32"] = {
    "s_addc_u32", 4, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0
    };
  EncodeTable["s_mul_i32"] = {
    "s_mul_i32", 36, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0
    };

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

  // SOPP instrucions
  EncodeTable["s_nop"] = {
    "s_nop", 0, FormatTable[SOPP], 0, ExeUnitSpecial, 0, 0, 0, 0, 0
    };
  EncodeTable["s_endpgm"] = {
    "s_endpgm", 1, FormatTable[SOPP], 0, ExeUnitSpecial, 0, 0, 0, 0, 0
    };
  EncodeTable["s_branch"] = {
    "s_branch", 2, FormatTable[SOPP], 0, ExeUnitBranch, 32, 0, 0, 0, 0
    };
}

