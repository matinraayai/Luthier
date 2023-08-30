#include "instr.hpp"
#include "error.h"
#include "hsa_intercept.hpp"



luthier::Operand::Operand(std::string op_str, OperandType type, int code,
        float floatValue, long int intValue, uint32_t literalConstant) {
    operand         = op_str;
    operandType     = type;
    operandCode     = code;
    operandFloatVal = floatValue;
    operandIntVal   = intValue;
    operandConst    = literalConstant;
}

std::string luthier::Operand::getOperand() const { return operand; }
luthier::OperandType luthier::Operand::getOperandType() const { return operandType; }
int luthier::Operand::getOperandCode() const { return operandCode; }
float luthier::Operand::getOperandFloatValue() const { return operandFloatVal; }
long int luthier::Operand::getOperandIntValue() const { return operandIntVal; }
uint32_t luthier::Operand::getOperandLiteralConstant() const { return operandConst; }

std::ostream& operator<<(std::ostream& os, const luthier::Operand& op) {
    os << op.getOperand();
    switch (op.getOperandCode()) {
        case luthier::OperandType::RegOperand:
        case luthier::OperandType::SpecialRegOperand:
            os << "\t(Register Encoding: " << op.getOperandCode() << ")" << std::endl;
            break;
        case luthier::OperandType::WaitCounter:
            os << "\t(Counter)" << std::endl;
            break;
        case luthier::OperandType::ImmOperand:
            os << "\t(Immediate)" << std::endl;
            break;
        case luthier::OperandType::LiteralConstant:
            os << "\t(Literal Constant)" << std::endl;
            break;
        case luthier::OperandType::SpecialOperand:
            os << "\t(This one means I didn't know how to encode it)" << std::endl;
            break;
        default:
            os << "\t(Invalid Operand)" << std::endl;
    }
    return os;
}

luthier_address_t luthier::Instr::getHostAddress() {
//    if (kd_ != nullptr && hostAddress_ == luthier_address_t{}) {
//        LUTHIER_HSA_CHECK(
//            LuthierHsaInterceptor::Instance().getHsaVenAmdLoaderTable().
//            hsa_ven_amd_loader_query_host_address(
//                reinterpret_cast<const void*>(deviceAddress_),
//                reinterpret_cast<const void**>(&hostAddress_)
//            )
//        );
//    }
    return hostAddress_;
}
hsa_executable_t luthier::Instr::getExecutable() {
    return executable_;
}
const kernel_descriptor_t *luthier::Instr::getKernelDescriptor() {
    const kernel_descriptor_t *kernelDescriptor{nullptr};

    auto coreApi = HsaInterceptor::Instance().getSavedHsaTables().core;
    LUTHIER_HSA_CHECK(coreApi.hsa_executable_symbol_get_info_fn(executableSymbol_,
                                                              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                              reinterpret_cast<luthier_address_t*>(&kernelDescriptor)));
    return kernelDescriptor;
}

luthier::Instr::Instr(std::string instStr, hsa_agent_t agent,
                    hsa_executable_t executable,
                    hsa_executable_symbol_t symbol,
                    luthier_address_t DeviceAccessibleInstrAddress,
                    size_t instrSize): executable_(executable), deviceAddress_(DeviceAccessibleInstrAddress),
                                       instStr_(std::move(instStr)), size_(instrSize),
                                       agent_(agent),
                                       executableSymbol_(symbol) {
    HsaInterceptor::Instance().getHsaVenAmdLoaderTable().hsa_ven_amd_loader_query_host_address(
                                                reinterpret_cast<const void*>(DeviceAccessibleInstrAddress),
                                                reinterpret_cast<const void**>(&hostAddress_)
                                                );
    GetOperandsFromString();
};

void luthier::Instr::GetOperandsFromString() {
    std::string instrstr_cpy(instStr_);
    std::string delim = " ";
    std::vector<std::string> op_str_vec;
    
    size_t i = instrstr_cpy.find(delim);
    while (i != std::string::npos) {
        i = instrstr_cpy.find(delim);
        op_str_vec.push_back(instrstr_cpy.substr(0, i));
        instrstr_cpy.erase(0, i + 1);
    }

    if (op_str_vec.size() > 0) {
        op_str_vec.erase(op_str_vec.begin());
        for (int i = 0; i < op_str_vec.size(); i++) {
            if (op_str_vec.at(i).find(",") != std::string::npos) {
                op_str_vec.at(i).erase(op_str_vec.at(i).find(","), 1);
            }
            if (op_str_vec.at(i).find(" ") != std::string::npos) {
                op_str_vec.at(i).erase(op_str_vec.at(i).find(" "), 1);
            }
            operands.push_back(EncodeOperand(op_str_vec.at(i)));
        }
    }
}

// IMPORTANT NOTE -- I use -1 in the operand code when I'm not sure what the ISA expects for encoding
// This is fine for now while I'm developing the instruction class because this isn't being passed
// into anything like an assembler. However, I hope to eventually have correct values for everything
// to prevent any problems down the road.
luthier::Operand luthier::Instr::EncodeOperand(std::string op) {
    std::string opstr(op);
    int code;

    // Wait Counter Operands
    // TO-DO: add export/mem-write-data count
    if (op.find("vmcnt")   != std::string::npos ||
        op.find("lgkmcnt") != std::string::npos) {
        code = -1;
        op.erase(0, op.find("(") + 1);
        op.erase(op.find(")"), op.length());
        return luthier::Operand(opstr, WaitCounter, code, -1, stoi(op, 0, 10), 0);
    }

    // Comp Register Operands
    if (!op.compare("vcc")) {
        code = 251;
        return luthier::Operand(opstr, SpecialRegOperand, code, -1, -1, 0);
    }
    // else if (!op.compare("exec")) { // Need to test this one
    //     code = 252;
    //     return {op, SpecialRegOperand, code, -1, -1, 0};
    // }
    else if (!op.compare("scc")) {
        code = 253;
        return luthier::Operand(opstr, SpecialRegOperand, code, -1, -1, 0);
    }

    // Register Operands (gpr)
    size_t sgpr_label_at = op.find("s");
    size_t vgpr_label_at = op.find("v");
    if (sgpr_label_at != std::string::npos) {
        op.erase(sgpr_label_at, 1);
        if (op.find("[") != std::string::npos) {
            op.erase(op.find("["), 1);
            op.erase(op.find(":"), op.length());
        }
        code = stoi(op, 0, 10);
        return luthier::Operand(opstr, RegOperand, code, -1, -1, 0);
    } else if (vgpr_label_at != std::string::npos) {
        op.erase(vgpr_label_at, 1);
        if (op.find("[") != std::string::npos) {
            op.erase(op.find("["), 1);
            op.erase(op.find(":"), op.length());
        }
        code = stoi(op, 0, 10) + 256;
        return luthier::Operand(opstr, RegOperand, code, -1, -1, 0);
    }

    // Inline Constants     
    // This probably doesn't count as an inline constant. The ISA literally says NOTHING about this operand
    // It also ONLY appears with Flat, Scratch, and Global instructions to tell you that the instruction is
    // using an offset
    if (!op.compare("off")) {
        return luthier::Operand(opstr, SpecialOperand, -1, -1, -1, 0);
    }

    // Assume all other operands are Immediates
    // Imm in Hex:
    if (op.find("0x") != std::string::npos) {
        op.erase(0, 2);
        code = -1;
        return luthier::Operand(opstr, ImmOperand, code, -1, stoi(op, 0, 16), 0);
    } 
    // Imm in Decimal
    else {
        code = -1;
        return luthier::Operand(opstr, ImmOperand, code, -1, stoi(op, 0, 10), 0);
    }
}

int luthier::Instr::getNumOperands() { return operands.size(); }

int luthier::Instr::getNumRegsUsed() {
    std::vector<luthier::Operand> reg_ops = getRegOperands();
    return int(reg_ops.size());
}

luthier::Operand luthier::Instr::getOperand(int op_num) { return operands.at(op_num); }

std::vector<luthier::Operand> luthier::Instr::getAllOperands() { return operands; }

std::vector<luthier::Operand> luthier::Instr::getImmOperands() {
    std::vector<luthier::Operand> imm_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).getOperandType() == ImmOperand) {
            imm_ops.push_back(operands.at(i));
        }
    }
    return imm_ops;
}

std::vector<luthier::Operand> luthier::Instr::getRegOperands() {
    std::vector<luthier::Operand> reg_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).getOperandType() == RegOperand || 
            operands.at(i).getOperandType() == SpecialRegOperand) {
            reg_ops.push_back(operands.at(i));
        }
    }
    return reg_ops;
}


// Edit operand instructions - implement later, as needed
// bool luthier::Instr::changeRegNum(int reg_op_num, int new_reg_code){
//     luthier::operand op = operands.at(reg_op_num);
    
//     if (op.type != RegOperand || op.type != SpecialRegOperand){
//         return false;
//     }

//     op.code = new_reg_code;

//     // If we want to change the operand string:
//     // if (new_reg_code < 102) {
        
//     // }
//     operands.at(reg_op_num) = op;
//     return true;
// }

// bool luthier::Instr::changeImmVal(int imm_op_num, int new_imm_val) {
//     luthier::operand op = operands.at(reg_op_num);
    
//     if (op.type != ImmOperand){
//         return false;
//     }

//     op.intValue = new_imm_val;
//     return true;
// }