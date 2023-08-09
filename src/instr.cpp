#include "instr.hpp"
#include "error.h"
#include "hsa_intercept.hpp"

luthier_address_t luthier::Instr::getHostAddress() const {
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
kernel_descriptor_t *luthier::Instr::getKernelDescriptor() {
    kernel_descriptor_t *kernelDescriptor{nullptr};

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

void sibir::Instr::GetOperandsFromString() {
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
sibir::operand sibir::Instr::EncodeOperand(std::string op) {
    std::string opstr(op);
    int code;

    // Wait Counter Operands
    if (op.find("vmcnt")   != std::string::npos ||
        op.find("lgkmcnt") != std::string::npos
        // TO-DO: add export/mem-write-data count
        ) {
        code = -1;
        op.erase(0, op.find("(") + 1);
        op.erase(op.find(")"), op.length());

        return {opstr, WaitCounter, code, -1, stoi(op, 0, 10), 0};
        // return {opstr, WaitCounter, code, -1, -1, 0};
    } 

    // Comp Register Operands
    if (!op.compare("vcc")) {
        code = 251;
        // return {opstr, SpecialRegOperand, code, -1, -1, 0};
        return {opstr, RegOperand, code, -1, -1, 0};
    } 
    // else if (!op.compare("exec")) {
    //     code = 252;
    //     return {op, SpecialRegOperand, code, -1, -1, 0};
    //    return {op, RegOperand, code, -1, -1, 0};
    // }
    else if (!op.compare("scc")) {
        code = 253;
        // return {op, SpecialRegOperand, code, -1, -1, 0};
        return {op, RegOperand, code, -1, -1, 0};
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
        return {opstr, RegOperand, code, -1, -1, 0};
    } else if (vgpr_label_at != std::string::npos) {
        op.erase(vgpr_label_at, 1);
        if (op.find("[") != std::string::npos) {
            op.erase(op.find("["), 1);
            op.erase(op.find(":"), op.length());
        }
        code = stoi(op, 0, 10) + 256;
        return {opstr, RegOperand, code, -1, -1, 0};
    }   


    // // Inline Constants 
    // else {
    //     code = -1;
    //     type = LiteralConstant;
    //     if (op.compare("off"))  // This operand is used for global mem instructions
    //         val = stoi(op, 0, 10);
    //     return {op, code, type, val};
    // }
    // return {opstr, InvalidOperandType, -1, -1, -1, 0};
    if (!op.compare("off")) {
        return {opstr, SpecialOperand, -1, -1, -1, 0};
    }

    // Assume all other operands are Immediates
    if (op.find("0x") != std::string::npos) {   // Imm in Hex 
        op.erase(0, 2);
        code = -1;
        return {opstr, ImmOperand, code, -1, stoi(op, 0, 16), 0};
    } else {    // Imm in Decimal
        code = -1;
        return {opstr, ImmOperand, code, -1, stoi(op, 0, 10), 0};
    }
}

int sibir::Instr::getNumOperands() { return operands.size(); }

sibir::operand sibir::Instr::getOperand(int num) { return operands.at(num); }

sibir :: OperandType 
sibir::Instr::getOperandType(int num) { return operands.at(num).type; }

std::vector<sibir::operand> sibir::Instr::getAllOperands() { return operands; }

std::vector<sibir::operand> sibir::Instr::getImmOperands() {
    std::vector<sibir::operand> imm_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).type == ImmOperand) {
            imm_ops.push_back(operands.at(i));
        }
    }
    return imm_ops;
}

std::vector<sibir::operand> sibir::Instr::getRegOperands() {
    std::vector<sibir::operand> reg_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).type == RegOperand) { // || 
            // operands.at(i).type == SpecialRegOperand) {
            reg_ops.push_back(operands.at(i));
        }
    }
    return reg_ops;
}

