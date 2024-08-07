//===-- WriteReg.hpp - Luthier WriteReg Intrinsic  ------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's <tt>WriteReg</tt> intrinsic, and how it should
/// be transformed from an extern function call into a set of
/// <tt>llvm::MachineInstr</tt>s.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INTRINSIC_WRITE_REG_HPP
#define LUTHIER_TOOLING_COMMON_INTRINSIC_WRITE_REG_HPP

#include "luthier/IntrinsicProcessor.h"
#include <llvm/Support/Error.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
writeRegIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &TM);

llvm::Error writeRegMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder);

} // namespace luthier

#endif