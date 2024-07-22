//===-- ReadReg.hpp - Luthier ReadReg Intrinsic  --------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's <tt>ReadReg</tt> intrinsic, and how it should
/// be transformed from an extern function call into a set of
/// <tt>llvm::MachineInstr</tt>s.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INTRINSIC_READ_REG_HPP
#define LUTHIER_TOOLING_COMMON_INTRINSIC_READ_REG_HPP

#include "IntrinsicLoweringInfo.hpp"
#include <llvm/Support/Error.h>

namespace luthier {


llvm::Expected<IntrinsicIRLoweringInfo>
readRegIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &TM);

}

#endif