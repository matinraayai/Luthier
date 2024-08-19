//===-- ReadReg.hpp - Luthier ReadReg Intrinsic  ---------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's <tt>ReadReg</tt> intrinsic, and how it should
/// be transformed from an extern function call into a set of
/// <tt>llvm::MachineInstr</tt>s.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INTRINSIC_READ_REG_HPP
#define LUTHIER_TOOLING_COMMON_INTRINSIC_READ_REG_HPP

#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/Support/Error.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
readRegIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &TM);

llvm::Error readRegMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder);

} // namespace luthier

#endif