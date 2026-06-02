//===-- UserArgPtr.h - Luthier user (explicit) arg access -------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// Luthier's <tt>userArgPtr</tt> intrinsic — returns the base of the tool's
/// explicit-argument region inside the Luthier-managed custom kernarg buffer
/// (i.e. <tt>USER_ARG_PTR + CustomKernargExplicitOffset</tt>). See
/// \c luthier/ToolCodeGen/CustomKernargLayout.h for the buffer layout.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSIC_USER_ARG_PTR_H
#define LUTHIER_INTRINSIC_INTRINSIC_USER_ARG_PTR_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/Error.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
userArgPtrIRProcessor(const llvm::Function &Intrinsic,
                      const llvm::CallInst &User,
                      const llvm::GCNTargetMachine &TM);

llvm::Error userArgPtrMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *Payload,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &);

} // namespace luthier

#endif
