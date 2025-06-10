//===-- ImplicitArgPtr.hpp - Luthier implicit arg access  -----------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This file describes Luthier's <tt>ImplicitArgPtr</tt> intrinsic, and how it
/// should be transformed from an extern function call into a set of
/// <tt>llvm::MachineInstr</tt>s.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_INTRINSICS_IMPLICIT_ARG_PTR_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_INTRINSICS_IMPLICIT_ARG_PTR_H
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/Error.h>
#include <luthier/Instrumentation/IntrinsicProcessor.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
implicitArgPtrIRProcessor(const llvm::Function &Intrinsic,
                          const llvm::CallInst &User,
                          const llvm::GCNTargetMachine &TM);

llvm::Error implicitArgPtrMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const std::function<llvm::Register(KernelArgumentType)> &KernArgAccessor,
    const llvm::MachineFunction &MF,
    const std::function<llvm::Register(llvm::MCRegister)> &PhysRegAccessor,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &PhysRegsToBeOverwritten);

} // namespace luthier

#endif