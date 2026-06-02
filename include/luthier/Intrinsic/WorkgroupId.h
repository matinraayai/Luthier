//===-- WorkgroupId.h - Luthier workgroup-id (blockIdx) intrinsic -*-C++-*-===//
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
/// Luthier's <tt>workgroupIdX/Y/Z</tt> intrinsics (HIP's \c blockIdx). They
/// read the \c WORKGROUP_ID_X/Y/Z scalar-value argument out of the SVA. The SVA
/// lane is filled at kernel entry by the prologue — from the preloaded
/// workgroup-id system SGPR on non-architected-SGPR targets, or from the TTMP
/// registers (TTMP9 / masked TTMP7) on architected-SGPR targets, where the
/// kernel's static 3-D-ness needed to mask TTMP7 correctly is known. The
/// intrinsic itself is therefore arch-independent (always an SVA read).
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSIC_WORKGROUP_ID_H
#define LUTHIER_INTRINSIC_INTRINSIC_WORKGROUP_ID_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/Error.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
workgroupIdXIRProcessor(const llvm::Function &Intrinsic,
                        const llvm::CallInst &User,
                        const llvm::GCNTargetMachine &TM);
llvm::Expected<IntrinsicIRLoweringInfo>
workgroupIdYIRProcessor(const llvm::Function &Intrinsic,
                        const llvm::CallInst &User,
                        const llvm::GCNTargetMachine &TM);
llvm::Expected<IntrinsicIRLoweringInfo>
workgroupIdZIRProcessor(const llvm::Function &Intrinsic,
                        const llvm::CallInst &User,
                        const llvm::GCNTargetMachine &TM);

llvm::Error workgroupIdXMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *Payload,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &);
llvm::Error workgroupIdYMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *Payload,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &);
llvm::Error workgroupIdZMIRProcessor(
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
