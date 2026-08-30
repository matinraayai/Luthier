//===-- LoadSVA.h - Luthier LoadSVA Intrinsic -------------------*- C++ -*-===//
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
/// Luthier's <tt>loadSVA</tt> intrinsic — returns a \c uint32_t VGPR holding
/// the wave's SVA courier value. Unlike <tt>readSVA</tt> (which reads a
/// single lane out of the SVA VGPR into an SGPR), \c loadSVA emits a
/// \c WWM_COPY of the SVA VGPR source itself.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSIC_LOAD_SVA_H
#define LUTHIER_INTRINSIC_INTRINSIC_LOAD_SVA_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/Error.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
loadSVAIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &TM);

llvm::Error loadSVAMIRProcessor(
    llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    llvm::Register SVAValueSource);

} // namespace luthier

#endif
