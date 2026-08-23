//===-- ReadSVA.h - Luthier ReadSVA Intrinsic  ------------------*- C++ -*-===//
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
/// Luthier's <tt>readSVA</tt> intrinsic — reads a scalar-value argument
/// (kernel-arg-derived SGPR) out of the SVA. Implementation is a thin
/// wrapper that declares the SA in the IR-stage effects map and delegates
/// the actual lowering to the IntrinsicMIRLoweringPass via
/// <tt>SVAScalarArgumentAccessor</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSIC_READ_SVA_H
#define LUTHIER_INTRINSIC_INTRINSIC_READ_SVA_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/Error.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
readSVAIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &TM);

llvm::Error readSVAMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SAAccessors);

} // namespace luthier

#endif
