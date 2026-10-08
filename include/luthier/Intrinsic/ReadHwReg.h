//===-- ReadHwReg.h - Luthier readHwReg intrinsic ---------------*- C++ -*-===//
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
/// This file describes Luthier's <tt>readHwReg</tt> intrinsic, which reads a
/// hardware register (e.g. \c HW_REG_HW_ID) with \c S_GETREG_B32, and how it
/// is transformed from an extern function call into a
/// <tt>llvm::MachineInstr</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_READ_HW_REG_H
#define LUTHIER_INTRINSIC_READ_HW_REG_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/Error.h>

namespace luthier {

/// The \c S_GETREG_B32 \c simm16 operand: bits [5:0] register id, [10:6]
/// bit offset, [15:11] bit count - 1.
constexpr uint16_t hwRegEncoding(unsigned Id, unsigned Offset = 0,
                                 unsigned Size = 32) {
  return uint16_t((Id & 0x3fu) | ((Offset & 0x1fu) << 6) |
                  (((Size - 1) & 0x1fu) << 11));
}

/// \c HW_REG_HW_ID on GFX9: wave, SIMD, CU, shader array and engine of the
/// executing wave.
inline constexpr uint16_t HwRegHwIdGfx9 = hwRegEncoding(4);

/// \c HW_REG_HW_ID1 on GFX10+ (RDNA): wave, SIMD, WGP, shader array and
/// engine of the executing wave.
inline constexpr uint16_t HwRegHwId1Gfx10 = hwRegEncoding(23);

llvm::Expected<IntrinsicIRLoweringInfo>
readHwRegIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                     const llvm::GCNTargetMachine &TM);

llvm::Error readHwRegMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder);

} // namespace luthier

#endif
