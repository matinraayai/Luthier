//===-- MIRToIRTranslator.h -------------------------------------*- C++ -*-===//
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
/// \file MIRToIRTranslator.h
/// Describes a set of APIs used to translate machine functions and
/// individual machine instructions to LLVM IR.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_MIR_TO_IR_TRANSLATOR_H
#define LUTHIER_TOOLING_MIR_TO_IR_TRANSLATOR_H
#include "luthier/Common/DenseMapInfo.h"
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>

namespace llvm {

class SIRegisterInfo;
class MachineInstr;
class MachineFunction;
class MachineBasicBlock;
class TargetRegisterInfo;
class MachineOperand;
class Error;

} // namespace llvm

namespace luthier {

/// \brief A utility class that lazily materializes LLVM Values from
/// physical register operands for use with the IR translator functions.
///
/// \details The tracker stores values keyed by the *exact* register that
/// was written in each machine basic block by the translated IR instructions.
/// When a different-sized overlapping register is requested (sub-register
/// or super-register), the tracker extracts or merges values on demand rather
/// than eagerly splitting every write into register-unit granularity. Registers
/// that overlap but are not strictly super or sub-registers (rare) will be
/// broken down to reg units and then processed via the sub and super reg logic.
///
/// On a write to register R, all overlapping entries (sub- and super-regs)
/// are invalidated so that subsequent reads through a different alias will
/// lazily recompute the value from R's entry. If a register being written only
/// partially overlaps with a tracked value, only the overlapped part is
/// invalidated.
///
/// If being used to translate an entire machine function, the tracker uses
/// the function's prototype and calling convention to figure out the initial
/// values for each register in the entry basic block (a process we also
/// referred to as "seeding"). The function prototype must meet the following
/// requirements:
/// - Kernels and other entry functions shouldn't have any arguments in their
/// prototype. Access to the kernel argument pointer, as well as other SGPR and
/// VGPR registers initialized by the GPU will be instead materialized by
/// emitting both LLVM and Luthier intrinsics at the beginning of the entry
/// basic block, all before the translation process starts. For example,
/// llvm.amdgcn.kernarg.segment.ptr() is used to initialize the SGPR pair's
/// value holding the kernel argument buffer's address.
/// - Any other non-entry functions must have all read/writable GPRs passed
/// as an argument to the function. This includes VGPRs, SGPRs, AGPRs,
/// Exec mask, SCC, VCC, M0, MODE, and FLAT_SCRATCH (on targets that support
/// writing to it). EXECZ and VCCZ are exceptions, as they are indirectly
/// calculated based on the values of EXEC and VCC, respectively. PC is not
/// modeled by the IR translator. Read access to privileged or read-only
/// registers are done via LLVM and Luthier intrinsics.
///
/// As the function prototype is very hard to change after a machine function is
/// assigned ot it, it is the caller's responsibility to have counted all
/// possible GPRs used by the function and adding them to its prototype. To do
/// so, caller must take into account the value of the kernel descriptor or the
/// AMD kernel code's fields to ensure a non-entry function's prototype is
/// properly configured. If a non-entry function is invoked with dynamically
/// allocated VGPRs, it must include the maximum possible number of VGPRs
/// available to each wave.
///
/// The process of seeding can also be done manually when translating individual
/// basic blocks or instructions
class MBBOperandTracker {

  using MCRegValueMap = llvm::DenseMap<llvm::MCRegister, llvm::Value *>;

  using BBValueMap =
      llvm::DenseMap<std::reference_wrapper<const llvm::MachineBasicBlock>,
                     MCRegValueMap>;

  const llvm::MachineFunction &MF;

  BBValueMap VM{};

  MCRegValueMap &getMap(const llvm::MachineBasicBlock &MBB) {
    auto It = VM.find(MBB);
    assert(It != VM.end() && "No value map for MBB");
    return It->second;
  }

  /// Handle overlapping entries in \p Map when \p Reg is about to be written.
  ///
  /// For each stored register that overlaps with \p Reg:
  ///  - If StoredReg is a *sub-register* of Reg (Reg fully covers it): erase.
  ///  - If StoredReg is a *super-register* of Reg (Reg partially overwrites
  ///    it): bitcast to vector, extractelement the non-overlapping sub-regs,
  ///    and preserve them as new entries.
  ///  - Otherwise (rare partial overlap): conservative erase.
  void invalidateOverlaps(MCRegValueMap &Map, llvm::MCRegister Reg,
                          llvm::IRBuilder<> &Builder);

  /// Search \p Map for a stored super-register that fully contains \p Reg.
  /// If found, bitcast to vector and extractelement the requested sub-reg.
  llvm::Value *tryExtractFromSuperReg(MCRegValueMap &Map, llvm::MCRegister Reg,
                                      llvm::IRBuilder<> &Builder);

  /// Try to compose \p Reg from stored sub-register entries in \p Map.
  /// Builds a vector via insertelement for each sub-reg, then bitcasts to
  /// the target integer type.
  llvm::Value *tryComposeFromSubRegs(MCRegValueMap &Map, llvm::MCRegister Reg,
                                     llvm::IRBuilder<> &Builder);

  llvm::Value *tryComposeFromOverlappingRegs(const llvm::MachineBasicBlock &MBB,
                                             MCRegValueMap &Map,
                                             llvm::MCRegister Reg,
                                             llvm::IRBuilder<> &Builder);

  /// If \p Reg is not a VGPR (i.e. SGPR, SCC, etc.) and \p V is an
  /// Instruction, attach \c !amdgpu.uniform metadata to mark it as uniform.
  void annotateUniformIfNeeded(llvm::Value *V, llvm::MCRegister Reg);

  /// Materialize the value of \p Reg in \p MBB.  Searches for an exact
  /// match first, then a containing super-register, then composable
  /// sub-registers, and finally falls back to predecessor PHI / undef.
  llvm::Value &materializeReg(const llvm::MachineBasicBlock &MBB,
                              llvm::MCRegister Reg);

public:
  explicit MBBOperandTracker(const llvm::MachineFunction &MF);

  const llvm::SIInstrInfo &getTII() const {
    return *MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  }

  const llvm::SIRegisterInfo &getTRI() const {
    return *MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
  }

  llvm::Value &getOperandAsValue(const llvm::MachineOperand &Op);

  /// Seed the value of \p Reg in \p MBB without invalidating overlaps.
  /// Used to initialize pre-loaded kernel entry registers before any
  /// instructions are raised.
  void seedRegValue(const llvm::MachineBasicBlock &MBB, llvm::MCRegister Reg,
                    llvm::Value *Val) {
    getMap(MBB)[Reg] = Val;
  }

  llvm::Value &getRegisterOperand(const llvm::MachineInstr &MI,
                                  llvm::MCRegister Reg);

  llvm::Value &getRegisterOperand(const llvm::MachineBasicBlock &MBB,
                                  llvm::MCRegister Reg);

  void setRegOperandValue(const llvm::MachineInstr &MI, llvm::MCRegister Reg,
                          llvm::Value *Val);

  void setRegOperandValue(const llvm::MachineOperand &Op, llvm::Value *Val);
};

llvm::Error translateMachineFunctionToIR(llvm::MachineFunction &MF);

/// Maps physical registers to IR values — used as input/output for
/// \c translateSingleInstr.
using PhysRegValueMap = llvm::DenseMap<llvm::MCRegister, llvm::Value *>;

/// Translate a single MachineInstr into LLVM IR using the TableGen-generated
/// semantic definitions.
///
/// \param MI           The instruction to translate.
/// \param Builder      IRBuilder positioned at the insertion point.
/// \param InputRegs    Pre-populated map of physical register → IR values for
///                     the instruction's input operands.
/// \param[out] OutputRegs  Filled with physical register → IR values for the
///                         instruction's output operands (explicit defs +
///                         implicit defs like SCC, VCC).
/// \returns Error on failure (e.g. unmodeled opcode).
llvm::Error translateSingleInstr(const llvm::MachineInstr &MI,
                                 llvm::IRBuilderBase &Builder,
                                 const PhysRegValueMap &InputRegs,
                                 PhysRegValueMap &OutputRegs);

} // namespace luthier

#endif