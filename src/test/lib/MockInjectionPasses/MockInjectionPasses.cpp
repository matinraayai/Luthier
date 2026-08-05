//===-- MockInjectionPasses.cpp -------------------------------------------===//
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
#include "MockInjectionPasses.h"

#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Error.h>

using namespace luthier;
using namespace luthier::test;

namespace {

llvm::cl::OptionCategory MockOpts("Luthier Mock Injection Pass Options");

llvm::cl::opt<std::string> MockHookName{
    "luthier-mock-hook-name", llvm::cl::init("_Z11bumpCounterv"),
    llvm::cl::desc("Mangled symbol name of the hook function to call from each "
                   "injected payload."),
    llvm::cl::cat(MockOpts)};

llvm::cl::opt<std::string> MockOpcodeMnemonic{
    "luthier-mock-inject-opcode", llvm::cl::init(""),
    llvm::cl::desc(
        "For luthier-mock-inject-at-opcode: case-sensitive substring of the "
        "MI mnemonic to match. Empty matches nothing."),
    llvm::cl::cat(MockOpts)};

} // namespace

namespace luthier::test {

llvm::StringRef getMockHookNameOpt() { return MockHookName; }
llvm::StringRef getMockOpcodeMnemonicOpt() { return MockOpcodeMnemonic; }

namespace {

/// Returns the unique VGPR Register defined by \p MI, or an invalid Register
/// if there is no such def or there is more than one.
llvm::Register firstVGPRDef(const llvm::MachineInstr &MI,
                            const llvm::MachineRegisterInfo &MRI,
                            const llvm::SIRegisterInfo &TRI) {
  llvm::Register Found;
  for (const llvm::MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;
    llvm::Register Reg = MO.getReg();
    if (!Reg.isPhysical())
      continue;
    const llvm::TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
    if (RC && llvm::SIRegisterInfo::isVGPRClass(RC))
      return Reg;
  }
  return Found;
}

} // namespace

InstrumentationPreservedAnalyses
MockInjectAtFunctionEntryPass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &,
    llvm::MachineFunction &TargetMF, llvm::FunctionAnalysisManager &) {
  llvm::Function *Hook = IModule.getFunction(MockHookName);
  if (!Hook || TargetMF.empty())
    return {};
  for (llvm::MachineInstr &MI : TargetMF.front()) {
    llvm::consumeError(createInjectedPayload(*Hook, MI, {}));
    break;
  }
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

InstrumentationPreservedAnalyses
MockInjectAtMBBEntryPass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &,
    llvm::MachineFunction &TargetMF, llvm::FunctionAnalysisManager &) {
  llvm::Function *Hook = IModule.getFunction(MockHookName);
  if (!Hook)
    return {};
  for (llvm::MachineBasicBlock &MBB : TargetMF) {
    if (MBB.empty())
      continue;
    llvm::consumeError(createInjectedPayload(*Hook, MBB.front(), {}));
  }
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

InstrumentationPreservedAnalyses
MockInjectAtMBBTerminatorPass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &,
    llvm::MachineFunction &TargetMF, llvm::FunctionAnalysisManager &) {
  llvm::Function *Hook = IModule.getFunction(MockHookName);
  if (!Hook)
    return {};
  for (llvm::MachineBasicBlock &MBB : TargetMF) {
    auto It = MBB.getFirstTerminator();
    if (It == MBB.end())
      continue;
    llvm::consumeError(createInjectedPayload(*Hook, *It, {}));
  }
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

InstrumentationPreservedAnalyses
MockInjectAtAllVALUPass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &,
    llvm::MachineFunction &TargetMF, llvm::FunctionAnalysisManager &) {
  llvm::Function *Hook = IModule.getFunction(MockHookName);
  if (!Hook)
    return {};
  const auto *TII = TargetMF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  for (llvm::MachineBasicBlock &MBB : TargetMF) {
    for (llvm::MachineInstr &MI : MBB) {
      if (TII->isVALU(MI))
        llvm::consumeError(createInjectedPayload(*Hook, MI, {}));
    }
  }
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

InstrumentationPreservedAnalyses
MockInjectAtAllScalarPass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &,
    llvm::MachineFunction &TargetMF, llvm::FunctionAnalysisManager &) {
  llvm::Function *Hook = IModule.getFunction(MockHookName);
  if (!Hook)
    return {};
  const auto *TII = TargetMF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  for (llvm::MachineBasicBlock &MBB : TargetMF) {
    for (llvm::MachineInstr &MI : MBB) {
      if (TII->isSALU(MI))
        llvm::consumeError(createInjectedPayload(*Hook, MI, {}));
    }
  }
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

InstrumentationPreservedAnalyses MockInjectAtOpcodePass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &,
    llvm::MachineFunction &TargetMF, llvm::FunctionAnalysisManager &) {
  llvm::Function *Hook = IModule.getFunction(MockHookName);
  if (!Hook || MockOpcodeMnemonic.empty())
    return {};
  const auto *TII = TargetMF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  for (llvm::MachineBasicBlock &MBB : TargetMF) {
    for (llvm::MachineInstr &MI : MBB) {
      llvm::StringRef Mnemonic = TII->getName(MI.getOpcode());
      if (Mnemonic.contains(MockOpcodeMnemonic.getValue()))
        llvm::consumeError(createInjectedPayload(*Hook, MI, {}));
    }
  }
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

InstrumentationPreservedAnalyses
MockInjectAtAllVGPRDefsWithRegArgPass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &,
    llvm::MachineFunction &TargetMF, llvm::FunctionAnalysisManager &) {
  llvm::Function *Hook = IModule.getFunction(MockHookName);
  if (!Hook)
    return {};
  const auto &MRI = TargetMF.getRegInfo();
  const auto &TRI = *static_cast<const llvm::SIRegisterInfo *>(
      TargetMF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo());
  llvm::Type *I32 = llvm::Type::getInt32Ty(IModule.getContext());
  for (llvm::MachineBasicBlock &MBB : TargetMF) {
    for (llvm::MachineInstr &MI : MBB) {
      llvm::Register VGPR = firstVGPRDef(MI, MRI, TRI);
      if (!VGPR.isValid())
        continue;
      llvm::SmallVector<PayloadArg, 1> Args;
      Args.push_back(RegArg{VGPR.asMCReg(), I32});
      llvm::consumeError(createInjectedPayload(*Hook, MI, Args));
    }
  }
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

} // namespace luthier::test
