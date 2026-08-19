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
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/PrototypeCallGraph.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/CommandLine.h>
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

/// \return the hook function named by \c -luthier-mock-hook-name in the
/// instrumentation module of \p P, or \c nullptr if the IModule under
/// instrumentation does not define it.
llvm::Function *getHook(Prototype &P) {
  return P.getInstrumentationModule().getFunction(MockHookName);
}

/// \return the \c PreservedAnalyses a mock pass reports after it has created
/// injected payloads.
///
/// Creating a payload adds a function to the IModule, declares an extern for
/// it in the target module, and emits a \c PATCHPOINT marker before the
/// target MI, so no prototype-level analysis survives. The inner
/// analysis-manager proxies must still be preserved: their invalidation
/// hooks clear the *entire* inner manager for both of the prototype's
/// modules (see \c Prototype.cpp), which would throw away the cached
/// \c MachineFunctionAnalysis results holding the target MIR that
/// \c CodeDiscoveryPass lifted, along with the IModule's
/// \c MachineModuleInfo, both of which the downstream pipeline still needs.
llvm::PreservedAnalyses payloadsCreatedPA() {
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.abandon<InjectedPayloadAndInstPointAnalysis>();
  PA.abandon<PrototypeCallGraphAnalysis>();
  return PA;
}

} // namespace

llvm::PreservedAnalyses
MockInjectAtFunctionEntryPass::run(Prototype &P,
                                  PrototypeAnalysisManager &PAM) {
  llvm::Function *Hook = getHook(P);
  if (!Hook)
    return llvm::PreservedAnalyses::all();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &TargetMF) {
    if (TargetMF.empty())
      return;
    for (llvm::MachineInstr &MI : TargetMF.front()) {
      llvm::consumeError(
          P.createInjectedPayload(
               *Hook, MI,
               PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
                   .getManager(),
               {})
              .takeError());
      break;
    }
  });
  return payloadsCreatedPA();
}

llvm::PreservedAnalyses
MockInjectAtMBBEntryPass::run(Prototype &P, PrototypeAnalysisManager &PAM) {
  llvm::Function *Hook = getHook(P);
  if (!Hook)
    return llvm::PreservedAnalyses::all();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &TargetMF) {
    for (llvm::MachineBasicBlock &MBB : TargetMF) {
      if (MBB.empty())
        continue;
      llvm::consumeError(
          P.createInjectedPayload(
               *Hook, MBB.front(),
               PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
                   .getManager(),
               {})
              .takeError());
    }
  });
  return payloadsCreatedPA();
}

llvm::PreservedAnalyses
MockInjectAtMBBTerminatorPass::run(Prototype &P,
                                  PrototypeAnalysisManager &PAM) {
  llvm::Function *Hook = getHook(P);
  if (!Hook)
    return llvm::PreservedAnalyses::all();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &TargetMF) {
    for (llvm::MachineBasicBlock &MBB : TargetMF) {
      auto It = MBB.getFirstTerminator();
      if (It == MBB.end())
        continue;
      llvm::consumeError(
          P.createInjectedPayload(
               *Hook, *It,
               PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
                   .getManager(),
               {})
              .takeError());
    }
  });
  return payloadsCreatedPA();
}

llvm::PreservedAnalyses
MockInjectAtAllVALUPass::run(Prototype &P, PrototypeAnalysisManager &PAM) {
  llvm::Function *Hook = getHook(P);
  if (!Hook)
    return llvm::PreservedAnalyses::all();
  llvm::FunctionAnalysisManager &IFAM =
      PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &TargetMF) {
    const auto *TII =
        TargetMF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
    for (llvm::MachineBasicBlock &MBB : TargetMF) {
      for (llvm::MachineInstr &MI : MBB) {
        if (TII->isVALU(MI))
          llvm::consumeError(
              P.createInjectedPayload(*Hook, MI, IFAM, {}).takeError());
      }
    }
  });
  return payloadsCreatedPA();
}

llvm::PreservedAnalyses
MockInjectAtAllScalarPass::run(Prototype &P, PrototypeAnalysisManager &PAM) {
  llvm::Function *Hook = getHook(P);
  if (!Hook)
    return llvm::PreservedAnalyses::all();
  llvm::FunctionAnalysisManager &IFAM =
      PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &TargetMF) {
    const auto *TII =
        TargetMF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
    for (llvm::MachineBasicBlock &MBB : TargetMF) {
      for (llvm::MachineInstr &MI : MBB) {
        if (TII->isSALU(MI))
          llvm::consumeError(
              P.createInjectedPayload(*Hook, MI, IFAM, {}).takeError());
      }
    }
  });
  return payloadsCreatedPA();
}

llvm::PreservedAnalyses
MockInjectAtOpcodePass::run(Prototype &P, PrototypeAnalysisManager &PAM) {
  llvm::Function *Hook = getHook(P);
  if (!Hook || MockOpcodeMnemonic.empty())
    return llvm::PreservedAnalyses::all();
  llvm::FunctionAnalysisManager &IFAM =
      PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &TargetMF) {
    const auto *TII =
        TargetMF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
    for (llvm::MachineBasicBlock &MBB : TargetMF) {
      for (llvm::MachineInstr &MI : MBB) {
        llvm::StringRef Mnemonic = TII->getName(MI.getOpcode());
        if (Mnemonic.contains(MockOpcodeMnemonic.getValue()))
          llvm::consumeError(
              P.createInjectedPayload(*Hook, MI, IFAM, {}).takeError());
      }
    }
  });
  return payloadsCreatedPA();
}

llvm::PreservedAnalyses
MockInjectAtAllVGPRDefsWithRegArgPass::run(Prototype &P,
                                           PrototypeAnalysisManager &PAM) {
  llvm::Function *Hook = getHook(P);
  if (!Hook)
    return llvm::PreservedAnalyses::all();
  llvm::Type *I32 =
      llvm::Type::getInt32Ty(P.getInstrumentationModule().getContext());
  llvm::FunctionAnalysisManager &IFAM =
      PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &TargetMF) {
    const auto &MRI = TargetMF.getRegInfo();
    const auto &TRI = *static_cast<const llvm::SIRegisterInfo *>(
        TargetMF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo());
    for (llvm::MachineBasicBlock &MBB : TargetMF) {
      for (llvm::MachineInstr &MI : MBB) {
        llvm::Register VGPR = firstVGPRDef(MI, MRI, TRI);
        if (!VGPR.isValid())
          continue;
        PayloadArg Args[]{RegArg{VGPR.asMCReg(), I32}};
        llvm::consumeError(
            P.createInjectedPayload(*Hook, MI, IFAM, Args).takeError());
      }
    }
  });
  return payloadsCreatedPA();
}

} // namespace luthier::test
