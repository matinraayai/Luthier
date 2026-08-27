//===-- StateValueArraySpecs.cpp ------------------------------------------===//
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
/// Implements the state value array specs and its Prototype-level analysis.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include <AMDGPU.h>
#include <AMDGPUTargetMachine.h>
#include <GCNSubtarget.h>
#include <MCTargetDesc/AMDGPUMCTargetDesc.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace luthier {

llvm::SmallVector<uint8_t, 4>
StateValueArraySpecs::findLowestFreeLanes(unsigned NumLanes,
                                          unsigned WaveSize) const {
  // Lane occupancy:
  //   0       — StackPointerRegSpillLane (SGPR0 of PRIVATE_SEGMENT_BUFFER)
  //   1       — FramePointerRegSSpillLane (SGPR1)
  //   2       — StackPointerStoreLane (instrumentation SGPR32)
  //   3..N-1  — BufferRsrcOrScratchSpillLane region (FS = 2 lanes, buffer
  //             rsrc = 4 lanes, architected-FS = 0 lanes)
  //   …       — Each ScalarArguments[SA] entry holds 1, 2, or 4 contiguous
  //             lanes starting at the stored base.
  llvm::BitVector Occupied(WaveSize, false);

  auto markRange = [&](unsigned Base, unsigned Count) {
    for (unsigned i = 0; i < Count; ++i) {
      unsigned L = Base + i;
      if (L < WaveSize)
        Occupied.set(L);
    }
  };

  markRange(StackPointerRegSpillLane, 1);
  markRange(FramePointerRegSSpillLane, 1);
  markRange(StackPointerStoreLane, 1);

  if (BufferRsrcSpillLane)
    markRange(*BufferRsrcSpillLane, /*PSB=*/4);
  if (ScratchSpillLane)
    markRange(*ScratchSpillLane, /*FLAT_SCR=*/2);
  for (const auto &[SA, Base] : ScalarArguments)
    markRange(Base, getArgumentLaneSize(SA));

  llvm::SmallVector<uint8_t, 4> Out;
  for (unsigned L = 0; L < WaveSize && Out.size() < NumLanes; ++L)
    if (!Occupied.test(L))
      Out.push_back(static_cast<uint8_t>(L));
  return Out;
}

unsigned StateValueArraySpecs::getArgumentLaneSize(ScalarValueArgument SA) {
  switch (SA) {
  case WAVEFRONT_PRIVATE_SEGMENT_BUFFER:
    return ScalarValueArgumentInfo<WAVEFRONT_PRIVATE_SEGMENT_BUFFER>::NumLanes;
  case KERNEL_ARG_PTR:
    return ScalarValueArgumentInfo<KERNEL_ARG_PTR>::NumLanes;
  case DISPATCH_ID:
    return ScalarValueArgumentInfo<DISPATCH_ID>::NumLanes;
  case FLAT_SCRATCH:
    return ScalarValueArgumentInfo<FLAT_SCRATCH>::NumLanes;
  case DISPATCH_PTR:
    return ScalarValueArgumentInfo<DISPATCH_PTR>::NumLanes;
  case QUEUE_PTR:
    return ScalarValueArgumentInfo<QUEUE_PTR>::NumLanes;
  case WORK_ITEM_PRIVATE_SEGMENT_SIZE:
    return ScalarValueArgumentInfo<WORK_ITEM_PRIVATE_SEGMENT_SIZE>::NumLanes;
  case IMPLICIT_ARG_BUFFER:
    return ScalarValueArgumentInfo<IMPLICIT_ARG_BUFFER>::NumLanes;
  case WORKGROUP_ID_X:
    return ScalarValueArgumentInfo<WORKGROUP_ID_X>::NumLanes;
  case WORKGROUP_ID_Y:
    return ScalarValueArgumentInfo<WORKGROUP_ID_Y>::NumLanes;
  case WORKGROUP_ID_Z:
    return ScalarValueArgumentInfo<WORKGROUP_ID_Z>::NumLanes;
  case WORKITEM_ID_X:
    return ScalarValueArgumentInfo<WORKITEM_ID_X>::NumLanes;
  case WORKITEM_ID_Y:
    return ScalarValueArgumentInfo<WORKITEM_ID_Y>::NumLanes;
  case WORKITEM_ID_Z:
    return ScalarValueArgumentInfo<WORKITEM_ID_Z>::NumLanes;
  }
  static_assert(SCALAR_VALUE_ARGUMENT_LAST == WORKITEM_ID_Z,
                "extend getArgumentLaneSize for new ScalarValueArgument");
  llvm_unreachable("Invalid scalar value argument");
}

bool StateValueArraySpecs::invalidate(Prototype &,
                                      const llvm::PreservedAnalyses &PA,
                                      PrototypeAnalysisManager::Invalidator &) {
  // Because this is read from the inner machine-passes pipeline via
  // PrototypeAnalysisManagerMachineFunctionProxy::getCachedResult, model
  // as a stateless outer analysis.
  auto PAC = PA.getChecker<StateValueArraySpecsAnalysis>();
  return !PAC.preservedWhenStateless();
}

llvm::AnalysisKey StateValueArraySpecsAnalysis::Key;

StateValueArraySpecsAnalysis::Result
StateValueArraySpecsAnalysis::run(Prototype &IP,
                                  PrototypeAnalysisManager &IPAM) {
  Result Out;

  llvm::Module &IModule = IP.getInstrumentationModule();
  llvm::Module &TargetModule = IP.getTargetModule();

  // Aggregate the SAs referenced by every luthier::readSVA use in the
  // instrumentation module. Both call shapes are recognized:
  //
  //   1. Direct calls to the intrinsic function itself — a Function marked
  //      with the "luthier-intrinsic" attribute whose value is
  //      "luthier::readSVA".
  //   2. Inline-asm placeholder CallInsts left behind by
  //      ProcessIntrinsicsAtIRLevelPass — an InlineAsm CallInst whose
  //      AsmString is "luthier::readSVA".
  //
  // In both shapes the first (and only) argument is the SA enum as a
  // ConstantInt — see src/lib/Intrinsic/ReadSVA.cpp for the IR-processor
  // contract, and ProcessIntrinsicsAtIRLevelPass for the shape it emits.
  auto ExtractSAFromCall = [](const llvm::CallInst &Call)
      -> std::optional<ScalarValueArgument> {
    if (Call.arg_size() != 1)
      return std::nullopt;
    const auto *SAConst =
        llvm::dyn_cast<llvm::ConstantInt>(Call.getArgOperand(0));
    if (!SAConst)
      return std::nullopt;
    uint64_t V = SAConst->getZExtValue();
    if (V > static_cast<uint64_t>(SCALAR_VALUE_ARGUMENT_LAST))
      return std::nullopt;
    return static_cast<ScalarValueArgument>(V);
  };

  auto IsReadSVACall = [](const llvm::CallInst &Call) -> bool {
    // Inline-asm placeholder path.
    if (const auto *IA = llvm::dyn_cast_or_null<llvm::InlineAsm>(
            Call.getCalledOperand())) {
      return IA->getAsmString() == "luthier::readSVA";
    }
    // Direct-call path: the callee is a Function marked with the Luthier
    // intrinsic attribute naming "luthier::readSVA".
    if (const llvm::Function *Callee = Call.getCalledFunction()) {
      if (Callee->hasFnAttribute(IntrinsicAttribute) &&
          Callee->getFnAttribute(IntrinsicAttribute).getValueAsString() ==
              "luthier::readSVA")
        return true;
    }
    return false;
  };

  llvm::SmallDenseSet<ScalarValueArgument> SAsUsed;
  for (const llvm::Function &F : IModule) {
    for (const llvm::BasicBlock &BB : F) {
      for (const llvm::Instruction &I : BB) {
        const auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);
        if (!Call || !IsReadSVACall(*Call))
          continue;
        if (auto SA = ExtractSAFromCall(*Call))
          SAsUsed.insert(*SA);
      }
    }
  }

  // If the initial entry point is not a kernel, then we are instrumenting
  // newly discovered code from an already-running instrumented kernel. In
  // that situation the SVA has already been initialized to save all
  // possible scalar arguments just to be safe.
  llvm::ModuleAnalysisManager &TargetMAM =
      IPAM.getResult<TargetModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager();
  bool IsInitialEntryPointKernel =
      TargetMAM.getResult<InitialEntryPointAnalysis>(TargetModule)
          .getInitialEntryPoint()
          .isKernel();
  if (!IsInitialEntryPointKernel) {
    for (std::underlying_type_t<ScalarValueArgument> I =
             SCALAR_VALUE_ARGUMENT_FIRST;
         I <= SCALAR_VALUE_ARGUMENT_LAST; ++I)
      SAsUsed.insert(static_cast<ScalarValueArgument>(I));
  }

  // The TargetMachine is reached the same way IntrinsicMIRLoweringPass
  // reaches it — via the IModule's MachineModuleAnalysis result.
  llvm::ModuleAnalysisManager &IMAM =
      IPAM.getResult<IModuleAnalysisManagerPrototypeProxy>(IP).getManager();
  const llvm::TargetMachine &TM =
      IMAM.getResult<llvm::MachineModuleAnalysis>(IModule).getMMI().getTarget();
  const auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(*IModule.begin());
  bool IsArchitectedFS = ST.hasArchitectedFlatScratch();
  bool HasFS = ST.enableFlatScratch();
#ifdef _DEBUG
  // Sanity: every function should share the same scratch-access requirements.
  for (const llvm::Function &F : IModule) {
    if (F.isDeclaration())
      continue;
    const auto &FuncST = TM.getSubtarget<llvm::GCNSubtarget>(F);
    bool FuncHasArchitectedFS = FuncST.hasArchitectedFlatScratch();
    bool FuncHasFS = FuncST.enableFlatScratch();
    assert(FuncHasArchitectedFS == IsArchitectedFS && FuncHasFS == HasFS &&
           "Functions have different scratch access requirements");
  }
#endif

  uint8_t NextLane = StateValueArraySpecs::StackPointerStoreLane + 1;
  if (!HasFS && !IsArchitectedFS) {
    Out.ScalarArguments.insert({WAVEFRONT_PRIVATE_SEGMENT_BUFFER, NextLane});
    NextLane +=
        ScalarValueArgumentInfo<WAVEFRONT_PRIVATE_SEGMENT_BUFFER>::NumLanes;
    Out.BufferRsrcSpillLane = NextLane;
    NextLane +=
        ScalarValueArgumentInfo<WAVEFRONT_PRIVATE_SEGMENT_BUFFER>::NumLanes;
  }
  if (!IsArchitectedFS) {
    Out.ScalarArguments.insert({FLAT_SCRATCH, NextLane});
    NextLane += ScalarValueArgumentInfo<FLAT_SCRATCH>::NumLanes;
    Out.ScratchSpillLane = NextLane;
    NextLane += ScalarValueArgumentInfo<FLAT_SCRATCH>::NumLanes;
  }

  using SVArgUnderlyingType = std::underlying_type_t<ScalarValueArgument>;

  // Assign per-SA lane bases in canonical enum order — the same order the
  // metadata-based factory used, so consumers see the same layout.
  auto AssignIfUsed = [&]<SVArgUnderlyingType SVArg>() {
    constexpr auto CastedSVArg = static_cast<ScalarValueArgument>(SVArg);
    if (!SAsUsed.contains(CastedSVArg))
      return;
    if constexpr (CastedSVArg == WAVEFRONT_PRIVATE_SEGMENT_BUFFER ||
                  CastedSVArg == FLAT_SCRATCH) {
      if (IsArchitectedFS) {
        return;
      }
    }
    Out.ScalarArguments.insert({CastedSVArg, NextLane});
    NextLane += ScalarValueArgumentInfo<CastedSVArg>::NumLanes;
  };

  // std::make_integer_sequence<T, N> produces [0, N-1]; the inclusive
  // SCALAR_VALUE_ARGUMENT_LAST sentinel is the highest valid enumerator, so
  // the count is LAST+1 to cover every SA in [FIRST, LAST].
  constexpr auto SVArgSequence =
      std::make_integer_sequence<SVArgUnderlyingType,
                                 SCALAR_VALUE_ARGUMENT_LAST + 1>{};
  [&]<SVArgUnderlyingType... SVArgs>(
      std::integer_sequence<SVArgUnderlyingType, SVArgs...>) {
    (AssignIfUsed.operator()<SVArgs>(), ...);
  }(SVArgSequence);

  return Out;
}

} // namespace luthier
