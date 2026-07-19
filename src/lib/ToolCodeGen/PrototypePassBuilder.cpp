//===-- PrototypePassBuilder.cpp -------------------------------===//
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
/// \file
/// Implements \c luthier::PrototypePassBuilder, including
/// \c buildInstrumentationPipeline and the Luthier-owned AMDGPU codegen
/// pass builder used to splice \c InjectedPayloadPEIPass into the machine
/// pipeline.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/PrototypePassBuilder.h"

#include "luthier/ToolCodeGen/ForwardISAStateToCalleesPass.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessIModulePass.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadPEIPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadPreserveLiveRegsPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include "luthier/ToolCodeGen/InstructionTracesAnalysis.h"
#include "luthier/ToolCodeGen/InstrumentationPMDriver.h"
#include "luthier/ToolCodeGen/IntrinsicMIRLoweringPass.h"
#include "luthier/ToolCodeGen/MIRToIRTranslationAnalysis.h"
#include "luthier/ToolCodeGen/PrePostAmbleEmitter.h"
#include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/SVAPhysVGPRPinPass.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/TargetModulePatcherPass.h"
#include "luthier/ToolCodeGen/TraceCallGraph.h"

#include <AMDGPUTargetMachine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/PEI.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassManagerInternal.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/CGPassBuilderOption.h>
#include <llvm/Target/TargetMachine.h>

using llvm::Error;
using llvm::ModulePassManager;
using llvm::PassBuilder;
using llvm::PassInstrumentationCallbacks;
using llvm::StringRef;

namespace luthier {

//===----------------------------------------------------------------------===//
// CodeGenAugmenter — opaque handle for AugmentCodeGenCallback.
//===----------------------------------------------------------------------===//

/// Opaque view into the codegen pass builder handed to
/// \c AugmentCodeGenCallback consumers. Kept minimal so consumers only touch
/// what they need. Extend as new augmentation surfaces are required.
class PrototypePassBuilder::CodeGenAugmenter {
public:
  CodeGenAugmenter() = default;

  /// Set to \c true by a callback that wants to skip Luthier's
  /// \c InjectedPayloadPEIPass on this run. Independent of the CL-level
  /// \c --disable-injected-payload-pei — either can force it off.
  bool SuppressInjectedPayloadPEI = false;
};

//===----------------------------------------------------------------------===//
// Post-hoc splice: locate LLVM's stock PrologEpilogInserterPass in the codegen
// pipeline built by AMDGPU and insert Luthier's InjectedPayloadPEIPass right
// after it.
//
// AMDGPU's own AMDGPUCodeGenPassBuilder lives in an anonymous namespace inside
// llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp and is unreachable from
// another TU. Rather than fork the whole subclass (and drag in ~30 TU-private
// static cl::opts), we let GCNTargetMachine::buildCodeGenPipeline populate a
// scratch ModulePassManager and then walk the resulting tree to splice our
// pass in. Walking requires access to the private/protected pass containers
// inside PassManager<> and the two-level adaptors; we reach them via the
// classical "Rob" pattern (explicit template instantiation of a template that
// friends the target member-pointer via ODR context).
//
// LLVM_UPSTREAM_SYNC — this file inspects the following LLVM internals:
//   * llvm::PassManager<T>::Passes            (protected member; PassManager.h)
//   * llvm::ModuleToFunctionPassAdaptor::Pass (private; PassManager.h)
//   * llvm::FunctionToMachineFunctionPassAdaptor::Pass
//                                             (private; MachinePassManager.h)
//   * llvm::detail::PassModel<...>            (PassManagerInternal.h)
//   * PassInfoMixin<T>::name() returning the class name minus "llvm::"
// If any of these change shape upstream, this walker must be updated.
//===----------------------------------------------------------------------===//

namespace {

//=== Rob pattern for reaching private/protected members. ==================//

template <typename Tag> struct Rob {
  static typename Tag::type Ptr;
};
template <typename Tag> typename Tag::type Rob<Tag>::Ptr;

template <typename Tag, typename Tag::type P> struct RobBinder {
  RobBinder() { Rob<Tag>::Ptr = P; }
  static RobBinder Instance;
};
template <typename Tag, typename Tag::type P>
RobBinder<Tag, P> RobBinder<Tag, P>::Instance;

//=== Type aliases for the concept types stored by each pass manager. ======//

using ModulePassConceptT =
    llvm::detail::PassConcept<llvm::Module, llvm::ModuleAnalysisManager>;
using FunctionPassConceptT =
    llvm::detail::PassConcept<llvm::Function, llvm::FunctionAnalysisManager>;
using MachineFunctionPassConceptT =
    llvm::detail::PassConcept<llvm::MachineFunction,
                              llvm::MachineFunctionAnalysisManager>;
using CGSCCPassConceptT = llvm::detail::PassConcept<
    llvm::LazyCallGraph::SCC, llvm::CGSCCAnalysisManager,
    llvm::LazyCallGraph &, llvm::CGSCCUpdateResult &>;

//=== Tags naming each member we want to reach. ============================//

struct MPMPassesTag {
  using type = std::vector<std::unique_ptr<ModulePassConceptT>>
      llvm::ModulePassManager::*;
};
struct FPMPassesTag {
  using type = std::vector<std::unique_ptr<FunctionPassConceptT>>
      llvm::FunctionPassManager::*;
};
struct MFPMPassesTag {
  using type = std::vector<std::unique_ptr<MachineFunctionPassConceptT>>
      llvm::MachineFunctionPassManager::*;
};
struct CGSCCPMPassesTag {
  using type = std::vector<std::unique_ptr<CGSCCPassConceptT>>
      llvm::CGSCCPassManager::*;
};
struct M2FAdaptorPassTag {
  using type = std::unique_ptr<llvm::ModuleToFunctionPassAdaptor::PassConceptT>
      llvm::ModuleToFunctionPassAdaptor::*;
};
struct F2MFAdaptorPassTag {
  using type = std::unique_ptr<
      llvm::FunctionToMachineFunctionPassAdaptor::PassConceptT>
      llvm::FunctionToMachineFunctionPassAdaptor::*;
};
struct M2CGSCCAdaptorPassTag {
  using type = std::unique_ptr<
      llvm::ModuleToPostOrderCGSCCPassAdaptor::PassConceptT>
      llvm::ModuleToPostOrderCGSCCPassAdaptor::*;
};
struct CGSCC2FAdaptorPassTag {
  using type = std::unique_ptr<llvm::CGSCCToFunctionPassAdaptor::PassConceptT>
      llvm::CGSCCToFunctionPassAdaptor::*;
};

//=== Explicit template instantiations that bind Rob<Tag>::Ptr. ============//
// (Placed inside the anon namespace so the ODR-context friend trick applies.)

template struct RobBinder<MPMPassesTag, &llvm::ModulePassManager::Passes>;
template struct RobBinder<FPMPassesTag, &llvm::FunctionPassManager::Passes>;
template struct RobBinder<MFPMPassesTag,
                          &llvm::MachineFunctionPassManager::Passes>;
template struct RobBinder<CGSCCPMPassesTag, &llvm::CGSCCPassManager::Passes>;
template struct RobBinder<M2FAdaptorPassTag,
                          &llvm::ModuleToFunctionPassAdaptor::Pass>;
template struct RobBinder<F2MFAdaptorPassTag,
                          &llvm::FunctionToMachineFunctionPassAdaptor::Pass>;
template struct RobBinder<M2CGSCCAdaptorPassTag,
                          &llvm::ModuleToPostOrderCGSCCPassAdaptor::Pass>;
template struct RobBinder<CGSCC2FAdaptorPassTag,
                          &llvm::CGSCCToFunctionPassAdaptor::Pass>;

//=== Cast a PassConcept<IRUnit>* to the PassModel wrapping AdaptorT.  =====//
// PassInfoMixin<T>::name() lets us verify the runtime type by comparing
// against AdaptorT::name() (which is the class name minus "llvm::") before
// doing the static_cast.
template <typename AdaptorT, typename IRUnitT, typename AMT,
          typename... ExtraArgTs>
AdaptorT *
asAdaptor(llvm::detail::PassConcept<IRUnitT, AMT, ExtraArgTs...> *Concept) {
  if (!Concept || Concept->name() != AdaptorT::name())
    return nullptr;
  auto *Model = static_cast<
      llvm::detail::PassModel<IRUnitT, AdaptorT, AMT, ExtraArgTs...> *>(
      Concept);
  return &Model->Pass;
}

//=== The walker itself. ===================================================//

/// Search \p MFPM for a pass whose \c name() matches
/// \c llvm::PrologEpilogInserterPass::name() and insert \p NewPass
/// immediately after it. Returns true on success.
bool spliceInMFPM(llvm::MachineFunctionPassManager &MFPM,
                  InjectedPayloadPEIPass NewPass) {
  auto &MFPasses = MFPM.*Rob<MFPMPassesTag>::Ptr;
  for (auto It = MFPasses.begin(); It != MFPasses.end(); ++It) {
    if ((*It)->name() != llvm::PrologEpilogInserterPass::name())
      continue;
    using NewPassModelT = llvm::detail::PassModel<
        llvm::MachineFunction, InjectedPayloadPEIPass,
        llvm::MachineFunctionAnalysisManager>;
    auto NewSlot = std::unique_ptr<MachineFunctionPassConceptT>(
        new NewPassModelT(std::move(NewPass)));
    MFPasses.insert(std::next(It), std::move(NewSlot));
    return true;
  }
  return false;
}

/// Descend into an FPM looking for a FunctionToMachineFunctionPassAdaptor.
bool spliceInFPM(llvm::FunctionPassManager &FPM,
                 InjectedPayloadPEIPass NewPass) {
  auto &FPasses = FPM.*Rob<FPMPassesTag>::Ptr;
  for (auto &Slot : FPasses) {
    if (auto *Adaptor =
            asAdaptor<llvm::FunctionToMachineFunctionPassAdaptor>(
                Slot.get())) {
      // Adaptor holds a unique_ptr<PassConcept<MachineFunction>>. The
      // concrete concept is a PassModel<MachineFunctionPassManager> when
      // an MFPM was wholesale added; unwrap it to reach the MFPM.
      auto &InnerSlot = *Adaptor.*Rob<F2MFAdaptorPassTag>::Ptr;
      (void)InnerSlot;
      // The MFPM was added to the adaptor via a raw PassConcept — retrieve
      // it via the PassModel wrapper.
      using MFPMModelT = llvm::detail::PassModel<
          llvm::MachineFunction, llvm::MachineFunctionPassManager,
          llvm::MachineFunctionAnalysisManager>;
      auto *InnerConcept =
          (Adaptor->*Rob<F2MFAdaptorPassTag>::Ptr).get();
      if (!InnerConcept)
        continue;
      auto *InnerModel = static_cast<MFPMModelT *>(InnerConcept);
      if (spliceInMFPM(InnerModel->Pass, std::move(NewPass)))
        return true;
    }
  }
  return false;
}

/// Descend into a CGSCC PM looking for a CGSCCToFunctionPassAdaptor.
bool spliceInCGSCCPM(llvm::CGSCCPassManager &CGPM,
                     InjectedPayloadPEIPass NewPass) {
  auto &CGPasses = CGPM.*Rob<CGSCCPMPassesTag>::Ptr;
  for (auto &Slot : CGPasses) {
    if (auto *Adaptor =
            asAdaptor<llvm::CGSCCToFunctionPassAdaptor>(Slot.get())) {
      using FPMModelT =
          llvm::detail::PassModel<llvm::Function, llvm::FunctionPassManager,
                                  llvm::FunctionAnalysisManager>;
      auto *InnerConcept =
          (Adaptor->*Rob<CGSCC2FAdaptorPassTag>::Ptr).get();
      if (!InnerConcept)
        continue;
      auto *InnerModel = static_cast<FPMModelT *>(InnerConcept);
      if (spliceInFPM(InnerModel->Pass, std::move(NewPass)))
        return true;
    }
  }
  return false;
}

/// Walk \p MPM and splice \c InjectedPayloadPEIPass immediately after LLVM's
/// stock \c PrologEpilogInserterPass. Returns true if the anchor was found.
bool spliceInjectedPayloadPEIAfterStockPEI(llvm::ModulePassManager &MPM) {
  auto &MPasses = MPM.*Rob<MPMPassesTag>::Ptr;
  for (auto &Slot : MPasses) {
    // Module → Function adaptor path.
    if (auto *Adaptor =
            asAdaptor<llvm::ModuleToFunctionPassAdaptor>(Slot.get())) {
      using FPMModelT =
          llvm::detail::PassModel<llvm::Function, llvm::FunctionPassManager,
                                  llvm::FunctionAnalysisManager>;
      auto *InnerConcept =
          (Adaptor->*Rob<M2FAdaptorPassTag>::Ptr).get();
      if (InnerConcept) {
        auto *InnerModel = static_cast<FPMModelT *>(InnerConcept);
        if (spliceInFPM(InnerModel->Pass, InjectedPayloadPEIPass()))
          return true;
      }
    }
    // Module → CGSCC → Function adaptor path (RequiresCodeGenSCCOrder).
    if (auto *Adaptor = asAdaptor<llvm::ModuleToPostOrderCGSCCPassAdaptor>(
            Slot.get())) {
      using CGPMModelT = llvm::detail::PassModel<
          llvm::LazyCallGraph::SCC, llvm::CGSCCPassManager,
          llvm::CGSCCAnalysisManager, llvm::LazyCallGraph &,
          llvm::CGSCCUpdateResult &>;
      auto *InnerConcept =
          (Adaptor->*Rob<M2CGSCCAdaptorPassTag>::Ptr).get();
      if (InnerConcept) {
        auto *InnerModel = static_cast<CGPMModelT *>(InnerConcept);
        if (spliceInCGSCCPM(InnerModel->Pass, InjectedPayloadPEIPass()))
          return true;
      }
    }
  }
  return false;
}

} // namespace

//===----------------------------------------------------------------------===//
// PrototypePassBuilder — construction / destruction.
//===----------------------------------------------------------------------===//

PrototypePassBuilder::PrototypePassBuilder(
    llvm::TargetMachine *TM, llvm::PipelineTuningOptions PTO,
    std::optional<llvm::PGOOptions> PGOOpt,
    PassInstrumentationCallbacks *PIC)
    : TM(TM), PIC(PIC),
      PB(std::make_unique<PassBuilder>(TM, PTO, PGOOpt, PIC)) {
  registerLuthierPasses();
  for (auto &Cb : PassBuilderAugmentationCallbacks)
    Cb(*PB);
}

PrototypePassBuilder::~PrototypePassBuilder() = default;

PassBuilder &PrototypePassBuilder::getPassBuilder() { return *PB; }

//===----------------------------------------------------------------------===//
// Luthier pass / analysis registration on the wrapped PassBuilder.
//===----------------------------------------------------------------------===//

void PrototypePassBuilder::registerLuthierPasses() {
  // Module passes — parsed via -passes=<...>.
  PB->registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &MPM,
         llvm::ArrayRef<PassBuilder::PipelineElement>) {
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"
        return false;
      });

  // Machine-function passes.
  PB->registerPipelineParsingCallback(
      [](StringRef Name, llvm::MachineFunctionPassManager &MFPM,
         llvm::ArrayRef<PassBuilder::PipelineElement>) {
#define MACHINE_FUNCTION_PASS(NAME, CREATE_PASS)                               \
  if (Name == NAME) {                                                          \
    MFPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"
        return false;
      });

  // TargetMachine-dependent passes cannot be registered through the plain
  // \c .def macro. Do them by hand here so they still show up in
  // -passes=<...> parsing.
  auto *ThisTM = TM;
  PB->registerPipelineParsingCallback(
      [ThisTM](StringRef Name, ModulePassManager &MPM,
               llvm::ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "luthier-forward-isa-state-to-callees" && ThisTM) {
          // Caller is expected to have constructed with a GCNTargetMachine.
          auto *GCNTM = static_cast<llvm::GCNTargetMachine *>(ThisTM);
          MPM.addPass(ForwardISAStateToCalleesPass(*GCNTM));
          return true;
        }
        return false;
      });
}

void PrototypePassBuilder::registerAllAnalyses(
    llvm::ModuleAnalysisManager &MAM, llvm::FunctionAnalysisManager &FAM,
    llvm::MachineFunctionAnalysisManager &MFAM,
    PrototypeAnalysisManager &IPAM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  FAM.registerPass([&] { return CREATE_PASS; });
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"

#define MACHINE_FUNCTION_ANALYSIS(NAME, CREATE_PASS)                           \
  MFAM.registerPass([&] { return CREATE_PASS; });
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"

#define PROTOTYPE_ANALYSIS(NAME, CREATE_PASS)                                  \
  IPAM.registerPass([&] { return CREATE_PASS; });
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"

  (void)MAM;
}

//===----------------------------------------------------------------------===//
// Cross-level proxy registration.
//===----------------------------------------------------------------------===//

void PrototypePassBuilder::crossRegisterProxies(
    llvm::ModuleAnalysisManager &MAM, llvm::FunctionAnalysisManager &FAM,
    llvm::MachineFunctionAnalysisManager &MFAM,
    PrototypeAnalysisManager &IPAM) {
  // The adaptors run pass instrumentation at the Prototype level, so
  // PassInstrumentationAnalysis must be available on its analysis manager.
  // Register it against the PIC held by the wrapped llvm::PassBuilder so
  // callbacks are shared with the nested Module/Function/MachineFunction
  // managers.
  PassInstrumentationCallbacks *ThePIC = PB->getPassInstrumentationCallbacks();
  IPAM.registerPass(
      [ThePIC] { return llvm::PassInstrumentationAnalysis(ThePIC); });

  IPAM.registerPass(
      [&] { return ModuleAnalysisManagerPrototypeProxy(MAM); });
  IPAM.registerPass(
      [&] { return FunctionAnalysisManagerPrototypeProxy(FAM); });
  IPAM.registerPass(
      [&] { return MachineFunctionAnalysisManagerPrototypeProxy(MFAM); });

  MAM.registerPass([&] { return PrototypeAnalysisManagerModuleProxy(IPAM); });
  FAM.registerPass(
      [&] { return PrototypeAnalysisManagerFunctionProxy(IPAM); });
  MFAM.registerPass(
      [&] { return PrototypeAnalysisManagerMachineFunctionProxy(IPAM); });
}

//===----------------------------------------------------------------------===//
// parsePipeline — target(...) / instrumentation(...) grammar.
//===----------------------------------------------------------------------===//

Error PrototypePassBuilder::parsePipeline(PrototypePassManager &PPM,
                                          StringRef PipelineText) {
  StringRef Remaining = PipelineText.trim();

  while (!Remaining.empty()) {
    bool IsTarget = Remaining.consume_front("target(");
    bool IsInstrumentation =
        !IsTarget && Remaining.consume_front("instrumentation(");

    if (!IsTarget && !IsInstrumentation) {
      return llvm::make_error<llvm::StringError>(
          "expected 'target(...)' or 'instrumentation(...)' at top level of "
          "-passes (bare pass names are not allowed)",
          llvm::inconvertibleErrorCode());
    }

    size_t Depth = 1;
    size_t Pos = 0;
    while (Pos < Remaining.size() && Depth > 0) {
      if (Remaining[Pos] == '(')
        Depth++;
      else if (Remaining[Pos] == ')')
        Depth--;
      if (Depth > 0)
        Pos++;
    }

    if (Depth != 0) {
      return llvm::make_error<llvm::StringError>(
          "unmatched parentheses in Prototype pass pipeline",
          llvm::inconvertibleErrorCode());
    }

    StringRef InnerText = Remaining.substr(0, Pos);
    Remaining = Remaining.substr(Pos + 1).ltrim();
    if (Remaining.consume_front(","))
      Remaining = Remaining.ltrim();

    bool Handled = false;
    for (auto &Cb : ParseCallbacks) {
      if (Cb(InnerText, PPM, IsTarget)) {
        Handled = true;
        break;
      }
    }
    if (Handled)
      continue;

    ModulePassManager InnerMPM;
    if (auto Err = PB->parsePassPipeline(InnerMPM, InnerText))
      return Err;

    if (IsTarget)
      PPM.addPass(createRunOnTargetModuleAdaptor(std::move(InnerMPM)));
    else
      PPM.addPass(
          createRunOnInstrumentationModuleAdaptor(std::move(InnerMPM)));
  }

  return Error::success();
}

//===----------------------------------------------------------------------===//
// IR pipeline (IModule).
//===----------------------------------------------------------------------===//

Error PrototypePassBuilder::buildIROptimizationPipeline(
    ModulePassManager &IMPM, const InstrumentationPMDriverOptions &Opts,
    llvm::OptimizationLevel Level) {
  // User-supplied IR pipeline override — take it as-is and skip the default.
  if (Opts.IModuleIRPasses.getNumOccurrences() > 0) {
    if (Opts.IModuleIRPasses.empty())
      return Error::success();
    return PB->parsePassPipeline(IMPM, Opts.IModuleIRPasses);
  }

  for (auto &Cb : PreIROptimizationCallbacks)
    Cb(IMPM, Level);

  IMPM.addPass(PB->buildPerModuleDefaultPipeline(Level));

  for (auto &Cb : PreIRIntrinsicLoweringCallbacks)
    Cb(IMPM);

  // Cache warm-up: force InjectedPayloadSideEffectsAnalysis to run before
  // ProcessIntrinsicsAtIRLevelPass rewrites intrinsic call sites to inline
  // asm placeholders. Mirrors InstrumentationPMDriver.cpp:436-440.
  IMPM.addPass(llvm::createModuleToFunctionPassAdaptor(
      llvm::RequireAnalysisPass<InjectedPayloadSideEffectsAnalysis,
                                llvm::Function>()));

  IMPM.addPass(ProcessIntrinsicsAtIRLevelPass());

  for (auto &Cb : PostIRIntrinsicLoweringCallbacks)
    Cb(IMPM);

  return Error::success();
}

//===----------------------------------------------------------------------===//
// Codegen pipeline (IModule).
//===----------------------------------------------------------------------===//

Error PrototypePassBuilder::buildCodeGenPipeline(
    ModulePassManager &IMPM, const InstrumentationPMDriverOptions &Opts,
    llvm::ModuleAnalysisManager &MAM, llvm::raw_pwrite_stream &Out,
    llvm::raw_pwrite_stream *DwoOut, llvm::CodeGenFileType FileType,
    llvm::MCContext &Ctx) {
  // GCNTargetMachine has no classof; the caller is expected to have
  // constructed this builder with a GCNTargetMachine.
  if (!TM)
    return llvm::make_error<llvm::StringError>(
        "PrototypePassBuilder: codegen pipeline requires a target machine",
        llvm::inconvertibleErrorCode());
  auto *GCNTM = static_cast<llvm::GCNTargetMachine *>(TM);

  CodeGenAugmenter Augmenter;
  for (auto &Cb : AugmentCodeGenCallbacks)
    Cb(Augmenter, *TM);

  const bool DisablePEI =
      Opts.DisableInjectedPayloadPEI || Augmenter.SuppressInjectedPayloadPEI;

  // Let AMDGPU own the codegen pipeline build (its
  // AMDGPUCodeGenPassBuilder subclass is anon-namespace and unreachable to
  // us). Populate a scratch MPM, then splice InjectedPayloadPEIPass in by
  // walking the resulting pass manager tree.
  ModulePassManager Scratch;
  if (auto Err = GCNTM->buildCodeGenPipeline(
          Scratch, MAM, Out, DwoOut, FileType,
          llvm::getCGPassBuilderOption(), Ctx, PIC))
    return Err;

  if (!DisablePEI) {
    if (!spliceInjectedPayloadPEIAfterStockPEI(Scratch))
      return llvm::make_error<llvm::StringError>(
          "PrototypePassBuilder: could not locate PrologEpilogInserterPass "
          "in AMDGPU codegen pipeline; InjectedPayloadPEIPass not inserted",
          llvm::inconvertibleErrorCode());
  }

  IMPM.addPass(std::move(Scratch));
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Top-level pipeline builder.
//===----------------------------------------------------------------------===//

Error PrototypePassBuilder::buildInstrumentationPipeline(
    PrototypePassManager &PPM, const InstrumentationPMDriverOptions &Opts,
    llvm::OptimizationLevel Level, llvm::ModuleAnalysisManager &MAM,
    llvm::raw_pwrite_stream &Out, llvm::raw_pwrite_stream *DwoOut,
    llvm::CodeGenFileType FileType, llvm::MCContext &Ctx) {
  // Assemble the IModule pipeline (IR + codegen).
  ModulePassManager IMPM;

  if (auto Err = buildIROptimizationPipeline(IMPM, Opts, Level))
    return Err;

  if (auto Err = buildCodeGenPipeline(IMPM, Opts, MAM, Out, DwoOut, FileType,
                                       Ctx))
    return Err;

  PPM.addPass(createRunOnInstrumentationModuleAdaptor(std::move(IMPM)));

  // Prototype-level passes. Legacy driver ordering placed
  // IntrinsicMIRLoweringPass / IModuleIPPredicatedLivenessAnalysis /
  // InjectedPayloadPreserveLiveRegsPass between ISel and MachinePasses, and
  // SVAPhysVGPRPinPass deep inside register allocation. Their new-PM
  // reincarnations are all Prototype-level per their headers, so they now
  // run against the fully-codegen'd Prototype.
  PPM.addPass(IntrinsicMIRLoweringPass());
  PPM.addPass(llvm::RequireAnalysisPass<IModuleIPPredicatedLivenessAnalysis,
                                        Prototype, PrototypeAnalysisManager>());
  PPM.addPass(InjectedPayloadPreserveLiveRegsPass());
  PPM.addPass(SVAPhysVGPRPinPass());

  // Final target-module patch step.
  PPM.addPass(TargetModulePatcherPass());

  return Error::success();
}

} // namespace luthier
