//===-- LuthierFile.cpp ---------------------------------------------------===//
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
/// Implements \c LuthierFileParser and the \c writeLuthierFile helpers.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/LuthierFile.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/CodeGen/MIRParser/MIRParser.h>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ModuleSlotTracker.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Base64.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/YAMLTraits.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

namespace luthier {

//===----------------------------------------------------------------------===//
// Internal YAML helper types
//===----------------------------------------------------------------------===//

/// String wrapper that round-trips through YAML as a literal block scalar
/// (the \c | style), preserving multi-line IR/MIR text without quoting or
/// escaping.  This is an internal implementation detail; callers of
/// \c LuthierFileParser interact only with \c llvm::StringRef
///
/// \c std::string already has \c ScalarTraits in LLVM's YAML library, which
/// escapes newlines into a single quoted line.  \c BlockScalarTraits cannot
/// be attached to \c std::string because that specialization is already
/// taken, so this distinct wrapper type is used instead
struct IRBlockString {
  std::string S;

  IRBlockString() = default;
  operator llvm::StringRef() const { return S; }
};

/// Internal YAML-visible mirror of the \c .luthier file schema
/// \c LuthierFileParser::create parses into this struct and then moves the
/// fields into the parser object; \c writeLuthierFile builds one from the
/// two modules before emitting YAML.  Each module has its own text blob and
/// its own \c ModuleFormat.
struct LuthierFileYaml {
  IRBlockString TargetModuleText;
  LuthierFileParser::ModuleFormat TargetModuleFormat =
      LuthierFileParser::ModuleFormat::MIR;
  IRBlockString InstrumentationModuleText;
  LuthierFileParser::ModuleFormat InstrumentationModuleFormat =
      LuthierFileParser::ModuleFormat::IR;
  std::vector<LuthierFileParser::MDSlotEntry> MDSlotMap;
};

} // namespace luthier

//===----------------------------------------------------------------------===//
// YAML traits
//===----------------------------------------------------------------------===//

namespace llvm::yaml {

template <> struct BlockScalarTraits<luthier::IRBlockString> {
  static void output(const luthier::IRBlockString &V, void *,
                     llvm::raw_ostream &OS) {
    OS << V.S;
  }
  static llvm::StringRef input(llvm::StringRef Str, void *,
                               luthier::IRBlockString &V) {
    V.S = Str.str();
    return {};
  }
};

template <> struct MappingTraits<luthier::LuthierFileParser::MDSlotEntry> {
  static void mapping(IO &IO, luthier::LuthierFileParser::MDSlotEntry &E) {
    IO.mapRequired("IModuleSlot", E.IModuleSlot);
    IO.mapRequired("TargetSlot", E.TargetSlot);
  }
};

template <>
struct ScalarEnumerationTraits<luthier::LuthierFileParser::ModuleFormat> {
  static void enumeration(IO &IO, luthier::LuthierFileParser::ModuleFormat &F) {
    IO.enumCase(F, "IR", luthier::LuthierFileParser::ModuleFormat::IR);
    IO.enumCase(F, "MIR", luthier::LuthierFileParser::ModuleFormat::MIR);
  }
};

template <> struct MappingTraits<luthier::LuthierFileYaml> {
  static void mapping(IO &IO, luthier::LuthierFileYaml &F) {
    IO.mapRequired("TargetModule", F.TargetModuleText);
    IO.mapOptional("TargetModuleFormat", F.TargetModuleFormat,
                   luthier::LuthierFileParser::ModuleFormat::MIR);
    IO.mapRequired("InstrumentationModule", F.InstrumentationModuleText);
    IO.mapOptional("InstrumentationModuleFormat", F.InstrumentationModuleFormat,
                   luthier::LuthierFileParser::ModuleFormat::IR);
    IO.mapOptional("MDSlotMap", F.MDSlotMap,
                   std::vector<luthier::LuthierFileParser::MDSlotEntry>{});
  }
};

} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(luthier::LuthierFileParser::MDSlotEntry)

namespace luthier {

namespace {

/// Resolves a \c Function to its \c MachineFunction, or null when it has none.
using MFAccessor = llvm::function_ref<llvm::MachineFunction *(llvm::Function &)>;

/// \brief Assigns every \c MDNode in \p M a slot number, and returns both
/// directions of the mapping.
///
/// \details Slot numbers are a name for a node that survives serialization: the
/// same walk over the same module — before writing, or after parsing it back —
/// visits the same nodes in the same order, so index N means the same node on
/// both sides. That is all a \c MDSlotMap entry needs.
///
/// The walk deliberately does not use \c ModuleSlotTracker's numbering. Two
/// reasons: its underlying \c SlotTracker is only initialized as a side effect
/// of printing (there is no public way to force it, and \c collectMDNodes
/// silently yields nothing until then), and more importantly it enumerates IR
/// only. Nodes hanging off a \c MachineInstr — a \c pcsections tag naming an
/// instrumentation point, which is exactly the kind of node the two modules
/// share — have no IR user at all and would never be named.
///
/// \p GetMF supplies the \c MachineFunction for a \c Function so MI-level
/// attachments can be reached. Passing an accessor that always returns null
/// restricts the walk to IR, which is all that is available before the MIR
/// parser has run.
struct MDNodeSlots {
  llvm::DenseMap<unsigned, llvm::MDNode *> SlotToMD;
  llvm::DenseMap<const llvm::MDNode *, unsigned> MDToSlot;

  void record(llvm::MDNode *MD) {
    if (!MD || MDToSlot.contains(MD))
      return;
    unsigned Slot = MDToSlot.size();
    MDToSlot[MD] = Slot;
    SlotToMD[Slot] = MD;
    /// Nested nodes are numbered too, so a shared node is nameable even when it
    /// is only reachable as an operand of something else.
    for (const llvm::MDOperand &Op : MD->operands())
      if (auto *Nested = llvm::dyn_cast_or_null<llvm::MDNode>(Op.get()))
        record(Nested);
  }

  void recordAttachments(const llvm::Value &V) {
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> Attachments;
    if (const auto *GO = llvm::dyn_cast<llvm::GlobalObject>(&V))
      GO->getAllMetadata(Attachments);
    else if (const auto *I = llvm::dyn_cast<llvm::Instruction>(&V))
      I->getAllMetadata(Attachments);
    for (auto &[KindID, MD] : Attachments)
      record(MD);
  }
};

MDNodeSlots buildMDNodeSlots(llvm::Module &M, MFAccessor GetMF) {
  MDNodeSlots Slots;

  for (llvm::NamedMDNode &NMD : M.named_metadata())
    for (llvm::MDNode *Op : NMD.operands())
      Slots.record(Op);

  for (llvm::GlobalVariable &GV : M.globals())
    Slots.recordAttachments(GV);

  for (llvm::Function &F : M) {
    Slots.recordAttachments(F);
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB)
        Slots.recordAttachments(I);

    llvm::MachineFunction *MF = GetMF(F);
    if (!MF)
      continue;
    for (llvm::MachineBasicBlock &MBB : *MF)
      for (llvm::MachineInstr &MI : MBB) {
        Slots.record(MI.getPCSections());
        for (llvm::MachineMemOperand *MMO : MI.memoperands()) {
          llvm::AAMDNodes AAInfo = MMO->getAAInfo();
          Slots.record(AAInfo.TBAA);
          Slots.record(AAInfo.TBAAStruct);
          Slots.record(AAInfo.Scope);
          Slots.record(AAInfo.NoAlias);
        }
      }
  }

  return Slots;
}

/// An accessor for modules whose MIR has not been parsed yet.
llvm::MachineFunction *noMachineFunctions(llvm::Function &) { return nullptr; }

/// Parses one module out of a text blob according to its \c ModuleFormat
llvm::Expected<std::unique_ptr<llvm::Module>> parseOneModule(
    llvm::LLVMContext &Ctx, llvm::StringRef Text, llvm::StringRef BufID,
    LuthierFileParser::ModuleFormat Format,
    std::function<std::optional<std::string>(llvm::StringRef, llvm::StringRef)>
        SetDataLayout,
    std::function<void(llvm::Function &)> SetMIRFunctionAttributes,
    std::unique_ptr<llvm::MIRParser> &OutMIRParser) {
  llvm::ParserCallbacks IRCallbacks;
  if (SetDataLayout)
    IRCallbacks.DataLayout =
        [SetDataLayout](llvm::StringRef TT,
                        llvm::StringRef OldDL) -> std::optional<std::string> {
      return SetDataLayout(TT, OldDL);
    };

  if (Format == LuthierFileParser::ModuleFormat::IR) {
    llvm::SMDiagnostic Err;
    llvm::MemoryBufferRef Buf(Text, BufID);
    auto M = llvm::parseIR(Buf, Err, Ctx, IRCallbacks);
    if (!M)
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to parse module '" +
                                        BufID.str() +
                                        "' as IR: " + Err.getMessage().str());
    return M;
  } else if (Format == LuthierFileParser::ModuleFormat::MIR) {
    auto Buf = llvm::MemoryBuffer::getMemBuffer(Text, BufID);
    auto FnAttrCB = [SetMIRFunctionAttributes](llvm::Function &F) {
      if (SetMIRFunctionAttributes)
        SetMIRFunctionAttributes(F);
    };
    auto Parser = llvm::createMIRParser(std::move(Buf), Ctx, FnAttrCB);
    if (!Parser)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Failed to create MIR parser for module '" + BufID.str() + "'");
    auto DataLayoutCB =
        [SetDataLayout](llvm::StringRef TT,
                        llvm::StringRef OldDL) -> std::optional<std::string> {
      if (SetDataLayout)
        return SetDataLayout(TT, OldDL);
      return std::nullopt;
    };
    auto M = Parser->parseIRModule(DataLayoutCB);
    if (!M)
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to parse module '" +
                                        BufID.str() + "' as MIR");
    OutMIRParser = std::move(Parser);
    return M;
  }
  return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
      "Invalid module format {0}", static_cast<unsigned int>(Format)));
}

/// Patches cross-module \c MDNode references so that instrumentation-module
/// metadata points back into the live target module's uniqued
/// \c MDNode s.  Uses \c MapMetadata (\c ValueMapper) rather than
/// \c replaceAllUsesWith because uniqued nodes can't be RAUW'd.
void patchIModuleMDNodeReferences(
    llvm::Module &IModule, llvm::Module &TargetModule,
    llvm::ArrayRef<LuthierFileParser::MDSlotEntry> MDSlotMap,
    MFAccessor GetTargetMF, MFAccessor GetIModuleMF) {
  if (MDSlotMap.empty())
    return;

  auto TargetSlotToMD = buildMDNodeSlots(TargetModule, GetTargetMF).SlotToMD;
  auto IModuleSlotToMD = buildMDNodeSlots(IModule, GetIModuleMF).SlotToMD;

  llvm::ValueToValueMapTy VM;
  for (auto &[IModSlot, TgtSlot] : MDSlotMap) {
    llvm::MDNode *IMD = IModuleSlotToMD.lookup(IModSlot);
    llvm::MDNode *TMD = TargetSlotToMD.lookup(TgtSlot);
    if (IMD && TMD && IMD != TMD)
      VM.MD()[IMD].reset(TMD);
  }
  if (VM.MD().empty())
    return;

  auto remapAttachments = [&](llvm::GlobalObject &GO) {
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> Attachments;
    GO.getAllMetadata(Attachments);
    for (auto &[KindID, MD] : Attachments)
      if (auto *NewMD =
              llvm::MapMetadata(MD, VM, llvm::RF_NoModuleLevelChanges))
        GO.setMetadata(KindID, NewMD);
  };

  for (llvm::Function &F : IModule) {
    remapAttachments(F);
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB) {
        llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> IMDs;
        I.getAllMetadata(IMDs);
        for (auto &[KindID, MD] : IMDs)
          if (auto *NewMD =
                  llvm::MapMetadata(MD, VM, llvm::RF_NoModuleLevelChanges))
            I.setMetadata(KindID, NewMD);
      }
  }

  for (llvm::GlobalVariable &GV : IModule.globals())
    remapAttachments(GV);

  for (llvm::NamedMDNode &NMD : IModule.named_metadata())
    for (unsigned I = 0, E = NMD.getNumOperands(); I != E; ++I)
      if (auto *NewMD = llvm::MapMetadata(NMD.getOperand(I), VM,
                                          llvm::RF_NoModuleLevelChanges))
        NMD.setOperand(I, NewMD);
}

/// Returns true if any \c Function in \p M has a cached
/// \c MachineFunctionAnalysis result on \p FAM.
bool moduleHasCachedMIR(llvm::Module &M, llvm::FunctionAnalysisManager &FAM) {
  for (llvm::Function &F : M)
    if (FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F))
      return true;
  return false;
}

/// Serializes \p M into \p Out using the MIR format, iterating over any
/// cached \c MachineFunctionAnalysis results on \p FAM.
void serializeModuleAsMIR(llvm::Module &M, llvm::FunctionAnalysisManager &FAM,
                          std::string &Out) {
  llvm::raw_string_ostream SS(Out);
  llvm::printMIR(SS, M);
  for (llvm::Function &F : M) {
    auto *MFRes = FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
    if (!MFRes)
      continue;
    llvm::printMIR(SS, FAM, MFRes->getMF());
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// LuthierFileParser
//===----------------------------------------------------------------------===//

llvm::Expected<LuthierFileParser>
LuthierFileParser::create(llvm::MemoryBufferRef Buffer) {
  LuthierFileYaml Y;
  llvm::yaml::Input YIN(Buffer.getBuffer());
  YIN >> Y;
  if (YIN.error())
    return llvm::createStringError(
        YIN.error(), "YAML parse error in .luthier file '" +
                         Buffer.getBufferIdentifier().str() + "'");
  LuthierFileParser P;
  P.Identifier = Buffer.getBufferIdentifier();
  P.TargetModuleText = std::move(Y.TargetModuleText.S);
  P.TargetModuleFormat = Y.TargetModuleFormat;
  P.InstrumentationModuleText = std::move(Y.InstrumentationModuleText.S);
  P.InstrumentationModuleFormat = Y.InstrumentationModuleFormat;
  P.MDSlotMap = std::move(Y.MDSlotMap);
  return P;
}

llvm::Expected<LuthierFileParser>
LuthierFileParser::create(llvm::StringRef Path) {
  auto MBOrErr = llvm::MemoryBuffer::getFile(Path);
  if (!MBOrErr)
    return llvm::createStringError(MBOrErr.getError(),
                                   "Failed to open .luthier file '" +
                                       Path.str() + "'");
  return create((*MBOrErr)->getMemBufferRef());
}

llvm::Expected<std::unique_ptr<Prototype>> LuthierFileParser::loadPrototype(
    llvm::LLVMContext &Ctx,
    const std::function<std::optional<std::string>(
        llvm::StringRef, llvm::StringRef)> &SetDataLayout,
    const std::function<void(llvm::Function &)> &SetMIRFunctionAttributes) {
  auto TargetMOrErr =
      parseOneModule(Ctx, TargetModuleText, Identifier, TargetModuleFormat,
                     SetDataLayout, SetMIRFunctionAttributes, TargetMIRParser);
  if (!TargetMOrErr)
    return TargetMOrErr.takeError();
  std::unique_ptr<llvm::Module> TargetM = std::move(*TargetMOrErr);

  auto IModuleMOrErr = parseOneModule(
      Ctx, InstrumentationModuleText, Identifier + ".instrumentation_module",
      InstrumentationModuleFormat, SetDataLayout, SetMIRFunctionAttributes,
      InstrumentationMIRParser);
  if (!IModuleMOrErr)
    return IModuleMOrErr.takeError();
  std::unique_ptr<llvm::Module> IModuleM = std::move(*IModuleMOrErr);

  /// Re-linking the two modules' shared MDNodes is deliberately deferred to
  /// loadMIR: the nodes that need re-linking hang off MachineInstrs, which do
  /// not exist until the MIR parser has run.

  return std::make_unique<Prototype>(std::move(TargetM), std::move(IModuleM));
}

llvm::Error LuthierFileParser::loadMIR(Prototype &P,
                                       PrototypeAnalysisManager &PAM) {
  llvm::Module &TargetModule = P.getTargetModule();
  llvm::Module &IModule = P.getInstrumentationModule();
  if (!PAM.isPassRegistered<ModuleAnalysisManagerPrototypeProxy>()) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Module analysis manager prototype proxy is not registered");
  }
  llvm::ModuleAnalysisManager &MAM =
      PAM.getResult<ModuleAnalysisManagerPrototypeProxy>(P).getManager();

  if (TargetMIRParser &&
      TargetMIRParser->parseMachineFunctions(TargetModule, MAM)) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to parse the target module machine functions");
  }
  if (InstrumentationMIRParser &&
      InstrumentationMIRParser->parseMachineFunctions(IModule, MAM)) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to parse the instrumentation module machine functions");
  }

  /// Now that both modules' MIR exists, restore the MDNodes the two of them
  /// shared before serialization — instrumentation-point tags above all. This
  /// has to happen here rather than at parse time because those tags are reached
  /// through MachineInstrs.
  llvm::FunctionAnalysisManager &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();
  auto GetMF = [&FAM](llvm::Function &F) -> llvm::MachineFunction * {
    auto *MFRes = FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
    return MFRes ? &MFRes->getMF() : nullptr;
  };
  patchIModuleMDNodeReferences(IModule, TargetModule, MDSlotMap, GetMF, GetMF);

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// writeLuthierFile
//===----------------------------------------------------------------------===//

llvm::Error writeLuthierFile(llvm::raw_ostream &OS, Prototype &IP,
                             PrototypeAnalysisManager &IPAM) {
  LuthierFileYaml Y;
  llvm::Module &TargetModule = IP.getTargetModule();
  llvm::Module &IModule = IP.getInstrumentationModule();

  if (!IPAM.isPassRegistered<FunctionAnalysisManagerPrototypeProxy>())
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Function analysis manager prototype proxy is not registered");

  llvm::FunctionAnalysisManager &FAM =
      IPAM.getResult<FunctionAnalysisManagerPrototypeProxy>(IP).getManager();

  // Target module: MIR if any function has a cached MFA, else IR text.
  if (moduleHasCachedMIR(TargetModule, FAM)) {
    Y.TargetModuleFormat = LuthierFileParser::ModuleFormat::MIR;
    serializeModuleAsMIR(TargetModule, FAM, Y.TargetModuleText.S);
  } else {
    Y.TargetModuleFormat = LuthierFileParser::ModuleFormat::IR;
    llvm::raw_string_ostream SS(Y.TargetModuleText.S);
    TargetModule.print(SS, nullptr);
  }

  // Instrumentation module: same test.
  if (moduleHasCachedMIR(IModule, FAM)) {
    Y.InstrumentationModuleFormat = LuthierFileParser::ModuleFormat::MIR;
    serializeModuleAsMIR(IModule, FAM, Y.InstrumentationModuleText.S);
  } else {
    Y.InstrumentationModuleFormat = LuthierFileParser::ModuleFormat::IR;
    llvm::raw_string_ostream SS(Y.InstrumentationModuleText.S);
    IModule.print(SS, nullptr);
  }

  // Record MDNode slot pairs shared between both modules so that loadMIR() can
  // restore the cross-module links on reload. Serializing the two modules
  // separately turns a node they share into two independent nodes — and the
  // nodes that matter here are `distinct`, so re-parsing cannot re-unify them
  // either. Without this map an instrumentation point survives as a payload
  // attachment and as a target pcsections tag that no longer compare equal, and
  // InjectedPayloadAndInstPointAnalysis (which matches on pointer identity) sees
  // no payloads at all.
  //
  // The walk has to reach MachineInstr attachments on both sides, since a
  // pcsections tag has no IR user.
  auto GetMF = [&FAM](llvm::Function &F) -> llvm::MachineFunction * {
    auto *MFRes = FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
    return MFRes ? &MFRes->getMF() : nullptr;
  };
  auto TargetSlots = buildMDNodeSlots(TargetModule, GetMF);
  auto IModuleSlots = buildMDNodeSlots(IModule, GetMF);

  for (auto &[IModSlot, MD] : IModuleSlots.SlotToMD) {
    auto It = TargetSlots.MDToSlot.find(MD);
    if (It != TargetSlots.MDToSlot.end())
      Y.MDSlotMap.push_back({IModSlot, It->second});
  }
  // DenseMap iteration order is unspecified; keep the file deterministic.
  llvm::sort(Y.MDSlotMap, [](const LuthierFileParser::MDSlotEntry &A,
                             const LuthierFileParser::MDSlotEntry &B) {
    return A.IModuleSlot < B.IModuleSlot;
  });

  llvm::yaml::Output Yout(OS);
  Yout << Y;
  return llvm::Error::success();
}

} // namespace luthier
