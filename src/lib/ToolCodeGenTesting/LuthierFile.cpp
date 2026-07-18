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
/// \file LuthierFile.cpp
/// Implements \c LuthierFileParser and the \c writeLuthierFile helpers.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGenTesting/LuthierFile.h"
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
  static void enumeration(IO &IO,
                          luthier::LuthierFileParser::ModuleFormat &F) {
    IO.enumCase(F, "IR", luthier::LuthierFileParser::ModuleFormat::IR);
    IO.enumCase(F, "Bitcode",
                luthier::LuthierFileParser::ModuleFormat::Bitcode);
    IO.enumCase(F, "MIR", luthier::LuthierFileParser::ModuleFormat::MIR);
  }
};

template <> struct MappingTraits<luthier::LuthierFileYaml> {
  static void mapping(IO &IO, luthier::LuthierFileYaml &F) {
    IO.mapRequired("TargetModule", F.TargetModuleText);
    IO.mapOptional("TargetModuleFormat", F.TargetModuleFormat,
                   luthier::LuthierFileParser::ModuleFormat::MIR);
    IO.mapRequired("InstrumentationModule", F.InstrumentationModuleText);
    IO.mapOptional("InstrumentationModuleFormat",
                   F.InstrumentationModuleFormat,
                   luthier::LuthierFileParser::ModuleFormat::IR);
    IO.mapOptional("MDSlotMap", F.MDSlotMap,
                   std::vector<luthier::LuthierFileParser::MDSlotEntry>{});
  }
};

} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(luthier::LuthierFileParser::MDSlotEntry)

namespace luthier {

namespace {

/// Returns a map from metadata slot number to \c MDNode* for every metadata
/// node reachable from \p M. The slot numbers match the IR printer's
/// assignment because both use \c ModuleSlotTracker
llvm::DenseMap<unsigned, llvm::MDNode *> buildSlotToMDNodeMap(llvm::Module &M) {
  llvm::ModuleSlotTracker MST(&M, /*ShouldInitializeAllMetadata=*/true);
  llvm::ModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMDNodes(MDList, 0, ~0u);
  llvm::DenseMap<unsigned, llvm::MDNode *> Out;
  Out.reserve(MDList.size());
  for (auto &[Slot, MD] : MDList)
    Out[Slot] = const_cast<llvm::MDNode *>(MD);
  return Out;
}

/// Parses one module out of a text blob according to its \c ModuleFormat.
/// Populates the corresponding \c MIRParser out-parameter for MIR-form
/// modules (left null otherwise).
llvm::Expected<std::unique_ptr<llvm::Module>>
parseOneModule(llvm::StringRef Text, llvm::StringRef BufID,
               LuthierFileParser::ModuleFormat Format, llvm::LLVMContext &Ctx,
               std::function<std::optional<std::string>(llvm::StringRef,
                                                        llvm::StringRef)>
                   SetDataLayout,
               std::function<void(llvm::Function &)> SetMIRFunctionAttributes,
               std::unique_ptr<llvm::MIRParser> &OutMIRParser) {
  llvm::ParserCallbacks IRCallbacks;
  if (SetDataLayout)
    IRCallbacks.DataLayout = [SetDataLayout](llvm::StringRef TT,
                                             llvm::StringRef OldDL)
        -> std::optional<std::string> { return SetDataLayout(TT, OldDL); };

  switch (Format) {
  case LuthierFileParser::ModuleFormat::IR: {
    llvm::SMDiagnostic Err;
    llvm::MemoryBufferRef Buf(Text, BufID);
    auto M = llvm::parseIR(Buf, Err, Ctx, IRCallbacks);
    if (!M)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Failed to parse module '" + BufID.str() +
          "' as IR: " + Err.getMessage().str());
    return M;
  }
  case LuthierFileParser::ModuleFormat::MIR: {
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
  case LuthierFileParser::ModuleFormat::Bitcode: {
    std::vector<char> Decoded;
    if (auto Err = llvm::decodeBase64(Text, Decoded))
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Failed to base64-decode module '" + BufID.str() +
          "': " + llvm::toString(std::move(Err)));
    auto DecodedBuf = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(Decoded.data(), Decoded.size()), BufID,
        /*RequiresNullTerminator=*/false);
    std::unique_ptr<llvm::Module> M;
    llvm::Error Err =
        llvm::parseBitcodeFile(DecodedBuf->getMemBufferRef(), Ctx).moveInto(M);
    if (Err)
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to parse module '" +
                                        BufID.str() + "' as bitcode: " +
                                        llvm::toString(std::move(Err)));
    return M;
  }
  }
  llvm_unreachable("unhandled ModuleFormat");
}

/// Patches cross-module \c MDNode references so that instrumentation-module
/// metadata points back into the live target module's uniqued
/// \c MDNode s.  Uses \c MapMetadata (\c ValueMapper) rather than
/// \c replaceAllUsesWith because uniqued nodes can't be RAUW'd.
void patchIModuleMDNodeReferences(
    llvm::Module &IModule, llvm::Module &TargetModule,
    llvm::ArrayRef<LuthierFileParser::MDSlotEntry> MDSlotMap) {
  if (MDSlotMap.empty())
    return;

  auto TargetSlotToMD = buildSlotToMDNodeMap(TargetModule);
  auto IModuleSlotToMD = buildSlotToMDNodeMap(IModule);

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

/// Fetches the \c FunctionAnalysisManager reachable from \p IPAM through the
/// cross-level proxies wired up by
/// \c PrototypePassBuilder::crossRegisterProxies.  Returns null if
/// the
/// proxy has not been registered yet.
llvm::FunctionAnalysisManager *
getFAM(Prototype &IP, PrototypeAnalysisManager &IPAM) {
  auto *Proxy =
      IPAM.getCachedResult<FunctionAnalysisManagerPrototypeProxy>(IP);
  if (Proxy)
    return &Proxy->getManager();
  // Force-construct the proxy result if it has been registered but not yet
  // materialized.
  return &IPAM
              .getResult<FunctionAnalysisManagerPrototypeProxy>(IP)
              .getManager();
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

llvm::Expected<LoadedPrototype> LuthierFileParser::load(
    llvm::LLVMContext &Ctx, PrototypeAnalysisManager & /*IPAM*/,
    std::function<std::optional<std::string>(llvm::StringRef, llvm::StringRef)>
        SetDataLayout,
    std::function<void(llvm::Function &)> SetMIRFunctionAttributes) const {
  std::unique_ptr<llvm::MIRParser> TargetMIRParser;
  auto TargetMOrErr = parseOneModule(
      TargetModuleText, "<luthier target>", TargetModuleFormat, Ctx,
      SetDataLayout, SetMIRFunctionAttributes, TargetMIRParser);
  if (!TargetMOrErr)
    return TargetMOrErr.takeError();
  std::unique_ptr<llvm::Module> TargetM = std::move(*TargetMOrErr);

  std::unique_ptr<llvm::MIRParser> IModuleMIRParser;
  auto IModuleMOrErr =
      parseOneModule(InstrumentationModuleText, "<luthier imodule>",
                     InstrumentationModuleFormat, Ctx,
                     /*SetDataLayout=*/nullptr, SetMIRFunctionAttributes,
                     IModuleMIRParser);
  if (!IModuleMOrErr)
    return IModuleMOrErr.takeError();
  std::unique_ptr<llvm::Module> IModuleM = std::move(*IModuleMOrErr);

  patchIModuleMDNodeReferences(*IModuleM, *TargetM, MDSlotMap);

  LoadedPrototype Out;
  Out.IP = std::make_unique<Prototype>(std::move(TargetM),
                                                 std::move(IModuleM));
  Out.TargetMIRParser = std::move(TargetMIRParser);
  Out.IModuleMIRParser = std::move(IModuleMIRParser);
  return Out;
}

llvm::Expected<std::pair<std::unique_ptr<llvm::Module>,
                         std::unique_ptr<llvm::MIRParser>>>
LuthierFileParser::loadIModule(llvm::LLVMContext &Ctx,
                               llvm::Module &TargetModule) const {
  std::unique_ptr<llvm::MIRParser> IModuleMIRParser;
  auto MOrErr =
      parseOneModule(InstrumentationModuleText, "<luthier imodule>",
                     InstrumentationModuleFormat, Ctx,
                     /*SetDataLayout=*/nullptr,
                     /*SetMIRFunctionAttributes=*/nullptr, IModuleMIRParser);
  if (!MOrErr)
    return MOrErr.takeError();
  std::unique_ptr<llvm::Module> M = std::move(*MOrErr);
  patchIModuleMDNodeReferences(*M, TargetModule, MDSlotMap);
  return std::make_pair(std::move(M), std::move(IModuleMIRParser));
}

//===----------------------------------------------------------------------===//
// writeLuthierFile
//===----------------------------------------------------------------------===//

llvm::Error writeLuthierFile(llvm::raw_ostream &OS, Prototype &IP,
                             PrototypeAnalysisManager &IPAM) {
  LuthierFileYaml Y;
  llvm::Module &TargetModule = IP.getTargetModule();
  llvm::Module &IModule = IP.getInstrumentationModule();

  llvm::FunctionAnalysisManager *FAM = getFAM(IP, IPAM);

  // Target module: MIR if any function has a cached MFA, else IR text.
  if (FAM && moduleHasCachedMIR(TargetModule, *FAM)) {
    Y.TargetModuleFormat = LuthierFileParser::ModuleFormat::MIR;
    serializeModuleAsMIR(TargetModule, *FAM, Y.TargetModuleText.S);
  } else {
    Y.TargetModuleFormat = LuthierFileParser::ModuleFormat::IR;
    llvm::raw_string_ostream SS(Y.TargetModuleText.S);
    TargetModule.print(SS, nullptr);
  }

  // Instrumentation module: same test.
  if (FAM && moduleHasCachedMIR(IModule, *FAM)) {
    Y.InstrumentationModuleFormat = LuthierFileParser::ModuleFormat::MIR;
    serializeModuleAsMIR(IModule, *FAM, Y.InstrumentationModuleText.S);
  } else {
    Y.InstrumentationModuleFormat = LuthierFileParser::ModuleFormat::IR;
    llvm::raw_string_ostream SS(Y.InstrumentationModuleText.S);
    IModule.print(SS, nullptr);
  }

  // Record MDNode slot pairs shared between both modules so that load()
  // can restore the cross-module links on reload.
  auto TargetSlotToMD = buildSlotToMDNodeMap(TargetModule);
  auto IModuleSlotToMD = buildSlotToMDNodeMap(IModule);

  llvm::DenseMap<const llvm::MDNode *, unsigned> TargetMDToSlot;
  TargetMDToSlot.reserve(TargetSlotToMD.size());
  for (auto &[Slot, MD] : TargetSlotToMD)
    TargetMDToSlot[MD] = Slot;

  for (auto &[IModSlot, MD] : IModuleSlotToMD) {
    auto It = TargetMDToSlot.find(MD);
    if (It != TargetMDToSlot.end())
      Y.MDSlotMap.push_back({IModSlot, It->second});
  }

  llvm::yaml::Output Yout(OS);
  Yout << Y;
  return llvm::Error::success();
}

llvm::Error writeLuthierFile(llvm::StringRef Path, Prototype &IP,
                             PrototypeAnalysisManager &IPAM) {
  std::error_code EC;
  llvm::ToolOutputFile OutFile(Path, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return LUTHIER_MAKE_GENERIC_ERROR("Failed to open .luthier output file '" +
                                      Path.str() + "': " + EC.message());
  if (auto Err = writeLuthierFile(OutFile.os(), IP, IPAM))
    return Err;
  OutFile.keep();
  return llvm::Error::success();
}

llvm::Error writeLuthierFile(llvm::StringRef Path, llvm::Module &TargetModule,
                             llvm::Module &IModule) {
  LuthierFileYaml Y;
  Y.TargetModuleFormat = LuthierFileParser::ModuleFormat::IR;
  {
    llvm::raw_string_ostream SS(Y.TargetModuleText.S);
    TargetModule.print(SS, nullptr);
  }
  Y.InstrumentationModuleFormat = LuthierFileParser::ModuleFormat::IR;
  {
    llvm::raw_string_ostream SS(Y.InstrumentationModuleText.S);
    IModule.print(SS, nullptr);
  }

  auto TargetSlotToMD = buildSlotToMDNodeMap(TargetModule);
  auto IModuleSlotToMD = buildSlotToMDNodeMap(IModule);

  llvm::DenseMap<const llvm::MDNode *, unsigned> TargetMDToSlot;
  TargetMDToSlot.reserve(TargetSlotToMD.size());
  for (auto &[Slot, MD] : TargetSlotToMD)
    TargetMDToSlot[MD] = Slot;

  for (auto &[IModSlot, MD] : IModuleSlotToMD) {
    auto It = TargetMDToSlot.find(MD);
    if (It != TargetMDToSlot.end())
      Y.MDSlotMap.push_back({IModSlot, It->second});
  }

  std::error_code EC;
  llvm::ToolOutputFile OutFile(Path, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return LUTHIER_MAKE_GENERIC_ERROR("Failed to open .luthier output file '" +
                                      Path.str() + "': " + EC.message());
  llvm::yaml::Output Yout(OutFile.os());
  Yout << Y;
  OutFile.keep();
  return llvm::Error::success();
}

} // namespace luthier
