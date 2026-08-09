//===-- MockLoadAMDGPUCodeObjects.cpp -------------------------------------===//
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
/// \file MockLoadAMDGPUCodeObjects.cpp
/// Implements the \c MockLoadAMDGPUCodeObjects class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MockLoadAMDGPUCodeObjects.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/CodeObjectManagerAnalysis.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/InitialExecutionPointAnalysis.h"
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-mock-load-amdgpu-code-objects"

namespace luthier {

bool MockAMDGPULoaderExternalVarParser::parse(
    llvm::cl::Option &O, llvm::StringRef ArgName, llvm::StringRef ArgValue,
    std::pair<std::string, uint64_t> &Val) {
  auto [ExternVarName, Addr] = ArgValue.split(':');
  Val.first = ExternVarName.str();
  if (Addr.getAsInteger(0, Val.second)) {
    return O.error("Failed to parse the address for variable " + Val.first +
                   ".");
  }
  return false;
}

bool MockAMDGPULoaderInitialEntryPointParser::parse(
    llvm::cl::Option &O, llvm::StringRef ArgName, llvm::StringRef ArgValue,
    MockAMDGPULoaderEntryPointSpec &Val) {
  auto [CodeObjectIndexStr, SymbolOrOffset] = ArgValue.split(':');
  if (CodeObjectIndexStr.getAsInteger(0, Val.first)) {
    return O.error("Failed to parse the code object index for " +
                   llvm::Twine(CodeObjectIndexStr) + ".");
  }
  uint64_t LoadOffset;
  if (SymbolOrOffset.getAsInteger(0, LoadOffset)) {
    Val.second = std::string(SymbolOrOffset);
  } else {
    Val.second = LoadOffset;
  }
  return false;
}

bool MockAMDGPULoaderInitialExecutionPointParser::parse(
    llvm::cl::Option &O, llvm::StringRef ArgName, llvm::StringRef ArgValue,
    std::pair<uint64_t, std::string> &Val) {
  auto [CodeObjectIndexStr, Symbol] = ArgValue.split(':');
  if (CodeObjectIndexStr.getAsInteger(0, Val.first)) {
    return O.error("Failed to parse the code object index for " +
                   llvm::Twine(CodeObjectIndexStr) + ".");
  }
  Val.second = Symbol;
  return false;
}

namespace {

/// \return the loaded code object at \p Index in \p Loader, or an error if
/// \p Index is out of range
llvm::Expected<const MockLoadedCodeObject *>
getLoadedCodeObjectAtIndex(const MockAMDGPULoader &Loader, uint64_t Index) {
  uint64_t CodeObjectIdx = 0;
  for (const auto &LCO : Loader.loaded_code_objects()) {
    if (CodeObjectIdx == Index)
      return &LCO;
    CodeObjectIdx++;
  }
  return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
      "Code object index {0} is out of range; the mock loader loaded {1} "
      "code object(s)",
      Index, CodeObjectIdx));
}

/// \return the loaded address of \p SymbolName within \p LCO, together with
/// whether the symbol is a kernel descriptor
llvm::Expected<std::pair<uint64_t, bool>>
resolveSymbol(const MockLoadedCodeObject &LCO, llvm::StringRef SymbolName,
              uint64_t CodeObjectIdx) {
  std::optional<object::AMDGCNElfSymbolRef> Symbol{std::nullopt};
  if (llvm::Error Err =
          LCO.getCodeObject().lookupSymbol(SymbolName).moveInto(Symbol))
    return std::move(Err);

  if (!Symbol.has_value())
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Failed to find the symbol {0} in code object index {1}",
                      SymbolName, CodeObjectIdx));

  uint64_t LoadOffset;
  if (llvm::Error Err = Symbol->getAddress().moveInto(LoadOffset))
    return std::move(Err);

  if (LoadOffset >= LCO.getLoadedRegion().size())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Symbol {0} resolves to offset {1:x}, outside code object index {2}",
        SymbolName, LoadOffset, CodeObjectIdx));

  /// isKernelDescriptor yields an Expected<bool>, so the value has to be
  /// unwrapped rather than tested directly — testing the Expected only asks
  /// whether the query succeeded.
  bool IsKernelDescriptor;
  if (llvm::Error Err =
          Symbol->isKernelDescriptor().moveInto(IsKernelDescriptor))
    return std::move(Err);

  return std::make_pair(
      reinterpret_cast<uint64_t>(LCO.getLoadedRegion().data()) + LoadOffset,
      IsKernelDescriptor);
}

/// Resolves \p Spec against \p Loader into an \c EntryPoint.
llvm::Expected<EntryPoint>
resolveEntryPoint(const MockAMDGPULoader &Loader,
                  const MockAMDGPULoaderEntryPointSpec &Spec) {
  const MockLoadedCodeObject *LCO = nullptr;
  if (llvm::Error Err =
          getLoadedCodeObjectAtIndex(Loader, Spec.first).moveInto(LCO))
    return std::move(Err);

  if (std::holds_alternative<uint64_t>(Spec.second)) {
    uint64_t LoadOffset = std::get<uint64_t>(Spec.second);
    if (LoadOffset > LCO->getLoadedRegion().size())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Offset {0:x} is outside the range of code object index {1}",
          LoadOffset, Spec.first));
    return EntryPoint{
        reinterpret_cast<uint64_t>(LCO->getLoadedRegion().data()) + LoadOffset};
  }

  std::pair<uint64_t, bool> Resolved;
  if (llvm::Error Err = resolveSymbol(*LCO, std::get<std::string>(Spec.second),
                                      Spec.first)
                            .moveInto(Resolved))
    return std::move(Err);

  auto [LoadAddr, IsKernelDescriptor] = Resolved;
  if (IsKernelDescriptor)
    return EntryPoint(
        *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(LoadAddr));
  return EntryPoint(LoadAddr);
}

} // namespace

MockLoadAMDGPUCodeObjects::MockLoadAMDGPUCodeObjects(
    MockAMDGPULoaderAnalysisOptions &Options)
    : Options(Options) {}

llvm::PreservedAnalyses
MockLoadAMDGPUCodeObjects::run(llvm::Module &M,
                               llvm::ModuleAnalysisManager &MAM) {
  llvm::LLVMContext &Ctx = M.getContext();
  /// Get the mock loader analysis and the code object analysis
  MockAMDGPULoader &Loader =
      MAM.getResult<MockAMDGPULoaderAnalysis>(M).getLoader();
  CodeObjectManagerAnalysis::Result CodeObjectManager =
      MAM.getResult<CodeObjectManagerAnalysis>(M);

  LLVM_DEBUG(luthier::dbgs()
             << "[MockLoadAMDGPUCodeObjects] "
             << Options.CodeObjectPathList.size() << " code object path(s), "
             << Options.ExternalVars.size()
             << " external variable definition(s)\n");

  /// Go over the code object paths and create buffers for each of them
  for (llvm::StringRef Path : Options.CodeObjectPathList) {
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Reading code object from "
               << Path << "\n");
    llvm::Expected<llvm::MemoryBuffer &> CodeObjectBufferOrErr =
        CodeObjectManager.readCodeObjectFromFile(Path);
    if (auto Err = CodeObjectBufferOrErr.takeError()) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto LoadedCodeObjectOrErr = Loader.loadCodeObject(*CodeObjectBufferOrErr);
    if (auto Err = LoadedCodeObjectOrErr.takeError()) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Loaded " << Path
               << " (load base "
               << llvm::format_hex(
                      reinterpret_cast<uint64_t>(
                          LoadedCodeObjectOrErr->getLoadedRegion().data()),
                      18)
               << ", size " << LoadedCodeObjectOrErr->getLoadedRegion().size()
               << ")\n");
  }

  /// Define the external variables
  for (auto &[SymName, SymAddr] : Options.ExternalVars) {
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Defining external "
                  "symbol "
               << SymName << " at " << llvm::format_hex(SymAddr, 18) << "\n");
    if (auto Err = Loader.defineExternalSymbol(
            SymName, reinterpret_cast<void *>(SymAddr))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
  }

  /// Finalize the loader
  LLVM_DEBUG(luthier::dbgs()
             << "[MockLoadAMDGPUCodeObjects] Finalizing loader ("
             << Loader.loaded_code_objects_size() << " LCO(s), "
             << Loader.external_symbol_size() << " external symbol(s))\n");
  if (auto Err = Loader.finalize()) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return llvm::PreservedAnalyses::all();
  }

  /// Record the initial entry and execution points on the module. Their
  /// spelling on the command line is loader-relative, so resolving them is this
  /// pass's business; from here on they are plain module metadata that
  /// InitialEntryPointAnalysis / InitialExecutionPointAnalysis parse without
  /// any knowledge of the loader.
  if (Options.InitialEntryPoint.getNumOccurrences() > 0) {
    EntryPoint EP;
    if (llvm::Error Err =
            resolveEntryPoint(Loader, Options.InitialEntryPoint).moveInto(EP)) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Initial entry point at "
               << llvm::format_hex(EP.getRawAddress(), 18)
               << (EP.isKernel() ? " (kernel descriptor)\n" : "\n"));
    setInitialEntryPoint(M, EP);
  }

  if (Options.InitialExecutionPoint.getNumOccurrences() > 0) {
    const std::pair<uint64_t, std::string> &Spec =
        Options.InitialExecutionPoint;
    const auto &[CodeObjectIdx, SymbolName] = Spec;
    const MockLoadedCodeObject *LCO = nullptr;
    if (llvm::Error Err =
            getLoadedCodeObjectAtIndex(Loader, CodeObjectIdx).moveInto(LCO)) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    std::pair<uint64_t, bool> Resolved;
    if (llvm::Error Err =
            resolveSymbol(*LCO, SymbolName, CodeObjectIdx).moveInto(Resolved)) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    auto [LoadAddr, IsKernelDescriptor] = Resolved;
    if (!IsKernelDescriptor) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
          "Initial execution point is not a kernel symbol")));
      return llvm::PreservedAnalyses::all();
    }
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Initial execution point at "
               << llvm::format_hex(LoadAddr, 18) << "\n");
    setInitialExecutionPoint(
        M, *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
               LoadAddr));
  }

  return llvm::PreservedAnalyses::all();
};

} // namespace luthier