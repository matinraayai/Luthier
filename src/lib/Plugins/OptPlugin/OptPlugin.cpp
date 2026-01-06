//===-- OptPlugin.cpp -----------------------------------------------------===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// Main file for the Luthier "opt" compiler plugin, which registers Luthier
/// passes and their names with the new pass manager's pass builder when loaded.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/AMDGPURegisterLiveness.h"
#include "luthier/Tooling/CodeObjectManagerAnalysis.h"
#include "luthier/Tooling/InstrumentationPMDriver.h"
#include "luthier/Tooling/IntrinsicMIRLoweringPass.h"
#include "luthier/Tooling/LRCallgraph.h"
#include "luthier/Tooling/MMISlotIndexesAnalysis.h"
#include "luthier/Tooling/MockAMDGPULoader.h"
#include "luthier/Tooling/MockLoadAMDGPUCodeObjects.h"
#include "luthier/Tooling/PhysRegsNotInLiveInsAnalysis.h"
#include "luthier/Tooling/PrePostAmbleEmitter.h"
#include "luthier/Tooling/SVStorageAndLoadLocations.h"
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Plugins/PassPlugin.h>
#include <luthier/Tooling/CodeDiscoveryPass.h>
#include <luthier/Tooling/InitialEntryPointAnalysis.h>
#include <luthier/Tooling/InstructionTracesAnalysis.h>
#include <luthier/Tooling/MetadataParserAnalysis.h>

namespace luthier {

static std::unique_ptr<MockAMDGPULoader> Loader{nullptr};

static InstrumentationPMDriverOptions InstrumentationPMOptions;

static amdgpu::hsamd::MetadataParser MetadataParser;

static MockAMDGPULoaderAnalysisOptions MockLoaderOptions;

struct MockAMDGPULoaderInitialEntryPointParser
    : public llvm::cl::parser<
          std::pair<uint64_t, std::variant<uint64_t, std::string>>> {
  MockAMDGPULoaderInitialEntryPointParser(llvm::cl::Option &O)
      : llvm::cl::parser<
            std::pair<uint64_t, std::variant<uint64_t, std::string>>>(O) {}

  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef ArgValue,
             std::pair<uint64_t, std::variant<uint64_t, std::string>> &Val) {
    auto [CodeObjectIndexStr, SymbolOrOffset] = ArgValue.split(':');
    if (CodeObjectIndexStr.getAsInteger(0, Val.first)) {
      return O.error("Failed to parse the code object index for " +
                     llvm::Twine(Val.first) + ".");
    }
    uint64_t LoadOffset;
    if (SymbolOrOffset.getAsInteger(0, LoadOffset)) {
      Val.second = LoadOffset;
    } else {
      Val.second = std::string(SymbolOrOffset);
    }

    return false;
  }
};

llvm::cl::opt<std::pair<uint64_t, std::variant<uint64_t, std::string>>, false,
              MockAMDGPULoaderInitialEntryPointParser>
    InitialEntryPoint{
        "initial-entrypoint",
        llvm::cl::desc("The initial entry point of the lifting process. "
                       "Formatted as <code-object-index>:<symbol-name> or "
                       "<code-object-index>:<load-offset>. Code objects are "
                       "zero indexed w.r.t the order they are specified to be "
                       "loaded into the mock loader"),
        llvm::cl::NotHidden};
}; // namespace luthier

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {

  const auto Callback = [](llvm::PassBuilder &PB) {
    /// Register Luthier module analysis passes
    PB.registerAnalysisRegistrationCallback([](llvm::ModuleAnalysisManager
                                                   &MAM) {
      MAM.registerPass([&]() {
        return luthier::InitialEntryPointAnalysis(
            [&](llvm::Module &M,
                llvm::ModuleAnalysisManager &MAM) -> luthier::EntryPoint {
              llvm::LLVMContext &Ctx = M.getContext();
              const auto &MockLoader =
                  MAM.getResult<luthier::MockAMDGPULoaderAnalysis>(M)
                      .getLoader();
              uint64_t CodeObjectIdx = 0;
              /// Find the
              for (const auto &LCO : MockLoader.loaded_code_objects()) {
                if (CodeObjectIdx == luthier::InitialEntryPoint.first) {
                  if (std::holds_alternative<uint64_t>(
                          luthier::InitialEntryPoint.second)) {
                    uint64_t LoadOffset =
                        std::get<uint64_t>(luthier::InitialEntryPoint.second);
                    if (LoadOffset > LCO.getLoadedRegion().size()) {
                      LUTHIER_CTX_EMIT_ON_ERROR(
                          Ctx, LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                                   "Offset {0:x} is outside the "
                                   "range of code object index {1}",
                                   LoadOffset, CodeObjectIdx)));
                    }
                    return luthier::EntryPoint{
                        reinterpret_cast<uint64_t>(
                            LCO.getLoadedRegion().data()) +
                        LoadOffset};
                  } else {
                    std::string SymbolName = std::get<std::string>(
                        luthier::InitialEntryPoint.second);
                    std::optional<luthier::object::AMDGCNElfSymbolRef> Symbol{
                        std::nullopt};
                    llvm::Error Err = LCO.getCodeObject()
                                          .lookupSymbol(SymbolName)
                                          .moveInto(Symbol);
                    LUTHIER_CTX_EMIT_ON_ERROR(Ctx, Err);

                    if (!Symbol.has_value()) {
                      LUTHIER_CTX_EMIT_ON_ERROR(
                          Ctx, LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                                   "Failed to find the symbol {0} in "
                                   "code object index {1}",
                                   SymbolName, CodeObjectIdx)));
                    }
                    uint64_t LoadOffset;
                    Err = Symbol->getAddress().moveInto(LoadOffset);
                    assert(LoadOffset < LCO.getLoadedRegion().size() &&
                           "Load offset falls outside of the code object");
                    uint64_t LoadAddr = reinterpret_cast<uint64_t>(
                                            LCO.getLoadedRegion().data()) +
                                        LoadOffset;
                    LUTHIER_CTX_EMIT_ON_ERROR(Ctx, Err);
                    if (Symbol->isKernelDescriptor()) {
                      auto &KD = *reinterpret_cast<
                          const llvm::amdhsa::kernel_descriptor_t *>(LoadAddr);
                      return luthier::EntryPoint(KD);
                    } else {
                      return luthier::EntryPoint(LoadAddr);
                    }
                  }
                }
                CodeObjectIdx++;
              };
              LUTHIER_CTX_EMIT_ON_ERROR(
                  Ctx, LUTHIER_MAKE_GENERIC_ERROR(
                           "Failed to get the entry point; Code "
                           "object index is out of range"));
              llvm_unreachable("Should have thrown an error by now");
            });
      });
      MAM.registerPass([]() { return luthier::CodeObjectManagerAnalysis(); });
      MAM.registerPass([]() { return luthier::AMDGPURegLivenessAnalysis(); });
      MAM.registerPass([]() { return luthier::LRCallGraphAnalysis(); });
      MAM.registerPass([]() { return luthier::MMISlotIndexesAnalysis(); });
      MAM.registerPass([]() {
        return luthier::LRStateValueStorageAndLoadLocationsAnalysis();
      });
      MAM.registerPass(
          []() { return luthier::FunctionPreambleDescriptorAnalysis(); });
      MAM.registerPass([]() { return luthier::MockAMDGPULoaderAnalysis(); });
      MAM.registerPass([&]() {
        return luthier::MetadataParserAnalysis(luthier::MetadataParser);
      });
    });
    /// Register Luthier machine function analysis passes
    PB.registerAnalysisRegistrationCallback(
        [](llvm::MachineFunctionAnalysisManager &MFAM) {
          MFAM.registerPass(
              []() { return luthier::InstructionTracesAnalysis(); });
          MFAM.registerPass(
              []() { return luthier::MachineFunctionEntryPoints(); });
        });

    PB.registerPipelineParsingCallback(
        [&](llvm::StringRef Name, llvm::ModulePassManager &MPM,
            llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
          if (Name == "luthier-mock-load-amdgpu-code-objects") {
            MPM.addPass(
                luthier::MockLoadAMDGPUCodeObjects(luthier::MockLoaderOptions));
          };
          if (Name == "luthier-code-discovery") {
            MPM.addPass(luthier::CodeDiscoveryPass());
          }
          if (Name == "luthier-apply-instrumentation") {
            MPM.addPass(luthier::InstrumentationPMDriver(
                luthier::InstrumentationPMOptions));
            return true;
          }
          return false;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "luthier-opt", LLVM_VERSION_STRING, Callback,
          nullptr};
}