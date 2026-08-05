//===-- ToolCodeGenPlugin.cpp ---------------------------------------------===//
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
/// \file ToolCodeGenPlugin.cpp
/// Main file for the Luthier "opt" compiler plugin, which registers Luthier
/// passes and their names with the new pass manager's pass builder when loaded.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/CodeDiscoveryPass.h"
#include "luthier/ToolCodeGen/DebugInfoPass.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/InstructionTracesAnalysis.h"
#include "luthier/ToolCodeGen/InstrumentationPMDriver.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorRegistry.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
#include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/ToolCodeGenTesting/AMDGPUMockLoaderPrinter.h"
#include "luthier/ToolCodeGenTesting/CodeObjectManagerAnalysis.h"
// #include "luthier/ToolCodeGen/IntrinsicMIRLoweringPass.h"
// #include "luthier/ToolCodeGen/LRCallgraph.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/MetadataParserAnalysis.h"
#include "luthier/ToolCodeGen/PrePostAmbleEmitter.h"
#include "luthier/ToolCodeGenTesting/MockAMDGPULoader.h"
#include "luthier/ToolCodeGenTesting/MockLoadAMDGPUCodeObjects.h"
#include "luthier/ToolCodeGenTesting/MockLoaderMemoryAccessor.h"
// #include "luthier/ToolCodeGen/IPVectorRegLiveness.h"
// #include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
// #include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/InitialExecutionPointAnalysis.h"
#include "luthier/ToolCodeGen/NewPMAsmPrinter.h"
#include "luthier/ToolCodeGen/TraceCallGraph.h"
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Plugins/PassPlugin.h>
#include <llvm/Support/TargetSelect.h>
// #include <luthier/ToolCodeGen/IndirectBranchResolverAnalysis.h>

namespace luthier {

static llvm::cl::OptionCategory OptPluginOptions{"Luthier Opt Plugin Options"};

static std::unique_ptr<MockAMDGPULoader> Loader{nullptr};

static InstrumentationPMDriverOptions InstrumentationPMOptions;

static amdgpu::hsamd::MetadataParser MetadataParser;

static MockAMDGPULoaderAnalysisOptions MockLoaderOptions;

static CodeDiscoveryPassOptions CodeDiscoveryOptions;

/// Per-process intrinsic processor registry used by the opt plugin's
/// instrumentation driver. The plugin has no \c HSATool to own one, so it
/// keeps a static; it is no longer a \c Singleton<> type.
static IntrinsicProcessorRegistry IntrinsicProcessorRegistryStorage;

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
      Val.second = std::string(SymbolOrOffset);
    } else {
      Val.second = LoadOffset;
    }

    return false;
  }
};

struct MockAMDGPULoaderInitialExecutionPointParser
    : public llvm::cl::parser<std::pair<uint64_t, std::string>> {
  MockAMDGPULoaderInitialExecutionPointParser(llvm::cl::Option &O)
      : llvm::cl::parser<std::pair<uint64_t, std::string>>(O) {}

  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef ArgValue, std::pair<uint64_t, std::string> &Val) {
    auto [CodeObjectIndexStr, Symbol] = ArgValue.split(':');
    if (CodeObjectIndexStr.getAsInteger(0, Val.first)) {
      return O.error("Failed to parse the code object index for " +
                     llvm::Twine(Val.first) + ".");
    }
    Val.second = Symbol;
    return false;
  }
};

llvm::cl::opt<std::pair<uint64_t, std::variant<uint64_t, std::string>>, false,
              MockAMDGPULoaderInitialEntryPointParser>
    InitialEntryPoint{
        "initial-entrypoint",
        llvm::cl::desc(
            "The initial entry point of the lifting process. "
            "Formatted as <code-object-index>:<mangled-symbol-name> or "
            "<code-object-index>:<load-offset>. \n"
            "Code objects are zero indexed w.r.t the order they are "
            "specified to be loaded into the mock loader."),
        llvm::cl::NotHidden, llvm::cl::cat(OptPluginOptions)};

llvm::cl::list<std::string> LuthierPluginPaths{
    "luthier-plugin",
    llvm::cl::desc(
        "Path to a Luthier pass plugin shared object to load and register "
        "with the Instrumentation PM Driver. May be specified multiple "
        "times."),
    llvm::cl::cat(OptPluginOptions)};

/// Maintains the loaded Luthier plugin handles
static llvm::SmallVector<PassPlugin, 4> LoadedLuthierPlugins;

/// Resolve every \c -luthier-plugin path to a \c PassPlugin, caching across
/// invocations so they remain loaded. Errors during load is skipped
static llvm::ArrayRef<PassPlugin> loadLuthierPluginsFromCLI() {
  for (const std::string &Path : LuthierPluginPaths) {
    PassPlugin *Already =
        llvm::find_if(LoadedLuthierPlugins, [&](const PassPlugin &P) {
          return P.getFilename() == Path;
        });
    if (Already != LoadedLuthierPlugins.end())
      continue;
    llvm::Expected<PassPlugin> LoadedPluginOrErr = PassPlugin::Load(Path);
    if (auto Err = LoadedPluginOrErr.takeError())
      llvm::errs() << "Failed to load plugin at path " << Path
                   << " with error: " << llvm::toString(std::move(Err)) << "\n";

    LoadedLuthierPlugins.push_back(std::move(*LoadedPluginOrErr));
  }
  return LoadedLuthierPlugins;
}

llvm::cl::opt<std::pair<uint64_t, std::string>, false,
              MockAMDGPULoaderInitialExecutionPointParser>
    InitialExecutionPoint{
        "initial-execution-point",
        llvm::cl::desc(
            "The initial execution point of the lifting process. "
            "Formatted as <code-object-index>:<mangled-symbol-name>. \n"
            "Code objects are zero indexed w.r.t the order they are "
            "specified to be loaded into the mock loader."),
        llvm::cl::NotHidden, llvm::cl::cat(OptPluginOptions)};
}; // namespace luthier

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  luthier::Loader = std::make_unique<luthier::MockAMDGPULoader>();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUTargetMCA();

  const auto Callback = [](llvm::PassBuilder &PB) {
    /// Register Luthier module analysis passes
    PB.registerAnalysisRegistrationCallback([](llvm::ModuleAnalysisManager
                                                   &MAM) {
      MAM.registerPass([&]() {
        return luthier::InitialEntryPointAnalysis(
            [&](llvm::Module &M,
                llvm::ModuleAnalysisManager &AM) -> luthier::EntryPoint {
              llvm::LLVMContext &Ctx = M.getContext();
              const auto &MockLoader =
                  AM.getResult<luthier::MockAMDGPULoaderAnalysis>(M)
                      .getLoader();
              uint64_t CodeObjectIdx = 0;
              for (const auto &LCO : MockLoader.loaded_code_objects()) {
                if (CodeObjectIdx == luthier::InitialEntryPoint.first) {
                  if (std::holds_alternative<uint64_t>(
                          luthier::InitialEntryPoint.second)) {
                    uint64_t LoadOffset =
                        std::get<uint64_t>(luthier::InitialEntryPoint.second);
                    if (LoadOffset > LCO.getLoadedRegion().size()) {
                      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
                          llvm::formatv("Offset {0:x} is outside the "
                                        "range of code object index {1}",
                                        LoadOffset, CodeObjectIdx))));
                      return luthier::EntryPoint{};
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
                    if (Err) {
                      Ctx.emitError(llvm::toString(std::move(Err)));
                      return luthier::EntryPoint{};
                    }

                    if (!Symbol.has_value()) {
                      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
                          llvm::formatv("Failed to find the symbol {0} in "
                                        "code object index {1}",
                                        SymbolName, CodeObjectIdx))));
                      return luthier::EntryPoint{};
                    }
                    uint64_t LoadOffset;
                    Err = Symbol->getAddress().moveInto(LoadOffset);
                    if (Err) {
                      Ctx.emitError(llvm::toString(std::move(Err)));
                      return luthier::EntryPoint{};
                    }
                    assert(LoadOffset < LCO.getLoadedRegion().size() &&
                           "Load offset falls outside of the code object");
                    uint64_t LoadAddr = reinterpret_cast<uint64_t>(
                                            LCO.getLoadedRegion().data()) +
                                        LoadOffset;
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
              Ctx.emitError(llvm::toString(
                  LUTHIER_MAKE_GENERIC_ERROR("Failed to get the entry point; "
                                             "Code object index is out of "
                                             "range")));
              llvm_unreachable("Should have thrown an error by now");
            });
      });
      MAM.registerPass([&]() {
        return luthier::InitialExecutionPointAnalysis(
            [&](llvm::Module &M, llvm::ModuleAnalysisManager &AM)
                -> const llvm::amdhsa::kernel_descriptor_t & {
              llvm::LLVMContext &Ctx = M.getContext();
              const auto &MockLoader =
                  AM.getResult<luthier::MockAMDGPULoaderAnalysis>(M)
                      .getLoader();
              uint64_t CodeObjectIdx = 0;
              for (const auto &LCO : MockLoader.loaded_code_objects()) {
                if (CodeObjectIdx == luthier::InitialExecutionPoint.first) {
                  std::optional<luthier::object::AMDGCNElfSymbolRef> Symbol{
                      std::nullopt};
                  llvm::Error Err =
                      LCO.getCodeObject()
                          .lookupSymbol(luthier::InitialExecutionPoint.second)
                          .moveInto(Symbol);
                  if (Err) {
                    Ctx.emitError(llvm::toString(std::move(Err)));
                    llvm_unreachable("Should have thrown an error by now");
                  }

                  if (!Symbol.has_value()) {
                    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
                        llvm::formatv("Failed to find the symbol {0} in "
                                      "code object index {1}",
                                      luthier::InitialExecutionPoint.second,
                                      CodeObjectIdx))));
                    llvm_unreachable("Should have thrown an error by now");
                  }
                  uint64_t LoadOffset;
                  Err = Symbol->getAddress().moveInto(LoadOffset);
                  if (Err) {
                    Ctx.emitError(llvm::toString(std::move(Err)));
                    llvm_unreachable("Should have thrown an error by now");
                  }
                  assert(LoadOffset < LCO.getLoadedRegion().size() &&
                         "Load offset falls outside of the code object");
                  uint64_t LoadAddr =
                      reinterpret_cast<uint64_t>(LCO.getLoadedRegion().data()) +
                      LoadOffset;
                  if (!Symbol->isKernelDescriptor()) {
                    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
                        "Initial execution point is not a kernel symbol")));
                    llvm_unreachable("Should have thrown an error by now");
                  }
                  auto &KD = *reinterpret_cast<
                      const llvm::amdhsa::kernel_descriptor_t *>(LoadAddr);
                  return KD;
                }
                CodeObjectIdx++;
              };
              Ctx.emitError(llvm::toString(
                  LUTHIER_MAKE_GENERIC_ERROR("Failed to get the entry point; "
                                             "Code object index is out of "
                                             "range")));
              llvm_unreachable("Should have thrown an error by now");
            });
      });
      MAM.registerPass([]() {
        return luthier::MemoryAllocationAnalysis(
            std::move(std::make_unique<luthier::MockLoaderMemoryAccessor>(
                *luthier::Loader)));
      });
      MAM.registerPass([]() { return luthier::CodeObjectManagerAnalysis(); });
      MAM.registerPass([]() { return luthier::TraceCallGraphAnalysis(); });
      // LRStateValueStorageAndLoadLocationsAnalysis is now a legacy
      // ModulePass on the IModule; the driver registers it directly.
      MAM.registerPass(
          []() { return luthier::FunctionPreambleDescriptorAnalysis(); });
      MAM.registerPass(
          []() { return luthier::MockAMDGPULoaderAnalysis(*luthier::Loader); });
      MAM.registerPass([&]() {
        return luthier::MetadataParserAnalysis(luthier::MetadataParser);
      });
      // MAM.registerPass([]() { return luthier::IPVectorRegLivenessAnalysis();
      // }); MAM.registerPass(
      // []() { return luthier::IndirectBranchResolverAnalysis(); });
      MAM.registerPass([]() { return luthier::IPPredCFGAnalysis(); });
    });
    /// Register Luthier machine function analysis passes
    PB.registerAnalysisRegistrationCallback(
        [](llvm::MachineFunctionAnalysisManager &MFAM) {
          MFAM.registerPass(
              []() { return luthier::InstructionTracesAnalysis(); });
        });

    PB.registerPipelineParsingCallback(
        [&](llvm::StringRef Name, llvm::ModulePassManager &MPM,
            llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
          if (Name == "luthier-asm-printer") {
            MPM.addPass(luthier::NewPMAsmPrinter(
                llvm::CodeGenFileType::AssemblyFile, llvm::outs(), false));
            return true;
          }
          if (Name == "luthier-mock-load-amdgpu-code-objects") {
            MPM.addPass(
                luthier::MockLoadAMDGPUCodeObjects(luthier::MockLoaderOptions));
            return true;
          };
          if (Name == "luthier-amdgpu-mock-loader-printer") {
            MPM.addPass(luthier::AMDGPUMockLoaderPrinter(llvm::outs()));
            return true;
          };
          if (Name == "luthier-code-discovery") {
            MPM.addPass(
                luthier::CodeDiscoveryPass(luthier::CodeDiscoveryOptions));
            return true;
          }
          if (Name == "luthier-debug-info") {
            MPM.addPass(luthier::DebugInfoPass());
            return true;
          }
          if (Name == "luthier-debug-info-printer") {
            MPM.addPass(luthier::DebugInfoPrinterPass(llvm::outs()));
            return true;
          }
          if (Name == "trace-callgraph-printer") {
            MPM.addPass(luthier::TraceCallGraphPrinter(llvm::outs()));
            return true;
          }
          // if (Name == "luthier-ip-vector-cfg-printer") {
          //   MPM.addPass(luthier::IPPredCFGPrinter(llvm::outs()));
          //   return true;
          // }
          // if (Name == "luthier-ip-vector-reg-liveness-printer") {
          //   MPM.addPass(luthier::IPVectorRegLivenessPrinter(llvm::outs()));
          //   return true;
          // }
          // if (Name == "luthier-ip-reaching-def-printer") {
          //   MPM.addPass(luthier::ReachingDefPrinterPass(llvm::outs()));
          //   return true;
          // }
          // Expose LLVM's PrintMIRPreparePass under a parseable name so users
          // can compose a full module-MIR dump as
          //   "print-mir-prepare,machine-function(print)"
          if (Name == "print-mir-prepare") {
            MPM.addPass(llvm::PrintMIRPreparePass(llvm::outs()));
            return true;
          }
          if (Name == "luthier-apply-instrumentation") {
            /// Called here so that CLI options are guaranteed populated.
            llvm::ArrayRef<luthier::PassPlugin> Plugins =
                luthier::loadLuthierPluginsFromCLI();
            MPM.addPass(luthier::InstrumentationPMDriver(
                luthier::InstrumentationPMOptions,
                luthier::IntrinsicProcessorRegistryStorage, Plugins));
            return true;
          }
          return false;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "luthier-opt", LLVM_VERSION_STRING, Callback,
          nullptr};
}