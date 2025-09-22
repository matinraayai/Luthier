//===-- ToolExecutableLoader.hpp - Luthier Tool Executable Loader ---------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Describes Luthier's Tool Executable Loader Singleton, in charge of:
/// - Managing all loaded instrumentation modules loaded automatically or
/// manually
/// - The lifetime of the instrumented executables
/// - Providing the instrumented versions of the original kernels
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_TOOL_EXECUTABLE_LOADER_HPP
#define LUTHIER_TOOLING_COMMON_TOOL_EXECUTABLE_LOADER_HPP
#include "InstrumentationModule.hpp"
#include "luthier/common/Singleton.h"
#include "luthier/hsa/Agent.h"
#include "luthier/hsa/CodeObjectReader.h"
#include "luthier/hsa/Executable.h"
#include "luthier/hsa/LoadedCodeObject.h"
#include "luthier/hsa/LoadedCodeObjectKernel.h"
#include "luthier/rocprofiler-sdk/ApiTableWrapperInstaller.h"
#include "luthier/types.h"
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Module.h>
#include <luthier/rocprofiler-sdk/ApiTableWrapperInstaller.h>
#include <vector>

namespace luthier {
namespace hsa {
class LoadedCodeObjectCache;
}

/// \brief A singleton object that keeps track of executables that belong to
/// Luthier, including instrumented executables and tool
/// instrumentation modules, plus launching instrumented kernels
class ToolExecutableLoader : public Singleton<ToolExecutableLoader> {
private:
  /// Mutex to protect internal state of the loader
  std::recursive_mutex Mutex;

  /// Table snapshot used to invoke HSA core operations
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot;

  /// Table snapshot used to invoke HSA loader operations
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &LoaderApiSnapshot;

  /// Used to install wrappers for executable freeze/destroy functions
  std::unique_ptr<
      const rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>
      CoreApiWrapperInstaller;

  /// Used to install a wrapper for __hipRegisterFunction
  std::unique_ptr<const rocprofiler::HipCompilerApiTableWrapperInstaller>
      HipCompilerWrapperInstaller;

  /// Reference to the code object cache used to support loading/unloading of
  /// instrumented kernels
  const hsa::LoadedCodeObjectCache &COC;

  /// The single static instrumentation module included in Luthier tool
  mutable StaticInstrumentationModule SIM;

  /// \brief a mapping between the loaded code objects instrumented and
  /// loaded by Luthier and their code object readers
  llvm::DenseMap<hsa_loaded_code_object_t, hsa_code_object_reader_t>
      InstrumentedLCOInfo{};

  llvm::DenseMap<hsa_executable_t, llvm::DenseSet<hsa_executable_t>>
      OriginalExecutablesWithKernelsInstrumented{};

  static t___hipRegisterFunction UnderlyingHipRegisterFn;

  static decltype(hsa_executable_freeze) *UnderlyingHsaExecutableFreezeFn;

  static decltype(hsa_executable_destroy) *UnderlyingHsaExecutableDestroyFn;

  static void
  hipRegisterFunctionWrapper(void **modules, const void *hostFunction,
                             char *deviceFunction, const char *deviceName,
                             unsigned int threadLimit, uint3 *tid, uint3 *bid,
                             dim3 *blockDim, dim3 *gridDim, int *wSize);

  static hsa_status_t hsaExecutableFreezeWrapper(hsa_executable_t Executable,
                                                 const char *Options);

  static hsa_status_t hsaExecutableDestroyWrapper(hsa_executable_t Executable);

public:
  ToolExecutableLoader(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &LoaderApiSnapshot,
      const hsa::LoadedCodeObjectCache &COC, llvm::Error &Err)
      : Singleton<luthier::ToolExecutableLoader>(),
        CoreApiSnapshot(CoreApiSnapshot), LoaderApiSnapshot(LoaderApiSnapshot),
        COC(COC), SIM(LoaderApiSnapshot) {

    CoreApiWrapperInstaller = std::make_unique<
        rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
        Err,
        std::make_tuple(&::CoreApiTable::hsa_executable_freeze_fn,
                        &UnderlyingHsaExecutableFreezeFn,
                        hsaExecutableFreezeWrapper),
        std::make_tuple(&::CoreApiTable::hsa_executable_destroy_fn,
                        &UnderlyingHsaExecutableDestroyFn,
                        hsaExecutableDestroyWrapper));
    if (Err)
      return;

    HipCompilerWrapperInstaller =
        std::make_unique<rocprofiler::HipCompilerApiTableWrapperInstaller>(
            Err, std::make_tuple(
                     &::HipCompilerDispatchTable::__hipRegisterFunction_fn,
                     &UnderlyingHipRegisterFn, hipRegisterFunctionWrapper));
    if (Err)
      return;
  };

  /// Loads a list of instrumented code objects into a new executable and
  /// freezes it, allowing the instrumented version of the \p OriginalKernel
  /// to run on its own
  /// This is useful for when the user wants to instrumentAndLoad a single
  /// kernel
  /// \param InstrumentedElfs a list of instrumented code objects that isolate
  /// the requirements of \p OriginalKernel in a single executable
  /// \param OriginalKernel the \c hsa::ExecutableSymbol of the original kernel
  /// \param Preset the preset name of the instrumentation
  /// \param ExternVariables a mapping between the name and the address of
  /// external variables of the instrumented code objects
  /// \return an \p llvm::Error if an issue was encountered in the process
  llvm::Error
  loadInstrumentedKernel(llvm::ArrayRef<uint8_t> InstrumentedElfs,
                         const hsa::LoadedCodeObjectKernel &OriginalKernel,
                         llvm::StringRef Preset,
                         const llvm::StringMap<const void *> &ExternVariables);

  /// Returns the instrumented kernel's \c hsa::ExecutableSymbol given its
  /// original un-instrumented version's \c hsa::ExecutableSymbol and the
  /// preset name it was instrumented under \n
  /// Used to run the instrumented version of the kernel when requested by the
  /// user
  /// \param OriginalKernel symbol of the un-instrumented original kernel
  /// \return symbol of the instrumented version of the target kernel, or
  /// \p llvm::Error
  [[nodiscard]] llvm::Expected<hsa_executable_symbol_t>
  getInstrumentedKernel(hsa_executable_symbol_t OriginalKernel,
                        llvm::StringRef Preset) const;

  /// Checks if the given \p Kernel is instrumented under the given \p Preset
  /// \return \c true if it's instrumented, \c false otherwise
  [[nodiscard]] bool
  isKernelInstrumented(const hsa::LoadedCodeObjectKernel &Kernel,
                       llvm::StringRef Preset) const;

  [[nodiscard]] const StaticInstrumentationModule &
  getStaticInstrumentationModule() const {
    return SIM;
  }

  ~ToolExecutableLoader() override;

private:
  void insertInstrumentedKernelIntoMap(
      const hsa_executable_t OriginalExecutable,
      const hsa_executable_symbol_t OriginalKernel, llvm::StringRef Preset,
      const hsa_executable_t InstrumentedExecutable,
      const hsa_executable_symbol_t InstrumentedKernel) {
    // Create an entry for the OriginalKernel if it doesn't already exist in the
    // map
    auto OriginalKernelEntry =
        OriginalToInstrumentedKernelsMap.find(OriginalKernel);
    if (OriginalKernelEntry == OriginalToInstrumentedKernelsMap.end()) {
      OriginalKernelEntry =
          OriginalToInstrumentedKernelsMap
              .emplace(OriginalKernel,
                       llvm::StringMap<hsa_executable_symbol_t>{})
              .first;
    }
    OriginalKernelEntry->second.insert({Preset, InstrumentedKernel});
    auto OriginalExecutableEntry =
        OriginalExecutablesWithKernelsInstrumented.find(OriginalExecutable);
    if (OriginalExecutableEntry ==
        OriginalExecutablesWithKernelsInstrumented.end()) {
      OriginalExecutableEntry = OriginalExecutablesWithKernelsInstrumented
                                    .insert({OriginalExecutable, {}})
                                    .first;
    }
    OriginalExecutableEntry->second.insert(InstrumentedExecutable);
  }
  /// \brief a mapping between the pair of an instrumented kernel, given
  /// its original kernel, and its instrumentation preset
  std::unordered_map<hsa_executable_symbol_t,
                     llvm::StringMap<hsa_executable_symbol_t>>
      OriginalToInstrumentedKernelsMap{};
};
}; // namespace luthier

#endif
