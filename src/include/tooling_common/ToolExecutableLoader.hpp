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
/// This file describes Luthier's Tool Executable Loader Singleton, which is
/// in charge of managing all loaded instrumentation modules loaded
/// automatically, the lifetime of the instrumented executables,
/// and loading instrumented kernels into a dispatch packet when the tool
/// requests it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_TOOL_EXECUTABLE_LOADER_HPP
#define LUTHIER_TOOLING_COMMON_TOOL_EXECUTABLE_LOADER_HPP
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Module.h>
#include <vector>

#include "InstrumentationModule.hpp"
#include "common/Singleton.hpp"
#include "hsa/CodeObjectReader.hpp"
#include "hsa/Executable.hpp"
#include "hsa/GpuAgent.hpp"
#include "luthier/hsa/LoadedCodeObjectKernel.h"
#include "luthier/types.h"

namespace luthier {

/// \brief A singleton object that keeps track of executables that belong to
/// Luthier, including instrumented executables and tool
/// instrumentation modules, plus launching instrumented kernels
class ToolExecutableLoader : public Singleton<ToolExecutableLoader> {
public:
  /// Registers the wrapper kernel of an instrumentation hook in a static
  /// instrumentation module
  /// For now, dummy empty kernels are used to give tool writers a handle to
  /// hooks on the host side (also referred to as the shadow host pointer) \n
  /// These handles are captured via the \c __hipRegisterFunction calls in
  /// HIP, and they have the same name as the hook they point to, with
  /// \c HOOK_HANDLE_PREFIX prefixed.
  /// \param WrapperShadowHostPtr shadow host pointer of the instrumentation
  /// hook's wrapper kernel
  /// \param HookWrapperName name of the wrapper kernel
  /// \sa LUTHIER_HOOK_ANNOTATE, LUTHIER_EXPORT_HOOK_HANDLE
  void registerInstrumentationHookWrapper(const void *WrapperShadowHostPtr,
                                          const char *HookWrapperName);

  /// Called right after the \p Exec is frozen by the application
  /// It mainly registers an \p Exec if it is a static instrumentation module
  /// executable
  /// \param Exec executable that was just frozen
  /// \return an \c llvm::Error indicating any issues encountered during the
  /// process
  llvm::Error registerIfLuthierToolExecutable(const hsa::Executable &Exec);

  /// Called right before the \p Exec is destroyed by the HSA runtime.
  /// It checks if \p Exec: \n
  /// 1. has been instrumented or not, and removes the instrumented versions
  /// of the executable
  /// 2. belongs to the static instrumentation module and removes it if so
  /// \param Exec the executable about to be destroyed
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  llvm::Error unregisterIfLuthierToolExecutable(const hsa::Executable &Exec);

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
  llvm::Error loadInstrumentedKernel(
      llvm::ArrayRef<
          std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>>
          InstrumentedElfs,
      const hsa::LoadedCodeObjectKernel &OriginalKernel, llvm::StringRef Preset,
      const llvm::StringMap<const void *> &ExternVariables);

  /// Loads a list of instrumented versions of the loaded code objects found in
  /// \p OriginalExecutable into a new executable and freezes it \n
  /// This is usually used when the user wants to instrument most of the kernels
  /// in the executable
  /// \param InstrumentedElfs a list of instrumented versions of the loaded code
  /// objects found in the original Executable
  /// \param OriginalExecutable the \c hsa::ExecutableSymbol of the original
  /// kernel
  /// \param ExternVariables a mapping between the name and the address of
  /// external variables of the instrumented code objects
  /// \return an \p llvm::Error if an issue was encountered in the process
  llvm::Error loadInstrumentedExecutable(
      llvm::ArrayRef<
          std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>>
          InstrumentedElfs,
      llvm::StringRef Preset,
      llvm::ArrayRef<std::tuple<hsa::GpuAgent, llvm::StringRef, const void *>>
          ExternVariables);

  /// Returns the instrumented kernel's \c hsa::ExecutableSymbol given its
  /// original un-instrumented version's \c hsa::ExecutableSymbol and the
  /// preset name it was instrumented under \n
  /// Used to run the instrumented version of the kernel when requested by the
  /// user
  /// \param OriginalKernel symbol of the un-instrumented original kernel
  /// \return symbol of the instrumented version of the target kernel, or
  /// \p llvm::Error
  llvm::Expected<const hsa::LoadedCodeObjectKernel &>
  getInstrumentedKernel(const hsa::LoadedCodeObjectKernel &OriginalKernel,
                        llvm::StringRef Preset) const;

  /// Checks if the given \p Kernel is instrumented under the given \p Preset
  /// \return \c true if it's instrumented, \c false otherwise
  bool isKernelInstrumented(const hsa::LoadedCodeObjectKernel &Kernel,
                            llvm::StringRef Preset) const;

  const StaticInstrumentationModule &getStaticInstrumentationModule() const {
    return SIM;
  }

  ~ToolExecutableLoader();

private:
  /// A private helper function to insert newly-instrumented versions of
  /// the \p OriginalKernel under the given \p Preset\n
  /// <b>Should be called after checking
  /// \c OriginalToInstrumentedKernelsMap doesn't have this entry already</b>
  /// \param OriginalKernel original kernel that was just instrumented
  /// \param Preset the preset name it was instrumented under
  /// \param InstrumentedKernel instrumented version of the original kernel
  void insertInstrumentedKernelIntoMap(
      const hsa::LoadedCodeObjectKernel &OriginalKernel, llvm::StringRef Preset,
      const hsa::LoadedCodeObjectKernel &InstrumentedKernel) {
    // Create an entry for the OriginalKernel if it doesn't already exist in the
    // map
    auto &OriginalKernelEntry =
        !OriginalToInstrumentedKernelsMap.contains(&OriginalKernel)
            ? OriginalToInstrumentedKernelsMap.insert({&OriginalKernel, {}})
                  .first->getSecond()
            : OriginalToInstrumentedKernelsMap[&OriginalKernel];
    OriginalKernelEntry.insert({Preset, &InstrumentedKernel});
  }

  /// The single static instrumentation module included in Luthier tool
  mutable StaticInstrumentationModule SIM{};

  /// \brief a mapping between the loaded code objects instrumented and
  /// loaded by Luthier and their code object readers
  llvm::DenseMap<hsa::LoadedCodeObject, hsa::CodeObjectReader>
      InstrumentedLCOInfo{};

  /// \brief a set of executables that has at least a single kernel of it
  /// instrumented
  llvm::DenseSet<hsa::Executable> OriginalExecutablesWithKernelsInstrumented{};

  /// \brief a mapping between the pair of an instrumented kernel, given
  /// its original kernel, and its instrumentation preset
  llvm::DenseMap<const hsa::LoadedCodeObjectKernel *,
                 llvm::StringMap<const hsa::LoadedCodeObjectKernel *>>
      OriginalToInstrumentedKernelsMap{};


};
}; // namespace luthier

#endif
