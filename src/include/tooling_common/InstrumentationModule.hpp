//===-- InstrumentationModule.hpp - Luthier Instrumentation Module --------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file describes Luthier's Instrumentation Module, which contains
/// an LLVM bitcode buffer as well as static variables loaded onto each GPU
/// device. The lifetime of an Instrumentation Module is managed by the
/// <tt>ToolExecutableManager</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INSTRUMENTATION_MODULE_HPP
#define LUTHIER_TOOLING_COMMON_INSTRUMENTATION_MODULE_HPP
#include "hsa/Executable.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h>
#include <optional>
#include <string>
#include <luthier/hsa/LoadedCodeObjectVariable.h>

namespace luthier {

namespace hsa {
class Executable;
}

class ToolExecutableLoader;

//===----------------------------------------------------------------------===//
// Instrumentation Module
//===----------------------------------------------------------------------===//

/// \brief Similar to HIP Modules in concept; Consists of an LLVM bitcode buffer
/// + All static variable addresses it uses on each GPU device
class InstrumentationModule {
public:
  /// Discriminator for LLVM-style RTTI (dyn_cast<> et al.)
  enum ModuleKind { MK_Static, MK_Dynamic };

protected:
  /// Only CodeObjectManager is allowed to create Instrumentation
  /// Modules
  friend ToolExecutableLoader;

  /// A map where indicates the "compatible" bitcode for each \c hsa::GpuAgent
  /// Compatible means that: \n
  /// 1. The bitcode has a compatible ISA with the target agent
  /// 2. The instrumentation module ensures that the external global variables
  /// of this module are loaded on the agent \n
  /// This map does not own the underlying bitcode buffer; If the module needs
  /// to take control of the buffer's lifetime it needs to save it in another
  /// field \n
  /// The instrumentation IR remains in bitcode format until the
  /// \c CodeGenerator asks for a copy of it in its \c llvm::LLVMContext
  /// to allow independent, parallelization-friendly compilation
  llvm::SmallDenseMap<hsa::GpuAgent, llvm::ArrayRef<char>, 2>
      PerAgentBitcodeBufferMap{};

  explicit InstrumentationModule(ModuleKind Kind) : Kind(Kind){};

  /// Compile Unit ID of the Module. This is an identifier generated
  /// by Clang to create a correspondence between the host and the device code.
  /// Presence of CUID is a requirement of all Luthier tool code
  std::string CUID{};

private:
  const ModuleKind Kind;

protected:
  /// List of static symbols without the agent information
  llvm::SmallVector<std::string> GlobalVariables{};

public:
  ModuleKind getKind() const { return Kind; }

  /// Reads the bitcode of this InstrumentationModule for the given \p ISA
  /// into a new \c llvm::orc::ThreadSafeModule backed by the
  /// passed \p Ctx \n
  /// The Context is locked during the process
  /// \param Ctx a thread-safe context to back the returned Module
  /// \return a thread-safe Module, or an \c llvm::Error if any problem was
  /// encountered during the process
  llvm::Expected<llvm::orc::ThreadSafeModule>
  readBitcodeIntoContext(llvm::orc::ThreadSafeContext &Ctx,
                         const hsa::GpuAgent &Agent) const;

  const llvm::SmallVector<std::string> &getGlobalVariableNames() const {
    return GlobalVariables;
  }

  /// Returns the loaded address of the global variable on the given \p Agent if
  /// already loaded, or \c std::nullopt if it is not loaded at the time of
  /// the query \n
  /// This is generally used when loading an instrumented executable
  /// \param GVName the name of the global variable queried
  /// \param Agent The \c hsa::GpuAgent to look for the global variable variable
  /// \return A \c luthier::address_t if the variable was located on the \p
  /// Agent, an \c std::nullopt if not loaded, or an \c llvm::Error if an issue
  /// was encountered
  /// \sa luthier::hsa::Executable::defineExternalAgentGlobalVariable
  virtual llvm::Expected<std::optional<luthier::address_t>>
  getGlobalVariablesLoadedOnAgent(llvm::StringRef GVName,
                                  const hsa::GpuAgent &Agent) const = 0;
};

//===----------------------------------------------------------------------===//
// Static Instrumentation Module
//===----------------------------------------------------------------------===//

/// \brief Keeps track of instrumentation code loaded via a static HIP FAT
/// binary
/// \details an implementation of \c InstrumentationModule which keeps track of
/// <b>the</b> static HIP FAT binary embedded in the shared object of a Luthier
/// tool.\n
/// For now we anticipate that only a single Luthier tool will be loaded at any
/// given time; i.e. we don't think there is a case to instrument an already
/// instrumented GPU device code; Therefore we assume only a single static
/// HIP FAT binary will be loaded at any given time. \c ToolExecutableManager
/// enforces this by keeping a single instance of this variable, as
/// well as keeping its constructor private to itself. \n
/// Furthermore, If two or more Luthier tools are loaded then
/// \c StaticInstrumentationModule will detect this by checking the compile unit
/// ID of each executable passed to it.\n
/// For each GPU Agent, the HIP runtime extracts an ISA-compatible
/// code object from the static FAT binary and loads it into a single
/// executable. This is done in a lazy fashion if deferred loading is enabled,
/// meaning the loading only occurs on a device if the app starts using it. \n
/// \c StaticInstrumentationModule gets notified when a new \c hsa::Executable
/// of the FAT binary gets loaded onto each device. On the first occurrence,
/// it will record the CUID of the module, and creates a list of global
/// variables in the module, as well as their associated \c
/// hsa::ExecutableSymbol on the loaded \c hsa::GpuAgent. On subsequent
/// executable loads, it only updates the global variable list. It should be
/// clear by now that \c StaticInstrumentationModule does not do any GPU memory
/// management and relies solely on HIP.\n A similar mechanism is in place to
/// detect unloading of the instrumentation module's executables; As they get
/// destroyed, the affected \c hsa::ExecutableSymbols get invalidated as well.\n
/// \c StaticInstrumentationModule also gets notified of the kernel shadow host
/// pointers of each hook, and converts them to the correct hook name to
/// be found in the module later on.
/// \sa InstrumentationModule::BitcodeBuffer, LUTHIER_HOOK_ANNOTATE,
/// LUTHIER_EXPORT_HOOK_HANDLE
class StaticInstrumentationModule final : public InstrumentationModule {
private:
  friend ToolExecutableLoader;
  /// Private default constructor only accessible by \c ToolExecutableManager
  StaticInstrumentationModule() : InstrumentationModule(MK_Static){};

  /// Each static HIP module gets loaded on each device as a single HSA
  /// executable \n
  /// This is a mapping from agents to said executables that belong to this
  /// static Module \n
  /// If HIP deferred loading is enabled, this map will be updated as the
  /// app utilizes multiple GPU devices and the HIP runtime loads the module on
  /// each utilized device \n
  /// Since HIP only loads a single LCO per executable, there's no need to save
  /// LCOs here
  llvm::DenseMap<hsa::GpuAgent, hsa::Executable> PerAgentModuleExecutables{};

  /// Keeps track of the copies of the bitcode's global variables on each device
  llvm::DenseMap<hsa::GpuAgent,
                 llvm::StringMap<const hsa::LoadedCodeObjectVariable *>>
      PerAgentGlobalVariables{};

  /// A mapping between the shadow host pointer of a hook and its name
  /// Gets updated whenever \c __hipRegisterFunction is called by
  /// \c ToolExecutableManager
  llvm::DenseMap<const void *, llvm::StringRef> HookHandleMap{};

  /// Registers this executable into the static Instrumentation Module \n
  /// On first invocation this function extracts the bitcode in the ELF of \p
  /// Exec 's LCO, and creates a list of global variables, as well as their
  /// \c hsa::ExecutableSymbol on the device the executable was loaded on \n
  /// On subsequent calls it only updates the global variable list for the
  /// new device \n
  /// This function is only called by \c ToolExecutableManager whenever
  /// it confirms a newly frozen executable is a copy of a Luthier
  /// static FAT binary for instrumentation
  /// \param Exec the static Luthier tool executable that was just frozen by
  /// the HIP runtime
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  llvm::Error registerExecutable(const hsa::Executable &Exec);

  /// Unregisters the executable from the Module \n
  /// As this function gets invoked for each executable on the device
  /// the instrumentation module was loaded on, the internal global variable
  /// list removes the defunct \c hsa::ExecutableSymbols. When the last
  /// executable of this module gets destroyed, the bitcode is wiped as well
  /// as any other internal state \n
  /// This function is only called by \c ToolExecutableManager whenever
  /// it confirms an executable that is about to be destroyed is a copy of
  /// a Luthier static FAT binary for instrumentation
  /// \param Exec handle to the module executable about to be destroyed
  /// \return an \c llvm::Error if any issue was encountered during the process
  llvm::Error unregisterExecutable(const hsa::Executable &Exec);

public:
  llvm::Expected<std::optional<luthier::address_t>>
  getGlobalVariablesLoadedOnAgent(llvm::StringRef GVName,
                                  const hsa::GpuAgent &Agent) const override;

  /// Same as \c
  /// luthier::StaticInstrumentationModule::getGlobalVariablesLoadedOnAgent,
  /// except it returns the ExecutableSymbols of the variables
  /// Use this function only if \c getGlobalVariablesLoadedOnAgent does not
  /// provide sufficient information.
  /// \param Agent The \c hsa::GpuAgent where a copy (executable) of this module
  /// is loaded
  /// \return a const reference to the mapping between variable names and their
  /// Executable Symbols, or an \c llvm::Error if an issue is encountered
  llvm::Expected<const llvm::StringMap<const hsa::LoadedCodeObjectVariable *> &>
  getGlobalHsaVariablesOnAgent(hsa::GpuAgent &Agent);

  /// Converts the shadow host pointer \p Handle to the name of the hook it
  /// represents
  /// \param Handle Shadow host pointer of the hook handle
  /// \return the name of the hook \c llvm::Function, or and \c llvm::Error if
  /// the \p Handle doesn't exist
  llvm::Expected<llvm::StringRef>
  convertHookHandleToHookName(const void *Handle) const;

  /// A helper function which detects if the passed executable is part of the
  /// static instrumentation module. \n
  /// Used by \c ToolExecutableManager to detect and register/unregister
  /// static instrumentation executables
  /// \param Exec an \c hsa::Executable
  /// \return \c true if this is a static instrumentation module copy, false if
  /// not, or an \c llvm::Error if any issues were encountered during the
  /// process
  static llvm::Expected<bool>
  isStaticInstrumentationModuleExecutable(const hsa::Executable &Exec);

  static bool classof(const InstrumentationModule *IM) {
    return IM->getKind() == MK_Static;
  }
};
} // namespace luthier
#endif