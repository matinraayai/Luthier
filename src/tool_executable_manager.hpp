//===-- tool_executable_manager.hpp - Luthier Tool Executable Manager -----===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's Tool Executable Manager Singleton, which is
/// in charge of managing all loaded instrumentation modules, as well as
/// the lifetime of the instrumented executables. It also describes Luthier's
/// instrumentation modules which are passed to the \c CodeGenerator.
//===----------------------------------------------------------------------===//
#ifndef TOOL_EXECUTABLE_MANAGER_HPP
#define TOOL_EXECUTABLE_MANAGER_HPP
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Module.h>
#include <vector>

#include "hsa_agent.hpp"
#include "hsa_code_object_reader.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "luthier/types.h"
#include "singleton.hpp"

namespace luthier {

class ToolExecutableManager;

//===----------------------------------------------------------------------===//
// Instrumentation Module
//===----------------------------------------------------------------------===//

/// \brief Similar to HIP Modules in concept; Consists of an LLVM bitcode buffer
/// + All static variable addresses it uses on each GPU device
class InstrumentationModule {
protected:
  /// Only CodeObjectManager is allowed to create Instrumentation
  /// Modules
  friend ToolExecutableManager;

  /// A buffer owned by the InstrumentationModule object to save a copy of its
  /// processed bitcode \n
  /// Upon creation, The bitcode of the module must be read and processed; This
  /// involves:
  /// 1. Removing any kernels that were used to create host pointers to hooks.
  /// \n
  /// 2. Extracting the device functions annotated as hooks and add a hook
  /// attribute to them. \n
  /// 3. Extracting the CUID of the module for its unique identification.
  /// 4. Removing the definition of all global variables.
  /// 5. Removing any managed global variable initializers i.e. variables
  /// suffixed with ".managed". The processed LLVM Module will then be converted
  /// back to a bitcode and stored here; This is so that the bitcode can be
  /// copied over to different LLVM Contexts, to allow independent,
  /// parallelization-friendly compilation.
  llvm::SmallVector<char> BitcodeBuffer{};

  InstrumentationModule() = default;

  /// Compile Unit ID of the Module. This is an identifier generated
  /// by Clang to create a correspondence between the host and the device code.
  /// Presence of CUID is a requirement of all Luthier tool code
  uint64_t CUID{0};

public:
  /// Reads the bitcode of this InstrumentationModule into a new
  /// \c llvm::orc::ThreadSafeModule backed by the passed \p Ctx
  /// \param Ctx a thread-safe context to back the returned Module
  /// \return a thread-safe Module, or an \c llvm::Error if any problem was
  /// encountered during the process
  llvm::Expected<llvm::orc::ThreadSafeModule>
  readBitcodeIntoContext(llvm::orc::ThreadSafeContext &Ctx);

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
                                  const hsa::GpuAgent &Agent) = 0;
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
/// instrumented GPU device code; Therefore we can assume only a single static
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
/// it will process the bitcode embedded inside its code object
/// (see \c InstrumentationModule::BitcodeBuffer)
/// and creates a list of global variables in the module, as well as their
/// associated
/// \c hsa::ExecutableSymbol on the loaded \c hsa::GpuAgent.
/// On subsequent executable loads, it only updates the global variable list.
/// It should be clear by now that \c StaticInstrumentationModule does not do
/// any GPU memory management and relies solely on HIP.\n
/// A similar mechanism is in place to detect unloading of the instrumentation
/// module's executables; As they get destroyed, the affected \c
/// hsa::ExecutableSymbols get invalidated as well. When the last \c
/// hsa::Executable of the module is destroyed, the bitcode buffer also gets
/// wiped.\n
/// \c StaticInstrumentationModule also gets notified of the kernel shadow host
/// pointers of each hook, and converts them to the correct hook name to
/// be found in the module later on.
/// \sa InstrumentationModule::BitcodeBuffer, LUTHIER_HOOK_ANNOTATE,
/// LUTHIER_EXPORT_HOOK_HANDLE
class StaticInstrumentationModule final : public InstrumentationModule {
private:
  friend ToolExecutableManager;
  /// Private default constructor only accessible by \c ToolExecutableManager
  StaticInstrumentationModule() = default;

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
  llvm::DenseMap<hsa::GpuAgent, llvm::StringMap<hsa::ExecutableSymbol>>
      PerAgentGlobalVariables{};

  /// List of static symbols without the agent information
  llvm::SmallVector<std::string> GlobalVariables{};

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
                                  const hsa::GpuAgent &Agent) override;

  /// Same as \c
  /// luthier::StaticInstrumentationModule::getGlobalVariablesLoadedOnAgent,
  /// except it returns the ExecutableSymbols of the variables
  /// Use this function only if \c getGlobalVariablesLoadedOnAgent does not
  /// provide sufficient information.
  /// \param Agent The \c hsa::GpuAgent where a copy (executable) of this module
  /// is loaded
  /// \return a const reference to the mapping between variable names and their
  /// Executable Symbols, or an \c llvm::Error if an issue is encountered
  llvm::Expected<const llvm::StringMap<hsa::ExecutableSymbol> &>
  getGlobalHsaVariablesOnAgent(hsa::GpuAgent &Agent);

  /// Converts the shadow host pointer \p Handle to the name of the hook it
  /// represents
  /// \param Handle Shadow host pointer of the hook handle
  /// \return the name of the hook \c llvm::Function, or and \c llvm::Error if
  /// the \p Handle doesn't exist
  llvm::Expected<llvm::StringRef>
  convertHookHandleToHookName(const void *Handle);

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
};

//===----------------------------------------------------------------------===//
// Tool Executable Manager
//===----------------------------------------------------------------------===//

/// \brief A singleton object that keeps track of executables that belong to
/// Luthier, including instrumented executables and tool instrumentation modules
class ToolExecutableManager : public Singleton<ToolExecutableManager> {
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
  /// This is useful for when the user wants to instrument a single kernel
  /// \param InstrumentedElfs a list of instrumented code objects that isolate
  /// the requirements of \p OriginalKernel in a single executable
  /// \param OriginalKernel the \c hsa::ExecutableSymbol of the original kernel
  /// \param Preset the preset name of the instrumentation
  /// \param ExternVariables a mapping between the name and the address of
  /// external variables of the instrumented code objects
  /// \return an \p llvm::Error if an issue was encountered in the process
  llvm::Error loadInstrumentedKernel(
      const llvm::ArrayRef<llvm::ArrayRef<uint8_t>> &InstrumentedElfs,
      const hsa::ExecutableSymbol &OriginalKernel, llvm::StringRef Preset,
      const llvm::ArrayRef<std::pair<llvm::StringRef, void *>>
          &ExternVariables);

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
      llvm::ArrayRef<std::pair<hsa::LoadedCodeObject, llvm::ArrayRef<uint8_t>>>
          InstrumentedElfs,
      llvm::StringRef Preset,
      llvm::ArrayRef<std::tuple<hsa::GpuAgent, llvm::StringRef, void *>>
          ExternVariables);

  /// Returns the instrumented kernel's \c hsa::ExecutableSymbol given its
  /// original un-instrumented version's \c hsa::ExecutableSymbol and the
  /// preset name it was instrumented under \n
  /// Used to run the instrumented version of the kernel when requested by the
  /// user
  /// \param OriginalKernel symbol of the un-instrumented original kernel
  /// \return symbol of the instrumented version of the target kernel, or
  /// \p llvm::Error
  llvm::Expected<const hsa::ExecutableSymbol &>
  getInstrumentedKernel(const hsa::ExecutableSymbol &OriginalKernel,
                        llvm::StringRef Preset) const;

  /// Checks if the given \p Kernel is instrumented under the given \p Preset
  /// \return \c true if it's instrumented, \c false otherwise
  bool isKernelInstrumented(const hsa::ExecutableSymbol &Kernel,
                            llvm::StringRef Preset) const;

  const StaticInstrumentationModule &getStaticInstrumentationModule() const {
    return SIM;
  }

  ~ToolExecutableManager();

private:
  /// A private helper function to insert newly-instrumented versions of
  /// the \p OriginalKernel under the given \p Preset\n
  /// <b>Should be called after checking
  /// \c OriginalToInstrumentedKernelsMap doesn't have this entry already</b>
  /// \param OriginalKernel original kernel that was just instrumented
  /// \param Preset the preset name it was instrumented under
  /// \param InstrumentedKernel instrumented version of the original kernel
  void insertInstrumentedKernelIntoMap(
      const hsa::ExecutableSymbol &OriginalKernel, llvm::StringRef Preset,
      const hsa::ExecutableSymbol &InstrumentedKernel) {
    // Create an entry for the OriginalKernel if it doesn't already exist in the
    // map
    auto &OriginalKernelEntry =
        !OriginalToInstrumentedKernelsMap.contains(OriginalKernel)
            ? OriginalToInstrumentedKernelsMap.insert({OriginalKernel, {}})
                  .first->getSecond()
            : OriginalToInstrumentedKernelsMap[OriginalKernel];
    OriginalKernelEntry.insert({Preset, InstrumentedKernel});
  }

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
  llvm::DenseMap<hsa::ExecutableSymbol, llvm::StringMap<hsa::ExecutableSymbol>>
      OriginalToInstrumentedKernelsMap{};
};
}; // namespace luthier

#endif
