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

/// \brief Similar to HIP Modules in concept; Consists of an LLVM bitcode buffer
/// + All static variable addresses it uses on each GPU device
/// \detail Luthier relies on HIP-written code for instrumentation;
class InstrumentationModule {
protected:
  /// Only CodeObjectManager is allowed to create Instrumentation
  /// Modules
  friend ToolExecutableManager;

  /// A buffer owned by the InstrumentationModule object to save a copy of its
  /// processed bitcode \n
  /// Upon creation, The bitcode of the module must be read and processed; This
  /// involves:
  /// 1. Removing any kernels that were used to create host pointers to hooks. \n
  /// 2. Extracting the device functions annotated as hooks and add a hook attribute
  /// to them. \n
  /// 3. Extracting the CUID of the module for its unique identification.
  /// 4. Removing the definition of all global variables.
  /// 5. Removing any managed global variable initializers i.e. variables suffixed
  /// with ".managed".
  /// The processed LLVM Module will then be converted back to a bitcode and
  /// stored here; This is so that the bitcode can be copied over to different
  /// LLVM Contexts, to allow independent, parallelization-friendly compilation.
  llvm::SmallVector<char> BitcodeBuffer{};

  InstrumentationModule() = default;

  /// Compile Unit ID of the Module. This is an identifier generated
  /// by Clang to create a correspondence between the host and the device code.
  /// Presence of CUID is a requirement of all Luthier tool code
  uint64_t CUID{0};

public:
  /// Copies the Module's bitcode into the passed \p Ctx
  /// \param Ctx a thread-safe context to read the bitcode into
  /// \return a thread-safe Module
  llvm::Expected<llvm::orc::ThreadSafeModule>
  readBitcodeIntoContext(llvm::orc::ThreadSafeContext &Ctx);

  /// Returns a mapping between the global variable name and their location \n
  /// This is generally used when loading an instrumented executable
  /// \param Agent The \c hsa::GpuAgent the variables are located on
  /// \param Out A mapping between the name of the global variable and its
  /// address on the \p Agent
  /// \return an \c llvm::Error if an issue was encountered
  /// \sa luthier::hsa::Executable::defineExternalAgentGlobalVariable
  virtual llvm::Error
  getGlobalVariablesOnAgent(hsa::GpuAgent &Agent,
                            llvm::StringMap<void *> &Out) = 0;
};

/// There's only a single instance of this
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

  /// List of "static" symbols; Static symbols come with Instrumentation Modules
  /// loaded with Loaded Code Objects, which means they are already loaded by
  /// the HSA runtime. This means Instrumentation Module doesn't need to worry
  /// about managing them
  llvm::DenseMap<hsa::GpuAgent, llvm::StringMap<hsa::ExecutableSymbol>>
      PerAgentGlobalVariables{};

  /// List of "static" symbols without the agent information
  llvm::SmallVector<std::string> GlobalVariables{};

  /// A mapping between the shadow host pointer of a hook and its name
  llvm::DenseMap<const void *, llvm::StringRef> HookHandleMap{};

  /// Registers the Executable as part of the Static Instrumentation Module
  /// \c hsa::Executable of the instrumentation module that was loaded on a
  /// device
  /// \param Exec the static Luthier tool executable
  llvm::Error registerExecutable(const hsa::Executable &Exec);

  /// Unregisters the executable from the Module.
  /// Called when the executable is about to be destroyed by HSA
  /// \param Exec handle to the executable about to be destroyed
  /// \return
  llvm::Error UnregisterExecutable(const hsa::Executable &Exec);

public:
  llvm::Error getGlobalVariablesOnAgent(hsa::GpuAgent &Agent,
                                        llvm::StringMap<void *> &Out) override;

  /// Same as \c
  /// luthier::StaticInstrumentationModule::getGlobalVariablesOnAgent, except it
  /// returns the ExecutableSymbols of the variables
  /// \param Agent The \c hsa::GpuAgent where a copy (executable) of this module
  /// is loaded
  /// \return a reference to the mapping between variable names and their
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

  static llvm::Expected<bool>
  isStaticInstrumentationModuleExecutable(const hsa::Executable &Exec);
};

/// \brief A singleton object that keeps track of executables that belong to
/// Luthier, including instrumented executables and tool instrumentation modules
class ToolExecutableManager : public Singleton<ToolExecutableManager> {
public:
  /// Registers the wrapper kernel of an instrumentation hook in a static
  /// tool code object
  /// For now, dummy empty kernels are used to give users a handle to hooks
  /// on the host side (also referred to as the shadow host pointer) \n
  /// These handles are captured via the \c __hipRegisterFunction calls in
  /// HIP, and they have the same name as the hook they point to, with
  /// \c HOOK_HANDLE_PREFIX prefixed.
  /// \param WrapperShadowHostPtr shadow host pointer of the instrumentation
  /// hook's wrapper kernel
  /// \param HookWrapperName name of the wrapper kernel
  /// \sa LUTHIER_HOOK_ANNOTATE, LUTHIER_EXPORT_HOOK_HANDLE
  void registerInstrumentationHookWrapper(const void *WrapperShadowHostPtr,
                                          const char *HookWrapperName);

  /// Called right after the \p Exec is frozen by the application.
  /// It mainly checks if \p Exec is a static tool executable, and registers it
  /// with the \c ToolExecutableManager
  /// \param Exec executable that was just frozen
  /// \return an \c llvm::Error indicating any issues encountered during the
  /// process
  llvm::Error registerIfLuthierToolExecutable(const hsa::Executable &Exec);

  /// Called right before the \p Exec is destroyed by the application.
  /// It checks if \p Exec: \n
  /// 1. has been instrumented or not, and removes the instrumented versions
  /// of the executable.
  /// 2. belongs to the static instrumentation module and removes it from
  /// the module.
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
  /// \param Profile the profile name of the instrumentation
  /// \param ExternVariables a mapping between the name and the address of
  /// external variables of the instrumented code objects
  /// \return an \p llvm::Error if an issue was encountered in the process
  llvm::Error loadInstrumentedKernel(
      const llvm::ArrayRef<llvm::ArrayRef<uint8_t>> &InstrumentedElfs,
      const hsa::ExecutableSymbol &OriginalKernel, llvm::StringRef Profile,
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
      llvm::StringRef Profile,
      llvm::ArrayRef<std::tuple<hsa::GpuAgent, llvm::StringRef, void *>>
          ExternVariables);

  /// Returns the instrumented kernel's \c hsa::ExecutableSymbol given its
  /// original un-instrumented version's \c hsa::ExecutableSymbol and the
  /// profile name it was instrumented under \n
  /// Used to run the instrumented version of the kernel when requested by the
  /// user
  /// \param OriginalKernel symbol of the un-instrumented original kernel
  /// \return symbol of the instrumented version of the target kernel, or
  /// \p llvm::Error
  llvm::Expected<const hsa::ExecutableSymbol &>
  getInstrumentedKernel(const hsa::ExecutableSymbol &OriginalKernel,
                        llvm::StringRef Profile) const;

  /// Checks if the given \p Kernel is instrumented under the given \p Profile
  /// \return \c true if it's instrumented, \c false otherwise
  bool isKernelInstrumented(const hsa::ExecutableSymbol &Kernel,
                            llvm::StringRef Profile) const;

  const StaticInstrumentationModule &getStaticInstrumentationModule() const {
    return SIM;
  }

  ~ToolExecutableManager();

private:
  mutable StaticInstrumentationModule SIM{};

  /// \brief a mapping between the loaded code objects instrumented and
  /// loaded by Luthier and their code object readers
  llvm::DenseMap<hsa::LoadedCodeObject, hsa::CodeObjectReader>
      InstrumentedLCOInfo;

  /// \brief a mapping between the pair of an instrumented kernel, given
  /// its original kernel, and its instrumentation profile
  llvm::DenseMap<hsa::ExecutableSymbol, llvm::StringMap<hsa::ExecutableSymbol>>
      InstrumentedKernels{};
};
}; // namespace luthier

#endif
