//===-- tool_executable_manager.hpp - Luthier Tool Executable Manager -----===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's Tool Executable Manager Singleton, which is
/// in charge of managing all loaded instrumentation modules, as well as
/// the lifetime of the instrumented executables.
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

class StaticInstrumentationModule;

/// \brief Consists of an LLVM bitcode buffer + All static variables it uses on
/// each GPU device
class InstrumentationModule {
protected:
  /// Only CodeObjectManager is allowed to create Instrumentation
  /// Modules
  friend ToolExecutableManager;

  /// A buffer owned by the InstrumentationModule object to save the
  /// processed bitcode \n
  /// The bitcode is first processed to make all its Global Variables external
  /// Then it is stored again in the bitcode format here. This is so that the
  /// bitcode can be copied over to different LLVM Contexts
  llvm::SmallVector<char> BitcodeBuffer{};

  InstrumentationModule() = default;

  /// Compile Unit ID of the Module
  uint64_t CUID{0};

public:
  /// Copies the Module's bitcode into the passed \p Ctx
  /// \param Ctx a thread-safe context to read the bitcode into
  /// \return a thread-safe Module
  llvm::Expected<llvm::orc::ThreadSafeModule>
  readBitcodeIntoContext(llvm::orc::ThreadSafeContext &Ctx);
};

/// There's only a single instance of this
class StaticInstrumentationModule final : public InstrumentationModule {
private:
  friend ToolExecutableManager;
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
  /// Called when the executable is destroyed by HSA
  /// \param Exec handle to the executable about to be destroyed
  /// \return
  llvm::Error UnregisterExecutable(const hsa::Executable &Exec);

public:
  llvm::Expected<const llvm::StringMap<hsa::ExecutableSymbol> &>
  getGlobalVariablesOnAgent(hsa::GpuAgent &Agent);

  llvm::Expected<llvm::StringRef>
  convertHookHandleToHookName(const void *Handle);
};

/// \brief A singleton object that keeps track of code objects related to
/// Luhtier.
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

  /// Called right after the \p Exec is frozen by HSA.
  /// Checks if \p Exec is a static tool executable, and registers it
  /// with the \c CodeObjectManager
  /// \param Exec executable that was just frozen
  /// \return an \c llvm::Error indicating any issues encountered during the
  /// process
  llvm::Error registerIfLuthierToolExecutable(const hsa::Executable &Exec);

  /**
   * Loads an instrumented \p hsa::Executable, containing the instrumented
   * version of the \p OriginalKernel
   * Called by \p CodeGenerator after it has compiled an instrumentation ELF
   * \param InstrumentedElf reference to the instrumented ELF file in memory
   * \param OriginalKernel the symbol of the target instrumented kernel
   * \return \p llvm::Error
   */
  llvm::Error loadInstrumentedKernel(
      const llvm::ArrayRef<uint8_t> &InstrumentedElf,
      const hsa::ExecutableSymbol &OriginalKernel,
      const std::vector<hsa::ExecutableSymbol> &ExternVariables);

  /**
   * Returns the instrumented kernel's \b hsa::ExecutableSymbol given its
   * original un-instrumented version's \b hsa::ExecutableSymbol
   * Used to run the instrumented version of the kernel when requested by the
   * user
   * \param OriginalKernel symbol of the un-instrumented original kernel
   * \return symbol of the instrumented version of the target kernel, or
   * \b llvm::Error
   */
  llvm::Expected<const hsa::ExecutableSymbol &>
  getInstrumentedKernel(const hsa::ExecutableSymbol &OriginalKernel) const;

  /**
   * checks if the given \p Kernel is instrumented
   * \param Kernel the queried kernel
   * \return \p true if it's instrumented, \p false otherwise
   */
  bool isKernelInstrumented(const hsa::ExecutableSymbol &Kernel) const;

  const StaticInstrumentationModule & getStaticInstrumentationModule() const {
    return SIM;
  }

  ~ToolExecutableManager();

private:

  mutable StaticInstrumentationModule SIM{};

  llvm::DenseMap<
      hsa::ExecutableSymbol,
      std::tuple<hsa::ExecutableSymbol, hsa::Executable, hsa::CodeObjectReader>>
      InstrumentedKernels{};

  mutable llvm::DenseMap<hsa::LoadedCodeObject, std::unique_ptr<llvm::Module>>
      ToolLCOEmbeddedIRModules{};
};
}; // namespace luthier

#endif
