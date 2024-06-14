#ifndef CODE_OBJECT_MANAGER_HPP
#define CODE_OBJECT_MANAGER_HPP
#include <llvm/ADT/DenseSet.h>
#include <llvm/IR/Module.h>
#include <vector>

#include "hsa_agent.hpp"
#include "hsa_code_object_reader.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "singleton.hpp"
#include <luthier/types.h>

namespace luthier {

/// \brief A singleton object that keeps track of code objects related to
/// Luhtier.
class CodeObjectManager : public Singleton<CodeObjectManager> {
public:

  /// Called right after \c hsa_executable_freeze is called on the passed
  /// \p Exec. Checks if \p Exec is a tool executable, and registers it
  /// with the \c CodeObjectManager
  /// \param Exec Freshly-frozen executable
  /// \return an \c llvm::Error indicating any issues encountered during the
  /// process
  llvm::Error registerIfLuthierToolExecutable(const hsa::Executable &Exec);

  /**
   * Returns the \p hsa::ExecutableSymbol of the instrumentation function, given
   * its HSA agent and wrapper kernel shadow host ptr
   * \param WrapperShadowHostPtr shadow host pointer of the wrapper kernel
   * \param Agent the GPU HSA agent where the instrumentation function is loaded
   * on
   * \return the instrumentation function symbol or \p llvm::Error
   */
  llvm::Expected<const hsa::ExecutableSymbol &>
  getInstrumentationFunction(const void *WrapperShadowHostPtr,
                             const hsa::GpuAgent &Agent) const;

  /**
   * Returns the \p hsa::ExecutableSymbol of the wrapper kernel associated with
   * an instrumentation function, given its wrapper kernel's shadow host pointer
   * and the HSA GPU Agent it is loaded on
   * \param WrapperHostPtr shadow host pointer of the wrapper kernel
   * \param Agent the HSA GPU Agent the wrapper kernel
   * (and instrumentation function) is loaded on
   * \return the instrumentation function wrapper kernel's
   * \p hsa::ExecutableSymbol, or \p llvm::Error
   */
  llvm::Expected<const hsa::ExecutableSymbol &>
  getInstrumentationFunctionWrapperKernel(const void *WrapperHostPtr,
                                          const hsa::GpuAgent &Agent) const;

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

  /**
   * Returns a new \c llvm::Module containing the LLVM IR of the given
   * \p Symbols.
   * \param Symbols A list of instrumentation functions to be copied over
   * to the returned Module. The symbols don't have to necessarily inside the
   * same Loaded Code Object; But the Loaded Code Objects should have the same
   * ISA.
   * \return a new \c llvm::Module which contains the IR of the given \p
   * Symbols, or an \c llvm::Error describing the issue encountered
   */
  llvm::Expected<std::unique_ptr<llvm::Module>>
  getModuleContainingInstrumentationFunctions(
      llvm::ArrayRef<hsa::ExecutableSymbol> Symbols) const;

  ~CodeObjectManager();

private:
  struct ToolFunctionInfo {
    const hsa::ExecutableSymbol InstrumentationFunction;
    const hsa::ExecutableSymbol WrapperKernel;
  };

  /**
   * A set of all \c hsa::Executable handles that belong to the Luthier tool,
   * containing the instrumentation function and their wrapper kernels
   */
  mutable llvm::DenseSet<std::pair<hsa::Executable, hsa::GpuAgent>>
      ToolExecutables{};

  /**
   * \brief A list of device functions captured by \c __hipRegisterFunction, not
   * yet processed by \c CodeObjectManager
   * The first \p std::tuple element is the "host shadow pointer" of
   * the instrumentation function's wrapper kernel, created via the macro
   * \p LUTHIER_EXPORT_FUNC
   * The second \p std::tuple element is the name of the dummy kernel
   */
  mutable llvm::DenseMap<const char *, const void *>
      StaticInstrumentationFunctions{};

  mutable llvm::DenseMap<std::pair<const void *, hsa::GpuAgent>,
                         ToolFunctionInfo>
      ToolFunctions{};

  llvm::DenseMap<
      hsa::ExecutableSymbol,
      std::tuple<hsa::ExecutableSymbol, hsa::Executable, hsa::CodeObjectReader>>
      InstrumentedKernels{};

  mutable llvm::DenseMap<hsa::LoadedCodeObject, std::unique_ptr<llvm::Module>>
      ToolLCOEmbeddedIRModules{};
};
}; // namespace luthier

#endif
