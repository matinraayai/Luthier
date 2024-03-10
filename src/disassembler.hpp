#ifndef CODE_LIFTER_HPP
#define CODE_LIFTER_HPP
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInst.h>
#include <llvm/Object/ELFObjectFile.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_instr.hpp"
#include "hsa_isa.hpp"
#include "luthier_types.h"
#include "object_utils.hpp"

namespace luthier {

/**
 * \brief a singleton class in charge of:
 * 1. disassembling device instructions inside functions (both direct and
 * indirect) and returning them as a \p std::vector of \p hsa::Instr or
 * plain \p llvm::MCInst, depending on its origin
 * 2. Creating a standalone \p llvm::Module and
 * \p llvm::MachineModuleInfoWrapperPass, which isolates the requirements
 * a single kernel function needs to run independently from its parent
 * un-instrumented \p hsa::Executable
 * Both operations are cached to the best ability to save execution time
 */
class CodeLifter {
private:
  /**
   * \brief Contains the constructs needed by LLVM for performing a disassembly
   * operation for each \p hsa::Isa.
   * Does not contain the constructs already created by \p TargetManager
   */
  struct DisassemblyInfo {
    std::unique_ptr<llvm::MCContext> Context;
    std::unique_ptr<llvm::MCDisassembler> DisAsm;

    DisassemblyInfo() : Context(nullptr), DisAsm(nullptr){};

    DisassemblyInfo(std::unique_ptr<llvm::MCContext> Context,
                    std::unique_ptr<llvm::MCDisassembler> DisAsm)
        : Context(std::move(Context)), DisAsm(std::move(DisAsm)){};
  };

  /**
   * \brief contains information regarding an \p llvm::Module and other
   * LLVM-based constructs associated with a kernel
   * The \p Module and \p MMIWP can be used to construct a standalone
   * executable, that when run instead of the original kernel, will produce
   * identical results
   */
  struct KernelModuleInfo {
    std::unique_ptr<llvm::Module> Module;
    std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP;
  };

  CodeLifter() = default;

  ~CodeLifter();

  llvm::Expected<DisassemblyInfo &> getDisassemblyInfo(const hsa::ISA &ISA);

  /**
   * Cached Disassembly Info for each \p hsa::ISA
   */
  llvm::DenseMap<hsa::ISA, DisassemblyInfo> DisassemblyInfoMap;

  /**
   * Holds info regarding addresses on the device belonging to instructions;
   * Contains all branch targets and branch instructions
   */
  llvm::DenseMap<std::pair<hsa::Executable, hsa::GpuAgent>,
                 llvm::DenseMap<luthier_address_t, llvm::SymbolInfoTy>>
      LabelAddressInfoMap;
  // TODO: Invalidate this cache once an Executable is destroyed

  std::optional<llvm::SymbolInfoTy>
  resolveAddressToLabel(const hsa::Executable &Executable,
                        const hsa::GpuAgent &Agent, luthier_address_t Address);

  llvm::DenseMap<std::pair<hsa::Executable, hsa::GpuAgent>,
                 llvm::DenseMap<luthier_address_t, hsa::ExecutableSymbol>>
      ExecutableSymbolAddressInfoMap;

  llvm::DenseMap<hsa::ExecutableSymbol, HSAMD::Kernel::Metadata>
      KernelsMetaData;

  llvm::DenseMap<std::pair<hsa::Executable, hsa::GpuAgent>, HSAMD::Metadata>
      ExecutableMetaData;

  llvm::Expected<std::optional<hsa::ExecutableSymbol>>
  resolveAddressToExecutableSymbol(const hsa::Executable &Executable,
                                   const hsa::GpuAgent &Agent,
                                   luthier_address_t Address);

  llvm::SymbolInfoTy
  getOrCreateNewAddressLabel(const hsa::Executable &Executable,
                             const hsa::GpuAgent &Agent,
                             luthier_address_t Address);

  llvm::Expected<const HSAMD::Kernel::Metadata &>
  getKernelMetaData(const hsa::ExecutableSymbol &Symbol);

  llvm::Expected<const HSAMD::Metadata &>
  getExecutableMetaData(const hsa::Executable &Exec,
                        const hsa::GpuAgent &Agent);

  llvm::Expected<llvm::Function *>
  createLLVMFunction(const hsa::ExecutableSymbol &Symbol, llvm::Module &Module);

  /**
   * Cache of \p hsa::ExecutableSymbol 's already disassembled by \p
   * CodeLifter The vectors have to be allocated as a smart pointer to stop
   * it from calling its destructor prematurely The disassembler is in
   * charge of clearing the map
   */
  llvm::DenseMap<hsa::ExecutableSymbol,
                 std::unique_ptr<std::vector<hsa::Instr>>>
      DisassembledSymbolsRaw;

  llvm::DenseMap<hsa::ExecutableSymbol,
                 std::unique_ptr<std::vector<hsa::Instr>>>
      DisassembledSymbolsSymbolized;

  /**
   * Cache of \p KernelModuleInfo for each kernel function lifted by the
   * \p CodeLifter
   */
  std::unordered_map<hsa::ExecutableSymbol, KernelModuleInfo> KernelModules;

public:
  // Delete the copy and assignment constructors for Singleton object

  CodeLifter(const CodeLifter &) = delete;
  CodeLifter &operator=(const CodeLifter &) = delete;

  /**
   * Singleton accessor, which constructs the instance on first invocation
   * \return the singleton \p CodeLifter
   */
  static inline CodeLifter &instance() {
    static CodeLifter Instance;
    return Instance;
  }

  /**
   * Disassembles the content of the given \p hsa::ExecutableSymbol
   * and returns a \p std::vector of \p hsa::Instr
   * Does not perform any control flow analysis
   * Further invocations will return the cached results
   * \param Symbol the \p hsa::ExecutableSymbol to be disassembled
   * \param Isa optional \p hsa::Isa. If not specified, will use the first
   * \p hsa::Isa return by the \p hsa::GpuAgent of the \p Symbol
   * Used when the \p hsa::GpuAgent supports multiple ISAs
   * \return on success, a const pointer to the cached \p std::vector of
   * \p hsa::Instr
   * \see luthier::hsa::Instr
   */
  llvm::Expected<const std::vector<hsa::Instr> *>
  disassemble(const hsa::ExecutableSymbol &Symbol,
              std::optional<hsa::ISA> Isa = std::nullopt);

  /**
   * Disassembles the code associated with the \p llvm::object::ELFSymbolRef
   * \param Symbol the symbol to disassemble. Must be of type
   * \p llvm::ELF::STT_FUNC
   * \param Size if given, will only disassemble the first \p Size-bytes of the
   * code
   * \return on Success, returns a \p std::vector of \p llvm::MCInst
   */
  llvm::Expected<std::vector<llvm::MCInst>>
  disassemble(const llvm::object::ELFSymbolRef &Symbol,
              std::optional<size_t> Size = std::nullopt);

  /**
   * Disassembles the machine code encapsulated by \p code for the given \p Isa
   * \param Isa the \p hsa::Isa for which the code should disassembled for
   * \param code an \p llvm::ArrayRef pointing to the beginning and end of the
   * machine code
   * \return on success, returns a \p std::vector of \p llvm::MCInst
   */
  llvm::Expected<std::vector<llvm::MCInst>>
  disassemble(const hsa::ISA &Isa, llvm::ArrayRef<uint8_t> Code);

  llvm::Error liftAndAddToModule(const hsa::ExecutableSymbol &Symbol,
                                 llvm::Module &Module,
                                 llvm::MachineModuleInfo &MMI);
};

} // namespace luthier

#endif
