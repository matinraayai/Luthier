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
#include "hsa_loaded_code_object.hpp"
#include "luthier_types.h"
#include "object_utils.hpp"

namespace luthier {

/**
 * \brief contains information regarding a lifted \ref hsa::ExecutableSymbol
 * of type \ref HSA_SYMBOL_KIND_INDIRECT_FUNCTION or
 * \ref HSA_SYMBOL_KIND_KERNEL, plus:
 * 1. All functions it can possibly call (i.e. related functions)
 * 2. All variables it can possibly address (i.e. related variables)
 * If the related variables or functions can't be statically determined, all
 * the functions and variables of the parent \ref hsa::LoadedCodeObject will
 * be considered related
 */
struct LiftedFunctionInfo {
  llvm::MachineFunction *MF{nullptr};
  llvm::DenseMap<hsa::ExecutableSymbol, llvm::MachineFunction *>
      RelatedFunctions;
  llvm::DenseMap<hsa::ExecutableSymbol, llvm::GlobalVariable *>
      RelatedGlobalVariables;
};

/**
 * \brief a singleton class in charge of:
 * 1. disassembling device instructions inside functions (both direct and
 * indirect) and returning them as a vector of \ref hsa::Instr or
 * plain \ref llvm::MCInst, depending on its origin
 * It doesn't try to symbolize any of the operands
 * 2. Creating a standalone \ref llvm::Module and
 * \ref llvm::MachineModuleInfo, which isolates the requirements
 * a single kernel function needs to run independently from its parent
 * un-instrumented \ref hsa::Executable and \ref hsa::LoadedCodeObject
 * Both operations are cached to the best ability to reduce execution time
 * All operands are symbolized
 */
class CodeLifter {

private:
  CodeLifter() = default;

  ~CodeLifter();

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

  /*****************************************************************************
   * \brief Disassembly Functionality
   ****************************************************************************/
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

  llvm::DenseMap<hsa::ISA, DisassemblyInfo>
      DisassemblyInfoMap; // < Contains the cached DisassemblyInfo for each ISA

  llvm::Expected<DisassemblyInfo &> getDisassemblyInfo(const hsa::ISA &ISA);

  /**
   * Cache of \ref hsa::ExecutableSymbol 's already disassembled by \ref
   * CodeLifter The vectors have to be allocated as a smart pointer to stop
   * it from calling its destructor prematurely The disassembler is in
   * charge of clearing the map
   */
  llvm::DenseMap<hsa::ExecutableSymbol,
                 std::unique_ptr<std::vector<hsa::Instr>>>
      DisassembledSymbolsRaw;

public:
  /**
   * Disassembles the content of the given \p hsa::ExecutableSymbol
   * and returns a \p std::vector of \p hsa::Instr
   * Does not perform any control flow analysis
   * Further invocations will return the cached results
   * \param Symbol the \p hsa::ExecutableSymbol to be disassembled
   * \return on success, a const pointer to the cached \p std::vector of
   * \p hsa::Instr
   * \see luthier::hsa::Instr
   */
  llvm::Expected<const std::vector<hsa::Instr> *>
  disassemble(const hsa::ExecutableSymbol &Symbol);

  /**
   * Disassembles the code associated with the \p llvm::object::ELFSymbolRef
   * \param Symbol the symbol to disassemble. Must be of type
   * \p llvm::ELF::STT_FUNC
   * \param Size if given, will only disassemble the first \p Size-bytes of the
   * code
   * \return on Success, returns a \p std::vector of \p llvm::MCInst and
   * a \p std::vector containing the address of every instruction
   */
  llvm::Expected<
      std::pair<std::vector<llvm::MCInst>, std::vector<luthier_address_t>>>
  disassemble(const llvm::object::ELFSymbolRef &Symbol,
              std::optional<size_t> Size = std::nullopt);

  /**
   * Disassembles the machine code encapsulated by \p code for the given \p Isa
   * \param ISA the \p hsa::Isa for which the code should disassembled for
   * \param code an \p llvm::ArrayRef pointing to the beginning and end of the
   * machine code
   * \return on success, returns a \p std::vector of \p llvm::MCInst and
   * a \p std::vector containing the start address of each instruction
   */
  llvm::Expected<
      std::pair<std::vector<llvm::MCInst>, std::vector<luthier_address_t>>>
  disassemble(const hsa::ISA &ISA, llvm::ArrayRef<uint8_t> Code);

  /*****************************************************************************
   * \brief Code Lifting Functionality
   ****************************************************************************/
public:
private:
  /**
   * \brief contains the \ref llvm::Module and \ref llvm::MachineModuleInfo
   * that stores the \ref LiftedFunctionInfo of every \ref hsa::ExecutableSymbol
   * in \ref hsa::Executable that has been lifted so far.
   * They will not be exposed to the \ref luthier::CodeGenerator, and only
   * serve as a cache.
   * When asked for a \ref hsa::ExecutableSymbol to be lifted, this entry
   * will have to be created first. The content of this Module will be cloned
   * into the passed Module and MachineModuleInfo
   * LLVM-based constructs associated with a kernel
   * The \p Module and \p MMIWP can be used to construct a standalone
   * executable, that when run instead of the original kernel, will produce
   * identical results
   */
  struct LiftedModuleInfo {
    std::unique_ptr<llvm::Module> Module;
    std::unique_ptr<llvm::MachineModuleInfo> MMI;
    llvm::DenseMap<hsa::ExecutableSymbol, llvm::MachineFunction *> Functions{};
    llvm::DenseMap<hsa::ExecutableSymbol, llvm::GlobalVariable *>
        GlobalVariables{};
    llvm::DenseMap<hsa::ExecutableSymbol, llvm::DenseSet<hsa::ExecutableSymbol>>
        RelatedFunctions{};
    llvm::DenseMap<hsa::ExecutableSymbol, llvm::DenseSet<hsa::ExecutableSymbol>>
        RelatedVariables{};
  };

  /**
   * Cache of \p KernelModuleInfo for each kernel function lifted by the
   * \p CodeLifter
   */
  llvm::DenseMap<hsa::Executable, LiftedModuleInfo> ExecutableModuleInfoEntries;

  // TODO: Invalidate these caches once an Executable is destroyed

  llvm::DenseMap<std::pair<hsa::Executable, hsa::GpuAgent>,
                 llvm::DenseSet<luthier_address_t>>
      BranchesAndTargetsLocations; // < Contains the addresses of the branch
                                   // targets and branch instructions

  bool isAddressBranchOrBranchTarget(const hsa::Executable &Executable,
                                     const hsa::GpuAgent &Agent,
                                     luthier_address_t Address);

  llvm::DenseMap<std::pair<hsa::Executable, hsa::GpuAgent>,
                 llvm::DenseMap<luthier_address_t, hsa::ExecutableSymbol>>
      ExecutableSymbolAddressInfoMap;

  llvm::DenseMap<hsa::ExecutableSymbol, HSAMD::Kernel::Metadata>
      KernelsMetaData;

  llvm::DenseMap<hsa::LoadedCodeObject, HSAMD::Metadata>
      LoadedCodeObjectsMetaData;

public:
  llvm::Expected<std::optional<hsa::ExecutableSymbol>>
  resolveAddressToExecutableSymbol(const hsa::Executable &Executable,
                                   const hsa::GpuAgent &Agent,
                                   luthier_address_t Address);

private:
  void addBranchOrBranchTargetAddress(const hsa::Executable &Executable,
                                      const hsa::GpuAgent &Agent,
                                      luthier_address_t Address);

  llvm::Expected<const HSAMD::Kernel::Metadata &>
  getKernelMetaData(const hsa::ExecutableSymbol &Symbol);

  llvm::Expected<const HSAMD::Metadata &>
  getLoadedCodeObjectMetaData(const hsa::LoadedCodeObject &LCO);

  llvm::Expected<llvm::Function *>
  createLLVMFunction(const hsa::ExecutableSymbol &Symbol, llvm::Module &Module);

  llvm::Expected<llvm::MachineFunction &>
  createLLVMMachineFunction(const hsa::ExecutableSymbol &Symbol,
                            llvm::MachineModuleInfo &MMI,
                            llvm::LLVMTargetMachine &TM, llvm::Function &F);

  struct LCORelocationInfo {
    hsa::ExecutableSymbol Symbol; // < The Symbol referenced by the relocation
    llvm::object::ELFRelocationRef RelocRef; // < Relocation Info
  };

  /**
   * Cache of relocation information, per LoadedCodeObject
   */
  llvm::DenseMap<hsa::LoadedCodeObject,            // < All LCOs lifted so far
                 llvm::DenseMap<luthier_address_t, // < Address of the
                                                   // relocation on the device
                                LCORelocationInfo  // < Relocation info per
                                                   // device address
                                >>
      Relocations;

  llvm::Expected<std::optional<LCORelocationInfo>>
  resolveRelocation(const hsa::LoadedCodeObject &LCO,
                    luthier_address_t Address);

public:
  llvm::Expected<LiftedFunctionInfo>
  liftAndAddToModule(const hsa::ExecutableSymbol &Symbol, llvm::Module &Module,
                     llvm::MachineModuleInfo &MMI);
};

} // namespace luthier

#endif
