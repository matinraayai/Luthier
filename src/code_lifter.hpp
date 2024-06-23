//===-- code_lifter.hpp - Luthier's Code Lifter  --------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's Code Lifter, a singleton in charge of
/// disassembling code objects into MC and MIR representations.
//===----------------------------------------------------------------------===//

#ifndef CODE_LIFTER_HPP
#define CODE_LIFTER_HPP
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineInstr.h>
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
#include "hsa_isa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "luthier/instr.h"
#include "luthier/lifted_representation.h"
#include "luthier/types.h"
#include "object_utils.hpp"
#include "singleton.hpp"

namespace luthier {

/// \brief a singleton class in charge of: \n
/// 1. disassembling an \c hsa::ExecutableSymbol of type \c KERNEL or
/// \c DEVICE_FUNCTION using LLVM MC and returning them as a vector of \c
/// hsa::Instr, without symbolizing the operands. \n
/// 2. Converting the disassembled information obtained from LLVM MC plus
/// additional information obtained from the backing \p hsa::Executable to
/// LLVM Machine IR (MIR) and exposing them as a \c LiftedRepresentation to the
/// user. \n
/// 3. TODO: In the presence of debug information in the disassembled/lifted
/// \p hsa::LoadedCodeObject, both the MC representation and MIR representation
/// will also contain the debug information, if requested.
/// \details The MIR lifted by the \p CodeLifter can have the following levels
/// of granularity:\n
/// 1. Kernel-level, in which the Module and MMI only contains enough
/// information to make a single kernel run independently from its parents, the
/// un-instrumented \c hsa::Executable and \c hsa::LoadedCodeObject.\n
/// 2. Executable-level, in which the Module and MMI contain all the information
/// that could be extracted from a single \p hsa::Executable.\n
/// All operations done by the \p CodeLifter is meant to be cached to the
/// best of ability, and invalidated once the \p hsa::Executable containing all
/// the inspected items are destroyed by the runtime.
class CodeLifter : public Singleton<CodeLifter> {

  //===----------------------------------------------------------------------===//
  // Generic and shared functionality among all components of the CodeLifter
  //===----------------------------------------------------------------------===//

private:
  /// Mutex to protect all cached items
  std::shared_mutex CacheMutex{};

public:
  /// Invoked by the \c Controller in the internal HSA callback to notify
  /// the \c CodeLifter that \p Exec has been destroyed by the HSA runtime;
  /// Therefore any cached information related to \p Exec must be removed since
  /// it is no longer valid
  /// \param Exec the \p hsa::Executable that is about to be destroyed by the
  /// HSA runtime
  /// \return \p llvm::Error describing whether the operation succeeded or
  /// faced an error
  llvm::Error invalidateCachedExecutableItems(hsa::Executable &Exec);

  //===----------------------------------------------------------------------===//
  // MC-backed Disassembly Functionality
  //===----------------------------------------------------------------------===//

private:
  /// \brief Contains the constructs needed by LLVM for performing a disassembly
  /// operation for each \c hsa::Isa. Does not contain the constructs already
  /// created by the \c TargetManager
  struct DisassemblyInfo {
    std::unique_ptr<llvm::MCContext> Context;
    std::unique_ptr<llvm::MCDisassembler> DisAsm;

    DisassemblyInfo() : Context(nullptr), DisAsm(nullptr){};

    DisassemblyInfo(std::unique_ptr<llvm::MCContext> Context,
                    std::unique_ptr<llvm::MCDisassembler> DisAsm)
        : Context(std::move(Context)), DisAsm(std::move(DisAsm)){};
  };

  /// Contains the cached \c DisassemblyInfo for each \c hsa::ISA
  llvm::DenseMap<hsa::ISA, DisassemblyInfo> DisassemblyInfoMap{};

  /// On success, returns a reference to the \c DisassemblyInfo associated with
  /// the given \p ISA. Creates the info if not already present in the \c
  /// DisassemblyInfoMap
  /// \param ISA the \c hsa::ISA of the \c DisassemblyInfo
  /// \return on success, a reference to the \c DisassemblyInfo associated with
  /// the given \p ISA, on failure, an \c llvm::Error describing the issue
  /// encountered during the process
  llvm::Expected<DisassemblyInfo &> getDisassemblyInfo(const hsa::ISA &ISA);

  /// Cache of kernel/device function symbols already disassembled by the
  /// \c CodeLifter.\n
  /// The vector handles themselves are allocated as a unique pointer to
  /// stop the map from calling its destructor prematurely.\n
  /// Entries get invalidated once the executable associated with the symbols
  /// get destroyed.
  llvm::DenseMap<hsa::ExecutableSymbol,
                 std::unique_ptr<std::vector<hsa::Instr>>>
      MCDisassembledSymbols{};

public:

  /// Disassembles the content of the given \p hsa::ExecutableSymbol and returns
  /// a reference to a \p std::vector<hsa::Instr>\n
  /// Does not perform any symbolization or control flow analysis\n
  /// The \c hsa::ISA of the backing \c hsa::LoadedCodeObject will be used to
  /// disassemble the \p Symbol\n
  /// The results of this operation gets cached on the first invocation
  /// \param Symbol the \p hsa::ExecutableSymbol to be disassembled. Must be of
  /// type \p KERNEL or \p DEVICE_FUNCTION
  /// \return on success, a const reference to the cached
  /// \c std::vector<hsa::Instr>. On failure, an \p llvm::Error
  /// \sa hsa::Instr
  llvm::Expected<const std::vector<hsa::Instr> &>
  disassemble(const hsa::ExecutableSymbol &Symbol);

  /**
   * Disassembles the machine code encapsulated by \p code for the given \p Isa
   * \param ISA the \p hsa::Isa for which the code should disassembled for
   * \param code an \p llvm::ArrayRef pointing to the beginning and end of the
   * machine code
   * \return on success, returns a \p std::vector of \p llvm::MCInst and
   * a \p std::vector containing the start address of each instruction
   */
  llvm::Expected<std::pair<std::vector<llvm::MCInst>, std::vector<address_t>>>
  disassemble(const hsa::ISA &ISA, llvm::ArrayRef<uint8_t> Code);

  /*****************************************************************************
   * \brief Beginning of Code Lifting Functionality
   ****************************************************************************/
private:
  // TODO: Invalidate these caches once an Executable is destroyed

  /*****************************************************************************
   * \brief \p llvm::MachineBasicBlock resolving
   ****************************************************************************/

  /**
   * \brief Contains the addresses of the instructions that are either branches
   * or target of other branch instructions
   * \details This map is used during lifting of MC instructions to MIR to
   * indicate start/end of each \p llvm::MachineBasicBlock
   */
  llvm::DenseMap<hsa::LoadedCodeObject, llvm::DenseSet<address_t>>
      BranchAndTargetLocations{};

  /**
   * Checks whether the given \p Address is the start of either a branch
   * instruction or a branch target of another branch instruction
   * in the given \p LCO
   * \param LCO an \p hsa::LoadedCodeObject that contains the \p Address
   * in its loaded region
   * @param Address a device address in the \p hsa::LoadedCodeObject
   * @return \p true if the Address
   */
  bool isAddressBranchOrBranchTarget(const hsa::LoadedCodeObject &LCO,
                                     address_t Address);
  /**
   * Used by the MC disassembler functionality to notify \p CodeLifter about the
   * loaded address of an instruction that is either a branch or the target
   * of another branch instruction
   * \param LCO an \p hsa::LoadedCodeObject that contains the \p Address
   * in its loaded region
   * @param Address a device address in the \p hsa::LoadedCodeObject
   */
  void addBranchOrBranchTargetAddress(const hsa::LoadedCodeObject &LCO,
                                      address_t Address);

  llvm::Expected<llvm::Function *>
  initializeLLVMFunctionFromSymbol(const hsa::ExecutableSymbol &Symbol,
                                   llvm::Module &Module);

  llvm::Expected<llvm::MachineFunction &> createLLVMMachineFunctionFromSymbol(
      const hsa::ExecutableSymbol &Symbol, llvm::MachineModuleInfo &MMI,
      llvm::LLVMTargetMachine &TM, llvm::Function &F);

  struct LCORelocationInfo {
    hsa::ExecutableSymbol Symbol; // < The Symbol referenced by the relocation
    int64_t Addend;
    uint64_t Type;
    // llvm::object::ELFRelocationRef RelocRef; // < Relocation Info
  };

  /**
   * Cache of relocation information, per LoadedCodeObject
   */
  std::unordered_map<hsa::LoadedCodeObject,        // < All LCOs lifted so far
                     std::unordered_map<address_t, // < Address of the
                                                   // relocation on the device
                                        LCORelocationInfo // < Relocation info
                                                          // per device address
                                        >>
      Relocations{};

  llvm::Expected<std::optional<LCORelocationInfo>>
  resolveRelocation(const hsa::LoadedCodeObject &LCO, address_t Address);

public:
  llvm::Expected<std::tuple<std::unique_ptr<llvm::Module>,
                            std::unique_ptr<llvm::MachineModuleInfoWrapperPass>,
                            LiftedRepresentation>>
  liftSymbol(const hsa::ExecutableSymbol &Symbol);

  llvm::Expected<LiftedRepresentation>
  liftSymbolAndAddToModule(const hsa::ExecutableSymbol &Symbol,
                           llvm::Module &Module, llvm::MachineModuleInfo &MMI);
};

} // namespace luthier

#endif
