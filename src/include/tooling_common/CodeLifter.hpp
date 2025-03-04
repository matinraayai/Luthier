//===-- CodeLifter.hpp - Luthier's Code Lifter  ---------------------------===//
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
/// This file describes Luthier's Code Lifter, a singleton in charge of
/// disassembling code objects into MC and MIR representations.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_TOOLING_COMMON_CODE_LIFTER_HPP
#define LUTHIER_TOOLING_COMMON_CODE_LIFTER_HPP
#include "AMDGPUTargetMachine.h"
#include "TargetManager.hpp"
#include "common/ObjectUtils.hpp"
#include "common/Singleton.hpp"
#include "hsa/Executable.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/ISA.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "hsa/hsa.hpp"
#include "luthier/tooling/LiftedRepresentation.h"
#include "luthier/types.h"
#include "llvm/Cloning.hpp"
#include <functional>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInst.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <luthier/hsa/Instr.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-lifter"

namespace luthier {

/// \brief A singleton class in charge of: \n
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

  //===--------------------------------------------------------------------===//
  // Generic and shared functionality among all components of the CodeLifter
  //===--------------------------------------------------------------------===//

private:
  /// Mutex to protect fields of the code lifter
  std::recursive_mutex CacheMutex{};

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

  //===--------------------------------------------------------------------===//
  // MC-backed Disassembly Functionality
  //===--------------------------------------------------------------------===//

private:
  /// \brief Contains the constructs needed by LLVM for performing a disassembly
  /// operation for each \c hsa::Isa. Does not contain the constructs already
  /// created by the \c TargetManager
  struct DisassemblyInfo {
    std::unique_ptr<llvm::MCContext> Context;
    std::unique_ptr<llvm::MCDisassembler> DisAsm;

    DisassemblyInfo() : Context(nullptr), DisAsm(nullptr) {};

    DisassemblyInfo(std::unique_ptr<llvm::MCContext> Context,
                    std::unique_ptr<llvm::MCDisassembler> DisAsm)
        : Context(std::move(Context)), DisAsm(std::move(DisAsm)) {};
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
  std::unordered_map<
      std::unique_ptr<hsa::LoadedCodeObjectSymbol>,
      std::unique_ptr<llvm::SmallVector<hsa::Instr>>,
      hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectSymbol>,
      hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectSymbol>>
      MCDisassembledSymbols{};

  /// The corrected version of LLVM's evaluate branch
  /// TODO: Merge this fix to upstream LLVM
  static bool evaluateBranch(const llvm::MCInst &Inst, uint64_t Addr,
                             uint64_t Size, uint64_t &Target);

public:
  /// Disassembles the contents of the function-type \p Symbol and returns
  /// a reference to its disassembled array of <tt>hsa::Instr</tt>s\n
  /// Does not perform any symbolization or control flow analysis\n
  /// The \c hsa::ISA of the backing \c hsa::LoadedCodeObject will be used to
  /// disassemble the \p Symbol\n
  /// The results of this operation gets cached on the first invocation
  /// \tparam ST type of the loaded code object symbol; Must be of
  /// type \p KERNEL or \p DEVICE_FUNCTION
  /// \param Symbol the symbol to be disassembled
  /// \return on success, a const reference to the cached disassembled
  /// instructions; On failure, an \p llvm::Error
  /// \sa hsa::Instr
  template <typename ST,
            typename = std::enable_if<
                std::is_same_v<ST, hsa::LoadedCodeObjectDeviceFunction> ||
                std::is_same_v<ST, hsa::LoadedCodeObjectKernel>>>
  llvm::Expected<llvm::ArrayRef<hsa::Instr>> disassemble(const ST &Symbol) {
    std::lock_guard Lock(CacheMutex);
    if (!MCDisassembledSymbols.contains(&Symbol)) {
      // Get the ISA associated with the Symbol
      auto LCO = hsa::LoadedCodeObject(Symbol.getLoadedCodeObject());
      auto ISA = LCO.getISA();
      LUTHIER_RETURN_ON_ERROR(ISA.takeError());
      // Locate the loaded contents of the symbol on the host
      auto MachineCodeOnDevice = Symbol.getLoadedSymbolContents();
      LUTHIER_RETURN_ON_ERROR(MachineCodeOnDevice.takeError());
      auto MachineCodeOnHost =
          hsa::convertToHostEquivalent(*MachineCodeOnDevice);
      LUTHIER_RETURN_ON_ERROR(MachineCodeOnHost.takeError());

      auto InstructionsAndAddresses = disassemble(*ISA, *MachineCodeOnHost);
      LUTHIER_RETURN_ON_ERROR(InstructionsAndAddresses.takeError());
      auto [Instructions, Addresses] = *InstructionsAndAddresses;

      auto &Out =
          MCDisassembledSymbols
              .emplace(Symbol.clone(),
                       std::make_unique<llvm::SmallVector<hsa::Instr>>())
              .first->second;
      Out->reserve(Instructions.size());

      auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
      LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

      auto MII = TargetInfo->getMCInstrInfo();

      auto BaseLoadedAddress =
          reinterpret_cast<luthier::address_t>(MachineCodeOnDevice->data());

      luthier::address_t PrevInstAddress = BaseLoadedAddress;

      for (unsigned int I = 0; I < Instructions.size(); ++I) {
        auto &Inst = Instructions[I];
        auto Address = Addresses[I] + BaseLoadedAddress;
        auto Size = Address - PrevInstAddress;
        if (MII->get(Inst.getOpcode()).isBranch()) {
          LLVM_DEBUG(

              llvm::dbgs() << "Instruction ";
              Inst.dump_pretty(llvm::dbgs(), TargetInfo->getMCInstPrinter(),
                               " ", TargetInfo->getMCRegisterInfo());
              llvm::dbgs() << llvm::formatv(
                  " at idx {0}, address {1:x}, size {2} is a branch; "
                  "Evaluating its target.\n",
                  I, Address, Size);

          );
          luthier::address_t Target;
          if (evaluateBranch(Inst, Address, Size, Target)) {
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                           "Evaluated address {0:x} as the branch target.\n",
                           Target););
            addDirectBranchTargetAddress(LCO, Target);
          } else {
            LLVM_DEBUG(llvm::dbgs()
                       << "Failed to evaluate the branch target.\n");
          }
        }
        PrevInstAddress = Address;
        Out->push_back(hsa::Instr(Inst, Symbol, Address, Size));
      }
    }
    return *MCDisassembledSymbols.find(&Symbol)->second;
  }

  /// Disassembles the machine code encapsulated by \p code for the given \p ISA
  /// \param ISA the \p hsa::Isa of the \p Code
  /// \param Code an \p llvm::ArrayRef pointing to the beginning and end of the
  ///  machine code
  /// \return on success, returns a \p std::vector of \p llvm::MCInst and
  /// a \p std::vector containing the start address of each instruction
  llvm::Expected<std::pair<std::vector<llvm::MCInst>, std::vector<address_t>>>
  disassemble(const hsa::ISA &ISA, llvm::ArrayRef<uint8_t> Code);

  //===--------------------------------------------------------------------===//
  // Beginning of Code Lifting Functionality
  //===--------------------------------------------------------------------===//

private:
  //===--------------------------------------------------------------------===//
  // MachineBasicBlock resolving
  //===--------------------------------------------------------------------===//

  /// \brief Contains the addresses of the HSA instructions that are
  /// target of other branch instructions, per
  /// \c hsa::LoadedCodeObject
  /// \details This map is used during lifting of MC instructions to MIR to
  /// indicate start/end of each \p llvm::MachineBasicBlock. It gets populated
  /// by during MC disassembly of functions
  llvm::DenseMap<hsa::LoadedCodeObject, llvm::DenseSet<address_t>>
      DirectBranchTargetLocations{};

  /// Checks whether the given \p Address is the start of a target of a
  /// direct branch instruction
  /// \param LCO an \p hsa::LoadedCodeObject that contains the \p Address
  /// in its loaded region
  /// \param \c Address a device address in the \p hsa::LoadedCodeObject
  /// \return true if the Address is the start of the target of another branch
  /// instruction; \c false otherwise
  bool isAddressDirectBranchTarget(const hsa::LoadedCodeObject &LCO,
                                   address_t Address);

  /// Used by the MC disassembler functionality to notify
  /// \c BranchLocations about the loaded address of an instruction
  /// that is the target of a direct branch instruction
  /// \param LCO an \p hsa::LoadedCodeObject that contains the \p Address
  /// in its loaded region
  /// \param Address a device address in the \p hsa::LoadedCodeObject
  void addDirectBranchTargetAddress(const hsa::LoadedCodeObject &LCO,
                                    address_t Address);

  //===--------------------------------------------------------------------===//
  // Relocation resolving
  //===--------------------------------------------------------------------===//

  typedef struct {
    std::unique_ptr<hsa::LoadedCodeObjectSymbol>
        Symbol; /// The HSA Executable Symbol
                /// referenced by the relocation
    llvm::object::ELFRelocationRef
        Relocation; /// The ELF relocation information
                    /// Safe to store directly since
                    /// LCO caches the ELF
  } LCORelocationInfo;

  /// Cache of \c LCORelocationInfo information per loaded address in each
  /// lifted \c hsa::LoadedCodeObject\n
  /// Combines relocation information from all sections into this map
  llvm::DenseMap<hsa::LoadedCodeObject,
                 llvm::DenseMap<address_t, LCORelocationInfo>>
      Relocations{};

  /// Returns an \c std::nullopt if the \p address doesn't have any relocation
  /// information associated with it, or the \c LCORelocationInfo associated
  /// with it otherwise
  /// \param LCO The \c hsa::LoadedCodeObject which contains \p Address inside
  /// its loaded range
  /// \param Address the loaded address being queried
  /// \return on success, the the \c LCORelocationInfo associated with
  /// the given \p Address if the address has a relocation info, or
  /// an \c std::nullopt otherwise; an \c llvm::Error on failure describing the
  /// issue encountered
  llvm::Expected<const CodeLifter::LCORelocationInfo *>
  resolveRelocation(const hsa::LoadedCodeObject &LCO, address_t Address);

  //===--------------------------------------------------------------------===//
  // Function-related code-lifting functionality
  //===--------------------------------------------------------------------===//

  /// Initializes an entry associated with the \p LCO inside the \p LR
  /// Creates an \c llvm::Module and \c llvm::MachineModuleInfo for the
  /// \p LCO
  /// \param [in] LCO the \c hsa::LoadedCodeObject to be lifted
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  llvm::Error initLR(LiftedRepresentation &LR,
                     const hsa::LoadedCodeObjectKernel &Kernel);

  /// Initializes a module entry associated with the \p GV inside the \p LR
  /// Does not check if the passed \p GV is indeed is of type variable
  /// \param [in] LCO the \c hsa::LoadedCodeObject \p GV belongs to
  /// \param [in] GV the \c hsa::ExecutableSymbol of type variable to be lifted
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  llvm::Error
  initLiftedGlobalVariableEntry(const hsa::LoadedCodeObject &LCO,
                                const hsa::LoadedCodeObjectSymbol &GV,
                                LiftedRepresentation &LR);

  /// Initializes a module entry associated with the \p Kernel inside the \p LR
  /// \p Func must be of type KERNEL
  /// Does not check if the passed symbol is indeed a kernel
  /// \param [in] LCO the \c hsa::LoadedCodeObject \p Kernel belongs to
  /// \param [in] Kernel the \c hsa::ExecutableSymbol to be initialized
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  llvm::Error initLiftedKernelEntry(const hsa::LoadedCodeObjectKernel &Kernel,
                                    LiftedRepresentation &LR);

  /// Initializes a module entry associated with the \p Func inside the \p LR
  /// \p Func must be of type DEVICE_FUNC
  /// Does not check if the passed symbol is indeed a device function
  /// \tparam HT underlying type of the lifted primitive
  /// \param [in] LCO the \c hsa::LoadedCodeObject \p Kernel belongs to
  /// \param [in] Func the \c hsa::ExecutableSymbol to be initialized
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  llvm::Error
  initLiftedDeviceFunctionEntry(const hsa::LoadedCodeObjectDeviceFunction &Func,
                                LiftedRepresentation &LR);

  llvm::Error liftFunction(const hsa::LoadedCodeObjectSymbol &Symbol,
                           llvm::MachineFunction &MF, LiftedRepresentation &LR);

  //===--------------------------------------------------------------------===//
  // Cached Lifted Representations
  //===--------------------------------------------------------------------===//

  std::unordered_map<
      std::unique_ptr<hsa::LoadedCodeObjectKernel>,
      std::unique_ptr<LiftedRepresentation>,
      hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectKernel>,
      hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectKernel>>
      LiftedKernelSymbols{};

  //===--------------------------------------------------------------------===//
  // Public-facing code-lifting functionality
  //===--------------------------------------------------------------------===//
public:
  /// Returns the \c LiftedRepresentation associated
  /// with the given \p Symbol\n
  /// The representation isolates the requirements of a single kernel can run
  /// interdependently from its parent \c hsa::LoadedCodeObject or \c
  /// hsa::Executable\n
  /// The representation gets cached on the first invocation
  /// \param KernelSymbol an \c hsa::ExecutableSymbol of type \c KERNEL
  /// \return on success, the lifted representation of the kernel symbol; an
  /// \c llvm::Error on failure, describing the issue encountered during the
  /// process
  /// \sa LiftedRepresentation
  llvm::Expected<const LiftedRepresentation &>
  lift(const hsa::LoadedCodeObjectKernel &KernelSymbol);

  llvm::Expected<std::unique_ptr<LiftedRepresentation>>
  cloneRepresentation(const LiftedRepresentation &SrcLR);
};

} // namespace luthier

#undef DEBUG_TYPE

#endif
