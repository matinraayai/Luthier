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
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/Object/ELFObjectFile.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "AMDGPUTargetMachine.h"
#include "hsa.hpp"
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
#include "target_manager.hpp"

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

  //===--------------------------------------------------------------------===//
  // Generic and shared functionality among all components of the CodeLifter
  //===--------------------------------------------------------------------===//

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

  /// \brief Contains the addresses of the HSA instructions that are either
  /// branches or target of other branch instructions, per
  /// \c hsa::LoadedCodeObject
  /// \details This map is used during lifting of MC instructions to MIR to
  /// indicate start/end of each \p llvm::MachineBasicBlock. It gets populated
  /// by during MC disassembly of functions.
  llvm::DenseMap<hsa::LoadedCodeObject, llvm::DenseSet<address_t>>
      BranchAndTargetLocations{};

  /// Checks whether the given \p Address is the start of either a branch
  /// instruction or a target of another branch instruction
  /// \param LCO an \p hsa::LoadedCodeObject that contains the \p Address
  /// in its loaded region
  /// \param \c Address a device address in the \p hsa::LoadedCodeObject
  /// \return true if the Address is the start of a branch instruction, or
  /// a target of another branch instruction; \c false otherwise
  bool isAddressBranchOrBranchTarget(const hsa::LoadedCodeObject &LCO,
                                     address_t Address);

  /// Used by the MC disassembler functionality to notify
  /// \c BranchAndTargetLocations about the loaded address of an instruction
  /// that is either a branch or the target of another branch instruction
  /// \param LCO an \p hsa::LoadedCodeObject that contains the \p Address
  /// in its loaded region
  /// \param Address a device address in the \p hsa::LoadedCodeObject
  void addBranchOrBranchTargetAddress(const hsa::LoadedCodeObject &LCO,
                                      address_t Address);

  //===--------------------------------------------------------------------===//
  // Relocation resolving
  //===--------------------------------------------------------------------===//

  typedef struct {
    hsa::ExecutableSymbol Symbol; // The HSA Executable Symbol referenced by
                                  // the relocation
    llvm::object::ELFRelocationRef Relocation; // The ELF relocation information
                                               // Safe to store directly since
                                               // LCO caches the ELF
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
  llvm::Expected<std::optional<LCORelocationInfo>>
  resolveRelocation(const hsa::LoadedCodeObject &LCO, address_t Address);

  //===--------------------------------------------------------------------===//
  // Function-related code-lifting functionality
  //===--------------------------------------------------------------------===//

  /// Initializes an entry associated with the \p LCO inside the \p LR
  /// Creates an \c llvm::Module and \c llvm::MachineModuleInfo for the
  /// \p LCO
  /// \tparam HT underlying type of the lifted primitive
  /// \param [in] LCO the \c hsa::LoadedCodeObject to be lifted
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  template <typename HT>
  llvm::Error initLiftedLCOEntry(const hsa::LoadedCodeObject &LCO,
                                 LiftedRepresentation<HT> &LR) {
    auto ISA = LCO.getISA();
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());

    auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
    LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

    llvm::GCNTargetMachine *TM = TargetInfo->getTargetMachine();

    // TODO: If debug information is available, the module's name must be
    // set to its source file
    llvm::orc::ThreadSafeModule TSModule{
        std::make_unique<llvm::Module>(llvm::to_string(LCO.hsaHandle()),
                                       *LR.Context.getContext()),
        LR.Context};

    auto Module = TSModule.getModuleUnlocked();
    // Set the data layout (very important)
    Module->setDataLayout(TM->createDataLayout());

    auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TM);
    auto &MMI = MMIWP->getMMI();

    auto &ModuleEntry =
        LR.Modules
            .insert(
                {LCO.hsaHandle(), std::move(std::make_pair(std::move(TSModule),
                                                           std::move(MMIWP)))})
            .first->getSecond();
    LR.RelatedLCOs.insert({LCO.hsaHandle(), {&ModuleEntry.first, &MMI}});
    return llvm::Error::success();
  }

  /// Initializes a module entry associated with the \p GV inside the \p LR
  /// Does not check if the passed \p GV is indeed is of type variable
  /// \tparam HT underlying type of the lifted primitive
  /// \param [in] LCO the \c hsa::LoadedCodeObject \p GV belongs to
  /// \param [in] GV the \c hsa::ExecutableSymbol of type variable to be lifted
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  template <typename HT>
  llvm::Error initLiftedGlobalVariableEntry(const hsa::LoadedCodeObject &LCO,
                                            const hsa::ExecutableSymbol &GV,
                                            LiftedRepresentation<HT> &LR) {
    auto &LLVMContext = *LR.getContext().getContext();
    auto &Module =
        *LR.RelatedLCOs.at(LCO.hsaHandle()).first->getModuleUnlocked();
    auto GVName = GV.getName();
    LUTHIER_RETURN_ON_ERROR(GVName.takeError());
    size_t GVSize = GV.getSize();
    // Lift each variable as an array of bytes, with a length of GVSize
    // We remove any initializers present in the LCO
    new llvm::GlobalVariable(
        Module,
        llvm::ArrayType::get(llvm::Type::getInt8Ty(LLVMContext), GVSize), false,
        llvm::GlobalValue::LinkageTypes::ExternalLinkage, nullptr, *GVName);
    return llvm::Error::success();
  }

  /// Initializes a module entry associated with the \p Kernel inside the \p LR
  /// \p Func must be of type KERNEL
  /// Does not check if the passed symbol is indeed a kernel
  /// \tparam HT underlying type of the lifted primitive
  /// \param [in] LCO the \c hsa::LoadedCodeObject \p Kernel belongs to
  /// \param [in] Kernel the \c hsa::ExecutableSymbol to be initialized
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  template <typename HT>
  llvm::Error initLiftedKernelEntry(const hsa::LoadedCodeObject &LCO,
                                    const hsa::ExecutableSymbol &Kernel,
                                    LiftedRepresentation<HT> &LR) {
    auto &LLVMContext = *LR.Context.getContext();
    auto &Module =
        *LR.RelatedLCOs.at(LCO.hsaHandle()).first->getModuleUnlocked();
    // Populate the Arguments ==================================================
    auto SymbolName = Kernel.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    auto KernelMD = Kernel.getKernelMetadata();
    LUTHIER_RETURN_ON_ERROR(KernelMD.takeError());

    // Kernel's return type is always void
    llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);

    // Create the Kernel's FunctionType with appropriate kernel Arguments
    // (if any)
    llvm::SmallVector<llvm::Type *> Params;
    if (KernelMD->Args.has_value()) {
      // Reserve the number of arguments in the Params vector
      Params.reserve(KernelMD->Args->size());
      // For now, we only rely on required argument metadata
      // This should be updated as new cases are encountered
      for (const auto &ArgMD : *KernelMD->Args) {
        llvm::Type *ParamType =
            llvm::Type::getIntNTy(Module.getContext(), ArgMD.Size);
        // if argument is not passed by value, then it's probably a pointer
        if (ArgMD.ValueKind != hsa::md::ValueKind::ByValue) {
          // AddressSpace is most likely global, but we check it anyway if
          // it's given
          unsigned int AddressSpace = ArgMD.AddressSpace.has_value()
                                          ? *ArgMD.AddressSpace
                                          : llvm::AMDGPUAS::GLOBAL_ADDRESS;
          // Convert the argument to a pointer
          ParamType =
              llvm::PointerType::get(ParamType, llvm::AMDGPUAS::GLOBAL_ADDRESS);
        }
        Params.push_back(ParamType);
      }
    }

    llvm::FunctionType *FunctionType =
        llvm::FunctionType::get(ReturnType, Params, false);

    auto *F = llvm::Function::Create(
        FunctionType, llvm::GlobalValue::ExternalLinkage,
        SymbolName->substr(0, SymbolName->rfind(".kd")), Module);

    // Populate the Attributes =================================================

    F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

    F->addFnAttr("uniform-work-group-size",
                 KernelMD->UniformWorkgroupSize ? "true" : "false");

    // Construct the attributes of the Function, which will result in the MF
    // attributes getting populated
    auto KD = Kernel.getKernelDescriptor();
    LUTHIER_RETURN_ON_ERROR(KD.takeError());

    auto KDOnHost = hsa::queryHostAddress(*KD);
    LUTHIER_RETURN_ON_ERROR(KDOnHost.takeError());

    F->addFnAttr(
        "amdgpu-lds-size",
        llvm::formatv("0, {0}", (*KDOnHost)->GroupSegmentFixedSize).str());
    // Private (scratch) segment size is determined by Analysis Usage pass
    // Kern Arg is determined via analysis usage + args set earlier
    if ((*KDOnHost)->getKernelCodePropertiesEnableSgprDispatchId() == 0) {
      F->addFnAttr("amdgpu-no-dispatch-id");
    }
    if ((*KDOnHost)->getKernelCodePropertiesEnableSgprDispatchPtr() == 0) {
      F->addFnAttr("amdgpu-no-dispatch-ptr");
    }
    if ((*KDOnHost)->getKernelCodePropertiesEnableSgprQueuePtr() == 0) {
      F->addFnAttr("amdgpu-no-queue-ptr");
    }
    F->addFnAttr("amdgpu-ieee",
                 (*KDOnHost)->getRsrc1EnableIeeeMode() ? "true" : "false");
    F->addFnAttr("amdgpu-dx10-clamp",
                 (*KDOnHost)->getRsrc1EnableDx10Clamp() ? "true" : "false");
    if ((*KDOnHost)->getRsrc2EnableSgprWorkgroupIdX() == 0) {
      F->addFnAttr("amdgpu-no-workgroup-id-x");
    }
    if ((*KDOnHost)->getRsrc2EnableSgprWorkgroupIdY() == 0) {
      F->addFnAttr("amdgpu-no-workgroup-id-y");
    }
    if ((*KDOnHost)->getRsrc2EnableSgprWorkgroupIdZ() == 0) {
      F->addFnAttr("amdgpu-no-workgroup-id-z");
    }
    switch ((*KDOnHost)->getRsrc2EnableVgprWorkitemId()) {
    case 0:
      F->addFnAttr("amdgpu-no-workitem-id-y");
    case 1:
      F->addFnAttr("amdgpu-no-workitem-id-z");
      break;
    default:
      llvm_unreachable("KD's VGPR workitem ID is not valid");
    }

    // TODO: Check the args metadata to set this correctly
    F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");

    // TODO: Set the rest of the attributes
    //    llvm::outs() << "Preloaded Args: " << (*KDOnHost)->KernArgPreload <<
    //    "\n";
    F->addFnAttr("amdgpu-calls");
    // Add dummy IR instructions ===============================================
    // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
    // won't run
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module.getContext(), "", F);
    LUTHIER_CHECK(BB);
    new llvm::UnreachableInst(Module.getContext(), BB);

    // Populate the MFI ========================================================

    llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs[LCO.hsaHandle()].second;

    auto &MF = MMI.getOrCreateMachineFunction(*F);

    // TODO: Fix alignment value depending on the function type
    MF.setAlignment(llvm::Align(4096));
    auto &TM = MMI.getTarget();

    auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
        TM.getSubtargetImpl(*F)->getRegisterInfo());
    auto MFI = MF.template getInfo<llvm::SIMachineFunctionInfo>();

    if ((*KDOnHost)->getKernelCodePropertiesEnableSgprPrivateSegmentBuffer() ==
        1) {
      MFI->addPrivateSegmentBuffer(*TRI);
    }
    if ((*KDOnHost)->getKernelCodePropertiesEnableSgprKernArgSegmentPtr() ==
        1) {
      MFI->addKernargSegmentPtr(*TRI);
    }
    if ((*KDOnHost)->getKernelCodePropertiesEnableSgprFlatScratchInit() == 1) {
      MFI->addFlatScratchInit(*TRI);
    }
    if ((*KDOnHost)->getRsrc2EnableSgprPrivateSegmentWaveByteOffset() == 1) {
      //    llvm::outs() << "Private segment Wave offset\n";
      MFI->addPrivateSegmentWaveByteOffset();
    }

    LR.RelatedFunctions.insert({Kernel.hsaHandle(), &MF});

    return llvm::Error::success();
  }

  /// Initializes a module entry associated with the \p Func inside the \p LR
  /// \p Func must be of type DEVICE_FUNC
  /// Does not check if the passed symbol is indeed a device function
  /// \tparam HT underlying type of the lifted primitive
  /// \param [in] LCO the \c hsa::LoadedCodeObject \p Kernel belongs to
  /// \param [in] Func the \c hsa::ExecutableSymbol to be initialized
  /// \param [in, out] LR the lifted representation to be updated
  /// \return an \c llvm::Error if any issues were encountered during the
  /// process
  template <typename HT>
  llvm::Error initLiftedDeviceFunctionEntry(const hsa::LoadedCodeObject &LCO,
                                            const hsa::ExecutableSymbol &Func,
                                            LiftedRepresentation<HT> &LR) {
    auto &LLVMContext = *LR.Context.getContext();
    auto &Module =
        *LR.RelatedLCOs.at(LCO.hsaHandle()).first->getModuleUnlocked();
    llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs.at(LCO.hsaHandle()).second;

    auto FuncName = Func.getName();
    llvm::Type *const ReturnType = llvm::Type::getVoidTy(Module.getContext());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ReturnType != nullptr));
    llvm::FunctionType *FunctionType =
        llvm::FunctionType::get(ReturnType, {}, false);

    auto *F = llvm::Function::Create(
        FunctionType, llvm::GlobalValue::PrivateLinkage, *FuncName, Module);
    F->setCallingConv(llvm::CallingConv::C);

    // Add dummy IR instructions ===============================================
    // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
    // won't run
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module.getContext(), "", F);
    LUTHIER_CHECK(BB);
    new llvm::UnreachableInst(Module.getContext(), BB);
    auto &MF = MMI.getOrCreateMachineFunction(*F);

    // TODO: Fix alignment value depending on the function type
    MF.setAlignment(llvm::Align(4096));
    LR.RelatedFunctions.insert({Func.hsaHandle(), &MF});
    return llvm::Error::success();
  }

  static llvm::Error verifyInstruction(llvm::MachineInstrBuilder &Builder);

  ///
  /// \tparam HT
  /// \param Symbol
  /// \param LR
  /// \return
  template <typename HT>
  llvm::Error
  liftFunction(const hsa::ExecutableSymbol &Symbol,
               LiftedRepresentation<HT> &LR,
               llvm::DenseMap<hsa::ExecutableSymbol, bool> &SymbolUsageMap) {
    auto LCO = Symbol.getDefiningLoadedCodeObject();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LCO.has_value()));

    llvm::Module &Module =
        *LR.RelatedLCOs.at(LCO->hsaHandle()).first->getModuleUnlocked();
    llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs.at(LCO->hsaHandle()).second;
    llvm::MachineFunction &MF = *LR.RelatedFunctions.at(Symbol.hsaHandle());
    auto &F = MF.getFunction();
    auto &TM = MMI.getTarget();

    auto ISA = LCO->getISA();
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());

    auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
    LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

    llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);
    auto MBBEntry = MBB;

    llvm::MCContext &MCContext = MMI.getContext();

    auto MCInstInfo = TargetInfo->getMCInstrInfo();

    llvm::DenseMap<luthier::address_t,
                   llvm::SmallVector<llvm::MachineInstr *>>
        UnresolvedBranchMIs; // < Set of branch instructions located at a
                             // luthier_address_t waiting for their
                             // target to be resolved after MBBs and MIs
                             // are created
    llvm::DenseMap<luthier::address_t, llvm::MachineBasicBlock *>
        BranchTargetMBBs; // < Set of MBBs that will be the target of the
                          // UnresolvedBranchMIs
    auto MIA = TargetInfo->getMCInstrAnalysis();

    auto TargetFunction = CodeLifter::instance().disassemble(Symbol);
    LUTHIER_RETURN_ON_ERROR(TargetFunction.takeError());

    llvm::SmallDenseSet<unsigned>
        LiveIns; // < Set of registers that are not explicitly defined by
                 // instructions (AKA instruction output operand), and
                 // have their value populated by the Driver using the
                 // Kernel Descriptor
    llvm::SmallDenseSet<unsigned> Defines; // < Set of registers defined by
                                           // instructions (output operands)

    for (const auto &Inst : *TargetFunction) {
      auto MCInst = Inst.getMCInst();
      const unsigned Opcode = MCInst.getOpcode();
      const llvm::MCInstrDesc &MCID = MCInstInfo->get(Opcode);
      bool IsDirectBranch = MCID.isBranch() && !MCID.isIndirectBranch();
      bool IsDirectBranchTarget =
          isAddressBranchOrBranchTarget(*LCO, Inst.getLoadedDeviceAddress()) &&
          !IsDirectBranch;

      if (IsDirectBranchTarget) {
        // Branch targets mark the beginning of an MBB
        auto OldMBB = MBB;
        MBB = MF.CreateMachineBasicBlock();
        MF.push_back(MBB);
        OldMBB->addSuccessor(MBB);
        BranchTargetMBBs.insert({Inst.getLoadedDeviceAddress(), MBB});
      }
      llvm::MachineInstrBuilder Builder =
          llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
      LR.MachineInstrToMCMap.insert(
          {Builder.getInstr(), const_cast<hsa::Instr *>(&Inst)});

      for (unsigned OpIndex = 0, E = MCInst.getNumOperands(); OpIndex < E;
           ++OpIndex) {
        //      llvm::outs() << "Number of operands in MCID: " <<
        //      MCID.operands().size()
        //                   << "\n";
        const llvm::MCOperand &Op = MCInst.getOperand(OpIndex);
        if (Op.isReg()) {
          //        llvm::outs() << "Reg Op detected \n";
          unsigned RegNum = Op.getReg();
          const bool IsDef = OpIndex < MCID.getNumDefs();
          unsigned Flags = 0;
          const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
          if (IsDef && !OpInfo.isOptionalDef()) {
            Flags |= llvm::RegState::Define;
            Defines.insert(RegNum);
          } else if (!Defines.contains(RegNum)) {
            LiveIns.insert(RegNum);
            //          llvm::outs() << "Live in detected: \n";
            //          llvm::outs() << "Register: ";
            //          Op.print(llvm::outs(), TargetInfo->getMCRegisterInfo());
            //          llvm::outs() << "\n";
            //          llvm::outs() << "Flags: " << Flags << "\n";
          }
          Builder.addReg(Op.getReg(), Flags);
        } else if (Op.isImm()) {
          // TODO: Resolve immediate load/store operands if they don't have
          // relocations associated with them (e.g. when they happen in the
          // text section)
          luthier::address_t InstAddr = Inst.getLoadedDeviceAddress();
          size_t InstSize = Inst.getSize();
          // Check if at any point in the instruction we need to apply
          // relocations
          bool RelocationApplied{false};
          for (luthier::address_t I = InstAddr; I <= InstAddr + InstSize; ++I) {
            auto RelocationInfo = resolveRelocation(*LCO, I);
            LUTHIER_RETURN_ON_ERROR(RelocationInfo.takeError());
            if (RelocationInfo->has_value()) {
              hsa::ExecutableSymbol TargetSymbol = RelocationInfo.get()->Symbol;

              auto TargetSymbolType = TargetSymbol.getType();

              auto Addend = RelocationInfo.get()->Relocation.getAddend();
              LUTHIER_RETURN_ON_ERROR(Addend.takeError());

              uint64_t Type = RelocationInfo.get()->Relocation.getType();

              SymbolUsageMap[TargetSymbol] = true;
              if (TargetSymbolType == hsa::VARIABLE) {
                auto *GV =
                    LR.RelatedGlobalVariables.at(TargetSymbol.hsaHandle());
                if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                  Type = llvm::SIInstrInfo::MO_GOTPCREL32_LO;
                else if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                  Type = llvm::SIInstrInfo::MO_GOTPCREL32_HI;
                Builder.addGlobalAddress(GV, *Addend, Type);
              } else if (TargetSymbolType == hsa::DEVICE_FUNCTION) {
                auto *UsedMF = LR.RelatedFunctions.at(TargetSymbol.hsaHandle());
                // Add the function as the operand
                if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                  Type = llvm::SIInstrInfo::MO_REL32_LO;
                if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                  Type = llvm::SIInstrInfo::MO_REL32_HI;
                Builder.addGlobalAddress(&UsedMF->getFunction(), *Addend, Type);
              } else {
                // For now, we don't handle calling kernels from kernels
                llvm_unreachable("not implemented");
              }
              RelocationApplied = true;
              break;
            }
          }
          if (!RelocationApplied) {
            Builder.addImm(Op.getImm());
          }

        } else if (!Op.isValid()) {
          llvm_unreachable("Operand is not set");
        } else {
          llvm_unreachable("Not yet implemented");
        }
      }
      LUTHIER_RETURN_ON_ERROR(verifyInstruction(Builder));
      // Basic Block resolving

      if (IsDirectBranch) {
        luthier::address_t BranchTarget;
        if (MIA->evaluateBranch(MCInst, Inst.getLoadedDeviceAddress(),
                                Inst.getSize(), BranchTarget)) {
          if (!UnresolvedBranchMIs.contains(BranchTarget)) {
            UnresolvedBranchMIs.insert({BranchTarget, {Builder.getInstr()}});
          } else {
            UnresolvedBranchMIs[BranchTarget].push_back(Builder.getInstr());
          }
        }
        //      MCInst.dump_pretty(llvm::outs(), TargetInfo->getMCInstPrinter(),
        //      "
        //      ",
        //                         TargetInfo->getMCRegisterInfo());
        auto OldMBB = MBB;
        MBB = MF.CreateMachineBasicBlock();
        MF.push_back(MBB);
        OldMBB->addSuccessor(MBB);
      }
    }

    // Resolve the branch and target MIs/MBBs
    for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
      MBB = BranchTargetMBBs[TargetAddress];
      for (auto &MI : BranchMIs) {
        MI->addOperand(llvm::MachineOperand::CreateMBB(MBB));
        MI->getParent()->addSuccessor(MBB);
        //      MI->print(llvm::outs());
        //      llvm::outs() << "\n";
      }
    }

    // Add the Live-ins to the first MBB
    auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
        TM.getSubtargetImpl(F)->getRegisterInfo());
    for (auto &LiveIn : LiveIns) {
      MF.getRegInfo().addLiveIn(LiveIn);
      MBBEntry->addLiveIn(LiveIn);
    }

    // Populate the properties of MF
    llvm::MachineFunctionProperties &Properties = MF.getProperties();
    Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
    Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
    Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
    Properties.set(llvm::MachineFunctionProperties::Property::TracksLiveness);
    Properties.set(llvm::MachineFunctionProperties::Property::Selected);
    return llvm::Error::success();
  }

  //===--------------------------------------------------------------------===//
  // Cached Lifted Representations
  //===--------------------------------------------------------------------===//

  llvm::DenseMap<hsa::Executable, LiftedRepresentation<hsa_executable_t>>
      LiftedExecutables{};

  llvm::DenseMap<hsa::ExecutableSymbol,
                 LiftedRepresentation<hsa_executable_symbol_t>>
      LiftedKernelSymbols{};

  //===--------------------------------------------------------------------===//
  // Public-facing code-lifting functionality
  //===--------------------------------------------------------------------===//
public:
  /// Returns the \c LiftedRepresentation<hsa_executable_symbol_t> associated
  /// with the given \p Symbol\n
  /// The representation isolates the requirements of a single kernel can run
  /// interdependently from its parent \c hsa::LoadedCodeObject or \c
  /// hsa::Executable\n
  /// The representation gets cached on the first invocation
  /// \param Symbol an \c hsa::ExecutableSymbol of type \c KERNEL
  /// \return on success, the lifted representation of the kernel symbol; an
  /// \c llvm::Error on failure, describing the issue encountered during the
  /// process
  /// \sa LiftedRepresentation
  llvm::Expected<const LiftedRepresentation<hsa_executable_symbol_t> &>
  liftKernelSymbol(const hsa::ExecutableSymbol &Symbol);

  /// Returns the \c LiftedRepresentation<hsa_executable_t> associated
  /// with the given \p Exec\n
  /// The representation gets cached on the first invocation
  /// \param Exec an \c hsa::Executable
  /// \return on success, the lifted representation of the executable; an
  /// \c llvm::Error on failure, describing the issue encountered during the
  /// process
  /// \sa LiftedRepresentation
  llvm::Expected<const LiftedRepresentation<hsa_executable_t> &>
  liftExecutable(const hsa::Executable &Exec);
};

} // namespace luthier

#endif
