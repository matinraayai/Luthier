
#include "luthier/Tooling/AnnotatedMachineInstr.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

namespace luthier {

constexpr auto LuthierInstrID = "luthier.instr.id";

inline llvm::Error findPCSectionEntry(const llvm::MDNode &PCSectionMD,
                                      llvm::StringRef HeaderName,
                                      llvm::MDString *&HeaderMD,
                                      llvm::MDNode *&ListMD) {
  const unsigned NumOperands = PCSectionMD.getNumOperands();
  for (auto [Idx, MDOperand] : llvm::enumerate(PCSectionMD.operands())) {
    if (auto *MDS = llvm::dyn_cast<llvm::MDString>(MDOperand);
        MDS && Idx != NumOperands - 1 && MDS->getString() == HeaderName) {
      auto *ConstList =
          llvm::dyn_cast<llvm::MDNode>(PCSectionMD.getOperand(Idx + 1));
      if (!ConstList) {
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "The MDNode after the header {0} is not an LLVM MDNode",
            HeaderName));
      }
      HeaderMD = MDS;
      ListMD = ConstList;
      return llvm::Error::success();
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<uint64_t>>
getInstrID(const llvm::MachineInstr &MI) {
  const llvm::MDNode *PCSections = MI.getPCSections();
  if (!PCSections)
    return std::nullopt;

  llvm::MDString *Header{nullptr};
  llvm::MDNode *ConstList{nullptr};

  LUTHIER_RETURN_ON_ERROR(
      findPCSectionEntry(*PCSections, LuthierInstrID, Header, ConstList));

  if (ConstList->getNumOperands() == 1) {
    const auto *IDMD =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(ConstList->getOperand(0));
    if (IDMD)
      return IDMD->getZExtValue();
  } else {
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Malformed {0} metadata", LuthierInstrID));
  }

  return std::nullopt;
}

llvm::Error assignInstrID(llvm::MachineInstr &MI) {
  /// We need to get the LLVM Context associated with the LLVM Function, of the
  /// MI as well as its LLVM Module
  llvm::MachineBasicBlock *MBB = MI.getParent();
  if (!MBB)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Machine instruction doesn't have a parent machine basic block");
  llvm::MachineFunction *MF = MBB->getParent();
  if (!MF)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Machine instruction's basic block doesn't have a parent "
        "machine function");

  llvm::Function &F = MF->getFunction();
  llvm::LLVMContext &Ctx = F.getContext();
  llvm::Module *M = F.getParent();
  if (!M) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "The LLVM Function of the machine instruction's machine function is "
        "not associated with an LLVM Module");
  }
  llvm::IRBuilder<> IRB(Ctx);
  llvm::MDBuilder MDB(Ctx);

  /// The ID that the MI will be assigned
  uint64_t InstrID = 0;

  /// Look for the next free instruction ID in the module
  llvm::NamedMDNode *NextInstrIDNamedMDNode =
      M->getOrInsertNamedMetadata("luthier.next.instr.id");

  llvm::Expected<llvm::ConstantInt &> NextInstrIDAsConstantValOrErr =
      [&]() -> llvm::Expected<llvm::ConstantInt &> {
    /// If there aren't any MD operands yet for the Next Instr ID, we start
    /// from 0
    if (NextInstrIDNamedMDNode->getNumOperands() == 0) {
      llvm::ConstantInt *Out = IRB.getInt64(0);
      NextInstrIDNamedMDNode->addOperand(
          llvm::MDNode::get(Ctx, MDB.createConstant(Out)));
      return *Out;
    } else if (NextInstrIDNamedMDNode->getNumOperands() == 1) {
      llvm::MDNode *NextInstrIDMDNode = NextInstrIDNamedMDNode->getOperand(0);
      if (NextInstrIDMDNode->getNumOperands() != 1) {
        return LUTHIER_MAKE_GENERIC_ERROR(
            "Malformed luthier.next.instr.id named metadata");
      }
      auto *NextIDAsConstantVal = llvm::mdconst::dyn_extract<llvm::ConstantInt>(
          NextInstrIDMDNode->getOperand(0));
      if (!NextIDAsConstantVal) {
        return LUTHIER_MAKE_GENERIC_ERROR(
            "Failed to convert luthier.next.instr.id named metadata to a "
            "constant int");
      }
      return *NextIDAsConstantVal;

    } else {
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Malformed luthier.next.instr.id named metadata");
    }
  }();
  LUTHIER_RETURN_ON_ERROR(NextInstrIDAsConstantValOrErr.takeError());
      /// Get the instruction ID of MI, and increment the next ID in the parent
      /// module

      /// Now that we have the ID of the instruction

      const llvm::MDNode *PCSections = MI.getPCSections();

  /// If we don't have a PCSection MD already, create one from scratch

      // if (!PCSections) {
      //
      //   MI.setPCSections(MDB.createPCSections({{LuthierInstrID, {}}}));
      //   return llvm::Error::success();
      // }
      return llvm::Error::success();
}

} // namespace luthier
