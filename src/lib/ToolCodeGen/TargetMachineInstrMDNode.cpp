//===-- TargetMachineInstrMDNode.cpp --------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file TargetMachineInstrMDNode.cpp
/// Implements the \c TargetMachineInstrMDNode class and its methods.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

namespace {

/// \brief A Dummy \c llvm::MDTuple subclass used to make the \c
/// llvm::MDNode::setOperand public instead of protected
class MutableMDTuple : public llvm::MDTuple {
public:
  static bool classof(const Metadata *Node) { return MDTuple::classof(Node); }

  void setOperand(unsigned I, Metadata *New) {
    MDTuple::setOperand(I, New);
  };
};

} // namespace

namespace luthier {
/// \brief set of target machine instruction annotations defined so far
enum TargetMachineInstrAnnotation : uint8_t {
  Tag = 0,
  TraceAddr = 1,
  CanRelaxDirectBranch = 2,
  NeedsRetranslation = 3,
  LastMachineInstrAnnotation = NeedsRetranslation
};

/// \brief This struct statically maps the enums in \c
/// TargetMachineInstrAnnotation to its additional information (e.g., its
/// string metadata section header)
template <TargetMachineInstrAnnotation Annotation>
struct MachineInstrAnnotationInfo;

template <> struct MachineInstrAnnotationInfo<Tag> {
  static constexpr auto MDName = "luthier.machine_instr.tag";
};

template <> struct MachineInstrAnnotationInfo<TraceAddr> {
  static constexpr auto MDName = "luthier.machine_instr.trace_addr";
};

template <> struct MachineInstrAnnotationInfo<CanRelaxDirectBranch> {
  static constexpr auto MDName =
      "luthier.machine_instr.can_relax_direct_branch";
};

template <> struct MachineInstrAnnotationInfo<NeedsRetranslation> {
  static constexpr auto MDName = "luthier.machine_instr.needs_retranslation";
};

/// Modified version of the \c llvm::MDBuilder::createPCSections that will force
/// any \c llvm::MDTuple created in the PCSections to be distinct to allow
/// modification/resizing in Luthier passes
static llvm::MDTuple *
createPCSections(llvm::LLVMContext &Ctx,
                 llvm::ArrayRef<llvm::MDBuilder::PCSection> Sections) {
  llvm::SmallVector<llvm::Metadata *, 2> Ops;
  llvm::MDBuilder MDB{Ctx};
  for (const auto &Entry : Sections) {
    const llvm::StringRef &Sec = Entry.first;
    Ops.push_back(MDB.createString(Sec));

    // If auxiliary data for this section exists, append it.
    const llvm::SmallVector<llvm::Constant *> &AuxConsts = Entry.second;
    if (!AuxConsts.empty()) {
      llvm::SmallVector<llvm::Metadata *, 1> AuxMDs;
      AuxMDs.reserve(AuxConsts.size());
      for (llvm::Constant *C : AuxConsts)
        AuxMDs.push_back(MDB.createConstant(C));
      Ops.push_back(llvm::MDNode::getDistinct(Ctx, AuxMDs));
    }
  }

  return llvm::MDNode::getDistinct(Ctx, Ops);
}

llvm::Expected<TargetMachineInstrMDNode &>
TargetMachineInstrMDNode::initializeMDNode(llvm::MachineInstr &MI) {
  llvm::MachineBasicBlock *ParentMBB = MI.getParent();
  if (!ParentMBB) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("MI {0} doesn't have a parent.", MI));
  }
  llvm::MachineFunction *MF = ParentMBB->getParent();
  if (!MF) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("MI {0}'s MBB doesn't have a parent", MI));
  }
  llvm::LLVMContext &Ctx = MF->getFunction().getContext();
  TargetMachineInstrMDNode &MD = create(Ctx);
  MI.setPCSections(*MF, &MD);
  return MD;
}

TargetMachineInstrMDNode &
TargetMachineInstrMDNode::create(llvm::LLVMContext &Ctx) {
  /// The Tag entry's aux tuple is the dynamic-annotation index list. Seed
  /// slot 0 (corresponding to Tag itself) with UndefValue to match the
  /// padding convention used by getOrCreateMDEntry — Tag is never looked up
  /// via the index list, so this entry is a placeholder.
  return *llvm::cast<TargetMachineInstrMDNode>(createPCSections(
      Ctx, {{MachineInstrAnnotationInfo<Tag>::MDName,
             {llvm::UndefValue::get(llvm::Type::getInt64Ty(Ctx))}}}));
}

TargetMachineInstrMDNode *
TargetMachineInstrMDNode::getInstrMDNodeIfExists(const llvm::MachineInstr &MI) {
  return llvm::dyn_cast<TargetMachineInstrMDNode>(MI.getPCSections());
}

template <TargetMachineInstrAnnotation Annotation>
static std::optional<std::pair<llvm::MDString &, llvm::MDTuple &>>
getMDEntryIfExists(const TargetMachineInstrMDNode &MDNode) {
  static_assert(Annotation != Tag,
                "Tag is not a dynamic annotation slot and cannot be queried "
                "through the index list");
  auto &IndexList = llvm::cast<llvm::MDTuple>(*MDNode.getOperand(Tag + 1));
  if (IndexList.getNumOperands() < Annotation + 1) {
    return std::nullopt;
  } else {
    const llvm::MDOperand &IdxMD = IndexList.getOperand(Annotation);
    if (const auto *CI = llvm::mdconst::dyn_extract<llvm::ConstantInt>(IdxMD)) {
      size_t Idx = CI->getZExtValue();
      auto &Header = llvm::cast<llvm::MDString>(*MDNode.getOperand(Idx));
      auto &ConstantList =
          llvm::cast<llvm::MDTuple>(*MDNode.getOperand(Idx + 1));
      return std::make_pair(std::ref(Header), std::ref(ConstantList));
    }
    return std::nullopt;
  }
}

template <TargetMachineInstrAnnotation Annotation>
static std::pair<llvm::MDString &, llvm::MDTuple &>
getOrCreateMDEntry(llvm::LLVMContext &Ctx, TargetMachineInstrMDNode &MDNode) {
  static_assert(Annotation != Tag,
                "Tag is not a dynamic annotation slot and cannot be allocated "
                "through the index list");
  auto &IndexList = llvm::cast<MutableMDTuple>(*MDNode.getOperand(Tag + 1));
  if (IndexList.getNumOperands() < Annotation + 1) {
    llvm::MDBuilder MDB{Ctx};
    unsigned CurrentIdx = IndexList.getNumOperands();
    while (CurrentIdx != Annotation + 1) {
      IndexList.push_back(MDB.createConstant(
          llvm::UndefValue::get(llvm::Type::getInt64Ty(Ctx))));
      CurrentIdx++;
    }
  }
  if (llvm::mdconst::dyn_extract<llvm::UndefValue>(
          IndexList.getOperand(Annotation).get())) {
    llvm::MDBuilder MDB{Ctx};
    IndexList.setOperand(Annotation, MDB.createConstant(llvm::ConstantInt::get(
                                         llvm::IntegerType::getInt64Ty(Ctx),
                                         MDNode.getNumOperands())));
    MDNode.push_back(
        MDB.createString(MachineInstrAnnotationInfo<Annotation>::MDName));
    MDNode.push_back(llvm::MDNode::getDistinct(Ctx, {}));
  }
  uint64_t SizeIdx = llvm::mdconst::extract<llvm::ConstantInt>(
                         IndexList.getOperand(Annotation))
                         ->getZExtValue();
  auto &Header = llvm::cast<llvm::MDString>(*MDNode.getOperand(SizeIdx));
  auto &ConstantList =
      llvm::cast<llvm::MDTuple>(*MDNode.getOperand(SizeIdx + 1));
  return {std::ref(Header), std::ref(ConstantList)};
}

void TargetMachineInstrMDNode::setTraceInstrAddress(llvm::LLVMContext &Ctx,
                                                    uint64_t Addr) {
  auto [StringHeader, AuxConstList] = getOrCreateMDEntry<TraceAddr>(Ctx, *this);
  llvm::MDBuilder MDB{Ctx};
  llvm::ConstantAsMetadata *NewAddressMD = MDB.createConstant(
      llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(Ctx), Addr));
  if (AuxConstList.getNumOperands() >= 1) {
    llvm::cast<MutableMDTuple>(AuxConstList).setOperand(0, NewAddressMD);
  } else {
    AuxConstList.push_back(NewAddressMD);
  }
}

std::optional<uint64_t> TargetMachineInstrMDNode::getTraceInstrAddress() const {
  auto TraceAddrHeaderAndEntry = getMDEntryIfExists<TraceAddr>(*this);
  if (!TraceAddrHeaderAndEntry.has_value())
    return std::nullopt;
  if (auto &ListMD = TraceAddrHeaderAndEntry->second;
      ListMD.getNumOperands() >= 1) {
    if (const auto *CI = llvm::mdconst::extract_or_null<llvm::ConstantInt>(
            ListMD.getOperand(0))) {
      return CI->getZExtValue();
    }
  }
  return std::nullopt;
}

void TargetMachineInstrMDNode::setCanRelaxDirectBranch(llvm::LLVMContext &Ctx,
                                                       bool CanRelaxBranch) {
  auto [StringHeader, AuxConstList] =
      getOrCreateMDEntry<CanRelaxDirectBranch>(Ctx, *this);
  llvm::MDBuilder MDB{Ctx};
  llvm::ConstantAsMetadata *CanRelaxBranchMD =
      MDB.createConstant(llvm::ConstantInt::getBool(Ctx, CanRelaxBranch));
  if (AuxConstList.getNumOperands() >= 1) {
    llvm::cast<MutableMDTuple>(AuxConstList).setOperand(0, CanRelaxBranchMD);
  } else {
    AuxConstList.push_back(CanRelaxBranchMD);
  }
}

bool TargetMachineInstrMDNode::canRelaxDirectBranch() const {
  auto CanRelaxHeaderAndEntry =
      getMDEntryIfExists<CanRelaxDirectBranch>(*this);
  if (!CanRelaxHeaderAndEntry.has_value())
    return true;
  if (auto &ListMD = CanRelaxHeaderAndEntry->second;
      ListMD.getNumOperands() >= 1) {
    if (auto *CB = llvm::mdconst::extract_or_null<llvm::ConstantInt>(
            ListMD.getOperand(0))) {
      return CB->getZExtValue();
    }
  }
  return true;
}

void TargetMachineInstrMDNode::setNeedsRetranslation(llvm::LLVMContext &Ctx,
                                                     bool NeedsRetranslate) {
  auto [StringHeader, AuxConstList] =
      getOrCreateMDEntry<NeedsRetranslation>(Ctx, *this);
  llvm::MDBuilder MDB{Ctx};
  llvm::ConstantAsMetadata *NeedsRetranslateMD =
      MDB.createConstant(llvm::ConstantInt::getBool(Ctx, NeedsRetranslate));
  if (AuxConstList.getNumOperands() >= 1) {
    llvm::cast<MutableMDTuple>(AuxConstList).setOperand(0, NeedsRetranslateMD);
  } else {
    AuxConstList.push_back(NeedsRetranslateMD);
  }
}

bool TargetMachineInstrMDNode::needsRetranslation() const {
  auto NeedsRetranslationHeaderAndEntry =
      getMDEntryIfExists<NeedsRetranslation>(*this);
  if (!NeedsRetranslationHeaderAndEntry.has_value())
    return false;
  if (auto &ListMD = NeedsRetranslationHeaderAndEntry->second;
      ListMD.getNumOperands() >= 1) {
    if (auto *NR = llvm::mdconst::extract_or_null<llvm::ConstantInt>(
            ListMD.getOperand(0))) {
      return NR->getZExtValue();
    }
  }
  return false;
}

bool TargetMachineInstrMDNode::classof(const Metadata *MD) {
  auto *MDAsTuple = llvm::dyn_cast<MDTuple>(MD);
  if (MDAsTuple && MDAsTuple->getNumOperands() >= Tag + 1) {
    if (auto *TagStringMD =
            llvm::dyn_cast<llvm::MDString>(MDAsTuple->getOperand(Tag))) {
      return TagStringMD->getString() ==
                 MachineInstrAnnotationInfo<Tag>::MDName &&
             llvm::isa<MDTuple>(MDAsTuple->getOperand(1));
    }
  }
  return false;
}

} // namespace luthier
