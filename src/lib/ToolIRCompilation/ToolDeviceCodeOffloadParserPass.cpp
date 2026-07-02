//===-- ToolDeviceCodeOffloadParserPass.cpp -------------------------------===//
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
/// \file
/// Implements the \c ToolDeviceCodeOffloadParserPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolIRCompilation/ToolDeviceCodeOffloadParserPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <string>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-tool-device-code-offload-parser-pass"

namespace luthier {

/// \brief Scan \c @llvm.global.annotations once, collecting annotated section
/// section-boundary slots and the CXX plugin's exported device functions.
static void collectAnnotatedSlots(
    llvm::Module &M,
    llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 2>> &Slots,
    llvm::SmallVectorImpl<llvm::Function *> &ExportedDeviceFnHandles) {
  const llvm::GlobalVariable *Annots =
      M.getGlobalVariable("llvm.global.annotations");
  if (!Annots || !Annots->hasInitializer())
    return;
  const auto *CA = llvm::dyn_cast<llvm::ConstantArray>(Annots->getOperand(0));
  if (!CA)
    return;
  for (const llvm::Value *Op : CA->operands()) {
    const auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Op);
    if (!CS || CS->getNumOperands() < 2)
      continue;
    llvm::Value *Annotatee = CS->getOperand(0)->stripPointerCasts();
    auto *NameGV = llvm::dyn_cast<llvm::GlobalVariable>(
        CS->getOperand(1)->stripPointerCasts());
    if (!NameGV)
      continue;
    llvm::StringRef Anno;
    if (!llvm::getConstantStringInfo(NameGV, Anno))
      continue;
    if (Anno == ExportFunctionHandleMarker) {
      if (auto *Fn = llvm::dyn_cast<llvm::Function>(Annotatee))
        ExportedDeviceFnHandles.push_back(Fn);
    } else if (Anno == OffloadSectionBeginAnnotation ||
               Anno == OffloadSectionEndAnnotation ||
               Anno == HipHandleSectionBeginAnnotation ||
               Anno == HipHandleSectionEndAnnotation) {
      if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(Annotatee))
        Slots[Anno].push_back(GV);
    }
  }
}

/// \brief Set the initializer of a pointer slot (e.g. \c FatBinarySectionBegin)
/// to \p Value.
static llvm::Error
populatePointerSlot(llvm::ArrayRef<llvm::GlobalVariable *> Slots,
                    llvm::StringRef SlotName, llvm::Constant *Value) {
  if (Slots.empty())
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("No slot found for {0}", SlotName));
  for (auto *OldVar : Slots) {
    if (!OldVar->getValueType()->isPointerTy())
      return LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Pointer slot {0} is not pointer-typed", SlotName));
    OldVar->setInitializer(Value);
  }
  return llvm::Error::success();
}

/// \brief Remove the \c __hip_module_ctor function from the
/// \c llvm.global_ctors array. The array is constant, so it is reconstructed
/// exactly as it was but without \c __hip_module_ctor, so the latter can be
/// deleted afterward.
static llvm::Error deleteModuleCtor(llvm::Module &M) {
  llvm::GlobalVariable *OldCtors = M.getGlobalVariable("llvm.global_ctors");
  /// If there are no global constructors there is nothing we should do
  if (!OldCtors)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "llvm.global_ctors is not present in this file");

  if (!OldCtors->hasInitializer())
    return LUTHIER_MAKE_GENERIC_ERROR(
        "llvm.global_ctors is missing the initializer");

  auto *CtorArray =
      llvm::dyn_cast<llvm::ConstantArray>(OldCtors->getInitializer());

  if (!CtorArray)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "llvm.global_ctors initializer is not a ConstantArray!");
  llvm::SmallVector<llvm::Constant *, 4> RemainingCtors;
  bool Found = false;
  /// We loop through all elements of the global ctor array and store them in a
  /// temporary array except __hip_module_ctor. We use this array to replace
  /// global_ctors with the new one which doesn't contain __hip_module_ctor. If
  /// it isn't in the array we do nothing
  for (auto &Op : CtorArray->operands()) {
    auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Op);
    if (!CS || CS->getNumOperands() < 2)
      continue;

    auto *F =
        llvm::dyn_cast<llvm::Function>(CS->getOperand(1)->stripPointerCasts());
    if (F && F->getName().contains("__hip_module_ctor")) {
      Found = true;
    } else {
      RemainingCtors.push_back(CS);
    }
  }
  /// If __hip_module_ctor is in the global_ctors, we reconstruct the array
  /// without it inside, we make sure it is the exact same in every other aspect
  /// to avoid errors
  if (Found) {
    if (RemainingCtors.empty()) {
      OldCtors->eraseFromParent();
    } else {

      llvm::ArrayType *NewATy = llvm::ArrayType::get(
          CtorArray->getType()->getElementType(), RemainingCtors.size());
      llvm::Constant *NewInit =
          llvm::ConstantArray::get(NewATy, RemainingCtors);

      auto *NewCtors = new llvm::GlobalVariable(
          M, NewATy, OldCtors->isConstant(), OldCtors->getLinkage(), NewInit,
          "", nullptr, OldCtors->getThreadLocalMode(),
          OldCtors->getAddressSpace(), OldCtors->isExternallyInitialized());

      NewCtors->copyAttributesFrom(OldCtors);

      if (OldCtors->getType() != NewCtors->getType()) {
        llvm::Constant *BitCast =
            llvm::ConstantExpr::getBitCast(NewCtors, OldCtors->getType());
        OldCtors->replaceAllUsesWith(BitCast);
      } else {
        OldCtors->replaceAllUsesWith(NewCtors);
      }
      OldCtors->eraseFromParent();
      NewCtors->setName("llvm.global_ctors");
    }
  }
  return llvm::Error::success();
}
/// \brief Deletes a function, this assumes there are no uses
static void deleteFunction(llvm::Function *Fun) {
  Fun->dropAllReferences();
  Fun->eraseFromParent();
}
/// \brief Deletes all uses of a function, so we can safely delete it after
static void deleteAllUses(llvm::Function *Fun) {
  for (auto *User : Fun->users()) {
    if (auto *Inst = llvm::dyn_cast<llvm::Instruction>(User)) {
      // If the call returns a value, we must "defuse" it first
      if (!Inst->use_empty()) {
        Inst->replaceAllUsesWith(llvm::UndefValue::get(Inst->getType()));
      }
      Inst->eraseFromParent(); // Delete the call line
    } else {
      // Handle ConstantExpr or GlobalAliases
      User->dropAllReferences();
    }
  }
}

/// \brief This pass harvests the host-handle / device-name pairs from the
/// \c __hipRegister* calls (and the CXX plugin's exported device functions)
/// into a \c { void*, const char* } array emitted in the
/// \c luthier_hip_handles section, and places the embedded bundle in the
/// \c luthier_fatbin section. It then points the trait's four annotated
/// section-boundary pointer slots (offload-section and HIP-handle-section
/// begin/end) at the linker's \c __start_/__stop_ symbols for those sections,
/// and deletes the host-side HIP registration machinery so the bundle is never
/// registered with the HIP runtime. The trait's slots are located via their
/// \c annotate attributes in \c @llvm.global.annotations.
llvm::PreservedAnalyses
ToolDeviceCodeOffloadParserPass::run(llvm::Module &M,
                                     llvm::ModuleAnalysisManager &MAM) {
  llvm::Triple T(M.getTargetTriple());
  /// Only operate on host code
  if (T.getArch() == llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();
  llvm::LLVMContext &C = M.getContext();

  llvm::Function *RFB = M.getFunction("__hipRegisterFatBinary");
  llvm::Function *RUFB = M.getFunction("__hipUnregisterFatBinary");
  llvm::Function *RFUN = M.getFunction("__hipRegisterFunction");
  llvm::Function *RMV = M.getFunction("__hipRegisterManagedVar");
  llvm::Function *RDV = M.getFunction("__hipRegisterVar");
  llvm::Function *RTX = M.getFunction("__hipRegisterTexture");
  llvm::Function *RSF = M.getFunction("__hipRegisterSurface");
  /// If there is no __hipRegisterFatBinary function, then there's no offload
  /// binary to deal with, so return early.
  if (!RFB)
    return llvm::PreservedAnalyses::all();

  llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 2>> SectionSlotsMap;
  llvm::SmallVector<llvm::Function *, 8> ExportedDeviceFnHandles;
  collectAnnotatedSlots(M, SectionSlotsMap, ExportedDeviceFnHandles);

  auto getOrCreateStruct =
      [&C](llvm::StringRef Name,
           llvm::ArrayRef<llvm::Type *> Fields) -> llvm::StructType * {
    if (auto *Existing = llvm::StructType::getTypeByName(C, Name))
      return Existing;
    return llvm::StructType::create(C, Fields, Name);
  };
  llvm::Type *PtrTy = llvm::PointerType::getUnqual(C);

  //===--------------------------------------------------------------------===//
  // Fat binary: place the bundle in the luthier_fatbin section and point the
  // FatBinaryStart / FatBinaryStop slots at the linker's section-boundary
  // symbols.
  //===--------------------------------------------------------------------===//

  /// Each \c __hipRegisterFatBinary call site hands us the host-side
  /// \c __CudaFatBinaryWrapper GV \c { i32 magic, i32 version, ptr binary,
  /// ptr dummy }. Chase \c binary through any constant-expr bitcasts to the
  /// bundle storage GV.
  llvm::GlobalVariable *BundleGV = nullptr;
  for (llvm::User *U : RFB->users()) {
    auto *CB = llvm::dyn_cast<llvm::CallBase>(U);
    if (!CB)
      continue;
    auto *WrapperGV =
        llvm::dyn_cast<llvm::GlobalVariable>(CB->getArgOperand(0));
    if (!WrapperGV || !WrapperGV->hasInitializer())
      continue;
    auto *WrapperInit =
        llvm::dyn_cast<llvm::ConstantStruct>(WrapperGV->getInitializer());
    if (!WrapperInit || WrapperInit->getNumOperands() < 3)
      continue;
    BundleGV = llvm::dyn_cast<llvm::GlobalVariable>(
        WrapperInit->getOperand(2)->stripPointerCasts());
    if (BundleGV)
      break;
  }

  /// Place the embedded bundle into the \c luthier_fatbin section and retain
  /// it, so the linker's \c __start_/__stop_luthier_fatbin boundary symbols
  /// bracket exactly these bytes and the section survives \c --gc-sections.
  /// Only possible when the bundle is defined in this TU (the
  /// \c -fcuda-include-gpubinary embedding case).
  if (BundleGV && !BundleGV->isDeclaration()) {
    BundleGV->setSection("luthier_fatbin");
    llvm::appendToUsed(M, {BundleGV});
  }

  /// External references to the linker-synthesized section-boundary symbols.
  /// Named at global scope (no namespace) so they carry the unmangled symbol
  /// names the linker defines; their addresses are stored into the slots.
  auto getBoundarySymbol = [&](llvm::StringRef Name) -> llvm::GlobalVariable * {
    if (auto *Existing = M.getGlobalVariable(Name))
      return Existing;
    return new llvm::GlobalVariable(M, llvm::Type::getInt8Ty(C),
                                    /*isConstant=*/true,
                                    llvm::GlobalValue::ExternalLinkage,
                                    /*Initializer=*/nullptr, Name);
  };
  if (!SectionSlotsMap[OffloadSectionBeginAnnotation].empty())
    LUTHIER_REPORT_FATAL_ON_ERROR(
        populatePointerSlot(SectionSlotsMap[OffloadSectionBeginAnnotation],
                            OffloadSectionBeginAnnotation,
                            getBoundarySymbol("__start_luthier_fatbin")));
  if (!SectionSlotsMap[OffloadSectionEndAnnotation].empty())
    LUTHIER_REPORT_FATAL_ON_ERROR(
        populatePointerSlot(SectionSlotsMap[OffloadSectionEndAnnotation],
                            OffloadSectionEndAnnotation,
                            getBoundarySymbol("__stop_luthier_fatbin")));

  //===--------------------------------------------------------------------===//
  // Handles: every __hipRegister* kind (kernels, device/managed vars, textures,
  // surfaces) plus the CXX plugin's exported device functions collapse into one
  // HipHandles array of { void* HostHandle, const char* DeviceName }.
  //===--------------------------------------------------------------------===//

  llvm::StructType *HandleInfoTy =
      getOrCreateStruct("struct.luthier::HipHandleInfo", {PtrTy, PtrTy});
  llvm::SmallVector<llvm::Constant *, 16> Handles;
  auto addHandle = [&](llvm::Constant *HostHandle, llvm::Constant *DeviceName) {
    Handles.push_back(
        llvm::ConstantStruct::get(HandleInfoTy, {HostHandle, DeviceName}));
  };

  /// Kernels: __hipRegisterFunction(modules, hostFun, deviceFun,
  /// deviceName,...)
  if (RFUN)
    for (llvm::User *U : RFUN->users())
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *Host = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *Name = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(3));
        if (Host && Name)
          addHandle(Host, Name);
      }

  /// Device vars / textures / surfaces all share the
  /// (modules, hostHandle, deviceName, ...) prologue: handle at arg1, name at
  /// arg2.
  auto harvestVarLike = [&](llvm::Function *F) {
    if (!F)
      return;
    for (llvm::User *U : F->users())
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *Host = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *Name = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(2));
        if (Host && Name)
          addHandle(Host, Name);
      }
  };
  harvestVarLike(RDV);
  harvestVarLike(RTX);
  harvestVarLike(RSF);

  /// Managed vars: __hipRegisterManagedVar(modules, pointer, init, name, ...).
  /// \c pointer is the host-side \c void** shadow. Register it as the handle,
  /// then make the shadow point to itself (\c *pointer = pointer) so that a
  /// host dereference of the managed variable yields the registered handle —
  /// which the runtime can then resolve back to the device-side name.
  if (RMV)
    for (llvm::User *U : RMV->users())
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *Ptr = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *Name = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(3));
        if (!Ptr || !Name)
          continue;
        addHandle(Ptr, Name);
        if (auto *ShadowGV =
                llvm::dyn_cast<llvm::GlobalVariable>(Ptr->stripPointerCasts());
            ShadowGV && !ShadowGV->isDeclaration() &&
            ShadowGV->getValueType()->isPointerTy())
          ShadowGV->setInitializer(ShadowGV);
      }

  /// Exported device functions: the CXX plugin tags plain \c __host__ functions
  /// with the \c luthier.export_function_handle AnnotateAttr;
  /// \c collectAnnotatedSlots gathered them above. The host sibling shares the
  /// original \c __device__ function's exact Itanium mangling, so its own IR
  /// symbol name IS the device-side name the loader looks up.
  for (llvm::Function *Fn : ExportedDeviceFnHandles) {
    llvm::Constant *DeviceNameStr =
        llvm::ConstantDataArray::getString(C, Fn->getName(), /*AddNull=*/true);
    auto *DeviceNameGV = new llvm::GlobalVariable(
        M, DeviceNameStr->getType(), /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, DeviceNameStr,
        ".luthier.device_fn_name");
    addHandle(Fn, DeviceNameGV);
  }

  /// Emit the harvested handles as one \c { void*, const char* } array into the
  /// \c luthier_hip_handles section, retained via \c llvm.used so it survives
  /// \c --gc-sections (\c SHF_GNU_RETAIN). The trait's HipHandleSection begin /
  /// end pointer slots are pointed at the linker's
  /// \c __start_/__stop_luthier_hip_handles boundary symbols, from which the
  /// runtime reconstructs the \c ArrayRef<HipHandleInfo>. Only done when the
  /// trait is instantiated in this TU (its slots are present); an empty handle
  /// list still yields a valid empty (zero-length) section.
  if (!SectionSlotsMap[HipHandleSectionBeginAnnotation].empty() ||
      !SectionSlotsMap[HipHandleSectionEndAnnotation].empty()) {
    llvm::ArrayType *HandlesArrTy =
        llvm::ArrayType::get(HandleInfoTy, Handles.size());
    auto *HandlesData = new llvm::GlobalVariable(
        M, HandlesArrTy, /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage,
        llvm::ConstantArray::get(HandlesArrTy, Handles),
        ".luthier.hip_handles");
    HandlesData->setSection("luthier_hip_handles");
    llvm::appendToUsed(M, {HandlesData});
    if (!SectionSlotsMap[HipHandleSectionBeginAnnotation].empty())
      LUTHIER_REPORT_FATAL_ON_ERROR(populatePointerSlot(
          SectionSlotsMap[HipHandleSectionBeginAnnotation],
          HipHandleSectionBeginAnnotation,
          getBoundarySymbol("__start_luthier_hip_handles")));
    if (!SectionSlotsMap[HipHandleSectionEndAnnotation].empty())
      LUTHIER_REPORT_FATAL_ON_ERROR(
          populatePointerSlot(SectionSlotsMap[HipHandleSectionEndAnnotation],
                              HipHandleSectionEndAnnotation,
                              getBoundarySymbol("__stop_luthier_hip_handles")));
  }

  //===--------------------------------------------------------------------===//
  // Tear down the host-side HIP registration machinery.
  //===--------------------------------------------------------------------===//

  /// Make sure we remove the hip module Ctor from llvm.global_ctors, not doing
  /// so the function cannot be deleted since it still would have a use
  LUTHIER_REPORT_FATAL_ON_ERROR(deleteModuleCtor(M));
  deleteFunction(M.getFunction("__hip_module_ctor"));
  deleteFunction(M.getFunction("__hip_register_globals"));
  ///  Delete all functions that call \c __hipRegisterFatBinary and then delete
  /// \c __hipRegisterFatBinary. We do the same for
  /// \c __hipUnregisterFatBinary as well below
  for (auto *User : RFB->users()) {
    if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User)) {
      auto *Fun = CallInst->getParent()->getParent();
      deleteFunction(Fun);
    }
  }
  deleteFunction(RFB);
  for (auto *User : RUFB->users()) {
    if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User)) {
      auto *Fun = CallInst->getParent()->getParent();
      deleteFunction(Fun);
    }
  }
  deleteFunction(RUFB);
  /// Delete all uses of these functions so we can safely delete them
  if (RMV)
    deleteAllUses(RMV);
  if (RDV)
    deleteAllUses(RDV);
  if (RTX)
    deleteAllUses(RTX);
  if (RSF)
    deleteAllUses(RSF);
  if (RFUN)
    deleteAllUses(RFUN);

  /// Now that they have ZERO users, safely erase them from the Module
  if (RMV)
    deleteFunction(RMV);
  if (RDV)
    deleteFunction(RDV);
  if (RTX)
    deleteFunction(RTX);
  if (RSF)
    deleteFunction(RSF);
  if (RFUN)
    deleteFunction(RFUN);
  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
