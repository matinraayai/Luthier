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
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Demangle/Demangle.h>
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

/// Names of the \c ToolDeviceCodeOffloadParserTrait static slots the pass
/// populates. The trait no longer annotates these members, so they are located
/// by their (demangled) C++ name instead of via \c llvm.global.annotations.
static constexpr llvm::StringLiteral FatBinaryStartSlot = "FatBinaryStart";
static constexpr llvm::StringLiteral FatBinaryStopSlot = "FatBinaryStop";
static constexpr llvm::StringLiteral HipHandlesSlot = "HipHandles";

/// \brief Collect every \c ToolDeviceCodeOffloadParserTrait<Derived> static
/// slot in \p M, keyed by the slot's member name (\c FatBinaryStart /
/// \c FatBinaryStop / \c HipHandles).
///
/// The slots are \c linkonce_odr template static members; one set exists per
/// \c Derived instantiated in this TU, so each name maps to a (usually
/// single-element) vector. They are matched by demangled name rather than an
/// annotation, since the trait no longer tags them.
static void collectTraitSlots(
    llvm::Module &M,
    llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 2>> &Slots) {
  for (llvm::GlobalVariable &GV : M.globals()) {
    if (!GV.hasName())
      continue;
    std::string Demangled = llvm::demangle(GV.getName());
    llvm::StringRef D(Demangled);
    if (!D.contains("luthier::ToolDeviceCodeOffloadParserTrait"))
      continue;
    if (D.ends_with("::" + FatBinaryStartSlot.str()))
      Slots[FatBinaryStartSlot].push_back(&GV);
    else if (D.ends_with("::" + FatBinaryStopSlot.str()))
      Slots[FatBinaryStopSlot].push_back(&GV);
    else if (D.ends_with("::" + HipHandlesSlot.str()))
      Slots[HipHandlesSlot].push_back(&GV);
  }
}

/// \brief Set the initializer of an \c llvm::ArrayRef<T> placeholder slot to a
/// constant view of \p TempArr.
///
/// The slot was emitted by Clang as a \c { ptr Data; i64 Length; } global
/// (matching \c ArrayRef's ABI). We side-load a private constant data array and
/// \c setInitializer the placeholder to point at it.
static llvm::Error
populateArrayRefSlot(llvm::ArrayRef<llvm::GlobalVariable *> Slots,
                     llvm::StringRef SlotName, llvm::Type *ElemTy,
                     llvm::ArrayRef<llvm::Constant *> TempArr,
                     llvm::Module &M) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !Slots.empty(), ("No slot found for " + SlotName).str()));
  llvm::LLVMContext &C = M.getContext();
  llvm::ArrayType *ArrayTy = llvm::ArrayType::get(ElemTy, TempArr.size());
  llvm::Constant *DataInit = llvm::ConstantArray::get(ArrayTy, TempArr);

  for (auto *OldVar : Slots) {
    /// Sanity-check the placeholder's IR type: it must be a two-element struct
    /// compatible with \c ArrayRef<T>::{Data, Length}.
    auto *SlotTy = llvm::dyn_cast<llvm::StructType>(OldVar->getValueType());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        SlotTy && SlotTy->getNumElements() == 2 &&
            SlotTy->getElementType(0)->isPointerTy() &&
            SlotTy->getElementType(1)->isIntegerTy(64),
        ("ArrayRef slot '" + SlotName +
         "' is not the expected { ptr, i64 } shape; the LLVM ABI for "
         "llvm::ArrayRef may have changed.")
            .str()));
    /// Each slot gets its own private data array — they share the same payload
    /// but one GV per slot keeps mangling simple and lets each ArrayRef view
    /// its own storage cleanly.
    auto *Data = new llvm::GlobalVariable(
        M, ArrayTy, /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage,
        DataInit, ".luthier.loader." + SlotName + ".data");
    llvm::Constant *Init = llvm::ConstantStruct::get(
        SlotTy, {Data, llvm::ConstantInt::get(llvm::Type::getInt64Ty(C),
                                              TempArr.size())});
    OldVar->setInitializer(Init);
    OldVar->setConstant(true);
  }
  return llvm::Error::success();
}

/// \brief Set the initializer of a pointer slot (e.g. \c FatBinaryStart) to
/// \p Value.
static llvm::Error
populatePointerSlot(llvm::ArrayRef<llvm::GlobalVariable *> Slots,
                    llvm::StringRef SlotName, llvm::Constant *Value) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !Slots.empty(), ("No slot found for " + SlotName).str()));
  for (auto *OldVar : Slots) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        OldVar->getValueType()->isPointerTy(),
        ("Pointer slot '" + SlotName + "' is not pointer-typed.").str()));
    OldVar->setInitializer(Value);
  }
  return llvm::Error::success();
}

/// \brief Remove the \c __hip_module_ctor function from the
/// \c llvm.global_ctors array. The array is constant, so it is reconstructed
/// exactly as it was but without \c __hip_module_ctor, so the latter can be
/// deleted afterwards.
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
static llvm::Error deleteFunction(llvm::Function *Fun) {
  Fun->dropAllReferences();
  Fun->eraseFromParent();
  return llvm::Error::success();
}
/// \brief Deletes all uses of a function, so we can safely delete it after
static llvm::Error deleteAllUses(llvm::Function *Fun) {
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
  return llvm::Error::success();
}

/// \brief This pass harvests the host-handle / device-name pairs from the
/// \c __hipRegister* calls (and the CXX plugin's exported device functions)
/// into the \c ToolDeviceCodeOffloadParserTrait's \c HipHandles slot, points
/// the trait's \c FatBinaryStart / \c FatBinaryStop slots at the embedded
/// bundle's
/// \c luthier_fatbin section boundaries, then deletes the host-side HIP
/// registration machinery so the bundle is never registered with the HIP
/// runtime.
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
  /// If there is no __hipRegisterFatBinary function, there is no point looking
  /// at the others
  if (!RFB)
    return llvm::PreservedAnalyses::all();

  llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 2>> Slots;
  collectTraitSlots(M, Slots);

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
  if (!Slots[FatBinaryStartSlot].empty())
    LUTHIER_REPORT_FATAL_ON_ERROR(
        populatePointerSlot(Slots[FatBinaryStartSlot], FatBinaryStartSlot,
                            getBoundarySymbol("__start_luthier_fatbin")));
  if (!Slots[FatBinaryStopSlot].empty())
    LUTHIER_REPORT_FATAL_ON_ERROR(
        populatePointerSlot(Slots[FatBinaryStopSlot], FatBinaryStopSlot,
                            getBoundarySymbol("__stop_luthier_fatbin")));

  //===--------------------------------------------------------------------===//
  // Handles: every __hipRegister* kind (kernels, device/managed vars, textures,
  // surfaces) plus the CXX plugin's exported device functions collapse into one
  // HipHandles array of { void* HostHandle, const char* DeviceName }.
  //===--------------------------------------------------------------------===//

  llvm::StructType *HandleInfoTy = getOrCreateStruct(
      "struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo",
      {PtrTy, PtrTy});
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

  /// Exported device functions: synthesized by the CXX plugin as plain
  /// \c __host__ functions tagged with the \c luthier.export_function_handle
  /// AnnotateAttr (in \c @llvm.global.annotations). The host sibling shares the
  /// original \c __device__ function's exact Itanium mangling, so its own IR
  /// symbol name IS the device-side name the loader looks up.
  if (const llvm::GlobalVariable *Annots =
          M.getGlobalVariable("llvm.global.annotations")) {
    if (const auto *CA =
            llvm::dyn_cast<llvm::ConstantArray>(Annots->getOperand(0))) {
      for (llvm::Value *Op : CA->operands()) {
        auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Op);
        if (!CS || CS->getNumOperands() < 2)
          continue;
        auto *Fn = llvm::dyn_cast<llvm::Function>(
            CS->getOperand(0)->stripPointerCasts());
        if (!Fn)
          continue;
        auto *NameGV = llvm::dyn_cast<llvm::GlobalVariable>(
            CS->getOperand(1)->stripPointerCasts());
        if (!NameGV)
          continue;
        llvm::StringRef AnnoStr;
        if (!llvm::getConstantStringInfo(NameGV, AnnoStr))
          continue;
        if (AnnoStr != ExportFunctionHandleMarker)
          continue;
        llvm::Constant *DeviceNameStr = llvm::ConstantDataArray::getString(
            C, Fn->getName(), /*AddNull=*/true);
        auto *DeviceNameGV = new llvm::GlobalVariable(
            M, DeviceNameStr->getType(), /*isConstant=*/true,
            llvm::GlobalValue::PrivateLinkage, DeviceNameStr,
            ".luthier.device_fn_name");
        addHandle(Fn, DeviceNameGV);
      }
    }
  }

  /// Populate the single unified HipHandles slot (when the tool instantiated
  /// the trait in this TU). An empty handle list still yields a valid empty
  /// ArrayRef.
  if (!Slots[HipHandlesSlot].empty())
    LUTHIER_REPORT_FATAL_ON_ERROR(populateArrayRefSlot(
        Slots[HipHandlesSlot], HipHandlesSlot, HandleInfoTy, Handles, M));

  //===--------------------------------------------------------------------===//
  // Tear down the host-side HIP registration machinery.
  //===--------------------------------------------------------------------===//

  /// Make sure we remove the hip module Ctor from llvm.global_ctors, not doing
  /// so the function cannot be deleted since it still would have a use
  LUTHIER_REPORT_FATAL_ON_ERROR(deleteModuleCtor(M));
  LUTHIER_REPORT_FATAL_ON_ERROR(
      deleteFunction(M.getFunction("__hip_module_ctor")));
  LUTHIER_REPORT_FATAL_ON_ERROR(
      deleteFunction(M.getFunction("__hip_register_globals")));
  ///  Delete all functions that call __hipRegisterFatBinary and then delete
  /// __hipRegisterFatBinary as well, we do the same for
  /// __hipUnregisterFatBinary as well below
  for (auto *User : RFB->users()) {
    if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User)) {
      auto *Fun = CallInst->getParent()->getParent();
      LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(Fun));
    }
  }
  LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(RFB));
  for (auto *User : RUFB->users()) {
    if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User)) {
      auto *Fun = CallInst->getParent()->getParent();
      LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(Fun));
    }
  }
  LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(RUFB));
  /// Delete all uses of these functions so we can safely delete them
  if (RMV)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteAllUses(RMV));
  if (RDV)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteAllUses(RDV));
  if (RTX)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteAllUses(RTX));
  if (RSF)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteAllUses(RSF));
  if (RFUN)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteAllUses(RFUN));

  /// Now that they have ZERO users, safely erase them from the Module
  if (RMV)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(RMV));
  if (RDV)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(RDV));
  if (RTX)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(RTX));
  if (RSF)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(RSF));
  if (RFUN)
    LUTHIER_REPORT_FATAL_ON_ERROR(deleteFunction(RFUN));
  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
