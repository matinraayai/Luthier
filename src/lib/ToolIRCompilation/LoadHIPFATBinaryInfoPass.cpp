//===-- LoadHIPFATBinaryInfoPass.cpp ----------------------------*- C++ -*-===//
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
/// \file LoadHIPFATBinaryInfoPass.cpp
/// This plugin removes all __hip_register functions from the IR and stores the
/// information inside the annotated variables. This is done so we can delegate
/// the registering of functions, fat bin wrappers, variables etc. to the Tool
/// Executable Loader instead of using HIP for it, whcih has proven to be
/// unreliable
//===----------------------------------------------------------------------===//
#include "luthier/ToolIRCompilation/LoadHIPFATBinaryInfoPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <cstring>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Plugins/PassPlugin.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-load-hip-fat-binary-info-pass"

namespace luthier {
/// \brief Fills out a StringMap with all annotated variables, we do this so we
/// can access them later without needing to search again.
/// \param M Module of the IR
/// \param NameToVar StringMap which we will fill out
static llvm::Error getAnnotatedValues(
    const llvm::Module &M,
    llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 4>> &NameToVar) {
  const llvm::GlobalVariable *V =
      M.getGlobalVariable("llvm.global.annotations");
  if (V == nullptr)
    return llvm::Error::success();
  const llvm::ConstantArray *CA = cast<llvm::ConstantArray>(V->getOperand(0));
  for (llvm::Value *Op : CA->operands()) {
    auto *CS = cast<llvm::ConstantStruct>(Op);
    /// The first field of the struct contains a pointer to the annotated
    /// variable.
    llvm::Value *AnnotatedVal = CS->getOperand(0)->stripPointerCasts();
    if (auto *Var = llvm::dyn_cast<llvm::GlobalVariable>(AnnotatedVal)) {
      /// The second field contains a pointer to a global annotation string.
      auto *GV =
          cast<llvm::GlobalVariable>(CS->getOperand(1)->stripPointerCasts());
      llvm::StringRef Content;
      llvm::getConstantStringInfo(GV, Content);
      if (Content.starts_with("luthier.loader.hip")) {
        /// Inline-static class-template slots in
        /// \c DeviceToolCodeFatBinaryParser<Derived> get emitted with
        /// linkonce_odr linkage; if multiple TUs ODR-use the class, the
        /// linker collapses the GVs but \c llvm.global.annotations (which
        /// has appending linkage) accumulates one entry per TU. De-dup so
        /// the later RAUW + erase loop doesn't touch a freed value.
        auto &Bucket = NameToVar[Content];
        if (llvm::find(Bucket, Var) == Bucket.end())
          Bucket.push_back(Var);
      }
    }
  }
  return llvm::Error::success();
}
/// \brief Set the initializer of an annotated \c llvm::ArrayRef<T> placeholder
/// to a constant view of \p TempArr.
///
/// The corresponding slot on \c DeviceToolCodeFatBinaryParser<Derived> is
/// declared \c inline \c static \c llvm::ArrayRef<T>{}, so Clang has already
/// emitted each placeholder as a global of struct type
/// \c { ptr Data; i64 Length; } (matching ArrayRef's ABI). We side-load a
/// private constant data array and \c setInitializer on the placeholder; no
/// retype/RAUW dance is needed.
///
/// \param Name Annotation string identifying the placeholder slot(s).
/// \param ElemTy IR type of each entry inside the data array.
/// \param TempArr The constants making up the data array.
/// \param M Containing module.
/// \param NameToVar Map of annotation string to placeholder GVs.
static llvm::Error populateArrayRefSlot(
    llvm::StringRef Name, llvm::Type *ElemTy,
    llvm::ArrayRef<llvm::Constant *> TempArr, llvm::Module &M,
    llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 4>> &NameToVar) {
  if (NameToVar[Name].empty()) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        ("There is no annotated variable for " + Name).str());
  }
  llvm::LLVMContext &C = M.getContext();
  llvm::ArrayType *ArrayTy = llvm::ArrayType::get(ElemTy, TempArr.size());
  llvm::Constant *DataInit = llvm::ConstantArray::get(ArrayTy, TempArr);

  for (auto *OldVar : NameToVar[Name]) {
    /// Sanity-check the placeholder's IR type: it must be a two-element
    /// struct compatible with ArrayRef<T>::{Data, Length}.
    auto *SlotTy = llvm::dyn_cast<llvm::StructType>(OldVar->getValueType());
    if (!SlotTy || SlotTy->getNumElements() != 2 ||
        !SlotTy->getElementType(0)->isPointerTy() ||
        !SlotTy->getElementType(1)->isIntegerTy(64)) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          ("Annotated ArrayRef slot '" + Name +
           "' is not the expected { ptr, i64 } shape; LLVM ABI for "
           "llvm::ArrayRef may have changed.")
              .str());
    }
    /// Each placeholder gets its own private data array — they share the
    /// same payload but having one GV per slot keeps mangling simple and
    /// lets each ArrayRef view its own storage cleanly.
    auto *Data = new llvm::GlobalVariable(
        M, ArrayTy, /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage,
        DataInit, ".luthier.loader." + Name + ".data");
    llvm::Constant *DataPtr = Data;
    if (DataPtr->getType() != SlotTy->getElementType(0))
      DataPtr =
          llvm::ConstantExpr::getBitCast(DataPtr, SlotTy->getElementType(0));
    llvm::Constant *Init = llvm::ConstantStruct::get(
        SlotTy, {DataPtr, llvm::ConstantInt::get(llvm::Type::getInt64Ty(C),
                                                 TempArr.size())});
    OldVar->setInitializer(Init);
    OldVar->setConstant(true);
  }
  return llvm::Error::success();
}

/// \brief Set the initializer of an annotated single-struct placeholder — a
/// two-element \c { ptr, i64 } slot such as \c HipFatBinaryInfo — directly to
/// \c { Ptr, Val }.
///
/// Unlike \c populateArrayRefSlot there is no side data array: the slot itself
/// IS the struct (the Luthier-side field is a lone \c HipFatBinaryInfo, not an
/// \c llvm::ArrayRef). The initializer is built with the slot's own type so it
/// matches whatever named struct Clang emitted for the field.
static llvm::Error populateStructSlot(
    llvm::StringRef Name, llvm::Constant *Ptr, uint64_t Val, llvm::Module &M,
    llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 4>> &NameToVar) {
  if (NameToVar[Name].empty()) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        ("There is no annotated variable for " + Name).str());
  }
  llvm::LLVMContext &C = M.getContext();
  for (auto *OldVar : NameToVar[Name]) {
    auto *SlotTy = llvm::dyn_cast<llvm::StructType>(OldVar->getValueType());
    if (!SlotTy || SlotTy->getNumElements() != 2 ||
        !SlotTy->getElementType(0)->isPointerTy() ||
        !SlotTy->getElementType(1)->isIntegerTy(64)) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          ("Annotated struct slot '" + Name +
           "' is not the expected { ptr, i64 } shape.")
              .str());
    }
    llvm::Constant *P = Ptr;
    if (P->getType() != SlotTy->getElementType(0))
      P = llvm::ConstantExpr::getBitCast(P, SlotTy->getElementType(0));
    OldVar->setInitializer(llvm::ConstantStruct::get(
        SlotTy, {P, llvm::ConstantInt::get(llvm::Type::getInt64Ty(C), Val)}));
    OldVar->setConstant(true);
  }
  return llvm::Error::success();
}

/// \brief Removed __hip_module_ctor function from the llvm.global_ctors array.
/// This array is constant so we have to reconstruct it exactly as it was but
/// without the __hip_module_ctor, so we can delete it after
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
/// \brief This pass first finds all instances of __hip_register functions and
/// fills out the annotated variables that will hold all the information the
/// Tool Executable Loader needs to register them with hip. After that it
/// deletes all functions that called __hipRegisterFatBinary and all
/// __hip_register functions
llvm::PreservedAnalyses
LoadHIPFATBinaryInfoPass::run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
  llvm::Triple T(M.getTargetTriple());
  bool Changed = false;
  // Only operate on host
  if (T.getArch() == llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();
  llvm::LLVMContext &C = M.getContext();
  llvm::StringMap<llvm::SmallVector<llvm::GlobalVariable *, 4>> NameToVar{};
  LUTHIER_REPORT_FATAL_ON_ERROR(getAnnotatedValues(M, NameToVar));

  /// We get all the functions we need to process, for each one we loop over the
  /// uses finding the call instances, and for each one we construct a struct
  /// holding all the information necessary to register the hip object with hip
  /// through the TEL
  llvm::Function *RFB = M.getFunction("__hipRegisterFatBinary");
  llvm::Function *RUFB = M.getFunction("__hipUnregisterFatBinary");
  llvm::Function *RFUN = M.getFunction("__hipRegisterFunction");
  llvm::Function *RMV = M.getFunction("__hipRegisterManagedVar");
  llvm::Function *RDV = M.getFunction("__hipRegisterVar");
  llvm::Function *RTX = M.getFunction("__hipRegisterTexture");
  llvm::Function *RSF = M.getFunction("__hipRegisterSurface");
  /// If there is no __hipRegisterFatBinary function, there is no point looking
  /// at the others
  if (!RFB) {
    return llvm::PreservedAnalyses::all();
  }
  /// There should only be one hip binary, but we generalize to assume there
  /// Helper to look up / create the named element struct type for a Hip*Info
  /// data array. The Luthier-side C++ struct definitions in
  /// \c DeviceToolCodeFatBinaryParser.h are the source of truth for the
  /// field shapes encoded here.
  auto getOrCreateStruct =
      [&C](llvm::StringRef Name,
           llvm::ArrayRef<llvm::Type *> Fields) -> llvm::StructType * {
    if (auto *Existing = llvm::StructType::getTypeByName(C, Name))
      return Existing;
    return llvm::StructType::create(C, Fields, Name);
  };
  llvm::Type *PtrTy = llvm::PointerType::getUnqual(C);
  llvm::Type *I64Ty = llvm::Type::getInt64Ty(C);
  llvm::Type *I32Ty = llvm::Type::getInt32Ty(C);

  /// Each \c __hipRegisterFatBinary call site hands us the host-side
  /// \c __CudaFatBinaryWrapper GV \c { i32 magic, i32 version, ptr binary,
  /// ptr dummy }. We chase \c binary through any constant-expr bitcasts to
  /// the bundle storage GV, and emit \c { ptr Bundle, i64 GV-array-size }
  /// — letting the runtime hand a correctly-sized \c MemoryBufferRef to
  /// LLVM's \c OffloadBundleFatBin parser without re-deriving the size.
  // Luthier embeds exactly one fat binary per tool TU, so the HipFatBinaries
  // slot is a single HipFatBinaryInfo, not an ArrayRef. Find the first
  // __hipRegisterFatBinary call's bundle GV and write { Bundle, Size } directly
  // into the slot.
  llvm::GlobalVariable *BundleGV = nullptr;
  uint64_t FatBinSize = 0;
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
    if (!BundleGV)
      continue;
    // HIP-Clang's dual-emission gives `__hip_fatbin` a fully-typed
    // `[N x i8]` initializer in the SAME TU as the
    // `__hipRegisterFatBinary` call, so the size is visible at IR time.
    // `luthier_create_offload_bundle`'s split-compile design can leave the
    // bytes separate from the registration TU, leaving `__hip_fatbin` as
    // `external constant i8` here — no ArrayType, no compile-time size. In
    // that case emit Size=0 and let
    // the loader discover the bundle extent at runtime by walking the
    // `.hip_fatbin` ELF section that contains the BundleGV's address.
    if (auto *ArrTy =
            llvm::dyn_cast<llvm::ArrayType>(BundleGV->getValueType())) {
      FatBinSize = ArrTy->getNumElements() *
                   ArrTy->getElementType()->getScalarSizeInBits() / 8;
    }
    break;
  }
  if (BundleGV)
    LUTHIER_REPORT_FATAL_ON_ERROR(populateStructSlot(
        HipFatBinariesAttr, BundleGV, FatBinSize, M, NameToVar));

  if (RFUN) {
    llvm::StructType *KernelInfoTy = getOrCreateStruct(
        "struct.luthier::DeviceToolCodeFatBinaryParser::HipKernelInfo",
        {PtrTy, PtrTy});
    llvm::SmallVector<llvm::Constant *, 10> KernelHandles{};
    for (llvm::User *U : RFUN->users()) {
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *Ptr = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *Name = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(3));
        if (Ptr && Name)
          KernelHandles.push_back(
              llvm::ConstantStruct::get(KernelInfoTy, {Ptr, Name}));
      }
    }
    LUTHIER_REPORT_FATAL_ON_ERROR(populateArrayRefSlot(
        HipKernelsAttr, KernelInfoTy, KernelHandles, M, NameToVar));
  }

  /// Harvest device-function handles. Unlike kernels, these don't go
  /// through \c __hipRegisterFunction — they're synthesized by the CXX
  /// plugin as plain \c __host__ functions tagged with the
  /// \c luthier.export_function_handle AnnotateAttr. The annotation
  /// lands in \c @llvm.global.annotations; scan it and collect each
  /// tagged \c Function as a \c { HostHandle, DeviceName } record where
  /// \c DeviceName is the function's own Itanium-mangled name (it's
  /// shared between the host sibling and the original \c __device__
  /// emission, so the loader can use it directly to look up the
  /// device-side hook in the IModule).
  {
    llvm::StructType *DevFnInfoTy = getOrCreateStruct(
        "struct.luthier::DeviceToolCodeFatBinaryParser::HipDeviceFunctionInfo",
        {PtrTy, PtrTy});
    llvm::SmallVector<llvm::Constant *, 10> DevFnEntries{};
    if (const llvm::GlobalVariable *Annots =
            M.getGlobalVariable("llvm.global.annotations")) {
      const auto *CA =
          llvm::dyn_cast<llvm::ConstantArray>(Annots->getOperand(0));
      if (CA) {
        for (llvm::Value *Op : CA->operands()) {
          auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Op);
          if (!CS || CS->getNumOperands() < 2)
            continue;
          /// Operand 0: pointer to the annotated GV (here, the host
          /// sibling Function); operand 1: pointer to the annotation
          /// string GV.
          auto *AnnotatedFn = llvm::dyn_cast<llvm::Function>(
              CS->getOperand(0)->stripPointerCasts());
          if (!AnnotatedFn)
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
          /// Compute the device-side name corresponding to this host
          /// sibling. The CXX plugin uses two different synthesis paths
          /// that we have to decode uniformly:
          ///
          /// * Non-templated dual-overload (most cases): the sibling
          ///   has the *same* source identifier as the original
          ///   \c __device__ function. Their IR symbols match
          ///   verbatim, so the sibling's IR symbol IS the device
          ///   function's name in the IModule. No further work
          ///
          /// * Templated per-specialization handle: the sibling has a
          ///   distinct source identifier of the form
          ///   \c <DevFuncHandlePrefix><orig-Itanium-mangling>, then
          ///   gets Itanium-mangled itself (C++ linkage at TU scope).
          ///   We have to recover the source identifier and strip
          ///   the prefix to get the original's mangled name
          ///
          /// We try to decode as a templated handle first: demangle
          /// the sibling's IR symbol (or use it directly if it isn't
          /// Itanium-mangled, e.g. inside an \c extern \c "C" block),
          /// and check whether its source identifier starts with
          /// \c DevFuncHandlePrefix. If yes, strip and use that. If
          /// no, fall through to the non-templated case: the IR symbol
          /// is already the device-side name
          llvm::StringRef SibFullName = AnnotatedFn->getName();
          llvm::ItaniumPartialDemangler Demangler;
          std::string SibName = SibFullName.str();
          char *BaseBuf = nullptr;
          llvm::StringRef SourceIdent;
          if (!Demangler.partialDemangle(SibName.c_str())) {
            size_t BaseSize = 0;
            BaseBuf = Demangler.getFunctionBaseName(/*Buf=*/nullptr, &BaseSize);
            if (BaseBuf)
              /// \c BaseSize is the buffer size including the trailing
              /// NUL (and possibly slack); use \c strlen to get the
              /// actual string length so we don't carry the NUL through
              /// into the DeviceName string we emit.
              SourceIdent = llvm::StringRef(BaseBuf, std::strlen(BaseBuf));
          } else {
            /// Not Itanium-mangled — the IR symbol IS the source
            /// identifier (C linkage sibling, e.g. inside extern "C").
            SourceIdent = SibFullName;
          }
          std::string OrigMangled;
          if (SourceIdent.starts_with(DevFuncHandlePrefix))
            OrigMangled =
                SourceIdent.drop_front(DevFuncHandlePrefix.size()).str();
          else
            OrigMangled = SibFullName.str();
          std::free(BaseBuf);
          llvm::Constant *DeviceNameStr = llvm::ConstantDataArray::getString(
              C, OrigMangled, /*AddNull=*/true);
          auto *DeviceNameGV = new llvm::GlobalVariable(
              M, DeviceNameStr->getType(), /*isConstant=*/true,
              llvm::GlobalValue::PrivateLinkage, DeviceNameStr,
              ".luthier.device_fn_name");
          DevFnEntries.push_back(llvm::ConstantStruct::get(
              DevFnInfoTy, {AnnotatedFn, DeviceNameGV}));
        }
      }
    }
    /// Skip the slot population entirely when no tagged device-function
    /// handles were found. Tests / TUs that don't use the export-handle
    /// machinery (or only have kernels) don't emit the
    /// \c HipDeviceFunctions slot, and \c populateArrayRefSlot would
    /// otherwise error trying to find a missing annotated GV.
    if (!DevFnEntries.empty())
      LUTHIER_REPORT_FATAL_ON_ERROR(populateArrayRefSlot(
          HipDeviceFunctionsAttr, DevFnInfoTy, DevFnEntries, M, NameToVar));
  }

  if (RMV) {
    llvm::StructType *ManagedVarTy = getOrCreateStruct(
        "struct.luthier::DeviceToolCodeFatBinaryParser::HipManagedVarInfo",
        {PtrTy, PtrTy, PtrTy, I64Ty, I32Ty});
    llvm::SmallVector<llvm::Constant *, 10> ManagedVars{};
    for (llvm::User *U : RMV->users()) {
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *ConstPtr = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *ConstInitValue =
            llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(2));
        auto *ConstName = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(3));
        auto *ConstSize = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(4));
        auto *ConstAlign = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(5));
        if (ConstPtr && ConstInitValue && ConstName && ConstSize && ConstAlign)
          ManagedVars.push_back(llvm::ConstantStruct::get(
              ManagedVarTy,
              {ConstPtr, ConstInitValue, ConstName, ConstSize, ConstAlign}));
      }
    }
    LUTHIER_REPORT_FATAL_ON_ERROR(populateArrayRefSlot(
        HipManagedVarsAttr, ManagedVarTy, ManagedVars, M, NameToVar));
  }

  if (RDV) {
    llvm::StructType *VarInfoTy = getOrCreateStruct(
        "struct.luthier::DeviceToolCodeFatBinaryParser::HipDeviceVarInfo",
        {PtrTy, PtrTy});
    llvm::SmallVector<llvm::Constant *, 10> DeviceVars{};
    for (llvm::User *U : RDV->users()) {
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *Ptr = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *ConstName = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(2));
        if (Ptr && ConstName)
          DeviceVars.push_back(
              llvm::ConstantStruct::get(VarInfoTy, {Ptr, ConstName}));
      }
    }
    LUTHIER_REPORT_FATAL_ON_ERROR(populateArrayRefSlot(
        HipDeviceVarsAttr, VarInfoTy, DeviceVars, M, NameToVar));
  }

  if (RTX) {
    llvm::StructType *TextureInfoTy = getOrCreateStruct(
        "struct.luthier::DeviceToolCodeFatBinaryParser::HipTextureInfo",
        {PtrTy, PtrTy});
    llvm::SmallVector<llvm::Constant *, 10> Textures{};
    for (llvm::User *U : RTX->users()) {
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *Ptr = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *ConstName = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(2));
        if (Ptr && ConstName)
          Textures.push_back(
              llvm::ConstantStruct::get(TextureInfoTy, {Ptr, ConstName}));
      }
    }
    LUTHIER_REPORT_FATAL_ON_ERROR(populateArrayRefSlot(
        HipTextureVarsAttr, TextureInfoTy, Textures, M, NameToVar));
  }

  if (RSF) {
    llvm::StructType *SurfaceInfoTy = getOrCreateStruct(
        "struct.luthier::DeviceToolCodeFatBinaryParser::HipSurfaceInfo",
        {PtrTy, PtrTy});
    llvm::SmallVector<llvm::Constant *, 10> Surfaces{};
    for (llvm::User *U : RSF->users()) {
      if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        auto *Ptr = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(1));
        auto *ConstName = llvm::dyn_cast<llvm::Constant>(CB->getArgOperand(2));
        if (Ptr && ConstName)
          Surfaces.push_back(
              llvm::ConstantStruct::get(SurfaceInfoTy, {Ptr, ConstName}));
      }
    }
    LUTHIER_REPORT_FATAL_ON_ERROR(populateArrayRefSlot(
        HipSurfaceVarsAttr, SurfaceInfoTy, Surfaces, M, NameToVar));
  }
  /// Make sure we remove the hip module Ctor from  llvm.global_ctors, not doing
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