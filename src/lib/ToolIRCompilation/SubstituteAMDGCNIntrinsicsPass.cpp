//===-- SubstituteAMDGCNIntrinsicsPass.cpp --------------------------------===//
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
#include "luthier/ToolIRCompilation/SubstituteAMDGCNIntrinsicsPass.h"
#include "luthier/Intrinsic/IntrinsicCalls.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <cstdint>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/Support/Alignment.h>
#include <utility>

namespace luthier {

namespace {

/// Byte offsets of \c hidden_group_size_{x,y} in the AMDGPU code-object-v5
/// implicit-arg buffer (see the AMDGPU implicit-arg ABI docs). Each field is a
/// \c uint16_t.
constexpr int64_t HiddenGroupSizeXOffset = 12;
constexpr int64_t HiddenGroupSizeYOffset = 14;

/// Emit \c call i32 @luthier::readSVA(i8 SVA) at the current builder
/// position and return its value. \p SVA is passed as its \c uint8_t
/// underlying value so the intrinsic-name mangling matches other
/// \c luthier::readSVA calls in the module (which use an \c i8 arg type).
llvm::Value *emitReadSVAI32(llvm::Module &M, llvm::IRBuilderBase &Builder,
                            ScalarValueArgument SVA) {
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(M.getContext());
  return insertCallToIntrinsic(M, Builder, "luthier::readSVA", *Int32Ty,
                               static_cast<uint8_t>(SVA));
}

/// Emit \c call i64 @luthier::readSVA(i8 SVA) at the current builder position
/// and return its value. Used for 2-lane 64-bit SVAs (e.g. \c DISPATCH_ID)
/// whose corresponding AMDGCN intrinsic returns an \c i64 directly.
llvm::Value *emitReadSVAI64(llvm::Module &M, llvm::IRBuilderBase &Builder,
                            ScalarValueArgument SVA) {
  llvm::Type *Int64Ty = llvm::Type::getInt64Ty(M.getContext());
  return insertCallToIntrinsic(M, Builder, "luthier::readSVA", *Int64Ty,
                               static_cast<uint8_t>(SVA));
}

/// Emit \c call i64 @luthier::readSVA(i8 SVA) at the current builder position,
/// then \c inttoptr the i64 to \c ptr addrspace(4) — the pointer type the
/// corresponding AMDGCN \c *.ptr intrinsic (e.g. \c llvm.amdgcn.implicitarg.ptr,
/// \c llvm.amdgcn.dispatch.ptr) returns.
llvm::Value *emitReadSVAConstPtr(llvm::Module &M, llvm::IRBuilderBase &Builder,
                                 ScalarValueArgument SVA,
                                 const llvm::Twine &Name) {
  auto &Ctx = M.getContext();
  llvm::PointerType *ConstPtrTy =
      llvm::PointerType::get(Ctx, llvm::AMDGPUAS::CONSTANT_ADDRESS);
  llvm::Value *PtrI64 = emitReadSVAI64(M, Builder, SVA);
  return Builder.CreateIntToPtr(PtrI64, ConstPtrTy, Name);
}

/// Emit an IR sequence that computes the current lane's \c threadIdx.<Dim>
/// value from the SVA-preserved lane-0 workitem IDs, the lane's position
/// within its wave, and the workgroup dimensions from the implicit-arg
/// buffer.
///
/// Given lane 0 of the wave has workitem coordinates (X0, Y0, Z0) and the
/// workgroup has X-major linear layout with sizes Wx, Wy, the current
/// lane at position \c lane in the wave has linear index
/// \c L = X0 + Y0*Wx + Z0*Wx*Wy + lane , and
/// \code
///   tid.x = L % Wx
///   tid.y = (L / Wx) % Wy
///   tid.z = L / (Wx * Wy)
/// \endcode
///
/// The setup lines (readSVA + implicit-arg loads + mbcnt) are identical
/// across dimensions; later CSE passes will merge repeated expansions
/// when a payload reads multiple dimensions.
llvm::Value *expandWorkitemId(llvm::Module &M, llvm::IRBuilderBase &Builder,
                              unsigned Dim) {
  auto &Ctx = M.getContext();
  llvm::Type *Int16Ty = llvm::Type::getInt16Ty(Ctx);
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);

  // Load Wx and Wy from the COV6 implicit-arg buffer.
  //   %iap = readSVA(IMPLICIT_ARG_BUFFER) + inttoptr
  //   %wx  = zext(load i16, %iap + 12) to i32
  //   %wy  = zext(load i16, %iap + 14) to i32
  llvm::Value *ImplicitArgPtr =
      emitReadSVAConstPtr(M, Builder, IMPLICIT_ARG_BUFFER, "iap");
  llvm::Value *WxPtr = Builder.CreateConstGEP1_64(
      Builder.getInt8Ty(), ImplicitArgPtr, HiddenGroupSizeXOffset, "wx.p");
  llvm::LoadInst *Wx16 =
      Builder.CreateAlignedLoad(Int16Ty, WxPtr, llvm::Align(2), "wx16");
  llvm::Value *Wx = Builder.CreateZExt(Wx16, Int32Ty, "wx");

  llvm::Value *WyPtr = Builder.CreateConstGEP1_64(
      Builder.getInt8Ty(), ImplicitArgPtr, HiddenGroupSizeYOffset, "wy.p");
  llvm::LoadInst *Wy16 =
      Builder.CreateAlignedLoad(Int16Ty, WyPtr, llvm::Align(2), "wy16");
  llvm::Value *Wy = Builder.CreateZExt(Wy16, Int32Ty, "wy");

  // Read lane-0's (X0, Y0, Z0) preserved in the SVA.
  //   %X0/Y0/Z0 = call i32 @luthier::readSVA(i8 WORKITEM_ID_{X,Y,Z})
  llvm::Value *X0 = emitReadSVAI32(M, Builder, WORKITEM_ID_X);
  llvm::Value *Y0 = emitReadSVAI32(M, Builder, WORKITEM_ID_Y);
  llvm::Value *Z0 = emitReadSVAI32(M, Builder, WORKITEM_ID_Z);

  // Compute lane-in-wave using the standard mbcnt idiom. Works on both
  // wave32 (mbcnt.hi(-1, x) returns x unchanged for lane_id < 32) and
  // wave64 (mbcnt.lo saturates at 32, mbcnt.hi contributes lane_id - 32).
  //   %lane_lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  //   %lane    = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane_lo)
  llvm::Value *NegOne = llvm::ConstantInt::getSigned(Int32Ty, -1);
  llvm::Value *Zero32 = Builder.getInt32(0);
  llvm::Value *LaneLo = Builder.CreateIntrinsic(
      Int32Ty, llvm::Intrinsic::amdgcn_mbcnt_lo, {NegOne, Zero32});
  LaneLo->setName("lane.lo");
  llvm::Value *Lane = Builder.CreateIntrinsic(
      Int32Ty, llvm::Intrinsic::amdgcn_mbcnt_hi, {NegOne, LaneLo});
  Lane->setName("lane");

  // Accumulate the per-lane linear workitem index.
  //   %YWx   = Y0 * Wx
  //   %WxWy  = Wx * Wy
  //   %ZWxWy = Z0 * WxWy
  //   %L     = X0 + YWx + ZWxWy + lane
  llvm::Value *YWx = Builder.CreateMul(Y0, Wx, "y.wx");
  llvm::Value *WxWy = Builder.CreateMul(Wx, Wy, "wx.wy");
  llvm::Value *ZWxWy = Builder.CreateMul(Z0, WxWy, "z.wx.wy");
  llvm::Value *L0a = Builder.CreateAdd(X0, YWx, "l0.a");
  llvm::Value *L0 = Builder.CreateAdd(L0a, ZWxWy, "l0");
  llvm::Value *L = Builder.CreateAdd(L0, Lane, "l");

  switch (Dim) {
  case 0:
    return Builder.CreateURem(L, Wx, "tid.x");
  case 1: {
    llvm::Value *Ldx = Builder.CreateUDiv(L, Wx, "l.dx");
    return Builder.CreateURem(Ldx, Wy, "tid.y");
  }
  case 2:
    return Builder.CreateUDiv(L, WxWy, "tid.z");
  default:
    llvm_unreachable("Dim must be 0, 1, or 2");
  }
}

} // namespace

llvm::PreservedAnalyses
SubstituteAMDGCNIntrinsicsPass::run(llvm::Module &M,
                                    llvm::ModuleAnalysisManager &) {
  llvm::Triple T(M.getTargetTriple());
  /// Only operate on device code
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();

  /// Substitution table. Each entry maps an amdgcn intrinsic name prefix to a
  /// rewriter that, given an insertion point on \c Builder, emits the
  /// replacement IR (routed through \c luthier::readSVA of the appropriate
  /// SVA slot) and returns the value that should replace uses of the original
  /// call.
  using Rewriter =
      std::function<llvm::Value *(llvm::Module &, llvm::IRBuilderBase &)>;
  const llvm::SmallVector<std::pair<llvm::StringRef, Rewriter>, 8>
      Substitutions = {
          {"llvm.amdgcn.workgroup.id.x",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAI32(M, B, WORKGROUP_ID_X);
           }},
          {"llvm.amdgcn.workgroup.id.y",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAI32(M, B, WORKGROUP_ID_Y);
           }},
          {"llvm.amdgcn.workgroup.id.z",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAI32(M, B, WORKGROUP_ID_Z);
           }},
          {"llvm.amdgcn.implicitarg.ptr",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAConstPtr(M, B, IMPLICIT_ARG_BUFFER, "iap");
           }},
          {"llvm.amdgcn.kernarg.segment.ptr",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAConstPtr(M, B, KERNEL_ARG_PTR, "kap");
           }},
          {"llvm.amdgcn.dispatch.ptr",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAConstPtr(M, B, DISPATCH_PTR, "dp");
           }},
          {"llvm.amdgcn.queue.ptr",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAConstPtr(M, B, QUEUE_PTR, "qp");
           }},
          {"llvm.amdgcn.dispatch.id",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return emitReadSVAI64(M, B, DISPATCH_ID);
           }},
          {"llvm.amdgcn.workitem.id.x",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return expandWorkitemId(M, B, 0);
           }},
          {"llvm.amdgcn.workitem.id.y",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return expandWorkitemId(M, B, 1);
           }},
          {"llvm.amdgcn.workitem.id.z",
           [](llvm::Module &M, llvm::IRBuilderBase &B) {
             return expandWorkitemId(M, B, 2);
           }},
      };

  for (llvm::Function &F : llvm::make_early_inc_range(M.functions())) {
    for (const auto &[Prefix, Emit] : Substitutions) {
      if (!F.getName().starts_with(Prefix))
        continue;
      for (llvm::User *U : llvm::make_early_inc_range(F.users())) {
        auto *CI = llvm::dyn_cast<llvm::CallInst>(U);
        if (CI == nullptr)
          continue;
        llvm::IRBuilder<> Builder(CI);
        llvm::Value *NewValue = Emit(M, Builder);
        CI->replaceAllUsesWith(NewValue);
        CI->eraseFromParent();
      }
      F.dropAllReferences();
      F.eraseFromParent();
      break;
    }
  }

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
