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
#include "luthier/ToolCodeGen/CustomKernargLayout.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <initializer_list>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <tuple>

namespace luthier {

namespace {

/// Emit IR that recomputes threadIdx.\p Dim (0=x,1=y,2=z) at \p Builder. The
/// preloaded work-item-id VGPR is only valid at kernel entry, so the value is
/// recovered from invariants available anywhere: the wave's lane-0 work-item id
/// (packed x|y<<10|z<<20), captured into the WORKITEM_ID_PACKED_LANE0 SVA at
/// entry, gives the wave's flat base; adding the lane index (mbcnt) gives this
/// lane's flat work-item id, which decomposes by blockDim (group_size from the
/// implicit args):
///   flat = (z0*Bx*By + y0*Bx + x0) + lane
///   x = flat % Bx ; y = (flat / Bx) % By ; z = flat / (Bx*By)
/// Emitted per use; the IModule's later -O pipeline CSEs the shared subgraph.
llvm::Value *emitWorkitemIdRecompute(llvm::Module &M, llvm::IRBuilderBase &B,
                                     unsigned Dim, bool Wave64) {
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::Type *I32 = B.getInt32Ty();
  llvm::Type *I16 = B.getInt16Ty();
  llvm::Type *I8 = B.getInt8Ty();
  llvm::PointerType *ConstPtr =
      llvm::PointerType::get(Ctx, llvm::AMDGPUAS::CONSTANT_ADDRESS);

  // The wave's lane-0 packed work-item id, captured into the SVA at entry.
  llvm::Value *Packed =
      insertCallToIntrinsic(M, B, "luthier::readSVA", *I32,
                            static_cast<uint8_t>(WORKITEM_ID_PACKED_LANE0));
  llvm::Value *X0 = B.CreateAnd(Packed, B.getInt32(0x3ff));
  llvm::Value *Y0 =
      B.CreateAnd(B.CreateLShr(Packed, B.getInt32(10)), B.getInt32(0x3ff));
  llvm::Value *Z0 =
      B.CreateAnd(B.CreateLShr(Packed, B.getInt32(20)), B.getInt32(0x3ff));

  // blockDim.x / .y from the COV5 group_size implicit args (u16).
  llvm::Value *IP =
      insertCallToIntrinsic(M, B, "luthier::implicitArgPtr", *ConstPtr);
  auto LoadGroupSize = [&](uint32_t Off) -> llvm::Value * {
    llvm::Value *P = B.CreateGEP(I8, IP, B.getInt32(Off));
    return B.CreateZExt(B.CreateLoad(I16, P), I32);
  };
  llvm::Value *Bx = LoadGroupSize(cov5::GroupSizeX);
  llvm::Value *By = LoadGroupSize(cov5::GroupSizeY);

  // lane id within the wave (EXEC-independent: mask = -1).
  llvm::Value *Lane = B.CreateIntrinsic(I32, llvm::Intrinsic::amdgcn_mbcnt_lo,
                                        {B.getInt32(-1), B.getInt32(0)});
  if (Wave64)
    Lane = B.CreateIntrinsic(I32, llvm::Intrinsic::amdgcn_mbcnt_hi,
                             {B.getInt32(-1), Lane});

  llvm::Value *BxBy = B.CreateMul(Bx, By);
  llvm::Value *Flat = B.CreateAdd(B.CreateMul(Z0, BxBy), B.CreateMul(Y0, Bx));
  Flat = B.CreateAdd(B.CreateAdd(Flat, X0), Lane);

  switch (Dim) {
  case 0:
    return B.CreateURem(Flat, Bx);
  case 1:
    return B.CreateURem(B.CreateUDiv(Flat, Bx), By);
  default:
    return B.CreateUDiv(Flat, BxBy);
  }
}

} // namespace

llvm::PreservedAnalyses
SubstituteAMDGCNIntrinsicsPass::run(llvm::Module &M,
                                    llvm::ModuleAnalysisManager &) {
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::PointerType *Int32PtrTy =
      llvm::PointerType::get(Ctx, llvm::AMDGPUAS::CONSTANT_ADDRESS);

  using Mapping = std::tuple<const char *, const char *, llvm::Type *>;
  for (llvm::Function &F : llvm::make_early_inc_range(M.functions())) {
    for (const auto &[LLVMName, LuthierName, ReturnType] :
         std::initializer_list<Mapping>{
             {"llvm.amdgcn.workgroup.id.x", "luthier::workgroupIdX", Int32Ty},
             {"llvm.amdgcn.workgroup.id.y", "luthier::workgroupIdY", Int32Ty},
             {"llvm.amdgcn.workgroup.id.z", "luthier::workgroupIdZ", Int32Ty},
             {"llvm.amdgcn.implicitarg.ptr", "luthier::implicitArgPtr",
              Int32PtrTy}}) {
      if (!F.getName().starts_with(LLVMName))
        continue;
      for (llvm::User *U : llvm::make_early_inc_range(F.users())) {
        auto *CI = llvm::dyn_cast<llvm::CallInst>(U);
        if (CI == nullptr)
          continue;
        llvm::IRBuilder<> Builder(CI);
        llvm::CallInst *NewCall =
            insertCallToIntrinsic(M, Builder, LuthierName, *ReturnType);
        CI->replaceAllUsesWith(NewCall);
        CI->eraseFromParent();
      }
      F.dropAllReferences();
      F.eraseFromParent();
      break;
    }
  }

  // workitem.id.{x,y,z} (threadIdx) is not a 1:1 substitution: it has no
  // stable register at the injection point, so each use is replaced by a
  // recompute from the entry-captured wave base + lane index + blockDim.
  static const char *WorkitemNames[3] = {"llvm.amdgcn.workitem.id.x",
                                         "llvm.amdgcn.workitem.id.y",
                                         "llvm.amdgcn.workitem.id.z"};
  for (unsigned Dim = 0; Dim < 3; ++Dim) {
    llvm::Function *F = M.getFunction(WorkitemNames[Dim]);
    if (!F)
      continue;
    for (llvm::User *U : llvm::make_early_inc_range(F->users())) {
      auto *CI = llvm::dyn_cast<llvm::CallInst>(U);
      if (CI == nullptr)
        continue;
      const bool Wave64 = CI->getFunction()
                              ->getFnAttribute("target-features")
                              .getValueAsString()
                              .contains("+wavefrontsize64");
      llvm::IRBuilder<> Builder(CI);
      llvm::Value *V = emitWorkitemIdRecompute(M, Builder, Dim, Wave64);
      CI->replaceAllUsesWith(V);
      CI->eraseFromParent();
    }
    F->dropAllReferences();
    F->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
