#include "IRKernelGen.h"

#include <luthier/IRTranslator/InstructionModel.h>

#include <SIRegisterInfo.h>

#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/Type.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier::test {

llvm::Expected<std::unique_ptr<llvm::Module>>
IRKernelGen::generate(const StandaloneMIContext &MICtx,
                      const InstrProfile &Profile, const KernargLayout &Layout,
                      llvm::LLVMContext &Ctx) const {
  auto M =
      std::make_unique<llvm::Module>(("trans_" + Profile.Name).str(), Ctx);
  M->setTargetTriple("amdgcn-amd-amdhsa");
  M->setDataLayout(TM.createDataLayout());

  std::string KName = getKernelName(Profile);

  llvm::Type *VoidTy = llvm::Type::getVoidTy(Ctx);
  llvm::Type *I8Ty = llvm::Type::getInt8Ty(Ctx);
  llvm::Type *I32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *I64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::PointerType *Ptr4 = llvm::PointerType::get(Ctx, 4);
  llvm::PointerType *Ptr1 = llvm::PointerType::get(Ctx, 1);

  auto *FTy = llvm::FunctionType::get(VoidTy, {Ptr4}, false);
  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                   KName, M.get());
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1");

  llvm::Argument *KernargBase = F->getArg(0);
  KernargBase->setName("kernarg");

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);

  const auto &TRI =
      *static_cast<const llvm::SIRegisterInfo *>(TM.getMCRegisterInfo());

  // -----------------------------------------------------------------------
  // Load inputs from kernarg and build the PhysRegValueMap.
  // -----------------------------------------------------------------------
  luthier::PhysRegValueMap InputRegs;

  unsigned FieldIdx = 0;
  for (unsigned I = 0; I < MICtx.InputRegs.size(); ++I) {
    llvm::MCRegister Reg = MICtx.InputRegs[I];
    if (!Reg) {
      ++FieldIdx; // Immediate — skip.
      continue;
    }

    const auto &Fld = Layout.Fields[FieldIdx++];
    const auto *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned Bits = TRI.getRegSizeInBits(*RC);

    llvm::Type *LoadTy = llvm::IntegerType::get(Ctx, Bits);
    llvm::Value *Ptr = B.CreateInBoundsGEP(
        I8Ty, KernargBase, B.getInt32(Fld.Offset),
        llvm::formatv("ka.in{0}.ptr", I));
    llvm::Value *Val =
        B.CreateLoad(LoadTy, Ptr, llvm::formatv("ka.in{0}", I));

    // For wide registers (64-bit+), seed sub-registers individually.
    if (Bits > 32) {
      unsigned NumSubs = Bits / 32;
      for (unsigned D = 0; D < NumSubs; ++D) {
        llvm::MCRegister Sub = TRI.getSubReg(Reg, AMDGPU::sub0 + D);
        llvm::Value *SubVal = B.CreateTrunc(
            B.CreateLShr(Val, D * 32), I32Ty,
            llvm::formatv("in{0}.sub{1}", I, D));
        InputRegs[Sub] = SubVal;
      }
    } else {
      InputRegs[Reg] = Val;
    }
  }

  // -----------------------------------------------------------------------
  // Call the semantic translation on the standalone MI.
  // -----------------------------------------------------------------------
  luthier::PhysRegValueMap OutputRegs;

  if (MICtx.MI) {
    if (auto Err = luthier::translateSingleInstr(*MICtx.MI, B, InputRegs,
                                                  OutputRegs))
      return std::move(Err);
  }

  // -----------------------------------------------------------------------
  // Load output pointer and store results.
  // -----------------------------------------------------------------------
  llvm::Value *OutPtrRaw = B.CreateLoad(
      I64Ty,
      B.CreateInBoundsGEP(I8Ty, KernargBase,
                          B.getInt32(Layout.OutputPtrOffset), "ka.outptr.raw"),
      "outptr.int");
  llvm::Value *OutPtr = B.CreateIntToPtr(OutPtrRaw, Ptr1, "outptr");

  // Store explicit output registers.
  for (unsigned I = 0; I < MICtx.OutputRegs.size() &&
                        I < Layout.OutputFields.size();
       ++I) {
    llvm::MCRegister Reg = MICtx.OutputRegs[I];
    const auto &OF = Layout.OutputFields[I];
    llvm::Type *OutTy = llvm::IntegerType::get(Ctx, OF.SizeBytes * 8);

    llvm::Value *Val = nullptr;
    auto It = OutputRegs.find(Reg);
    if (It != OutputRegs.end())
      Val = It->second;

    if (!Val)
      Val = llvm::Constant::getNullValue(OutTy);

    // Type adjustment.
    if (Val->getType() != OutTy) {
      unsigned VBits = Val->getType()->getIntegerBitWidth();
      unsigned OBits = OutTy->getIntegerBitWidth();
      if (VBits > OBits)
        Val = B.CreateTrunc(Val, OutTy);
      else if (VBits < OBits)
        Val = B.CreateZExt(Val, OutTy);
      else
        Val = B.CreateBitCast(Val, OutTy);
    }

    llvm::Value *StorePtr = B.CreateInBoundsGEP(
        I8Ty, OutPtr, B.getInt32(OF.Offset),
        llvm::formatv("out.{0}.ptr", OF.Name));
    B.CreateStore(Val, StorePtr);
  }

  // Store implicit defs (SCC, VCC, etc.).
  for (unsigned I = MICtx.OutputRegs.size(); I < Layout.OutputFields.size();
       ++I) {
    const auto &OF = Layout.OutputFields[I];
    llvm::Type *OutTy = llvm::IntegerType::get(Ctx, OF.SizeBytes * 8);
    llvm::Value *Val = llvm::Constant::getNullValue(OutTy);

    for (const auto &[Reg, RegVal] : OutputRegs) {
      if (TRI.getName(Reg) == OF.Name) {
        Val = RegVal;
        if (Val->getType() != OutTy) {
          unsigned VBits = Val->getType()->getIntegerBitWidth();
          unsigned OBits = OutTy->getIntegerBitWidth();
          if (VBits > OBits)
            Val = B.CreateTrunc(Val, OutTy);
          else
            Val = B.CreateZExt(Val, OutTy);
        }
        break;
      }
    }

    llvm::Value *StorePtr = B.CreateInBoundsGEP(
        I8Ty, OutPtr, B.getInt32(OF.Offset),
        llvm::formatv("out.{0}.ptr", OF.Name));
    B.CreateStore(Val, StorePtr);
  }

  B.CreateRetVoid();
  return std::move(M);
}

} // namespace luthier::test
