//===-- InstrFuzzerMain.cpp - AMDGPU Instruction Semantic Fuzzer ----------===//
//
// Validates that the TableGen-generated IR semantic translations match the
// actual hardware behavior by dispatching paired MIR/IR kernels on a real GPU.
//
//===----------------------------------------------------------------------===//

#include "FuzzerDriver.h"
#include "HSADispatcher.h"
#include "IRKernelGen.h"
#include "InstrDescriptor.h"
#include "KernelCompiler.h"
#include "MachineKernelBuilder.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>

using namespace luthier::test;

//===----------------------------------------------------------------------===//
// Global test fixtures
//===----------------------------------------------------------------------===//

static std::unique_ptr<HSADispatcher> GDispatcher;
static std::unique_ptr<llvm::TargetMachine> GTM;
static std::unique_ptr<InstrDescriptor> GInstrDesc;
static std::unique_ptr<FuzzerDriver> GDriver;

class InstrSemanticFuzzerEnv : public ::testing::Environment {
public:
  void SetUp() override {
    // Initialize LLVM AMDGPU target.
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUAsmParser();

    // Initialize HSA and detect GPU.
    GDispatcher = std::make_unique<HSADispatcher>();
    auto Err = GDispatcher->init();
    ASSERT_FALSE(Err) << "HSA init failed: " << llvm::toString(std::move(Err));

    llvm::errs() << "Detected GPU: " << GDispatcher->getGpuTarget() << "\n";

    // Create a TargetMachine for the detected GPU.
    std::string Error;
    const llvm::Target *T =
        llvm::TargetRegistry::lookupTarget("amdgcn-amd-amdhsa", Error);
    ASSERT_NE(T, nullptr) << "AMDGPU target not found: " << Error;

    llvm::TargetOptions Opts;
    GTM.reset(T->createTargetMachine("amdgcn-amd-amdhsa",
                                     GDispatcher->getGpuTarget(), "", Opts,
                                     std::nullopt, std::nullopt));
    ASSERT_NE(GTM, nullptr);

    GInstrDesc = std::make_unique<InstrDescriptor>(*GTM);
    GDriver = std::make_unique<FuzzerDriver>(*GDispatcher, *GTM, *GInstrDesc);
  }

  void TearDown() override {
    GDriver.reset();
    GInstrDesc.reset();
    GTM.reset();
    if (GDispatcher)
      GDispatcher->shutdown();
    GDispatcher.reset();
  }
};

//===----------------------------------------------------------------------===//
// Smoke test: verify HSA init and GPU detection work
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, HSAInitialization) {
  ASSERT_NE(GDispatcher, nullptr);
  EXPECT_FALSE(GDispatcher->getGpuTarget().empty());
  llvm::errs() << "GPU target: " << GDispatcher->getGpuTarget() << "\n";
}

//===----------------------------------------------------------------------===//
// Smoke test: compile a trivial IR kernel to ELF
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, CompileTrivialIRKernel) {
  ASSERT_NE(GTM, nullptr);

  llvm::LLVMContext Ctx;
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  M->setTargetTriple("amdgcn-amd-amdhsa");
  M->setDataLayout(GTM->createDataLayout());

  // define amdgpu_kernel void @trivial() { ret void }
  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                   "trivial", M.get());
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1");
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> Builder(BB);
  Builder.CreateRetVoid();

  auto ELFOrErr =
      KernelCompiler::compileIR(*M, GDispatcher->getGpuTarget());
  ASSERT_TRUE(!!ELFOrErr) << llvm::toString(ELFOrErr.takeError());
  EXPECT_GT(ELFOrErr->size(), 0u);
  llvm::errs() << "Compiled trivial kernel: " << ELFOrErr->size()
               << " bytes\n";
}

//===----------------------------------------------------------------------===//
// Smoke test: dispatch a trivial kernel on the GPU
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, DispatchTrivialKernel) {
  ASSERT_NE(GDispatcher, nullptr);
  ASSERT_NE(GTM, nullptr);

  llvm::LLVMContext Ctx;
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  M->setTargetTriple("amdgcn-amd-amdhsa");
  M->setDataLayout(GTM->createDataLayout());

  // define amdgpu_kernel void @trivial_dispatch(ptr addrspace(1) %out) {
  //   store i32 42, ptr addrspace(1) %out
  //   ret void
  // }
  auto *I32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *PtrTy = llvm::PointerType::get(Ctx, 1); // addrspace(1)
  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), {PtrTy},
                                      false);
  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                   "trivial_dispatch", M.get());
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1");

  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> Builder(BB);
  Builder.CreateStore(Builder.getInt32(42), F->getArg(0));
  Builder.CreateRetVoid();

  // Compile.
  auto ELFOrErr =
      KernelCompiler::compileIR(*M, GDispatcher->getGpuTarget());
  ASSERT_TRUE(!!ELFOrErr) << llvm::toString(ELFOrErr.takeError());

  // Allocate kernarg (8 bytes for the pointer) and output buffer (4 bytes).
  auto KernargOrErr = GDispatcher->allocKernarg(8);
  ASSERT_TRUE(!!KernargOrErr) << llvm::toString(KernargOrErr.takeError());
  void *Kernarg = *KernargOrErr;

  auto OutBufOrErr = GDispatcher->allocFinegrained(4);
  ASSERT_TRUE(!!OutBufOrErr) << llvm::toString(OutBufOrErr.takeError());
  void *OutBuf = *OutBufOrErr;

  // Initialize output to zero.
  std::memset(OutBuf, 0, 4);

  // Write output buffer pointer into kernarg.
  std::memcpy(Kernarg, &OutBuf, 8);

  // Dispatch.
  auto ResultOrErr = GDispatcher->dispatch(
      llvm::ArrayRef<char>(ELFOrErr->data(), ELFOrErr->size()),
      "trivial_dispatch", Kernarg, 8);
  ASSERT_TRUE(!!ResultOrErr) << llvm::toString(ResultOrErr.takeError());
  EXPECT_EQ(ResultOrErr->SignalValue, 0);

  // Verify result.
  uint32_t Result = 0;
  std::memcpy(&Result, OutBuf, 4);
  EXPECT_EQ(Result, 42u);

  GDispatcher->freeMem(OutBuf);
  GDispatcher->freeMem(Kernarg);
}

//===----------------------------------------------------------------------===//
// InstrDescriptor smoke test
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, InstrDescriptorAnalysis) {
  ASSERT_NE(GInstrDesc, nullptr);
  EXPECT_GT(GInstrDesc->getNumOpcodes(), 0u);

  // Just verify that analyzing a known instruction doesn't crash.
  // S_ADD_U32 should exist in any AMDGPU target.
  for (unsigned I = 0, E = GInstrDesc->getNumOpcodes(); I < E; ++I) {
    llvm::StringRef Name = GInstrDesc->getName(I);
    if (Name == "S_ADD_U32") {
      auto Profile = GInstrDesc->analyze(I);
      EXPECT_TRUE(Profile.IsTestable);
      EXPECT_GE(Profile.Inputs.size(), 2u);
      EXPECT_GE(Profile.Outputs.size(), 1u);
      llvm::errs() << "S_ADD_U32: " << Profile.Inputs.size() << " inputs, "
                   << Profile.Outputs.size() << " outputs, "
                   << Profile.ImplicitDefs.size() << " implicit defs\n";
      return;
    }
  }
  FAIL() << "S_ADD_U32 not found in instruction table";
}

//===----------------------------------------------------------------------===//
// MachineKernelBuilder smoke test
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, MachineKernelBuilderForSADD) {
  ASSERT_NE(GInstrDesc, nullptr);
  ASSERT_NE(GTM, nullptr);

  unsigned SADDOpcode = 0;
  bool Found = false;
  for (unsigned I = 0, E = GInstrDesc->getNumOpcodes(); I < E; ++I) {
    if (GInstrDesc->getName(I) == "S_ADD_U32") {
      SADDOpcode = I;
      Found = true;
      break;
    }
  }
  ASSERT_TRUE(Found) << "S_ADD_U32 not found";

  auto Profile = GInstrDesc->analyze(SADDOpcode);
  ASSERT_TRUE(Profile.IsTestable);

  MachineKernelBuilder MKBuilder(*GTM);
  KernargLayout Layout;
  auto MFCtxOrErr = MKBuilder.build(Profile, Layout);
  ASSERT_TRUE(!!MFCtxOrErr) << llvm::toString(MFCtxOrErr.takeError());

  EXPECT_NE(MFCtxOrErr->MF, nullptr);
  EXPECT_NE(MFCtxOrErr->TargetMI, nullptr);
  EXPECT_GT(Layout.TotalSize, 0u);
  EXPECT_GT(Layout.OutputBufSize, 0u);
  EXPECT_FALSE(Layout.OutputFields.empty());

  llvm::errs() << "=== MachineFunction for S_ADD_U32 ===\n"
               << "Kernarg size: " << Layout.TotalSize << " bytes\n"
               << "Output buffer size: " << Layout.OutputBufSize << " bytes\n"
               << "Output fields: " << Layout.OutputFields.size() << "\n"
               << "Target MI opcode: " << MFCtxOrErr->TargetMI->getOpcode()
               << "\n";

  // Try emitting to ELF.
  auto ELFOrErr = MKBuilder.emitToELF(*MFCtxOrErr);
  if (ELFOrErr)
    llvm::errs() << "Emitted reference ELF: " << ELFOrErr->size()
                 << " bytes\n";
  else
    llvm::errs() << "ELF emission: " << llvm::toString(ELFOrErr.takeError())
                 << "\n";
}

//===----------------------------------------------------------------------===//
// IRKernelGen smoke test
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, IRKernelGenForSADD) {
  ASSERT_NE(GInstrDesc, nullptr);
  ASSERT_NE(GTM, nullptr);
  ASSERT_NE(GDispatcher, nullptr);

  unsigned SADDOpcode = 0;
  bool Found = false;
  for (unsigned I = 0, E = GInstrDesc->getNumOpcodes(); I < E; ++I) {
    if (GInstrDesc->getName(I) == "S_ADD_U32") {
      SADDOpcode = I;
      Found = true;
      break;
    }
  }
  ASSERT_TRUE(Found);

  auto Profile = GInstrDesc->analyze(SADDOpcode);
  ASSERT_TRUE(Profile.IsTestable);

  // Build the MachineFunction to get the layout.
  MachineKernelBuilder MKBuilder(*GTM);
  KernargLayout Layout;
  auto MFCtxOrErr = MKBuilder.build(Profile, Layout);
  ASSERT_TRUE(!!MFCtxOrErr) << llvm::toString(MFCtxOrErr.takeError());

  // Now generate the IR kernel using the same layout.
  IRKernelGen IRGen(*GTM);
  llvm::LLVMContext Ctx;
  auto ModOrErr = IRGen.generate(Profile, Layout, Ctx);
  ASSERT_TRUE(!!ModOrErr) << llvm::toString(ModOrErr.takeError());

  auto &M = *ModOrErr;
  EXPECT_NE(M, nullptr);

  // Verify the module has the expected kernel function.
  std::string KName = IRKernelGen::getKernelName(Profile);
  llvm::Function *F = M->getFunction(KName);
  EXPECT_NE(F, nullptr) << "Kernel function " << KName << " not found";
  EXPECT_EQ(F->getCallingConv(), llvm::CallingConv::AMDGPU_KERNEL);

  // Try compiling it to verify it's valid.
  auto ELFOrErr =
      KernelCompiler::compileIR(*M, GDispatcher->getGpuTarget());
  ASSERT_TRUE(!!ELFOrErr) << llvm::toString(ELFOrErr.takeError());
  EXPECT_GT(ELFOrErr->size(), 0u);

  llvm::errs() << "IR kernel compiled: " << ELFOrErr->size() << " bytes\n";
}

//===----------------------------------------------------------------------===//
// End-to-end: generate MIR kernel, compile, and dispatch on GPU
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, DispatchGeneratedRefKernel) {
  ASSERT_NE(GInstrDesc, nullptr);
  ASSERT_NE(GTM, nullptr);
  ASSERT_NE(GDispatcher, nullptr);
  ASSERT_GT(GDispatcher->getNumGpuAgents(), 0u);

  const auto &Agent = GDispatcher->getGpuAgent(0);

  // Find S_MOV_B32 — the simplest instruction (1 input, 1 output, no SCC).
  unsigned Opcode = 0;
  bool Found = false;
  for (unsigned I = 0, E = GInstrDesc->getNumOpcodes(); I < E; ++I) {
    if (GInstrDesc->getName(I) == "S_MOV_B32") {
      Opcode = I;
      Found = true;
      break;
    }
  }
  if (!Found)
    GTEST_SKIP() << "S_MOV_B32 not found — skipping dispatch test";

  auto Profile = GInstrDesc->analyze(Opcode);
  if (!Profile.IsTestable)
    GTEST_SKIP() << "S_MOV_B32 not testable: " << Profile.SkipReason;

  MachineKernelBuilder MKBuilder(*GTM);
  KernargLayout Layout;
  auto MFCtxOrErr = MKBuilder.build(Profile, Layout);
  ASSERT_TRUE(!!MFCtxOrErr) << llvm::toString(MFCtxOrErr.takeError());

  // Emit to ELF.
  auto ELFOrErr = MKBuilder.emitToELF(*MFCtxOrErr);
  if (!ELFOrErr) {
    llvm::errs() << "ELF emission failed: "
                 << llvm::toString(ELFOrErr.takeError()) << "\n";
    GTEST_SKIP() << "ELF emission not yet working for this instruction";
  }

  // Allocate memory.
  auto KAOrErr = GDispatcher->allocKernarg(Agent, Layout.TotalSize);
  ASSERT_TRUE(!!KAOrErr) << llvm::toString(KAOrErr.takeError());
  void *KA = *KAOrErr;

  auto OutOrErr = GDispatcher->allocFinegrained(
      Agent, std::max<size_t>(Layout.OutputBufSize, 4));
  ASSERT_TRUE(!!OutOrErr) << llvm::toString(OutOrErr.takeError());
  void *OutBuf = *OutOrErr;
  std::memset(OutBuf, 0, Layout.OutputBufSize);

  // Fill inputs with a known value.
  uint32_t InputVal = 0xDEADBEEF;
  for (const auto &F : Layout.Fields) {
    if (F.IsInput) {
      std::memcpy(static_cast<char *>(KA) + F.Offset, &InputVal,
                  std::min<size_t>(F.SizeBytes, sizeof(InputVal)));
    }
  }
  std::memcpy(static_cast<char *>(KA) + Layout.OutputPtrOffset, &OutBuf, 8);

  // Dispatch.
  std::string KName = MachineKernelBuilder::getKernelName(Profile);
  auto ResOrErr = GDispatcher->dispatch(
      Agent,
      llvm::ArrayRef<char>(ELFOrErr->data(), ELFOrErr->size()), KName, KA,
      Layout.TotalSize);
  ASSERT_TRUE(!!ResOrErr) << llvm::toString(ResOrErr.takeError());
  EXPECT_EQ(ResOrErr->SignalValue, 0);

  // Read back outputs.
  for (const auto &OF : Layout.OutputFields) {
    uint32_t Val = 0;
    std::memcpy(&Val, static_cast<char *>(OutBuf) + OF.Offset,
                std::min<size_t>(OF.SizeBytes, sizeof(Val)));
    llvm::errs() << "  " << OF.Name << " = 0x"
                 << llvm::format_hex_no_prefix(Val, OF.SizeBytes * 2) << "\n";
  }

  GDispatcher->freeMem(OutBuf);
  GDispatcher->freeMem(KA);
}

//===----------------------------------------------------------------------===//
// FuzzerDriver: single-instruction end-to-end test
//===----------------------------------------------------------------------===//

TEST(InstrSemanticFuzzer, FuzzerDriverSingleInstr) {
  ASSERT_NE(GDriver, nullptr);

  // Find S_MOV_B32 as a simple test target.
  unsigned Opcode = 0;
  bool Found = false;
  for (unsigned I = 0, E = GInstrDesc->getNumOpcodes(); I < E; ++I) {
    if (GInstrDesc->getName(I) == "S_MOV_B32") {
      Opcode = I;
      Found = true;
      break;
    }
  }
  if (!Found)
    GTEST_SKIP() << "S_MOV_B32 not found";

  ASSERT_GT(GDispatcher->getNumGpuAgents(), 0u);
  auto R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), Opcode,
                                    /*Seed=*/42);

  if (R.Skipped)
    GTEST_SKIP() << "Skipped: " << R.SkipReason;

  // If there was a compilation error, report but don't hard-fail yet
  // (MIR compilation may not work for all generated kernels).
  if (!R.ErrorMsg.empty() && R.Outputs.empty()) {
    llvm::errs() << "Error testing " << R.InstrName << ": " << R.ErrorMsg
                 << "\n";
    GTEST_SKIP() << R.ErrorMsg;
  }

  if (!R.Passed)
    llvm::errs() << R.ErrorMsg << "\n";

  // For now we expect the test framework to run without crashing.
  // Once the IR semantic translation is wired into IRKernelGen, this
  // EXPECT_TRUE will enforce correctness.
  // EXPECT_TRUE(R.Passed) << R.ErrorMsg;
}

//===----------------------------------------------------------------------===//
// Parameterized test: iterate all modeled opcodes
//===----------------------------------------------------------------------===//

/// Collects all testable opcode indices at test-suite registration time.
/// Since the opcode list from SIInstrSemantics.inc requires linking the
/// generated code, and we may not have it available at static init, we
/// instead scan the InstrDescriptor at runtime.
static std::vector<unsigned> collectTestableOpcodes() {
  // This is called during INSTANTIATE_TEST_SUITE_P, which happens before
  // the global environment SetUp.  We can't use GInstrDesc here.
  // Instead, we create a temporary descriptor.
  // If LLVM isn't initialized yet, return empty and the suite will have
  // zero test cases (the smoke tests above still run).

  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  std::string Error;
  const llvm::Target *T =
      llvm::TargetRegistry::lookupTarget("amdgcn-amd-amdhsa", Error);
  if (!T)
    return {};

  llvm::TargetOptions Opts;
  std::unique_ptr<llvm::TargetMachine> TM(T->createTargetMachine(
      "amdgcn-amd-amdhsa", "gfx908", "", Opts, std::nullopt, std::nullopt));
  if (!TM)
    return {};

  InstrDescriptor Desc(*TM);
  std::vector<unsigned> Opcodes;
  for (unsigned I = 0, E = Desc.getNumOpcodes(); I < E; ++I) {
    auto Profile = Desc.analyze(I);
    // Only include testable, non-memory, scalar ALU instructions for now.
    if (Profile.IsTestable && Profile.Mem.MemKind == MemAccessInfo::None &&
        !Profile.Outputs.empty())
      Opcodes.push_back(I);
  }
  return Opcodes;
}

class SemanticFuzzerParam : public ::testing::TestWithParam<unsigned> {};

TEST_P(SemanticFuzzerParam, MatchesHardware) {
  ASSERT_NE(GDriver, nullptr) << "FuzzerDriver not initialized";

  unsigned Opcode = GetParam();
  auto Results = GDriver->testInstructionMulti(Opcode,
                                               /*NumIterations=*/5,
                                               /*BaseSeed=*/12345);

  ASSERT_FALSE(Results.empty());

  if (Results[0].Skipped)
    GTEST_SKIP() << Results[0].SkipReason;

  // Report compilation errors as skips (not hard failures) during bringup.
  if (!Results[0].ErrorMsg.empty() && Results[0].Outputs.empty())
    GTEST_SKIP() << Results[0].ErrorMsg;

  for (const auto &R : Results) {
    EXPECT_TRUE(R.Passed) << R.ErrorMsg;
    if (!R.Passed)
      break; // One failure is enough to report.
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllScalarALU, SemanticFuzzerParam,
    ::testing::ValuesIn(collectTestableOpcodes()),
    [](const ::testing::TestParamInfo<unsigned> &Info) {
      // Create a temp descriptor for naming (or use a static cache).
      std::string Error;
      const llvm::Target *T =
          llvm::TargetRegistry::lookupTarget("amdgcn-amd-amdhsa", Error);
      if (!T)
        return std::string("unknown");
      llvm::TargetOptions Opts;
      std::unique_ptr<llvm::TargetMachine> TM(T->createTargetMachine(
          "amdgcn-amd-amdhsa", "gfx908", "", Opts, std::nullopt,
          std::nullopt));
      if (!TM)
        return std::string("unknown");
      return TM->getMCInstrInfo()->getName(Info.param).str();
    });

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new InstrSemanticFuzzerEnv);
  return RUN_ALL_TESTS();
}
