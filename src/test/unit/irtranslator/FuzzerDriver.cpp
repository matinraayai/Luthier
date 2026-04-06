#include "FuzzerDriver.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include <cstring>
#include <random>

namespace luthier::test {

// ============================================================================
// Input randomization
// ============================================================================

void FuzzerDriver::fillRandomInputs(void *KernargPtr,
                                     const KernargLayout &Layout,
                                     uint64_t Seed) {
  std::mt19937_64 RNG(Seed);
  auto *Base = static_cast<uint8_t *>(KernargPtr);

  for (const auto &F : Layout.Fields) {
    if (!F.IsInput)
      continue;
    for (unsigned B = 0; B < F.SizeBytes; ++B)
      Base[F.Offset + B] = static_cast<uint8_t>(RNG());
  }
}

// ============================================================================
// Output comparison
// ============================================================================

bool FuzzerDriver::compareOutputs(
    const void *RefBuf, const void *TransBuf,
    const void *RefDataBuf, const void *TransDataBuf,
    const KernargLayout &Layout,
    std::vector<TestResult::OutputComparison> &Out) {
  bool AllMatch = true;
  const auto *Ref = static_cast<const uint8_t *>(RefBuf);
  const auto *Trans = static_cast<const uint8_t *>(TransBuf);

  // Compare register output fields.
  for (const auto &OF : Layout.OutputFields) {
    TestResult::OutputComparison C;
    C.Name = OF.Name.str();
    C.SizeBytes = OF.SizeBytes;
    C.Reference = 0;
    C.Translated = 0;

    unsigned CopySize = std::min<unsigned>(OF.SizeBytes, 8);
    std::memcpy(&C.Reference, Ref + OF.Offset, CopySize);
    std::memcpy(&C.Translated, Trans + OF.Offset, CopySize);

    C.Matches = std::memcmp(Ref + OF.Offset, Trans + OF.Offset,
                            OF.SizeBytes) == 0;
    if (!C.Matches)
      AllMatch = false;

    Out.push_back(C);
  }

  // Compare data buffer contents (for store/atomic instructions).
  if (Layout.DataBufIsOutput && RefDataBuf && TransDataBuf &&
      Layout.DataBufSizeBytes > 0) {
    const auto *RD = static_cast<const uint8_t *>(RefDataBuf);
    const auto *TD = static_cast<const uint8_t *>(TransDataBuf);

    // Compare in dword granularity for readable diagnostics.
    for (uint32_t Off = 0; Off < Layout.DataBufSizeBytes; Off += 4) {
      uint32_t RefVal = 0, TransVal = 0;
      unsigned Chunk = std::min<uint32_t>(4, Layout.DataBufSizeBytes - Off);
      std::memcpy(&RefVal, RD + Off, Chunk);
      std::memcpy(&TransVal, TD + Off, Chunk);

      if (RefVal != TransVal) {
        TestResult::OutputComparison C;
        C.Name = llvm::formatv("mem[{0}]", Off);
        C.SizeBytes = Chunk;
        C.Reference = RefVal;
        C.Translated = TransVal;
        C.Matches = false;
        Out.push_back(C);
        AllMatch = false;
      }
    }
  }

  return AllMatch;
}

// ============================================================================
// Mismatch formatting
// ============================================================================

std::string FuzzerDriver::formatMismatch(const TestResult &R) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  OS << "MISMATCH for " << R.InstrName << " (seed=" << R.Seed << "):\n";
  for (const auto &C : R.Outputs) {
    if (!C.Matches) {
      OS << "  " << C.Name << ": ref=0x"
         << llvm::format_hex_no_prefix(C.Reference, C.SizeBytes * 2)
         << "  trans=0x"
         << llvm::format_hex_no_prefix(C.Translated, C.SizeBytes * 2) << "\n";
    }
  }
  return S;
}

// ============================================================================
// Core test logic
// ============================================================================

TestResult FuzzerDriver::testInstruction(const GpuAgentInfo &Agent,
                                         unsigned Opcode, uint64_t Seed) {
  TestResult R;
  R.Opcode = Opcode;
  R.InstrName = Desc.getName(Opcode).str();
  R.Seed = Seed;

  // --- Analyze ---
  InstrProfile Profile = Desc.analyze(Opcode);
  if (!Profile.IsTestable) {
    R.Skipped = true;
    R.SkipReason = Profile.SkipReason.str();
    return R;
  }

  // --- Classify memory requirements ---
  MemoryTestSetup MemSetup = classifyMemoryTest(Profile);
  AddressConfig AddrCfg = randomizeAddress(MemSetup, Profile, Seed);

  // --- Step 1: Build standalone MachineInstr ---
  auto SAMIOrErr = SAMIBuilder.build(Profile);
  if (!SAMIOrErr) {
    R.ErrorMsg = "StandaloneMIBuilder failed: " +
                 llvm::toString(SAMIOrErr.takeError());
    return R;
  }

  // --- Step 2: Build reference kernel from the standalone MI ---
  KernargLayout Layout;
  auto KCtxOrErr = MKBuilder.build(*SAMIOrErr, Profile, Layout);
  if (!KCtxOrErr) {
    R.ErrorMsg = "MachineKernelBuilder failed: " +
                 llvm::toString(KCtxOrErr.takeError());
    return R;
  }

  // Apply memory setup to layout.
  Layout.DataBufSizeBytes = MemSetup.DataBufSizeBytes;
  Layout.DataBufNeedsPreFill = MemSetup.NeedsPreFill;
  Layout.DataBufIsOutput = MemSetup.DataBufIsOutput;
  Layout.GroupSegmentSize = MemSetup.GroupSegmentSize;
  Layout.PrivateSegmentSize = MemSetup.PrivateSegmentSize;

  // Emit reference kernel → ELF.
  auto RefELFOrErr = MKBuilder.emitToELF(*KCtxOrErr);
  if (!RefELFOrErr) {
    R.ErrorMsg = "Reference ELF emission failed: " +
                 llvm::toString(RefELFOrErr.takeError());
    return R;
  }

  // --- Step 3: Build translated IR kernel from the same standalone MI ---
  llvm::LLVMContext Ctx;
  auto TransModOrErr =
      IRGen.generate(*SAMIOrErr, Profile, Layout, Ctx);
  if (!TransModOrErr) {
    R.ErrorMsg = "IR generation failed: " +
                 llvm::toString(TransModOrErr.takeError());
    return R;
  }

  auto TransELFOrErr = KernelCompiler::compileIR(**TransModOrErr, Agent.Name);
  if (!TransELFOrErr) {
    R.ErrorMsg = "IR compilation failed: " +
                 llvm::toString(TransELFOrErr.takeError());
    return R;
  }

  // --- Allocate memory ---
  auto KAOrErr = Dispatcher.allocKernarg(Agent, Layout.TotalSize);
  if (!KAOrErr) {
    R.ErrorMsg = "Kernarg alloc: " + llvm::toString(KAOrErr.takeError());
    return R;
  }
  void *KA = *KAOrErr;

  size_t OutBufSize = std::max<size_t>(Layout.OutputBufSize, 4);
  auto RefOutOrErr = Dispatcher.allocFinegrained(Agent, OutBufSize);
  if (!RefOutOrErr) {
    R.ErrorMsg = "Ref out alloc: " + llvm::toString(RefOutOrErr.takeError());
    Dispatcher.freeMem(KA);
    return R;
  }
  void *RefOut = *RefOutOrErr;

  auto TransOutOrErr = Dispatcher.allocFinegrained(Agent, OutBufSize);
  if (!TransOutOrErr) {
    R.ErrorMsg = "Trans out alloc: " + llvm::toString(TransOutOrErr.takeError());
    Dispatcher.freeMem(KA);
    Dispatcher.freeMem(RefOut);
    return R;
  }
  void *TransOut = *TransOutOrErr;

  // Data buffers for memory instructions (two copies for ref/trans).
  void *RefDataBuf = nullptr;
  void *TransDataBuf = nullptr;
  if (MemSetup.DataBufSizeBytes > 0) {
    auto B1 = Dispatcher.allocFinegrained(Agent, MemSetup.DataBufSizeBytes);
    auto B2 = Dispatcher.allocFinegrained(Agent, MemSetup.DataBufSizeBytes);
    if (!B1 || !B2) {
      if (B1)
        Dispatcher.freeMem(*B1);
      if (B2)
        Dispatcher.freeMem(*B2);
      Dispatcher.freeMem(TransOut);
      Dispatcher.freeMem(RefOut);
      Dispatcher.freeMem(KA);
      R.ErrorMsg = "Data buf alloc failed";
      return R;
    }
    RefDataBuf = *B1;
    TransDataBuf = *B2;
  }

  // --- Fill inputs ---
  std::memset(KA, 0, Layout.TotalSize);
  fillRandomInputs(KA, Layout, Seed);

  // Fill data buffers with deterministic pattern.
  if (MemSetup.NeedsPreFill && RefDataBuf && TransDataBuf) {
    fillDeterministicPattern(RefDataBuf, MemSetup.DataBufSizeBytes, Seed);
    fillDeterministicPattern(TransDataBuf, MemSetup.DataBufSizeBytes, Seed);
  } else if (RefDataBuf && TransDataBuf) {
    std::memset(RefDataBuf, 0, MemSetup.DataBufSizeBytes);
    std::memset(TransDataBuf, 0, MemSetup.DataBufSizeBytes);
  }

  // Write data buffer pointer into kernarg (if applicable).
  if (Layout.DataBufPtrOffset != UINT32_MAX && RefDataBuf) {
    std::memcpy(static_cast<char *>(KA) + Layout.DataBufPtrOffset,
                &RefDataBuf, 8);
  }

  // Write V# descriptor into kernarg (if applicable).
  if (Layout.VSharpOffset != UINT32_MAX && RefDataBuf) {
    VSharpDescriptor V = VSharpDescriptor::createRaw(
        reinterpret_cast<uint64_t>(RefDataBuf), MemSetup.DataBufSizeBytes);
    std::memcpy(static_cast<char *>(KA) + Layout.VSharpOffset, V.Words, 16);
  }

  // --- Dispatch reference kernel ---
  std::memset(RefOut, 0, OutBufSize);
  std::memcpy(static_cast<char *>(KA) + Layout.OutputPtrOffset, &RefOut, 8);

  std::string RefKName = MachineKernelBuilder::getKernelName(Profile);
  auto RefResOrErr = Dispatcher.dispatch(
      Agent,
      llvm::ArrayRef<char>(RefELFOrErr->data(), RefELFOrErr->size()),
      RefKName, KA, Layout.TotalSize, /*Grid=*/1, /*WG=*/1,
      Layout.GroupSegmentSize, Layout.PrivateSegmentSize);
  if (!RefResOrErr) {
    R.ErrorMsg = "Ref dispatch: " + llvm::toString(RefResOrErr.takeError());
    goto cleanup;
  }
  if (RefResOrErr->SignalValue != 0) {
    R.ErrorMsg = llvm::formatv("Ref signal={0}", RefResOrErr->SignalValue);
    goto cleanup;
  }

  // --- Dispatch translated kernel ---
  // Swap data buffer pointer to the trans copy.
  if (Layout.DataBufPtrOffset != UINT32_MAX && TransDataBuf) {
    std::memcpy(static_cast<char *>(KA) + Layout.DataBufPtrOffset,
                &TransDataBuf, 8);
  }
  if (Layout.VSharpOffset != UINT32_MAX && TransDataBuf) {
    VSharpDescriptor V = VSharpDescriptor::createRaw(
        reinterpret_cast<uint64_t>(TransDataBuf), MemSetup.DataBufSizeBytes);
    std::memcpy(static_cast<char *>(KA) + Layout.VSharpOffset, V.Words, 16);
  }

  std::memset(TransOut, 0, OutBufSize);
  std::memcpy(static_cast<char *>(KA) + Layout.OutputPtrOffset, &TransOut, 8);

  {
    std::string TransKName = IRKernelGen::getKernelName(Profile);
    auto TransResOrErr = Dispatcher.dispatch(
        Agent,
        llvm::ArrayRef<char>(TransELFOrErr->data(), TransELFOrErr->size()),
        TransKName, KA, Layout.TotalSize, /*Grid=*/1, /*WG=*/1,
        Layout.GroupSegmentSize, Layout.PrivateSegmentSize);
    if (!TransResOrErr) {
      R.ErrorMsg = "Trans dispatch: " + llvm::toString(TransResOrErr.takeError());
      goto cleanup;
    }
    if (TransResOrErr->SignalValue != 0) {
      R.ErrorMsg = llvm::formatv("Trans signal={0}", TransResOrErr->SignalValue);
      goto cleanup;
    }
  }

  // --- Compare outputs ---
  R.Passed = compareOutputs(RefOut, TransOut, RefDataBuf, TransDataBuf,
                            Layout, R.Outputs);
  if (!R.Passed)
    R.ErrorMsg = formatMismatch(R);

cleanup:
  Dispatcher.freeMem(TransDataBuf);
  Dispatcher.freeMem(RefDataBuf);
  Dispatcher.freeMem(TransOut);
  Dispatcher.freeMem(RefOut);
  Dispatcher.freeMem(KA);
  return R;
}

std::vector<TestResult>
FuzzerDriver::testInstructionMulti(const GpuAgentInfo &Agent, unsigned Opcode,
                                   unsigned NumIterations, uint64_t BaseSeed) {
  std::vector<TestResult> Results;
  Results.reserve(NumIterations);

  for (unsigned I = 0; I < NumIterations; ++I) {
    uint64_t IterSeed = BaseSeed ^ (static_cast<uint64_t>(Opcode) << 32) ^
                        (static_cast<uint64_t>(I) << 48);
    TestResult R = testInstruction(Agent, Opcode, IterSeed);

    if (R.Skipped || (!R.Passed && R.Outputs.empty()))
      return {R};

    Results.push_back(std::move(R));

    if (!Results.back().Passed)
      break;
  }
  return Results;
}

std::vector<TestResult>
FuzzerDriver::testInstructionAllAgents(unsigned Opcode,
                                       unsigned NumIterations,
                                       uint64_t BaseSeed) {
  std::vector<TestResult> AllResults;

  for (size_t A = 0; A < Dispatcher.getNumGpuAgents(); ++A) {
    const auto &Agent = Dispatcher.getGpuAgent(A);
    auto Results = testInstructionMulti(Agent, Opcode, NumIterations, BaseSeed);
    for (auto &R : Results) {
      // Tag the result with the agent name for multi-GPU reporting.
      R.InstrName = llvm::formatv("[{0}] {1}", Agent.Name, R.InstrName);
      AllResults.push_back(std::move(R));
    }
  }

  return AllResults;
}

} // namespace luthier::test
