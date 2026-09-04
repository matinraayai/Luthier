//===-- FuzzerDriver.cpp ----------------------------------------*- C++ -*-===//
#include "FuzzerDriver.h"

#include "DiagnosticDump.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include <cstring>
#include <memory>
#include <random>

namespace luthier::test {

namespace {

/// RAII holder for a device allocation.
class Buffer {
public:
  Buffer(HSADispatcher &D, void *P) : D(D), P(P) {}
  ~Buffer() {
    if (P)
      D.freeMem(P);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void *get() const { return P; }

private:
  HSADispatcher &D;
  void *P;
};

} // namespace

std::string formatResult(const TestResult &R) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  OS << R.InstrName << " (seed=" << R.Seed << ")";
  if (R.Skipped) {
    OS << " SKIPPED: " << R.SkipReason;
    return S;
  }
  if (!R.ErrorMsg.empty()) {
    OS << " ERROR: " << R.ErrorMsg;
    // An output divergence reports itself through ErrorMsg but still carries
    // the per-output values -- which are the whole point of the report, so
    // fall through and print them. Only a failure that never got as far as
    // running (no outputs captured) has nothing more to say.
    if (R.Outputs.empty())
      return S;
    OS << "\n";
  } else {
    OS << (R.Passed ? " PASSED" : " FAILED") << "\n";
  }
  for (unsigned I = 0; I < R.InputValues.size(); ++I)
    OS << llvm::formatv("    src{0} = 0x{1:x-8}\n", I, R.InputValues[I]);
  for (const auto &C : R.Outputs)
    OS << llvm::formatv("    {0}: run1=0x{1:x-8} run2=0x{2:x-8} {3}\n", C.Name,
                        C.Reference, C.Translated, C.Matches ? "==" : "!=");
  return S;
}

TestResult FuzzerDriver::testInstruction(const GpuAgentInfo &Agent,
                                         unsigned Opcode, uint64_t Seed,
                                         llvm::ArrayRef<uint32_t> FixedInputs,
                                         bool CompareTranslation) {
  TestResult R;
  R.Opcode = Opcode;
  R.InstrName = Desc.getName(Opcode).str();
  R.Seed = Seed;

  // Collects the reference MIR and the lifted IR so a failing opcode can be
  // inspected afterwards. The report is written before each dispatch and
  // removed once the comparison succeeds, because a faulting translated kernel
  // aborts the process from inside HSA and never returns here.
  DiagnosticDump Dump(R.InstrName, Seed);

  //===--------------------------------------------------------------------===//
  // Analyze.
  //===--------------------------------------------------------------------===//
  InstrProfile Profile = Desc.analyze(Opcode);
  if (!Profile.IsTestable) {
    R.Skipped = true;
    R.SkipReason = Profile.SkipReason.str();
    return R;
  }

  //===--------------------------------------------------------------------===//
  // Reference path: build the MI + kernel MachineFunction, then emit an ELF.
  //===--------------------------------------------------------------------===//
  KernargLayout Layout;
  auto KCtxOrErr = MKBuilder.build(Profile, Layout);
  if (!KCtxOrErr) {
    R.ErrorMsg = "MachineKernelBuilder: " +
                 llvm::toString(KCtxOrErr.takeError());
    return R;
  }

  auto RefELFOrErr = MKBuilder.emitToELF(*KCtxOrErr);
  if (!RefELFOrErr) {
    R.ErrorMsg = "reference ELF emission: " +
                 llvm::toString(RefELFOrErr.takeError());
    return R;
  }
  llvm::ArrayRef<char> RefELF(RefELFOrErr->data(), RefELFOrErr->size());

  //===--------------------------------------------------------------------===//
  // Translation path: raise the reference kernel back to LLVM IR via
  // TraceFunctionTranslator and recompile. A distinct MachineFunction is required
  // (emitToELF above consumed the reference context's pipeline, and the
  // translator rewrites the IR body in place), so build a second context. Both
  // builds are deterministic, so they share the same kernarg `Layout`.
  //
  // If the translation path is not requested, the second dispatch re-runs the
  // reference kernel (a determinism check).
  //===--------------------------------------------------------------------===//
  llvm::SmallVector<char, 0> TransELFStorage;
  llvm::ArrayRef<char> Run2ELF = RefELF;
  std::string Run2KernelName = KCtxOrErr->KernelName;
  // Without the translation path nothing else captures the MIR, so record it
  // here to keep a crash report meaningful.
  if (Dump.enabled() && !CompareTranslation) {
    llvm::raw_string_ostream MIROS(Dump.artifacts().ReferenceMIR);
    KCtxOrErr->MF->print(MIROS);
  }
  if (CompareTranslation) {
    KernargLayout TransLayout;
    auto TransKCtxOrErr = MKBuilder.build(Profile, TransLayout);
    if (!TransKCtxOrErr) {
      R.ErrorMsg = "translation build: " +
                   llvm::toString(TransKCtxOrErr.takeError());
      return R;
    }
    // Raising and recompiling runs entirely on the host, and a translator that
    // crashes there takes the process down just as surely as a faulting kernel
    // does. Record the MIR going in, so such a crash still names the
    // instruction that caused it.
    if (Dump.enabled()) {
      llvm::raw_string_ostream MIROS(Dump.artifacts().ReferenceMIR);
      TransKCtxOrErr->MF->print(MIROS);
      Dump.write("TRANSLATING (raising MIR to IR)",
                 "The process did not get past TraceFunctionTranslator. If this "
                 "report survived, raising or recompiling this instruction "
                 "crashed on the host — the lifted IR section is empty because "
                 "it never got produced.");
    }
    auto TransELFOrErr = MKBuilder.emitTranslatedToELF(
        *TransKCtxOrErr, Dump.enabled() ? &Dump.artifacts() : nullptr);
    if (!TransELFOrErr) {
      R.ErrorMsg = "translation path: " +
                   llvm::toString(TransELFOrErr.takeError());
      Dump.write("FAILED (translation emission)", R.ErrorMsg);
      return R;
    }
    TransELFStorage = std::move(*TransELFOrErr);
    Run2ELF = llvm::ArrayRef<char>(TransELFStorage.data(),
                                   TransELFStorage.size());
    Run2KernelName = TransKCtxOrErr->KernelName;
  }

  //===--------------------------------------------------------------------===//
  // Allocate. The kernarg buffer is padded because the kernel descriptor's
  // kernarg segment size may exceed the bytes the kernel actually reads.
  //===--------------------------------------------------------------------===//
  auto KAOrErr = Dispatcher.allocKernarg(Agent, std::max(Layout.TotalSize, 1024u));
  if (!KAOrErr) {
    R.ErrorMsg = "kernarg alloc: " + llvm::toString(KAOrErr.takeError());
    return R;
  }
  Buffer KA(Dispatcher, *KAOrErr);

  const size_t OutSize = std::max<size_t>(Layout.OutputBufSize, 4);
  auto Out1OrErr = Dispatcher.allocFinegrained(Agent, OutSize);
  if (!Out1OrErr) {
    R.ErrorMsg = "output alloc: " + llvm::toString(Out1OrErr.takeError());
    return R;
  }
  Buffer Out1(Dispatcher, *Out1OrErr);

  auto Out2OrErr = Dispatcher.allocFinegrained(Agent, OutSize);
  if (!Out2OrErr) {
    R.ErrorMsg = "output alloc: " + llvm::toString(Out2OrErr.takeError());
    return R;
  }
  Buffer Out2(Dispatcher, *Out2OrErr);

  // Optional host-visible data buffer for FLAT/GLOBAL ops. The kernel
  // initializes it and reads it back into the output buffer itself, so the
  // host only has to allocate it and hand its device pointer to the kernel.
  std::unique_ptr<Buffer> DataBuf;
  if (Layout.DataBufPtrOffset != UINT32_MAX) {
    auto DBOrErr = Dispatcher.allocFinegrained(
        Agent, std::max<size_t>(Layout.DataBufSizeBytes, 4));
    if (!DBOrErr) {
      R.ErrorMsg = "data buffer alloc: " + llvm::toString(DBOrErr.takeError());
      return R;
    }
    DataBuf = std::make_unique<Buffer>(Dispatcher, *DBOrErr);
    std::memset(DataBuf->get(), 0, std::max<size_t>(Layout.DataBufSizeBytes, 4));
  }

  //===--------------------------------------------------------------------===//
  // Fill inputs.
  //===--------------------------------------------------------------------===//
  if (!FixedInputs.empty() && FixedInputs.size() != Layout.Inputs.size()) {
    R.ErrorMsg = llvm::formatv("{0} fixed inputs supplied but the instruction "
                               "has {1} input operands",
                               FixedInputs.size(), Layout.Inputs.size());
    return R;
  }

  std::memset(KA.get(), 0, Layout.TotalSize);
  std::mt19937_64 RNG(Seed);
  for (unsigned I = 0; I < Layout.Inputs.size(); ++I) {
    uint32_t V = FixedInputs.empty() ? static_cast<uint32_t>(RNG())
                                     : FixedInputs[I];
    R.InputValues.push_back(V);
    std::memcpy(static_cast<char *>(KA.get()) + Layout.Inputs[I].Offset, &V,
                sizeof(V));
  }

  if (DataBuf) {
    void *P = DataBuf->get();
    std::memcpy(static_cast<char *>(KA.get()) + Layout.DataBufPtrOffset, &P,
                sizeof(P));
  }

  //===--------------------------------------------------------------------===//
  // Dispatch twice with identical inputs and compare.
  //===--------------------------------------------------------------------===//
  auto RunOnce = [&](llvm::ArrayRef<char> ELF, llvm::StringRef KernelName,
                     void *OutBuf) -> llvm::Error {
    std::memset(OutBuf, 0, OutSize);
    std::memcpy(static_cast<char *>(KA.get()) + Layout.OutputPtrOffset, &OutBuf,
                sizeof(OutBuf));
    // Dispatch geometry and LDS/scratch sizes are single-sourced from the
    // layout (1 work-item for VALU; a full wave + LDS for DS).
    auto ResOrErr = Dispatcher.dispatch(
        Agent, ELF, KernelName, KA.get(),
        std::max(Layout.TotalSize, 1024u), Layout.GridSizeX,
        Layout.WorkgroupSizeX, Layout.GroupSegmentSize,
        Layout.PrivateSegmentSize);
    if (!ResOrErr)
      return ResOrErr.takeError();
    if (ResOrErr->SignalValue != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          llvm::formatv("dispatch completed with signal {0}",
                        ResOrErr->SignalValue));
    return llvm::Error::success();
  };

  Dump.setLayout(Layout);
  Dump.setInputs(R.InputValues);

  // A GPU memory fault aborts the process inside the HSA runtime, so the report
  // has to be on disk before the dispatch that might trigger it. If the process
  // survives, the report is rewritten with the real verdict below.
  Dump.write("DISPATCHING (reference kernel)",
             "The process did not get past the reference dispatch. If this "
             "report survived, the reference kernel faulted on the device.");
  if (auto Err = RunOnce(RefELF, KCtxOrErr->KernelName, Out1.get())) {
    R.ErrorMsg = "reference dispatch: " + llvm::toString(std::move(Err));
    Dump.write("FAILED (reference dispatch)", R.ErrorMsg);
    return R;
  }

  Dump.write(CompareTranslation ? "DISPATCHING (translated kernel)"
                                : "DISPATCHING (second reference run)",
             "The process did not get past the second dispatch. If this report "
             "survived, that kernel faulted on the device — see the lifted IR "
             "below and the automated analysis above.");
  if (auto Err = RunOnce(Run2ELF, Run2KernelName, Out2.get())) {
    R.ErrorMsg = (CompareTranslation ? "translation dispatch: "
                                     : "second dispatch: ") +
                 llvm::toString(std::move(Err));
    Dump.write("FAILED (second dispatch)", R.ErrorMsg);
    return R;
  }

  R.Passed = true;
  for (const auto &OF : Layout.Outputs) {
    TestResult::OutputComparison C;
    C.Name = OF.Name;
    C.SizeBytes = OF.SizeBytes;
    std::memcpy(&C.Reference, static_cast<char *>(Out1.get()) + OF.Offset,
                OF.SizeBytes);
    std::memcpy(&C.Translated, static_cast<char *>(Out2.get()) + OF.Offset,
                OF.SizeBytes);
    C.Matches = std::memcmp(static_cast<char *>(Out1.get()) + OF.Offset,
                            static_cast<char *>(Out2.get()) + OF.Offset,
                            OF.SizeBytes) == 0;
    if (!C.Matches)
      R.Passed = false;
    R.Outputs.push_back(std::move(C));
  }

  if (!R.Passed) {
    R.ErrorMsg = CompareTranslation
                     ? "translation-path output differs from the reference"
                     : "outputs differ between two identical dispatches";
    Dump.write("FAILED (output mismatch)", formatResult(R));
  } else {
    Dump.discard();
  }
  return R;
}

} // namespace luthier::test
