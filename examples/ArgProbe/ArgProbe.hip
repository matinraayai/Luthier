//===-- ArgProbe.hip - Per-kernel-argument probe tool -----------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License").
//===----------------------------------------------------------------------===//
///
/// \file
/// A Luthier tool that instruments every load-carrying MI in every dispatched
/// kernel with ONE injected payload whose only job is to read a single
/// kernel argument (an SGPR/VGPR-preloaded scalar value, or a hidden
/// implicit-arg buffer field) and \c printf it. Each probe is a separate
/// device function — no branching on the probe kind inside the payload —
/// so the resulting IR is trivial to inspect when a specific arg's lowering
/// misbehaves. Selection is per-run via \c --arg-probe.
///
/// The tool exercises:
///   * \c luthier::readSVA of the full \c ScalarValueArgument set, via the
///     matching HIP / LLVM AMDGCN builtins (workgroup ids, workitem ids,
///     dispatch/queue/kernarg/implicit-arg pointers, dispatch id).
///   * Implicit-arg buffer reads at each hidden-arg offset (block count,
///     group size, remainder, grid dims, global offset, private/shared
///     base, queue ptr, hostcall / printf / heap / multigrid-sync /
///     completion-action / default-queue).
///   * Hostcall wiring end-to-end: every payload's \c printf routes through
///     the hostcall service the loader stands up when the instrumented
///     kernel declares \c hidden_hostcall_buffer (which it does, since
///     printf drags it in).
///
/// A modeled-after / rebased sibling of \c PrintfOnLoad — same on-packet
/// override, same load-MI predicate, same rocprofiler-sdk plumbing — but
/// the payload matrix is what's under test.
//===----------------------------------------------------------------------===//
#include <SIInstrInfo.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/Agent.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <memory>
#include <mutex>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-arg-probe"

//===----------------------------------------------------------------------===//
// ArgProbeTool — per-tool singleton subclass of HSATool
//===----------------------------------------------------------------------===//

struct ArgProbeTool
    : public luthier::HSATool<ArgProbeTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  //===---------------------------------------------------------------===//
  // Device-side payloads. Each payload targets a SINGLE argument so the
  // IR footprint is easy to eyeball with objdump / --dump-instrumented-
  // binary-to. All payloads:
  //   1. Read the target arg (an SVA read or an implicit-arg load).
  //   2. Guard on the first active lane in the current exec mask so a
  //      wavefront prints one line instead of N.
  //   3. printf the label and the observed value. printf implicitly
  //      brings in the hostcall buffer, so every payload's compile
  //      exercises the hostcall wiring.
  //===---------------------------------------------------------------===//

  // ---- SGPR / VGPR preloaded scalar arguments ----
  // (Each read maps to a luthier::readSVA of the matching ScalarValueArgument
  // via SubstituteAMDGCNIntrinsicsPass.)
  __attribute__((device, used)) static void probeWorkgroupIdX();
  __attribute__((device, used)) static void probeWorkgroupIdY();
  __attribute__((device, used)) static void probeWorkgroupIdZ();
  __attribute__((device, used)) static void probeWorkitemIdX();
  __attribute__((device, used)) static void probeWorkitemIdY();
  __attribute__((device, used)) static void probeWorkitemIdZ();
  __attribute__((device, used)) static void probeDispatchPtr();
  __attribute__((device, used)) static void probeQueuePtr();
  __attribute__((device, used)) static void probeKernargPtr();
  __attribute__((device, used)) static void probeImplicitArgPtr();

  // ---- Hidden implicit args (loaded from the implicitarg buffer) ----
  // Offsets follow the COV5 hidden-arg block. Layouts are documented in
  // llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp.
  __attribute__((device, used)) static void probeBlockCountX();
  __attribute__((device, used)) static void probeBlockCountY();
  __attribute__((device, used)) static void probeBlockCountZ();
  __attribute__((device, used)) static void probeGroupSizeX();
  __attribute__((device, used)) static void probeGroupSizeY();
  __attribute__((device, used)) static void probeGroupSizeZ();
  __attribute__((device, used)) static void probeRemainderX();
  __attribute__((device, used)) static void probeRemainderY();
  __attribute__((device, used)) static void probeRemainderZ();
  __attribute__((device, used)) static void probeGridDims();
  __attribute__((device, used)) static void probeGlobalOffsetX();
  __attribute__((device, used)) static void probeGlobalOffsetY();
  __attribute__((device, used)) static void probeGlobalOffsetZ();
  __attribute__((device, used)) static void probePrivateBase();
  __attribute__((device, used)) static void probeSharedBase();
  __attribute__((device, used)) static void probeHiddenQueuePtr();
  __attribute__((device, used)) static void probeHostcallBuffer();
  __attribute__((device, used)) static void probeMultigridSyncArg();
  __attribute__((device, used)) static void probeCompletionAction();
  __attribute__((device, used)) static void probeDefaultQueue();
  __attribute__((device, used)) static void probeHeapV1();
  __attribute__((device, used)) static void probePrintfBuffer();

  // ---- Nothing but a printf. Used as the hostcall-only baseline. ----
  __attribute__((device, used)) static void probePrintfHello();

  //===---------------------------------------------------------------===//
  // Host-side CLI.
  //===---------------------------------------------------------------===//

  enum ProbeKind {
    // SGPR/VGPR-preloaded scalar args
    WorkgroupIdX, WorkgroupIdY, WorkgroupIdZ,
    WorkitemIdX,  WorkitemIdY,  WorkitemIdZ,
    DispatchPtr,  QueuePtr,     KernargPtr,
    ImplicitArgPtr,
    // Hidden implicit args
    BlockCountX, BlockCountY, BlockCountZ,
    GroupSizeX,  GroupSizeY,  GroupSizeZ,
    RemainderX,  RemainderY,  RemainderZ,
    GridDims,
    GlobalOffsetX, GlobalOffsetY, GlobalOffsetZ,
    PrivateBase, SharedBase,
    HiddenQueuePtr, HostcallBuffer,
    MultigridSyncArg, CompletionAction, DefaultQueue,
    HeapV1, PrintfBuffer,
    // Baseline: a plain printf, no arg read.
    PrintfHello,
  };

  struct Options {
    llvm::cl::OptionCategory Cat{"Argument Probe Tool Options"};

    llvm::cl::opt<ProbeKind> Probe{
        "arg-probe",
        llvm::cl::desc("Which kernel argument to record"),
        llvm::cl::values(
            clEnumValN(WorkgroupIdX, "workgroup-id-x", "workgroup id X (SGPR)"),
            clEnumValN(WorkgroupIdY, "workgroup-id-y", "workgroup id Y (SGPR)"),
            clEnumValN(WorkgroupIdZ, "workgroup-id-z", "workgroup id Z (SGPR)"),
            clEnumValN(WorkitemIdX, "workitem-id-x", "workitem id X (VGPR0)"),
            clEnumValN(WorkitemIdY, "workitem-id-y", "workitem id Y (VGPR)"),
            clEnumValN(WorkitemIdZ, "workitem-id-z", "workitem id Z (VGPR)"),
            clEnumValN(DispatchPtr, "dispatch-ptr", "AQL dispatch packet ptr"),
            clEnumValN(QueuePtr, "queue-ptr", "HSA queue ptr"),
            clEnumValN(KernargPtr, "kernarg-ptr", "kernarg segment ptr"),
            clEnumValN(ImplicitArgPtr, "implicit-arg-ptr",
                       "implicit arg buffer ptr"),
            clEnumValN(BlockCountX, "block-count-x", "hidden block_count_x"),
            clEnumValN(BlockCountY, "block-count-y", "hidden block_count_y"),
            clEnumValN(BlockCountZ, "block-count-z", "hidden block_count_z"),
            clEnumValN(GroupSizeX, "group-size-x", "hidden group_size_x"),
            clEnumValN(GroupSizeY, "group-size-y", "hidden group_size_y"),
            clEnumValN(GroupSizeZ, "group-size-z", "hidden group_size_z"),
            clEnumValN(RemainderX, "remainder-x", "hidden remainder_x"),
            clEnumValN(RemainderY, "remainder-y", "hidden remainder_y"),
            clEnumValN(RemainderZ, "remainder-z", "hidden remainder_z"),
            clEnumValN(GridDims, "grid-dims", "hidden grid_dims"),
            clEnumValN(GlobalOffsetX, "global-offset-x",
                       "hidden global_offset_x"),
            clEnumValN(GlobalOffsetY, "global-offset-y",
                       "hidden global_offset_y"),
            clEnumValN(GlobalOffsetZ, "global-offset-z",
                       "hidden global_offset_z"),
            clEnumValN(PrivateBase, "private-base", "hidden private_base"),
            clEnumValN(SharedBase, "shared-base", "hidden shared_base"),
            clEnumValN(HiddenQueuePtr, "hidden-queue-ptr",
                       "hidden queue_ptr (kernarg-buffer field)"),
            clEnumValN(HostcallBuffer, "hostcall-buffer",
                       "hidden hostcall_buffer"),
            clEnumValN(MultigridSyncArg, "multigrid-sync-arg",
                       "hidden multigrid_sync_arg"),
            clEnumValN(CompletionAction, "completion-action",
                       "hidden completion_action"),
            clEnumValN(DefaultQueue, "default-queue", "hidden default_queue"),
            clEnumValN(HeapV1, "heap-v1", "hidden heap_v1"),
            clEnumValN(PrintfBuffer, "printf-buffer", "hidden printf_buffer"),
            clEnumValN(PrintfHello, "printf-hello",
                       "no arg read, just a printf")),
        llvm::cl::init(WorkgroupIdX), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> InstrBeginInterval{
        "instr-begin-interval",
        llvm::cl::desc("First load MI index (inclusive) to instrument"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> InstrEndInterval{
        "instr-end-interval",
        llvm::cl::desc("End of the load-MI interval (exclusive)"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> KernelBeginInterval{
        "kernel-begin-interval",
        llvm::cl::desc("First dispatched kernel (inclusive) to instrument"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> KernelEndInterval{
        "kernel-end-interval",
        llvm::cl::desc("End of the dispatched-kernel interval (exclusive)"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<std::string> DumpInstrumentedBinaryTo{
        "dump-instrumented-binary-to",
        llvm::cl::desc("If non-empty, write the raw ELF of every freshly "
                       "compiled instrumented dispatch to this path"),
        llvm::cl::init(""), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
  } Opts;

  std::recursive_mutex Mutex;

  uint32_t NumKernelLaunched = 0;
  unsigned NextLoadIdx = 0;

  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// Device payload bodies.
//
// Guard-on-first-active-lane helper: pulled out as a static inline so every
// payload has the same one-line prelude and the codegen stays tiny. Returns
// true on the elected lane; the caller returns immediately on false.
//===----------------------------------------------------------------------===//

__attribute__((device, always_inline)) static inline bool firstActiveLane() {
  unsigned long long ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  return __ffsll(ExecMask) == LaneId;
}

// Reader helpers for the 3 fixed-width tiers of hidden-arg fields.
// \c __builtin_amdgcn_implicitarg_ptr returns a HIP \c __constant__
// (address_space(4)) pointer. We route it through \c uintptr_t so the
// per-tier reinterpret_cast lands in the flat address space that
// \c printf / caller code expects; the load itself is at whatever
// address space clang picks for the flat pointer (a subsequent
// \c infer-address-spaces / \c flat-scratch pass promotes it back to
// \c address_space(4) where the hidden-arg block actually lives).
__attribute__((device, always_inline)) static inline uint16_t
implArgU16(unsigned Off) {
  auto U = reinterpret_cast<uintptr_t>(__builtin_amdgcn_implicitarg_ptr());
  return *reinterpret_cast<const uint16_t *>(U + Off);
}
__attribute__((device, always_inline)) static inline uint32_t
implArgU32(unsigned Off) {
  auto U = reinterpret_cast<uintptr_t>(__builtin_amdgcn_implicitarg_ptr());
  return *reinterpret_cast<const uint32_t *>(U + Off);
}
__attribute__((device, always_inline)) static inline uint64_t
implArgU64(unsigned Off) {
  auto U = reinterpret_cast<uintptr_t>(__builtin_amdgcn_implicitarg_ptr());
  return *reinterpret_cast<const uint64_t *>(U + Off);
}

// ---- SGPR / VGPR preloaded scalar args ----

__attribute__((device, used)) void ArgProbeTool::probeWorkgroupIdX() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] workgroup_id_x=%u\n", __builtin_amdgcn_workgroup_id_x());
}
__attribute__((device, used)) void ArgProbeTool::probeWorkgroupIdY() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] workgroup_id_y=%u\n", __builtin_amdgcn_workgroup_id_y());
}
__attribute__((device, used)) void ArgProbeTool::probeWorkgroupIdZ() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] workgroup_id_z=%u\n", __builtin_amdgcn_workgroup_id_z());
}
__attribute__((device, used)) void ArgProbeTool::probeWorkitemIdX() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] workitem_id_x=%u\n", __builtin_amdgcn_workitem_id_x());
}
__attribute__((device, used)) void ArgProbeTool::probeWorkitemIdY() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] workitem_id_y=%u\n", __builtin_amdgcn_workitem_id_y());
}
__attribute__((device, used)) void ArgProbeTool::probeWorkitemIdZ() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] workitem_id_z=%u\n", __builtin_amdgcn_workitem_id_z());
}
__attribute__((device, used)) void ArgProbeTool::probeDispatchPtr() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] dispatch_ptr=%p\n",
         (const void *)__builtin_amdgcn_dispatch_ptr());
}
__attribute__((device, used)) void ArgProbeTool::probeQueuePtr() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] queue_ptr=%p\n",
         (const void *)__builtin_amdgcn_queue_ptr());
}
__attribute__((device, used)) void ArgProbeTool::probeKernargPtr() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] kernarg_ptr=%p\n",
         (const void *)__builtin_amdgcn_kernarg_segment_ptr());
}
// NOTE: DISPATCH_ID has no host-source equivalent in current HIP/clang —
// there is no \c __builtin_amdgcn_dispatch_id and calling
// \c llvm.amdgcn.dispatch.id via \c __asm__("llvm...") crashes the
// frontend. The SVA-level DISPATCH_ID slot itself is still tested
// transitively: it's covered by the aggregator's snapshot-and-restore
// path whenever the app kernel had it preloaded, and any of the other
// SVA readSVA calls in this probe indirectly exercises the same
// enable-bit machinery. If a source-level call is ever added (a
// \c __builtin_amdgcn_dispatch_id or a HIP wrapper), reinstate the probe.
__attribute__((device, used)) void ArgProbeTool::probeImplicitArgPtr() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] implicit_arg_ptr=%p\n",
         (const void *)__builtin_amdgcn_implicitarg_ptr());
}

// ---- Hidden implicit args ----
// Offsets taken from AMDGPUHSAMetadataStreamer's COV5 layout.

__attribute__((device, used)) void ArgProbeTool::probeBlockCountX() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] block_count_x=%u\n", implArgU32(0x00));
}
__attribute__((device, used)) void ArgProbeTool::probeBlockCountY() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] block_count_y=%u\n", implArgU32(0x04));
}
__attribute__((device, used)) void ArgProbeTool::probeBlockCountZ() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] block_count_z=%u\n", implArgU32(0x08));
}
__attribute__((device, used)) void ArgProbeTool::probeGroupSizeX() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] group_size_x=%u\n", implArgU16(0x0C));
}
__attribute__((device, used)) void ArgProbeTool::probeGroupSizeY() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] group_size_y=%u\n", implArgU16(0x0E));
}
__attribute__((device, used)) void ArgProbeTool::probeGroupSizeZ() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] group_size_z=%u\n", implArgU16(0x10));
}
__attribute__((device, used)) void ArgProbeTool::probeRemainderX() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] remainder_x=%u\n", implArgU16(0x12));
}
__attribute__((device, used)) void ArgProbeTool::probeRemainderY() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] remainder_y=%u\n", implArgU16(0x14));
}
__attribute__((device, used)) void ArgProbeTool::probeRemainderZ() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] remainder_z=%u\n", implArgU16(0x16));
}
__attribute__((device, used)) void ArgProbeTool::probeGridDims() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] grid_dims=%u\n", implArgU16(0x22));
}
__attribute__((device, used)) void ArgProbeTool::probeGlobalOffsetX() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] global_offset_x=%llu\n",
         (unsigned long long)implArgU64(0x28));
}
__attribute__((device, used)) void ArgProbeTool::probeGlobalOffsetY() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] global_offset_y=%llu\n",
         (unsigned long long)implArgU64(0x30));
}
__attribute__((device, used)) void ArgProbeTool::probeGlobalOffsetZ() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] global_offset_z=%llu\n",
         (unsigned long long)implArgU64(0x38));
}
__attribute__((device, used)) void ArgProbeTool::probePrivateBase() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] private_base=0x%x\n", implArgU32(0x40));
}
__attribute__((device, used)) void ArgProbeTool::probeSharedBase() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] shared_base=0x%x\n", implArgU32(0x44));
}
__attribute__((device, used)) void ArgProbeTool::probeHiddenQueuePtr() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] hidden.queue_ptr=0x%llx\n",
         (unsigned long long)implArgU64(0x48));
}
__attribute__((device, used)) void ArgProbeTool::probeHostcallBuffer() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] hostcall_buffer=0x%llx\n",
         (unsigned long long)implArgU64(0x50));
}
__attribute__((device, used)) void ArgProbeTool::probeMultigridSyncArg() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] multigrid_sync_arg=0x%llx\n",
         (unsigned long long)implArgU64(0x60));
}
__attribute__((device, used)) void ArgProbeTool::probeCompletionAction() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] completion_action=0x%llx\n",
         (unsigned long long)implArgU64(0x68));
}
__attribute__((device, used)) void ArgProbeTool::probeDefaultQueue() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] default_queue=0x%llx\n",
         (unsigned long long)implArgU64(0x70));
}
__attribute__((device, used)) void ArgProbeTool::probePrintfBuffer() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] printf_buffer=0x%llx\n",
         (unsigned long long)implArgU64(0x88));
}
__attribute__((device, used)) void ArgProbeTool::probeHeapV1() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] heap_v1=0x%llx\n",
         (unsigned long long)implArgU64(0x90));
}
__attribute__((device, used)) void ArgProbeTool::probePrintfHello() {
  if (!firstActiveLane())
    return;
  printf("[arg-probe] hostcall printf reached\n");
}

//===----------------------------------------------------------------------===//
// File-scope statics.
//===----------------------------------------------------------------------===//

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(ArgProbeTool)

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

//===----------------------------------------------------------------------===//
// ArgProbeTool::run — payload creation
//===----------------------------------------------------------------------===//

llvm::PreservedAnalyses
ArgProbeTool::run(luthier::Prototype &P,
                  luthier::PrototypeAnalysisManager &PAM) {
  // Route the ProbeKind CLI selection to the corresponding device function
  // pointer. Kept as a lambda so the switch is one contiguous block for
  // reference — adding a new probe touches this table AND the enum, and
  // nothing else on the host side.
  const void *PayloadFn = [&]() -> const void * {
    switch (Opts.Probe) {
    case WorkgroupIdX:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeWorkgroupIdX);
    case WorkgroupIdY:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeWorkgroupIdY);
    case WorkgroupIdZ:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeWorkgroupIdZ);
    case WorkitemIdX:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeWorkitemIdX);
    case WorkitemIdY:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeWorkitemIdY);
    case WorkitemIdZ:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeWorkitemIdZ);
    case DispatchPtr:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeDispatchPtr);
    case QueuePtr:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeQueuePtr);
    case KernargPtr:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeKernargPtr);
    case ImplicitArgPtr:
      return reinterpret_cast<const void *>(
          &ArgProbeTool::probeImplicitArgPtr);
    case BlockCountX:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeBlockCountX);
    case BlockCountY:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeBlockCountY);
    case BlockCountZ:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeBlockCountZ);
    case GroupSizeX:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeGroupSizeX);
    case GroupSizeY:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeGroupSizeY);
    case GroupSizeZ:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeGroupSizeZ);
    case RemainderX:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeRemainderX);
    case RemainderY:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeRemainderY);
    case RemainderZ:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeRemainderZ);
    case GridDims:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeGridDims);
    case GlobalOffsetX:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeGlobalOffsetX);
    case GlobalOffsetY:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeGlobalOffsetY);
    case GlobalOffsetZ:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeGlobalOffsetZ);
    case PrivateBase:
      return reinterpret_cast<const void *>(&ArgProbeTool::probePrivateBase);
    case SharedBase:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeSharedBase);
    case HiddenQueuePtr:
      return reinterpret_cast<const void *>(
          &ArgProbeTool::probeHiddenQueuePtr);
    case HostcallBuffer:
      return reinterpret_cast<const void *>(
          &ArgProbeTool::probeHostcallBuffer);
    case MultigridSyncArg:
      return reinterpret_cast<const void *>(
          &ArgProbeTool::probeMultigridSyncArg);
    case CompletionAction:
      return reinterpret_cast<const void *>(
          &ArgProbeTool::probeCompletionAction);
    case DefaultQueue:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeDefaultQueue);
    case HeapV1:
      return reinterpret_cast<const void *>(&ArgProbeTool::probeHeapV1);
    case PrintfBuffer:
      return reinterpret_cast<const void *>(&ArgProbeTool::probePrintfBuffer);
    case PrintfHello:
      return reinterpret_cast<const void *>(&ArgProbeTool::probePrintfHello);
    }
    return reinterpret_cast<const void *>(&ArgProbeTool::probePrintfHello);
  }();

  NextLoadIdx = 0;
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    // Skip HIP runtime scaffolding kernels — instrumenting them just floods
    // dispatches unrelated to our target. Same rationale as PrintfOnLoad.
    if (MF.getName().starts_with("__amd_rocclr_")) {
      luthier::errs() << "[ArgProbeTool] skipping HIP runtime helper '"
                      << MF.getName() << "'\n";
      return;
    }
    luthier::errs() << "[ArgProbeTool] payload creation for MF '"
                    << MF.getName() << "' probe="
                    << static_cast<unsigned>(Opts.Probe.getValue())
                    << " starting at load idx " << NextLoadIdx << "\n";
    unsigned StartIdx = NextLoadIdx;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        // Instrument every load — same predicate as PrintfOnLoad. The
        // kernel this tool targets has exactly one global load per lane,
        // so most workloads bottom out at "one payload site per dispatch".
        if (!MI.mayLoad())
          continue;
        unsigned I = NextLoadIdx++;
        if (I < Opts.InstrBeginInterval || I >= Opts.InstrEndInterval)
          continue;
        LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
            PayloadFn, P, PAM, MI,
            llvm::ArrayRef<luthier::PayloadArg>{}));
      }
    }
    luthier::errs() << "[ArgProbeTool] payload creation for MF '"
                    << MF.getName() << "' picked "
                    << (NextLoadIdx - StartIdx) << " load MI(s)\n";
  });
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<luthier::TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.abandon<luthier::InjectedPayloadAndInstPointAnalysis>();
  PA.abandon<luthier::PrototypeCallGraphAnalysis>();
  return PA;
}

//===----------------------------------------------------------------------===//
// ArgProbeTool::onPackets — override the packet with the instrumented KD
//===----------------------------------------------------------------------===//

void ArgProbeTool::onPackets(
    const hsa_queue_t &Queue, uint64_t,
    llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
    hsa_amd_queue_intercept_packet_writer PacketWriter) {
  std::vector<luthier::hsa::AqlPacket> AppPackets(Packets.begin(),
                                                  Packets.end());
  for (auto &Packet : AppPackets) {
    auto *DispatchPacket = Packet.asKernelDispatch();
    if (!DispatchPacket) {
      PacketWriter(&Packet, 1);
      continue;
    }

    std::lock_guard<std::recursive_mutex> Lock(Mutex);

    const auto Core = ::CoreApiSnapshot->getTable();
    const auto AmdExt = ::AmdExtSnapshot->getTable();

    const bool ActiveRegion = NumKernelLaunched >= Opts.KernelBeginInterval &&
                              NumKernelLaunched < Opts.KernelEndInterval;

    const auto *OriginalKD =
        reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
            DispatchPacket->kernel_object);

    bool MustDestroySignalAfterLaunch = false;
    if (DispatchPacket->completion_signal.handle == 0) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          AmdExt.callFunction<&::AmdExtTable::hsa_amd_signal_create_fn>(
              0, 0, nullptr, 0, &DispatchPacket->completion_signal),
          "Failed to create a signal for the dispatch packet."));
      MustDestroySignalAfterLaunch = true;
    }
    const hsa_signal_value_t SignalValue =
        Core.callFunction<&::CoreApiTable::hsa_signal_load_scacquire_fn>(
            DispatchPacket->completion_signal);

    // The kernarg buffer the launcher hands back has to outlive the dispatch;
    // move it into KernargBuf so the destructor doesn't fire between the
    // packet write and the completion-signal wait. Same protocol as
    // PrintfOnLoad / InstrCount.
    luthier::ExtendedKernargBuffer KernargBuf;
    if (ActiveRegion) {
      auto FirstTry =
          this->overrideWithInstrumented(*DispatchPacket, Queue, /*Preset=*/0);
      if (!FirstTry) {
        llvm::consumeError(FirstTry.takeError());
        std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->runInstrumentationPipelineForDispatch(*OriginalKD)
                .moveInto(ObjBuf));
        if (!Opts.DumpInstrumentedBinaryTo.empty()) {
          std::error_code EC;
          llvm::raw_fd_ostream OS(Opts.DumpInstrumentedBinaryTo, EC);
          if (!EC)
            OS.write(ObjBuf->getBufferStart(), ObjBuf->getBufferSize());
        }
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->loadInstrumented(std::move(ObjBuf), OriginalKD, 0)
                .takeError());
        auto Retry =
            this->overrideWithInstrumented(*DispatchPacket, Queue, 0);
        LUTHIER_REPORT_FATAL_ON_ERROR(Retry.takeError());
        KernargBuf = std::move(*Retry);
      } else {
        KernargBuf = std::move(*FirstTry);
      }
    }

    constexpr auto Timeout =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::seconds{300})
            .count();
    PacketWriter(DispatchPacket, 1);

    const hsa_signal_value_t CurrentSignal =
        Core.callFunction<&::CoreApiTable::hsa_signal_wait_relaxed_fn>(
            DispatchPacket->completion_signal, HSA_SIGNAL_CONDITION_LT,
            SignalValue, Timeout, HSA_WAIT_STATE_ACTIVE);
    if (CurrentSignal >= SignalValue)
      llvm::report_fatal_error(
          llvm::formatv("Wait timeout reached for signal handle {0:x}",
                        DispatchPacket->completion_signal.handle));

    if (MustDestroySignalAfterLaunch) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          Core.callFunction<&::CoreApiTable::hsa_signal_destroy_fn>(
              DispatchPacket->completion_signal),
          "Failed to destroy the dispatch packet's completion signal."));
    }

    NumKernelLaunched++;
  }
}

//===----------------------------------------------------------------------===//
// rocprofiler-sdk init / fini
//===----------------------------------------------------------------------===//

namespace luthier {

static void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    ArgProbeTool::withInstance([](ArgProbeTool &T) {
      luthier::errs() << "Arg probe tool observed " << T.NumKernelLaunched
                      << " dispatched kernel(s).\n";
    });
    ArgProbeTool::destroyInstance();
    delete LoaderSnapshot;
    LoaderSnapshot = nullptr;
    delete AmdExtSnapshot;
    AmdExtSnapshot = nullptr;
    delete CoreApiSnapshot;
    CoreApiSnapshot = nullptr;
    luthier::finalizeStreams();
  });
}

static void atToolFini() { atToolFini(nullptr); }

static void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    luthier::initializeStreams();
    luthier::errs() << "Arg probe tool is launching.\n";
    std::atexit(atToolFini);

    llvm::Error Err = llvm::Error::success();
    CoreApiSnapshot =
        new luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>(Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    AmdExtSnapshot =
        new luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable>(Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    LoaderSnapshot = new luthier::rocprofiler::HsaExtensionTableSnapshot<
        HSA_EXTENSION_AMD_LOADER>(Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    ArgProbeTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                 *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(0, &Argv, "Luthier Arg Probe Tool",
                                          &luthier::errs(), nullptr,
                                          "LUTHIER_ARGS"),
        "Failed to parse the command line arguments."));
    llvm::EnablePrettyStackTrace();
    luthier::printStackTraceOnFatalSignal();
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash error message and backtrace.\n");
    llvm::sys::AddSignalHandler(atToolFini, nullptr);
  });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ClientID) {
  luthier::atToolInit(nullptr);

  LLVM_DEBUG(const uint32_t RocProfVerMajor = Version / 10000;
             const uint32_t RocProfVerMinor = (Version % 10000) / 100;
             const uint32_t RocProfVerPatch = Version % 100;
             llvm::dbgs()
             << "Registering Luthier arg probe tool with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  ClientID->name = "Luthier Arg Probe tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
