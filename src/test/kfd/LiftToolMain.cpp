//===-- LiftToolMain.cpp - a read-only Luthier tool below the runtime ------===//
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
///
/// \file
/// A preloadable Luthier tool that lifts every kernel an application dispatches,
/// and does nothing else.
///
/// \par What this is for
/// It is the end-to-end check that the KFD path joins up: the wrapper's packet
/// chain delivers a dispatch, the driver-level resolver finds the kernel's
/// allocation, the topology names the device, and code discovery disassembles
/// what it finds. Each of those has unit tests; none of them proves the four
/// work together against a live GPU, which is what this does.
///
/// \par Why it is a separate shared object rather than part of the harness
/// Two reasons, and the second is the load-bearing one.
///
/// It matches deployment: a tool is preloaded alongside the wrapper into an
/// application it did not build, and the application here --
/// \c kfd-nonhsa-tests -- is used completely unmodified, which is a stronger
/// statement than a harness instrumenting itself.
///
/// And it keeps \c LuthierTooling out of the application's binary.
/// \c LuthierTooling links \c hsa-runtime64, which carries its own copy of
/// hsakmt's state, while the harness links \c libhsakmt directly. Two copies in
/// one process each want their own \c /dev/kfd descriptor and their own DRM VM,
/// and the kernel permits one VM per GPU per process -- so the second to
/// initialize fails. Nothing here calls \c hsa_init, so the two copies stay
/// dormant; keeping them in separate objects means the arrangement is at least
/// visible rather than accidental.
///
/// \par What it reports, and why it reports failures loudly
/// A count, on exit. The failure this whole module is most prone to is attaching
/// successfully and then observing nothing, which looks exactly like an
/// application that dispatched nothing -- so a zero count is an error here, not
/// a quiet success.
///
/// The summary separates \c functions from \c invalid deliberately. Code
/// discovery returning no error means only that: it does not mean the lifted MIR
/// is well formed, and an earlier version of this tool counted the two as one
/// thing and reported 100% on workloads that contained malformed kernels. The
/// machine verifier is run per function, non-fatally, so the difference is
/// visible without the tool being able to take the application down.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/KFDTool.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/ToolCodeGen/Prototype.h"

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <cstdlib>

namespace {

/// Set by \c LUTHIER_KFD_LIFT_BLIND, which substitutes an accessor that knows
/// nothing. The negative control: on a GPU whose device pointers the host can
/// also read -- which is every GPU the application maps its own allocations on,
/// and every APU regardless -- code discovery could appear to succeed while
/// never consulting the accessor at all. If blinding the accessor does not break
/// it, the accessor was not what it was reading.
bool blindMode() {
  static const bool Blind = getenv("LUTHIER_KFD_LIFT_BLIND") != nullptr;
  return Blind;
}


/// Set by LUTHIER_KFD_LIFT_HSA. Brings HSA up on the first dispatch, which a
/// read-only tool does not need -- it is here because instrumenting *does* need
/// it, and this is the cheapest place to check that the sequence works before a
/// tool depends on it.
bool wantHsa() {
  static const bool Want = getenv("LUTHIER_KFD_LIFT_HSA") != nullptr;
  return Want;
}

/// A resolver that is present and knows nothing.
class BlindResolver final : public luthier::DriverAllocationResolver {
public:
  llvm::Expected<Allocation> resolve(uint64_t) const override {
    return Allocation();
  }
  bool isAvailable() const override { return true; }
};

std::atomic<uint64_t> Dispatches{0};
std::atomic<uint64_t> Lifted{0};
std::atomic<uint64_t> Failed{0};
std::atomic<uint64_t> Functions{0};
std::atomic<uint64_t> Invalid{0};

class LiftTool : public luthier::KFDTool<LiftTool> {
public:
  using luthier::KFDTool<LiftTool>::KFDTool;

  /// Deliberately overrides the base's accessor so the negative control can be
  /// selected at run time rather than at build time -- the point of a control is
  /// that it exercises the same binary.
  std::unique_ptr<luthier::MemoryAllocationAccessor>
  createMemoryAllocationAccessor() {
    if (blindMode())
      return std::make_unique<luthier::DriverOnlyMemoryAllocationAccessor>(
          std::make_unique<BlindResolver>());
    return luthier::KFDTool<LiftTool>::createMemoryAllocationAccessor();
  }

  void onDispatchPacket(const luthier::kfd::QueueInfo &Q, uint64_t,
                        luthier::hsa::AqlPacket &Packet) {
    const auto *Dispatch = Packet.asKernelDispatch();
    if (Dispatch == nullptr)
      return; // a barrier or something else; not ours
    Dispatches++;

    if (wantHsa()) {
      static std::once_flag Once;
      std::call_once(Once, [this] {
        if (llvm::Error Err = ensureHsaInitialized()) {
          fprintf(stderr, "[luthier-lift] HSA did not come up: %s\n",
                  llvm::toString(std::move(Err)).c_str());
          return;
        }
        fprintf(stderr, "[luthier-lift] HSA is up; snapshots core=%d amd_ext=%d "
                        "loader=%d\n",
                getCoreApiTableSnapshot().wasRegistrationCallbackInvoked(),
                getAmdExtTableSnapshot().wasRegistrationCallbackInvoked(),
                getLoaderTableSnapshot().wasRegistrationCallbackInvoked());
        auto AgentOrErr = agentForCurrentDispatch();
        if (!AgentOrErr) {
          fprintf(stderr, "[luthier-lift] no agent for this dispatch: %s\n",
                  llvm::toString(AgentOrErr.takeError()).c_str());
          return;
        }
        fprintf(stderr, "[luthier-lift] dispatch GPU maps to agent 0x%llx\n",
                static_cast<unsigned long long>(AgentOrErr->handle));
      });
    }

    // kernel_object is a device address, and the kernel descriptor lives at it.
    // Dereferencing it here rests on the application having mapped its own
    // allocation for host access, which every KFD application observed so far
    // does -- hsakmt and tinygrad both mmap the render node over the
    // allocation's own virtual address right after creating it. The lifting
    // below does not rest on that: it reads through the accessor's own mapping.
    const auto *KD =
        reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
            Dispatch->kernel_object);

    unsigned Functions_ = 0, Invalid_ = 0;
    llvm::Error Err = runCodeDiscoveryForDispatch(
        *KD, [&](luthier::Prototype &IP, luthier::PrototypeAnalysisManager &,
                 llvm::ModuleAnalysisManager &TargetMAM) -> llvm::Error {
          llvm::Module &M = IP.getTargetModule();

          // Reach the MachineFunctions the way CodeDiscoveryPass created them:
          // through the module's function analysis manager. Asking
          // MachineModuleInfo instead does not work and fails *silently* --
          // MMI only knows functions handed to it via insertFunction, which on
          // this path nothing does, so getMachineFunction returns null, the
          // verify call never runs, and the tool reports invalid=0 on every
          // workload including ones full of malformed kernels. That is exactly
          // what an earlier version of this file did.
          auto &FAM =
              TargetMAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M)
                  .getManager();

          for (llvm::Function &F : M) {
            if (F.isDeclaration())
              continue;
            Functions_++;

            // getCachedResult, not getResult: we want to verify what discovery
            // actually produced, not create an empty MachineFunction here and
            // then congratulate ourselves on it verifying.
            auto *Cached = FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
            if (Cached == nullptr) {
              llvm::errs() << "[luthier-lift] no MIR was produced for "
                           << F.getName() << "\n";
              Invalid_++;
              continue;
            }

            // Run the machine verifier, but never let it abort. The pass form
            // (MachineVerifierPass) calls report_fatal_error, which is right for
            // a standalone compiler and wrong here: this library is preloaded
            // into an application that did not ask for it, and taking that
            // application down because a kernel lifted to malformed MIR would be
            // a far worse outcome than reporting it.
            //
            // Worth doing rather than trusting the absence of an llvm::Error:
            // "discovery returned no error" and "the MIR is well formed" are
            // different claims, and the LDS/gds disassembler bug on gfx90a and
            // gfx942 produces exactly the second failure without the first.
            // Reports through llvm::errs() rather than a raw_string_ostream.
            // Not a style choice: handing it a raw_string_ostream makes it
            // return false while writing nothing, so every kernel came back
            // invalid with no explanation. Measured; the mechanism is not
            // understood, and errs() is where this tool's other output goes
            // anyway.
            if (!Cached->getMF().verify(nullptr, "luthier-lift", &llvm::errs(),
                                        /*AbortOnError=*/false))
              Invalid_++;
          }
          return llvm::Error::success();
        });

    if (Err) {
      Failed++;
      fprintf(stderr, "[luthier-lift] gpu=%u kernel_object=0x%llx FAILED: %s\n",
              Q.GpuId,
              static_cast<unsigned long long>(Dispatch->kernel_object),
              llvm::toString(std::move(Err)).c_str());
      return;
    }
    // Producing nothing counts as a failure, not a success with an empty result.
    // CodeDiscoveryPass reports an unresolvable address through the LLVM error
    // handler and returns normally, so "no error came back" is a much weaker
    // statement than it looks -- it is exactly what a blinded accessor produces.
    if (Functions_ == 0) {
      Failed++;
      fprintf(stderr,
              "[luthier-lift] gpu=%u kernel_object=0x%llx lifted NOTHING\n",
              Q.GpuId,
              static_cast<unsigned long long>(Dispatch->kernel_object));
      return;
    }
    Lifted++;
    Functions += Functions_;
    Invalid += Invalid_;
    fprintf(stderr,
            "[luthier-lift] gpu=%u kernel_object=0x%llx lifted %u function(s)"
            "%s\n",
            Q.GpuId, static_cast<unsigned long long>(Dispatch->kernel_object),
            Functions_, Invalid_ != 0 ? " [INVALID MIR]" : "");
  }
};

luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable> *CoreSnap{nullptr};
luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnap{nullptr};
luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnap{nullptr};

/// Stand the tool up, from a library constructor.
///
/// It cannot wait for \c rocprofiler_configure. That fires while HSA is
/// initializing, and in this design HSA initializes on the first dispatch -- a
/// dispatch the tool only sees because constructing it registered a packet
/// callback. Waiting for configure would mean never registering, never seeing a
/// dispatch, and never initializing HSA.
///
/// Requesting the API-table snapshots this early is allowed: the requirement is
/// that it happens before rocprofiler is fully configured, which a library
/// constructor satisfies. \c rocprofiler_configure is still exported, so
/// rocprofiler knows the process contains a client at all.
///
/// This says nothing about \e when HSA initializes -- still the first dispatch,
/// so the application keeps its claim on the driver's per-process resources. See
/// \c KFDTool::ensureHsaInitialized.
void attach() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    llvm::Error Err = llvm::Error::success();
    CoreSnap =
        new luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>(Err);
    AmdExtSnap =
        new luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable>(Err);
    LoaderSnap = new luthier::rocprofiler::HsaExtensionTableSnapshot<
        HSA_EXTENSION_AMD_LOADER>(Err);
    if (Err) {
      fprintf(stderr, "[luthier-lift] could not request API tables: %s\n",
              llvm::toString(std::move(Err)).c_str());
      return;
    }

    LiftTool::createInstance(*CoreSnap, *AmdExtSnap, *LoaderSnap, Err);
    if (Err) {
      fprintf(stderr, "[luthier-lift] could not attach: %s\n",
              llvm::toString(std::move(Err)).c_str());
      // Not fatal: the application is not ours to take down because optional
      // instrumentation failed. The summary reports zero, which the test treats
      // as failure -- so this cannot pass silently either.
      return;
    }
    fprintf(stderr, "[luthier-lift] attached%s\n",
            blindMode() ? " (blind: the accessor reports nothing)" : "");
  });
}

__attribute__((destructor)) void report() {
  fprintf(stderr,
          "[luthier-lift] SUMMARY dispatches=%llu lifted=%llu failed=%llu "
          "functions=%llu invalid=%llu\n",
          static_cast<unsigned long long>(Dispatches.load()),
          static_cast<unsigned long long>(Lifted.load()),
          static_cast<unsigned long long>(Failed.load()),
          static_cast<unsigned long long>(Functions.load()),
          static_cast<unsigned long long>(Invalid.load()));
}

/// Preloaded, so a constructor is the only entry point that runs early enough.
__attribute__((constructor)) void attachFromConstructor() { attach(); }

} // namespace

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
                      rocprofiler_client_id_t *ClientID) {
  ClientID->name = "Luthier KFD kernel-lifting tool";
  // Deliberately does no work: the tool was already stood up by the constructor,
  // because by the time this fires HSA is mid-initialization. Exported so that
  // rocprofiler recognises a client in the process.
  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr, nullptr, nullptr};
  return &Cfg;
}
