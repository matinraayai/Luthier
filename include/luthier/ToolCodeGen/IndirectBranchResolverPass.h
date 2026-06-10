//===-- IndirectBranchResolverPass.h ----------------------------*- C++ -*-===//
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
/// Public interface of the indirect-branch resolution feature: the shared POD
/// ABI with the device-side resolver, plus the \c IndirectBranchResolverPass
/// declaration. All of the pass's logic — and the device resolver itself — live
/// in \c IndirectBranchResolverPass.hip.
///
/// \par Why a pimpl
/// The embedded device resolver is reached through a
/// \c DeviceToolCodeFatBinaryParser, whose \c inline \c static
/// \c [[gnu::used]] fat-binary slots get emitted (as COMDAT) in every TU that
/// completes that parser type — and only the bundle's host TU
/// (\c IndirectBranchResolverPass.hip) has those slots populated by
/// \c LoadHIPFATBinaryInfoPass. To keep the parser instantiated in exactly that
/// one TU, the pass holds the parser-deriving \c Impl behind a
/// \c std::unique_ptr to an \b incomplete type: \c Impl is only defined in the
/// \c .hip, so no other TU (the driver included) instantiates the slots, yet
/// this header stays freely includable.
///
/// \par Device-compilation guard
/// During device compilation (\c __HIP_DEVICE_COMPILE__) only the POD ABI below
/// is visible, so the device resolver pulls in no LLVM headers.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INDIRECT_BRANCH_RESOLVER_PASS_H
#define LUTHIER_TOOL_CODE_GEN_INDIRECT_BRANCH_RESOLVER_PASS_H

#include "luthier/ToolCodeGen/InjectedPayloadCreationPass.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <memory>

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace luthier {

/// \brief Rewrites every computed-target branch/call in the lifted trace so it
/// jumps to the *instrumented* (load-domain) copy of its callee.
///
/// \details On the first target \c MachineFunction that has an indirect site,
/// the pass links the arch-matching device resolver bitcode (carried in its own
/// embedded fat binary via \c Impl) into the IModule; the resolver function
/// \c __luthier_resolve_indirect then becomes a non-payload IModule function
/// that \c TargetModulePatcherPass clones into the target module. For each
/// indirect branch/call MI the pass injects a payload that reads the
/// computed-target register (a trace address), calls
/// \c __luthier_resolve_indirect, and writes the returned load address back
/// into that register before the transfer executes. It also drops a
/// \c .luthier.uses_hostcall marker so the launcher provisions a hostcall
/// buffer for the resolver's slow path.
///
/// \note Indirect *returns* are intentionally NOT patched: a return address is
/// already a load-domain address, not a callee trace entry, so resolving it
/// through the trace->load map is wrong. Uniform return handling is deferred.
///
/// All member definitions live in \c IndirectBranchResolverPass.hip (see the
/// file header for the pimpl rationale).
class IndirectBranchResolverPass
    : public InjectedPayloadCreationPass<IndirectBranchResolverPass,
                                         llvm::Module> {
public:
  /// Parses this pass's embedded resolver fat binary. \p Err is set on a bundle
  /// parse failure (a build bug — the bundle is produced at compile time).
  explicit IndirectBranchResolverPass(llvm::Error &Err);

  // Defaulted in the .hip, where \c Impl is complete (the unique_ptr deleter
  // needs the complete type). Move-only — the new-PM pass manager move-
  // constructs the pass into its model.
  IndirectBranchResolverPass(IndirectBranchResolverPass &&) noexcept;
  ~IndirectBranchResolverPass();

  /// \c InstrumentationPass entry point (granularity: one target
  /// \c MachineFunction at a time). Links the resolver lazily on the first MF
  /// that has an indirect site, then injects a dispatcher before each site.
  InstrumentationPreservedAnalyses runInstrumentationPass(
      llvm::Module &IModule, llvm::ModuleAnalysisManager &IMAM,
      llvm::Module &TargetModule, llvm::ModuleAnalysisManager &TargetMAM);

private:
  /// Parse + link the arch-matching resolver bitcode into \p IModule (once).
  /// \p TM is the target module's target machine (for the ISA tuple).
  llvm::Error ensureResolverLinked(llvm::Module &IModule,
                                   const llvm::TargetMachine &TM);

  /// Inject the dispatcher payload on the computed-target transfer \p MI.
  ///
  /// Reads the target register, then walks an if/else chain over the
  /// statically-known \p CallTargets, comparing the incoming trace address
  /// against each callee's entry-point trace address and, on a match, writing
  /// that callee's (instrumented) address into the target register and
  /// returning. If \p IsIncomplete (the call site has unresolved targets), the
  /// default arm calls \c __luthier_resolve_indirect and writes its result. If
  /// \p MI is a call, it is also rewritten into a plain indirect jump.
  /// \p TM supplies the payload's subtarget attributes.
  llvm::Error injectDispatcher(llvm::Module &IModule, llvm::MachineInstr &MI,
                               const llvm::TargetMachine &TM,
                               llvm::ArrayRef<llvm::Function *> CallTargets,
                               bool IsIncomplete);

  /// The parser holding this pass's embedded resolver fat binary. Held by
  /// pointer to an incomplete type so only the \c .hip TU instantiates its
  /// \c [[gnu::used]] fat-binary slots (see the file header).
  struct Impl;
  std::unique_ptr<Impl> PImpl;

  /// Set once the resolver bitcode has been linked into the IModule.
  bool Linked = false;
  /// The linked \c __luthier_resolve_indirect function in the IModule.
  llvm::Function *ResolverFn = nullptr;
};

} // namespace luthier

#endif // LUTHIER_TOOL_CODE_GEN_INDIRECT_BRANCH_RESOLVER_PASS_H
