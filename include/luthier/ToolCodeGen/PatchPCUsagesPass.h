//===-- PatchPCUsagesPass.h -------------------------------------*- C++ -*-===//
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
/// \file
/// Declares \c PatchPCUsagesPass — a Prototype-level pass that
/// rewrites the PC-operating machine instructions of the target module so the
/// lifted code can execute at a runtime address different from the trace
/// address it was discovered at. Concretely, it patches every trace
/// \c S_GETPC_B64 / \c S_SETPC_B64 / \c S_SWAPPC_B64 / \c S_CALL_B64 /
/// \c S_ADD_PC_i64 in the target module. \c S_CBRANCH_*_FORK and
/// \c S_CBRANCH_JOIN are skipped.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_PATCH_PC_USAGES_PASS_H
#define LUTHIER_TOOL_CODE_GEN_PATCH_PC_USAGES_PASS_H
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h"
#include <cstdint>
#include <llvm/IR/PassManager.h>
#include <memory>

namespace luthier {

class PrototypeCallGraph;

/// \brief Prototype-level pass that rewrites the target module's trace
/// PC-operating instructions so the code becomes relocatable.
class PatchPCUsagesPass
    : public llvm::RequiredPassInfoMixin<PatchPCUsagesPass> {
public:

  /// Signature of the host callback. Matches the argument layout the AMD
  /// hostcall \c FUNCTION_CALL service uses: \p Out is a 2-word out
  /// buffer, \p In is a 7-word in buffer.
  ///
  ///   \c In[0] — unresolved trace address the device is asking about;
  ///   \c In[1] — address of the AQL dispatch packet that launched the
  ///              calling wave (read on-device via
  ///              \c __builtin_amdgcn_dispatch_ptr) — disambiguates
  ///              simultaneous dispatches when the pass's loaded modules
  ///              collide;
  ///   \c In[2] — address of \c Parser::EntryPointToTraceFunctionAddrMap (a
  ///              \c EntryPointTraceFunctionEntry**), so the callback can grow
  ///              / replace the pointer if the table needs reallocating;
  ///   \c In[3] — address of \c Parser::EntryPointToTraceFunctionAddrMapSize
  ///              (a \c uint32_t*), so the callback can advance the
  ///              live-entry count after appending;
  ///   \c In[4] — address of \c Parser::EntryPointToTraceFunctionAddrMapMaxSize
  ///              (a \c uint32_t*), so the callback knows when it has to
  ///              reallocate the underlying array;
  ///   \c Out[0] — resolved LLVM \c Function's handle address the device
  ///              should jump to.
  ///
  /// The device-side spinlock \c Parser::EntryPointToTraceFunctionAddrMapLock
  /// is held across the whole hostcall — the callback is safe to read and
  /// mutate all three of the above without acquiring anything itself.
  using TargetAddressHostResolverFn = void (*)(uint64_t Out[2],
                                               const uint64_t In[7]);

  /// \param TargetAddressHostResolver raw host function pointer instrumentation
  /// logic has to call in order to resolve missing target function addresses
  /// \param Err Error in case of failure to construct the pass
  explicit PatchPCUsagesPass(
      TargetAddressHostResolverFn TargetAddressHostResolver, llvm::Error &Err);

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);

private:
  class Parser : public ToolDeviceCodeOffloadParserTrait<Parser> {
  public:
    using ToolDeviceCodeOffloadParserTrait::ToolDeviceCodeOffloadParserTrait;
  };

  llvm::Error emitEntryPointSeed(Prototype &P) const;

  llvm::Error bindRuntimeTableToEnclosingKernel(Prototype &P) const;

  llvm::Error handleEntryPointToTraceFunctionAddrMapCtorDtor(
    Prototype &P, const EntryPoint &EP, const PrototypeCallGraph &CG) const;

  /// Resolver method called when there are no instrumented entries for an
  /// entry point
  TargetAddressHostResolverFn TargetAddressHostResolver;

  /// \c ToolDeviceCodeOffloaderParser instance used to parse the device bitcode
  std::unique_ptr<Parser> OffloadParser;
};

} // namespace luthier

#endif // LUTHIER_TOOL_CODE_GEN_PATCH_PC_USAGES_PASS_H
