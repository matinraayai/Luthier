//===-- PatchPCUsagesPass.h ----------------------------------*- C++ -*-===//
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

/// \brief Prototype-level pass that rewrites the target module's trace
/// PC-operating instructions so the code becomes relocatable.
class PatchPCUsagesPass : public llvm::PassInfoMixin<PatchPCUsagesPass> {
public:
  /// One entry of the \c EntryPointToTraceFunctionAddrMap
  /// table: a trace function's entry point address and its associated
  /// LLVM \c Function's handle address in the loaded tool module.
  struct EntryPointTraceFunctionEntry {
    uint64_t TraceAddr;
    uint64_t FnHandleAddr;
  };

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

  class Parser : public ToolDeviceCodeOffloadParserTrait<Parser> {
  public:
    using luthier::ToolDeviceCodeOffloadParserTrait<
        Parser>::ToolDeviceCodeOffloadParserTrait;

    /// Writes a single, statically-known target address into
    /// \p Reg. Emitted at every site with only a single possible target.
    __attribute__((device, used)) static void
    patchRegSingle(llvm::MCRegister Reg, std::uint64_t FnHandleAddr);

    /// Variadic-template multi-target hook. Every pack element is a
    /// \c EntryPointTraceFunctionEntry enumerating one candidate branch target
    /// known statically. At runtime the hook reads \p Reg, checks each
    /// compile-time pair, writes the matching \c FnHandleAddr on hit, or drops
    /// through to the \c EntryPointToTraceFunctionAddrMap + hostcall fallback
    /// when nothing matches.
    template <typename... Pairs>
    __attribute__((device, used)) static void
    patchRegMulti(llvm::MCRegister Reg, std::uint64_t HostFnPtr, Pairs... Ps);

    /// Runtime resolver table the \c patchRegMulti fallback path walks.
    __attribute__((device, used)) static EntryPointTraceFunctionEntry
        *EntryPointToTraceFunctionAddrMap;
    __attribute__((
        device,
        used)) static std::uint32_t EntryPointToTraceFunctionAddrMapSize;
    __attribute__((
        device,
        used)) static std::uint32_t EntryPointToTraceFunctionAddrMapMaxSize;
    /// Wave-uniform spinlock. Zero when free, one when held. The
    /// \c patchRegMulti fallback holds it across the entire read-then-
    /// hostcall region so the host callback can grow the table in place
    /// without racing concurrent readers.
    __attribute__((
        device,
        used)) static std::uint32_t EntryPointToTraceFunctionAddrMapLock;

    /// A set of statically-known \c EntryPointTraceFunctionEntry table +
    /// entry count. Used to initialize the \c EntryPointToTraceFunctionAddrMap
    /// value at load time. Only emitted when the initial entry point is a
    /// kernel.
    __attribute__((device, used,
                   constant)) static const EntryPointTraceFunctionEntry
        *EntryPointToTraceFunctionAddrMapSeed;
    __attribute__((device, used, constant)) static std::uint32_t
        EntryPointToTraceFunctionAddrMapSeedSize;

    /// ctor used to dynamically allocate the
    /// \c EntryPointToTraceFunctionAddrMap
    __attribute__((device, used, constructor)) static void
    initEntryPointToTraceFunctionAddrMap();

    /// dtor used to release the table \c initEntryPointToTraceFunctionAddrMap
    /// allocated.
    __attribute__((device, used, destructor)) static void
    finiEntryPointToTraceFunctionAddrMap();
  };

  /// \param TargetAddressHostResolver raw host function pointer instrumentation
  /// logic has to call in order to resolve missing target function addresses
  /// \param Err Error in case of failure to construct the pass
  explicit PatchPCUsagesPass(
      TargetAddressHostResolverFn TargetAddressHostResolver, llvm::Error &Err);

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);

private:
  TargetAddressHostResolverFn TargetAddressHostResolver;

  std::unique_ptr<Parser> OffloadParser;
};

} // namespace luthier

#endif // LUTHIER_TOOL_CODE_GEN_PATCH_PC_USAGES_PASS_H
