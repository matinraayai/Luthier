//===-- RebaseAppScratchAccessesPass.h --------------------------*- C++ -*-===//
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
/// Declares \c RebaseAppScratchAccessesPass — a Prototype-level pass that
/// displaces every application access to the wavefront's private segment
/// upward, past the region Luthier reserves for its own instrumentation stack.
///
/// Luthier's instrumentation stack is pinned to offset zero of the wavefront's
/// private segment: the instrumentation stack pointer is the constant zero, the
/// state value array's emergency slots sit at offsets 0 and 4, and a payload's
/// own frame starts just above them. That region is
/// \c WORK_ITEM_INSTRUMENTATION_PRIVATE_SEGMENT_SIZE bytes wide and is decided
/// by \c TargetModulePatcherPass , which writes it into its own state value
/// array lane. Everything the *application* does with private memory therefore
/// has to move up by exactly that much, and this pass is what moves it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_REBASE_APP_SCRATCH_ACCESSES_PASS_H
#define LUTHIER_TOOL_CODE_GEN_REBASE_APP_SCRATCH_ACCESSES_PASS_H
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h"
#include <llvm/IR/PassManager.h>
#include <memory>

namespace luthier {

/// \brief Prototype-level pass that rewrites every application instruction
/// capable of reaching the private segment so that it addresses the
/// application's displaced frame rather than Luthier's reserve.
///
/// \details The rewrite is expressed as ordinary Luthier instrumentation rather
/// than as hand-written MIR. For each such instruction the pass reads the
/// operands it needs off the \c MachineInstr on the host, forwards the register
/// ones as \c RegArg (which \c Prototype::createInjectedPayload lowers into
/// \c luthier::readReg calls inside the payload) and the immediate ones as
/// \c llvm::Constant s, attaches one of the device-side payloads that live in
/// this pass's own translation unit, and then **erases the original
/// instruction** — the payload performs the access in its place. Register
/// allocation and preservation around the payload are the codegen pipeline's
/// job; this pass scavenges nothing.
///
/// Three classes of instruction are handled, and the address arithmetic differs
/// for each:
///
///   * <b>\c SCRATCH_* </b> (flat scratch, every addressing form) always
///     targets the private segment, so the displacement is added
///     unconditionally. The effective per-lane private offset is
///     <tt>vaddr + saddr + inst_offset</tt> per the ISA's
///     <tt>SWIZZLE(SGPR_offset + VGPR_offset + INST_OFFSET, TID)</tt>; forms
///     that lack an operand pass zero for it.
///
///   * <b>\c FLAT_* </b> may target the private segment or the global one
///     depending on which aperture the runtime address falls in, so the payload
///     tests it with \c __builtin_amdgcn_is_private (a comparison of the
///     address's high half against \c SRC_PRIVATE_BASE ) and adds the
///     displacement only on the private side. \c GLOBAL_* cannot reach scratch
///     and is left alone.
///
///   * <b>\c BUFFER_* </b> reaches scratch only through the application's
///     per-wave scratch resource descriptor. The aperture test does not apply —
///     a scratch V#'s base is the scratch backing memory's address, not an
///     aperture address — so the payload instead compares the instruction's V#
///     base against the wavefront's flat scratch base, which Luthier's kernel
///     prologue has already parked in the state value array's \c FLAT_SCRATCH
///     lanes. On a match the displacement is added to \c soffset , scaled by
///     the wavefront size because \c soffset is applied outside the swizzle.
///
/// The pass runs between the tool's own injection pass and
/// \c PatchPCUsagesPass . That is the only correct window: the target module
/// still holds nothing but application code (the instrumentation module is not
/// merged in until \c TargetModulePatcherPass ), and PC-usage patching has to
/// see the final instruction layout.
class RebaseAppScratchAccessesPass
    : public llvm::PassInfoMixin<RebaseAppScratchAccessesPass> {
public:
  /// \brief Owns this translation unit's offload bundle, which carries the
  /// device-side payloads, and gives the host side a way to map a payload's
  /// address to its device symbol name.
  ///
  /// The payloads themselves are free functions in the pass's \c .hip rather
  /// than members here: their signatures mention \c __amdgpu_buffer_rsrc_t ,
  /// which only exists in a HIP translation unit, and this header is included
  /// by plenty that are not.
  class Parser : public ToolDeviceCodeOffloadParserTrait<Parser> {
  public:
    using luthier::ToolDeviceCodeOffloadParserTrait<
        Parser>::ToolDeviceCodeOffloadParserTrait;
  };

  /// Constructor.
  /// \param [out] Err set if this translation unit's offload bundle could not
  /// be parsed, which is a build-configuration problem rather than a runtime
  /// one
  explicit RebaseAppScratchAccessesPass(llvm::Error &Err);

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);

  static llvm::StringRef name() {
    return "luthier-rebase-app-scratch-accesses";
  }

  static bool isRequired() { return true; }

private:
  std::unique_ptr<Parser> OffloadParser;
};

} // namespace luthier

#endif
