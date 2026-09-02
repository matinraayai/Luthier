//===-- TargetRegisterBudget.h ----------------------------------*- C++ -*-===//
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
/// The register budget an application's wavefront was launched with, and the
/// queries target-module passes should use in place of
/// \c MachineRegisterInfo::isReserved .
///
/// \c CodeDiscoveryPass lifts the launch budget out of the kernel descriptor
/// and records it on every target-module function as
/// \c luthier-app-num-sgpr / \c luthier-app-num-vgpr. Those attributes are
/// the ground truth for "what the application owns" and are never rewritten.
///
/// They are deliberately distinct from \c amdgpu-num-sgpr /
/// \c amdgpu-num-vgpr, which the AMDGPU backend reads as an allocation *cap*
/// when the target module is finally compiled, and which Luthier widens to
/// the subtarget's addressable maximum so instrumentation can allocate past
/// the application's original budget (see \c luthier-max-num-regs-attrs ).
///
/// Conflating the two is what makes \c MachineRegisterInfo::isReserved the
/// wrong query for target-module passes: with the widened cap it reports
/// nothing above the budget as reserved, and with the true cap it reports
/// *everything* above the budget as reserved. Neither answers the question
/// those passes are actually asking, which is one of:
///
///   * "does the application own this register?" -- use
///     \c addAppOwnedRegisters ; registers the wave was never launched with
///     hold no application-visible state and must not be treated as live.
///   * "may instrumentation take this register?" -- use
///     \c isAvailableForInstrumentation .
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TARGET_REGISTER_BUDGET_H
#define LUTHIER_TOOL_CODE_GEN_TARGET_REGISTER_BUDGET_H

#include <llvm/MC/MCRegister.h>

namespace llvm {
class Function;
class GCNSubtarget;
class LivePhysRegs;
class MachineFunction;
} // namespace llvm

namespace luthier {

/// Attribute holding the number of SGPRs the application's wavefront was
/// launched with, as lifted from the kernel descriptor.
inline constexpr const char *AppNumSGPRsAttribute = "luthier-app-num-sgpr";

/// Attribute holding the number of VGPRs the application's wavefront was
/// launched with, as lifted from the kernel descriptor.
inline constexpr const char *AppNumVGPRsAttribute = "luthier-app-num-vgpr";

/// The GPR budget the application's wavefront was launched with. Registers
/// at or above these counts physically exist on the device but were not
/// allocated to the wave, so they hold no application state and are free for
/// instrumentation to take.
struct TargetRegisterBudget {
  /// Application-owned SGPRs: \c SGPR0 .. \c SGPR(NumSGPRs-1) .
  unsigned NumSGPRs = 0;
  /// Application-owned VGPRs: \c VGPR0 .. \c VGPR(NumVGPRs-1) .
  unsigned NumVGPRs = 0;
  /// Application-owned AGPRs, zero on subtargets without them.
  unsigned NumAGPRs = 0;
};

/// Records \p NumSGPRs / \p NumVGPRs on \p F as the application's launch
/// budget. Called by \c CodeDiscoveryPass for every lifted target-module
/// function; the values survive any later rewrite of \c amdgpu-num-sgpr /
/// \c amdgpu-num-vgpr.
void setTargetRegisterBudget(llvm::Function &F, unsigned NumSGPRs,
                             unsigned NumVGPRs);

/// \return the application launch budget recorded on \p MF 's function,
/// clamped to what \p MF 's subtarget can address. Falls back to the
/// \c amdgpu-num-* attributes, and then to the subtarget maximums, for
/// functions that predate \c setTargetRegisterBudget .
TargetRegisterBudget
getTargetRegisterBudget(const llvm::MachineFunction &MF);

/// Adds to \p Out every register whose contents are application-visible
/// across an instrumentation point in \p MF : the launch-budget GPRs, plus
/// \c SCC / \c VCC, plus the runtime-owned registers outside the GPR files
/// (\c EXEC , \c FLAT_SCR , \c M0 , the \c TTMP / \c TBA / \c TMA family,
/// and the \c SRC_* hardware registers).
///
/// This is the conservative "assume everything is live" seed used when a
/// target function is not fully discovered. GPR membership comes from the
/// launch budget rather than from \c MachineRegisterInfo::isReserved , so
/// GPRs the wave was never launched with are correctly left out.
void addAppOwnedRegisters(const llvm::MachineFunction &MF,
                          llvm::LivePhysRegs &Out);

/// \return \c true if any 32-bit constituent of \p Reg lies inside \p MF 's
/// application launch budget, i.e. the wave owns at least part of \p Reg and
/// its contents are application-visible.
bool isAppOwnedGPR(const llvm::MachineFunction &MF, llvm::MCRegister Reg);

/// \return \c true if \p Reg is reserved for a purpose instrumentation must
/// respect. Unlike \c MachineRegisterInfo::isReserved this does *not* report
/// GPRs that are merely above the function's \c amdgpu-num-* allocation cap:
/// those are the registers scavenging exists to find.
bool isReservedForApp(const llvm::MachineFunction &MF, llvm::MCRegister Reg);

/// \return \c true if instrumentation may claim \p Reg in \p MF -- i.e. it
/// belongs to an allocatable class, is not reserved for a special purpose by
/// the subtarget, and no application code in \p MF touches it.
///
/// Register-scavenging target-module passes should use this instead of
/// spelling out \c MachineRegisterInfo::isAllocatable / \c isPhysRegUsed , so
/// that the definition of "free" lives in one place.
bool isAvailableForInstrumentation(const llvm::MachineFunction &MF,
                                   llvm::MCRegister Reg);

} // namespace luthier

#endif
