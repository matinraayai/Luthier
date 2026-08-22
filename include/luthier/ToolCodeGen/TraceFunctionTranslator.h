//===-- TraceFunctionTranslator.h -------------------------------*- C++ -*-===//
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
/// \file TraceFunctionTranslator.h
/// Describes the \c TraceFunctionTranslator class which is used to translate
/// discovered machine function traces by the \c CodeDiscoveryPass to LLVM IR.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TRACE_FUNCTION_TRANSLATOR_H
#define LUTHIER_TOOL_CODE_GEN_TRACE_FUNCTION_TRANSLATOR_H
#include "luthier/Common/DenseMapInfo.h"
#include "luthier/ToolCodeGen/MIInlineAsmEmitter.h"
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/ValueHandle.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>

namespace llvm {

class SIRegisterInfo;
class MachineInstr;
class MachineFunction;
class MachineBasicBlock;
class TargetRegisterInfo;
class MachineOperand;
class Error;

} // namespace llvm

namespace luthier {

class TraceFunctionTranslator;

/// Individual machine instruction to LLVM IR translation functions emitted
/// by the tablegen backend. Each of these function is specialized on
/// \p MI's opcode. \c TraceFunctionTranslator::raiseMachineInstr is in charge
/// of dispatching the correct specialization
template <uint16_t Opcode>
void raiseMachineInstr(const llvm::MachineInstr &MI,
                       llvm::IRBuilderBase &Builder,
                       TraceFunctionTranslator &Translator);

/// \brief Translates machine instructions inside trace functions to LLVM IR and
/// inserts the translated IR into the associated LLVM Function handle.
/// The translated IR can then be used to query value and semantics-related
/// information about the instructions inside the target module
/// \details The translator performs a per-basic block translation of the
/// machine function. Each basic block maintains a mapping between registers
/// and a list of available LLVM values hashed based on their type. The map
/// gets updated after translating each instruction in the basic block
/// from the basic block's beginning. The entry basic block's initial values
/// are initialized based on the calling convention of the function; Kernels'
/// initial values are populated according to the AMD GPU kernel initial
/// state described by the AMDGPU backend documentation; Device (non-entry)
/// function's entry basic blocks obtain their initial values from their
/// arguments. Other basic blocks emit unresolved PHI nodes for any registers
/// that have been used for the first time by an instruction. After all
/// basic blocks have been translated, the PHI nodes will be then be resolved
/// by connecting them to their incoming values in their predecessor blocks if
/// they already exist, or making additional PHIs to request them from///
/// predecessors further away. If making additional PHI nodes results in
/// requesting an uninitialized value from the entry basic block, a
/// \c freeze(poision) value is emitted to serve as a place holder.
/// This process is performed until all unresolved PHI nodes have been
/// resolved. \n
/// Instead of directly using MC register enums to map registers to their
/// associated set of values, This class instead uses the following three
/// things to distinguish each register entry: 1. The base physical MC register
/// of a "register file", 2. The base hardware register index offset in units of
/// half-words (16-bits), and 3. The size of the register in half-words
/// (16-bits). This was chosen instead of utilizing the register's MC reg enum
/// as a key, to allow for assigning values to arbitrarily-sized regions inside
/// the register file arbitrarily-sized registers inside the value map,
/// (which includes the entire register file itself). The hardware
/// register index (opcode encoding) of each register can be queried from
/// both LLVM's AMDGPU register records (see \c llvm::AMDGPU::getMCReg and
/// \c llvm::SIRegisterInfo::getEncodingValue) and the AMD GPU public ISA docs
/// (see section named "Scalar ALU Operands").
/// Possible register files include:
/// - <b>User SGPR register file</b>, with \c SGPR0 as its base index.
/// This file covers the entire number of SGPRs the kernel descriptor requests
/// in its \c RSRC1 field via the `amdgpu-num-sgpr` function attribute.
/// Special registers \c VCC, and on GFX9 and earlier targets,
/// \c FLAT_SCR and \c XNACK_MASK are placed at the end of this file, mimicing
/// documented hardware behavior. This implies that these special registers
/// can be addressed with both their special name (e.g. \c VCC) and their
/// logical name (e.g. \c SGPR16), also mimicing how the hardware works.
/// The translator's hash function detects if a logical SGPR encoding was used
/// in a machine instruction to refer to where these special registers and
/// aliases them to the same entry. The aliasing behavior has been removed in
/// GFX10+, but \c VCC still remains in the same \c SGPR file.
/// The maximum size of this region at the time of writing is 108 DWORDs,
/// ending right after \c VCC_HI
/// - <b>The Trap handler SGPR register file</b>, which starts from the
/// first encoded trap handler register located right after \c VCC_HI
/// (index 108) in all targets. On older targets, the base register of this
/// file is \c TBA_LO and on newer targets, \c TTMP0. This region includes all
/// trap handler registers. This region usually ends right before index 124.
/// - <b>\c EXEC mask, \c MO, and \c SGPR_NULL (if available on the target)
/// register file</b>, starts from the last trap handler index (124) to
/// the end of the \c EXEC_HI's encoding, which is 128 (exclusive).
/// - <b>Aperature register file</b>, which includes \c SRC_SHARED_BASE,
/// \c SRC_SHARED_LIMIT, \c SRC_PRIVATE_BASE, \c SRC_PRIVATE_LIMIT, and \c
/// (on GFX9 and GFX10) \c SRC_POPS_EXITING_WAVE_ID. On GFX8 and earlier
/// targets, these registers are not available, hence this register file
/// is empty on those targets. The hardware encoding range of this file is from
/// 235 to 239, inclusive.
/// - <b>Status register SGPR file</b>, which includes \c SRC_VCCZ, \c
/// SRC_EXECZ, and \c SRC_SCC, ranging from hardware index of 251 to 253 on
/// all targets, inclusive. These are modelled as 32-bit SGPR registers in the
/// AMDGPU backend, and the translator honors that in its register files.
/// All reads/writes for the pseudo register \c SCC will be resolved into
/// \c SRC_SCC.
/// - <b>Hardware register file</b>, with the \c MODE register as its base,
/// is meant to model the value of the hardware registers that can be
/// read/written to via the \c S_GETREG_B32 and \c S_SETREG_B32 instructions.
/// As of right now, only the \c MODE register is supported, as it is the
/// only register with an MC enum in the AMD GPU backen. Additional hardware
/// register will be added in the future as conversion of \c S_GETREG_B32
/// and \c S_SETREG_B32 instructions to simple read/writes to hardware
/// register values is implemented.
/// - <b>VGPR register file</b>, with \c VGPR0 as its base register. The number
/// of VGPRs is determined by the `amdgpu-num-vgpr` function attribute. If
/// it is determined that the function is using the dynamic VGPR feature
/// introduced in GFX12+ (how to detect this is TBD), maximum number of
/// addressable VGPRs for the sub target is assumed.
/// - <b>AGPR register file</b>, with \c AGPR0 as its base register. On targets
/// that support it, its size should be the same as the number of VGPRs
/// requested by the kernel. \n
/// The register files' formats are deduced primarily by looking at certain
/// function attributes populated by the \c CodeDiscoveryPass as well as the
/// features of the function's sub target. \n
/// Following Luthier's machine function trace semantics, transfer of control
/// from one trace function to another results in a call instruction that pass
/// the entire register file of the caller to the callee as argument. The
/// prototype of the callee can be queried via
/// \c TraceFunctionTranslator::getStandardDeviceFunctionType function.
/// When passing device (i.e. non-entry) function to the translator the
/// \p CodeDiscoveryPass must ensure the function's prototype matches the
/// type returned by \c getStandardDeviceFunctionType, since the function's
/// prototype cannot be changed later by the translator. Kernel and other
/// entry functions don't have to follow this rule, as their initial/// register
/// values are initialized in the body of the function according to the initial
/// kernel state documented by the AMDGPU LLVM backend; For example, \c
/// llvm.amdgcn.kernarg.segment.ptr() is used to initialize the SGPR pair's
/// value holding the kernel argument buffer's address. \n The translator also
/// takes care of materializing register values as the instruction semantics
/// translator functions (i.e. \c raiseMachineInstr) requests them. The
/// translator materializes sub-registers of register values via bitcasting the
/// available register values into vector types and using the \c
/// `extractelement` IR instruction to retrieve the requested sub-register. The
/// same process is also done for when a super register of a requested register
/// already has a value materialized in the value map. Registers that overlap
/// but are not strictly super or sub-registers (rare) will be broken down to
/// reg units and then processed via the super reg logic.
/// \n
/// On a write to a register, all overlapping entries (sub- and super-regs)
/// are invalidated so that subsequent reads through a different alias will
/// lazily recompute the value from the register's entry. If a register being
/// written only partially overlaps with a tracked value, only the
/// overlapped part is invalidated. \n
/// As of right now, Luthier does not model read/write privilege of registers.
/// Luthier also does not currently model VGPR indexing operand access
/// used in GFX9.
/// \TODO The initial basic block in the machine function must not have any
/// predecessors. The \c CodeDiscoveryPass is in charge of detecting this
/// issue and fixing it up in the discovered trace before passing the
/// machine function for translation.
/// \TODO The \c TraceFunctionTranslator was designed to exepect
/// well-formed Luthier trace machine functions; In other words, during
/// translation, it doesn't return a \c llvm::Error that can be later checked,
/// and only has assertions for sanity checks for performance reasons and
/// readabilty; Which is why the \p CodeDiscoveryPass if prompted runs the
/// machine verifier on the trace machine function before passing it
/// down to the translator.
class TraceFunctionTranslator {
  template <uint16_t Opcode>
  friend void luthier::raiseMachineInstr(const llvm::MachineInstr &MI,
                                         llvm::IRBuilderBase &Builder,
                                         TraceFunctionTranslator &Translator);

  /// Primary interface for raising machine instructions to LLVM IR in the
  /// translator loop. If a semantic for the \p MI's opcode exists, the
  /// appropriate specialization of raiseMachineInstr is dispatched; If not,
  /// an inline assembly version of the instruction the appropriate number of
  /// input and output operands are emitted
  void raiseMachineInstr(const llvm::MachineInstr &MI,
                         llvm::IRBuilderBase &Builder);

  /// Translates all instructions of \p MBB into its BodyBB and closes the
  /// block with a fall-through branch or \c unreachable safety net. Shared
  /// between the initial \c translate pass and \c retranslateMBB
  void translateMBBBody(llvm::MachineBasicBlock &MBB);

  llvm::MachineFunction &MF;

  /// IR Inline assembly instruction emitter used to emit place holder
  /// inline assembly IR instructions for machine instructions with no
  /// semantics
  std::unique_ptr<MIInlineAsmEmitter> InlineAsmEmitter{};

  const llvm::SIRegisterInfo &TRI;

  const llvm::SIInstrInfo &TII;

  const llvm::GCNSubtarget &ST;

  /// We keep track of the same physical register's value per its available
  /// type inside each basic block; For example, if the translation requires a
  /// register value of i32 to be cast to a f32, we cache both values. This way,
  /// when a later instruction requests the f32 version, we don't emit a
  /// redundant cast instruction. Allowed types are ints, FP, and pointer types.
  /// Pointer types don't have to check for size compatibility, as they will
  /// use the \c llvm::IntToPtrInst which takes care of truncating/extending
  /// the output pointer size
  using ValueTypeMap = llvm::DenseMap<llvm::Type *, llvm::Value *>;

  /// Register file key used to hash register values:
  /// - First element: the base register of the register file
  /// - Second element: the hardware index offset of the register
  /// from the base (in half-word units)
  /// - Third element: the size of the register (in half-word units)
  using RegFileKey = std::tuple<llvm::MCRegister, unsigned, unsigned>;

  /// Register size granularity; Remains 16 until something changes in AMD
  /// GPUs
  static constexpr unsigned RegGranule = 16;

  /// Cache for registers in a basic block
  using RegValueMap = llvm::DenseMap<RegFileKey, ValueTypeMap>;

  using BBStateMap = llvm::DenseMap<const llvm::BasicBlock *, RegValueMap>;

  /// Total 16-bit-lane footprint of each register file.
  llvm::SmallDenseMap<llvm::MCRegister, unsigned> RegFileSize{};

  /// Indicates the index where each register file is placed in
  /// the argument list in case of a function call
  llvm::SmallVector<llvm::MCRegister> FunctionCallArgOrder{};

  /// Logical SGPR that aliases \c VCC_LO for applicable targets
  llvm::MCRegister VccLoSgpr{llvm::MCRegister::NoRegister};

  /// Logical SGPR that aliases \c XNACK_MASK_LO on targets where xnack is
  /// supported
  llvm::MCRegister XnackMaskLoSgpr{llvm::MCRegister::NoRegister};

  /// Logical SGPR that aliases \c FLAT_SCR_LO on targets that store
  /// \c FLAT_SRC before the VCC with the rest of the normal SGPRs
  llvm::MCRegister FlatScrLoSgpr{llvm::MCRegister::NoRegister};

  /// Marks the start of the TTMP0 to EXEC HI register file
  llvm::MCRegister TTMPBaseReg = {llvm::AMDGPU::TTMP0};

  unsigned ExecBaseReg = 0;

  /// Encoding offset (in 16-bit lanes) of the lane right after the EXEC_HI
  /// register is encoded in the register file.
  unsigned ExecHiHalfWordOffset = 0;

  /// Encoding offset (in 16-bit lanes) of where the current subtarget encodes
  /// the first status register in the SGPR file (i.e., VCCZ)
  unsigned SrcVCCZHalfWordOffset = 0;

  /// SGPR half-word index where the status registers are encoded in the
  /// SGPR file; This usually comes after the aperture registers on GFX9+
  /// targets
  unsigned SGPRStatusRegHWordOffsetStart = 0;

  struct ToBeFixedRegValuePhiInfo {
    const llvm::BasicBlock *BB;
    RegFileKey RegKey;
    llvm::PHINode *Phi;
  };

  /// Register value placeholder PHIs that need to be fixed up after the
  /// function has been translated
  llvm::SmallVector<ToBeFixedRegValuePhiInfo> ToBeFixedPhis{};

  /// Per-basic block register file value mapping; This is updated every
  /// time a single register/file is read/written to
  BBStateMap VM{};

  /// For each vector MBB, the synthetic IR "check" basic block inserted
  /// between the MBB's IR predecessors and its BodyBB. The check BB holds
  /// the EXEC mask PHI, computes the per-lane active predicate, and branches
  /// to either the BodyBB (lane active) or the synthetic skip block (lane
  /// inactive). Membership in this map indicates the MBB is a vector MBB
  llvm::DenseMap<const llvm::MachineBasicBlock *, llvm::BasicBlock *>
      VectorCheckBBs{};

  /// All synthetic IR basic blocks (CheckBB + SkipBB) emitted as part of
  /// the EXEC mask predicate scaffolding. Populated by Pass 3 and left in
  /// place so downstream analyses can still discover the scaffold shape
  /// via CFG inspection; \c optimizeNonTraceInsts no longer treats these
  /// blocks specially, so their instructions are subject to normal DCE
  /// like any non-trace IR
  llvm::SmallPtrSet<const llvm::BasicBlock *, 8> ExecScaffoldBBs{};

  /// Per-BB \c WeakTrackingVH shadow of the tracked register-file state
  /// as it exits each LLVM \c BasicBlock. Entries are populated at two
  /// distinct moments:
  ///   - Primary body BB per MBB: populated in \c snapshotBBExitStates
  ///     from \c VM after Pass 3 finishes but before \c fixupPhis runs.
  ///   - Synthetic \c CheckBB per vector MBB: populated inline in
  ///     \c emitExecPredicateCheck from the placeholder PHIs hoisted
  ///     into that CheckBB.
  ///   - Synthetic \c SkipBB: intentionally never populated; SkipBB gets
  ///     no exit-reg-map metadata.
  /// \c WeakTrackingVH follows RAUW (single-value PHIs collapsed by
  /// \c fixupPhis end up pointing at their surviving incoming value) and
  /// nulls on erase (so DCE by \c optimizeNonTraceInsts leaves the
  /// shadow safe to dereference during post-translation emission — and
  /// dropped entries are what tells the emitter to omit slices whose
  /// SSA source was optimized away)
  llvm::DenseMap<
      const llvm::BasicBlock *,
      llvm::DenseMap<
          RegFileKey,
          llvm::SmallVector<std::pair<llvm::Type *, llvm::WeakTrackingVH>, 2>>>
      ExitStateShadow{};

  /// Build the EXEC-mask per-lane active predicate in \p CheckBB and emit a
  /// conditional branch to either \p BodyBB (active lane) or \p SkipBB
  /// (inactive lane). The CheckBB receives a placeholder EXEC PHI that is
  /// resolved by \c fixupPhis from \p VectorMBB's MIR predecessors
  void emitExecPredicateCheck(llvm::BasicBlock *CheckBB,
                              llvm::BasicBlock *BodyBB,
                              llvm::BasicBlock *SkipBB);

  /// After translation is complete, run a worklist-driven
  /// \c llvm::simplifyInstruction + trivial-dead removal sweep over all
  /// non-trace instructions in the translated function. Trace instructions
  /// are preserved as-is
  void optimizeNonTraceInsts();

  /// For every synthetic vector-MBB \c CheckBB, if the EXEC value
  /// dominating its per-lane predicate is provably all-ones (a
  /// \c ConstantInt with all bits set, or a \c PHINode whose incoming
  /// values are all such constants), replace the conditional branch
  /// with an unconditional branch to the body BB. This trivially makes
  /// the entire mbcnt/lshr/and/trunc chain dead, so
  /// \c optimizeNonTraceInsts strips it. Runs after \c foldHwregIntrinsics
  /// (so EXEC PHIs have their incoming values populated) and before
  /// \c optimizeNonTraceInsts (which then cleans up the resulting dead
  /// IR). Also erases the now-unreachable \c SkipBB if safe.
  void foldTriviallyActiveExecChecks();

  /// Copy \c VM into \c ExitStateShadow for the primary body BB of each
  /// MBB, converting each raw \c Value* into a \c WeakTrackingVH so
  /// subsequent RAUW / erasure by \c fixupPhis, \c foldHwregIntrinsics,
  /// and \c optimizeNonTraceInsts is followed automatically. Called
  /// after Pass 3 completes but before \c fixupPhis. Does not touch
  /// scaffold BB entries — those are populated inline by
  /// \c emitExecPredicateCheck.
  void snapshotBBExitStates();

  /// Attach \c luthier.bb_exit_reg_map to the translated function using
  /// \c ExitStateShadow. Iterates \c F's BBs in order and emits one
  /// tuple per BB that has a shadow entry: primary body BBs (from
  /// \c snapshotBBExitStates) and scaffold CheckBBs (from
  /// \c emitExecPredicateCheck). Scaffold SkipBBs, having no shadow
  /// entry, contribute no tuple. Called as the final step of
  /// \c translate() after \c optimizeNonTraceInsts.
  void emitBBExitRegMapMetadata();

  /// True iff a VALU MI's named-register access should be wrapped with
  /// the conditional indexed-VGPR-access expansion: subtarget supports
  /// \c FeatureVGPRIndexMode (GFX9 family), MI is a VALU instruction,
  /// and \p Reg is a VGPR. SGPR / AGPR / special-register operands and
  /// non-VALU instructions take the direct path.
  bool shouldEmitGPRIndexAccess(const llvm::MachineInstr &MI,
                                llvm::MCRegister Reg) const;

  /// Emit IR for an indexed VGPR read: returns the value of
  /// \c VGPR[Reg + (gpr_idx_en ? M0[7:0] : 0)] as \p OutType.
  /// `gpr_idx_en` is the bit-27 field of MODE; M0[7:0] is the index.
  /// When MODE.GPR_IDX_EN folds to 0 at the call site (e.g. after the
  /// kernel-entry MODE constant flows through \c foldHwregIntrinsics),
  /// the indexed branch becomes dead and the optimizer collapses the
  /// select to the direct read.
  llvm::Value &emitIndexedVGPRSrc(const llvm::MachineInstr &MI,
                                  llvm::MCRegister Reg, llvm::Type *OutType);

  /// Emit IR for an indexed VGPR write:
  /// \c VGPR[Reg + (gpr_idx_en ? M0[7:0] : 0)] = Val. Implemented by
  /// reading the register-file slice starting at \p Reg, inserting
  /// \p Val at the conditional index, and writing the slice back. The
  /// slice scope means the write touches the register-tracking state
  /// for every VGPR in the slice; the read+insert+write trio preserves
  /// values for all but the targeted index.
  void emitIndexedVGPRDst(const llvm::MachineInstr &MI, llvm::MCRegister Reg,
                          llvm::Value *Val);

  /// Rewrite each call to \c @llvm.amdgcn.s.getreg.i32 or
  /// \c @llvm.amdgcn.s.setreg.i32 whose hwreg encoding is a compile-time
  /// constant naming a register the translator tracks (currently only
  /// \c AMDGPU::MODE) into a direct read/write of that register's
  /// tracked SSA value with explicit bitfield extract/insert logic.
  ///
  /// Run before \c optimizeNonTraceInsts so the constant kernel-entry
  /// MODE value (assembled by \c buildInitialModeValue) flows through
  /// the bitfield ops and lets \c simplifyInstruction fold them. The
  /// replacement instructions inherit the original call's pcsections
  /// metadata so trace vs. non-trace status is preserved.
  ///
  /// Calls whose hwreg ID does not map to a tracked register (STATUS,
  /// TRAPSTS, HW_ID, etc. — none of these have AMDGPU register enum
  /// entries today) are left as opaque intrinsic calls.
  void foldHwregIntrinsics();

  /// Maps the physical pseudo register to its hardware register
  /// \note Use this instead of \c llvm::AMDGPU::getMCReg directly
  llvm::MCRegister getPhysReg(llvm::MCRegister Reg) const;

  /// Provides the register's physical size
  /// \note Use this instead of \c llvm::TargetRegisterInfo::getRegSizeInBits
  unsigned getPhysRegisterSize(llvm::MCRegister Reg) const;

  /// Given the base register of the register file, returns the value name
  /// for that register file; Primarily used for creating value entries
  /// covering the entirety of the register file
  std::string getRegfileValueName(llvm::MCRegister BaseReg);

  /// Used to get the final name of the value associated with a physical
  /// register
  std::string getRegValueName(llvm::MCRegister Reg) const {
    return llvm::StringRef(TRI.getName(Reg)).lower() + "_val";
  }

  /// Resolve \p Reg to a \c RegFileKey (register base index + 16-bit-lane
  /// offset + number of 16-bit halves). Performs GFX9- alias translation for
  /// <tt>VCC</tt>/<tt>XNACK_MASK</tt>/<tt>FLAT_SCR</tt> before encoding the
  /// final key
  RegFileKey getRegFileKey(llvm::MCRegister Reg) const;

  /// Initializes fields related to the layout of the register files based
  /// on the number of SGPRs/VGPRs and other fields of the function
  /// being translated
  llvm::Error initRegFileLayouts();

  /// Initializes the initial values in the entry basic block if it's
  /// determined that the function is a kernel
  void initKernelEntryRegs(llvm::IRBuilderBase &Builder);

  /// Initializes the initial values in the entry basic block if it's
  /// determined that the function is not a kernel
  void initDeviceFunctionEntryRegs(llvm::IRBuilderBase &Builder);

  /// Retrieves the named operand \p OpName in \p MI, retrieves it as a
  /// \c llvm::MachineBasicBlock. It then returns the MBB's \c llvm::BasicBlock
  /// Primarily used with branch instructions
  llvm::BasicBlock &getOperandAsBasicBlock(const llvm::MachineInstr &MI,
                                           llvm::AMDGPU::OpName OpName);

  /// Retrieves the \c llvm::MachineBasicBlock stored in the \p Op, and
  /// returns its associated \c llvm::BasicBlock
  llvm::BasicBlock &getOperandAsBasicBlock(const llvm::MachineOperand &Op);

  /// Retrieves the named operand \p OpName in \p MI as an \c llvm::Function.
  /// The operand must be of kind \c MO_GlobalAddress and its global value must
  /// be a Function — typically set by \c CodeDiscoveryPass for direct-call
  /// instructions (S_CALL_B64) after the callgraph resolves the target.
  /// Returns nullptr if the operand is not a Function reference.
  llvm::Function *getOperandAsFunction(const llvm::MachineInstr &MI,
                                       llvm::AMDGPU::OpName OpName);

  /// Returns the fall-through BasicBlock (next block after the current MI's
  /// block)
  llvm::BasicBlock *getNextBB(const llvm::MachineInstr &MI);

  /// Decode the cache-policy immediate value into a SyncScope::ID.
  /// \p CPolVal must be an integer constant carrying the cpol bits as encoded
  /// by AMDGPU FLAT/MUBUF/MTBUF/SMEM instructions. The bit layout varies by
  /// subtarget:
  ///   - gfx7-gfx11 (pre-CDNA3, pre-gfx12): bit 0=GLC, bit 1=SLC, bit 2=DLC.
  ///     SLC=1 ⇒ system; GLC=1 SLC=0 ⇒ agent; GLC=0 ⇒ workgroup.
  ///   - CDNA3 (gfx940/942/950): bits[1:0]=SC1:SC0 (4-level scope):
  ///     00 wavefront, 01 workgroup, 10 agent, 11 system.
  ///   - gfx12 (RDNA4): bits[4:3]=scope (4-level), bits[2:0]=th.
  /// Falls back to SyncScope::System if the value isn't a ConstantInt.
  llvm::SyncScope::ID getSyncScope(const llvm::Value *CPolVal) const;

  /// Decode the cache-policy immediate value into an AtomicOrdering.
  /// AMDGPU atomic instructions are monotonic at the HW level: acquire /
  /// release / sequential consistency are produced by SIMemoryLegalizer
  /// surrounding the atomic with barrier instructions, not by the atomic
  /// itself. Always returns Monotonic for now; the cpol value is reserved
  /// for future per-bit ordering refinement.
  llvm::AtomicOrdering getOrdering(const llvm::Value *CPolVal) const;

  /// Retrieves and translates the named operand \p OpName to a value and caches
  /// it for the \p MI's basic block
  /// If \p OutType is given, appropriately converts the returned
  /// value to match the <tt>OutType</tt>; Otherwise, the returned type will
  /// have an integer type.
  /// \note \p OutType's scalar type must be either integer, floating point,
  /// or pointer
  /// \note The named operand represented by \p OpName must be either a
  /// register or an integer
  llvm::Value &getOperandAsValue(const llvm::MachineInstr &MI,
                                 llvm::AMDGPU::OpName OpName,
                                 llvm::Type *OutType = nullptr);

  /// Translates the machine operand \p Op of type register or immediate to a
  /// value and caches it for the the basic block of the \p Op's instruction
  /// \see getOperandAsValue
  llvm::Value &getOperandAsValue(const llvm::MachineOperand &Op,
                                 llvm::Type *OutType = nullptr);

  /// Convenience method for obtaining the basic block of \p MI first
  /// before asking for the associated value with \p Reg using
  /// \c getOperandAsValue
  /// \see getOperandAsValue
  llvm::Value &getOperandAsValue(const llvm::MachineInstr &MI,
                                 llvm::MCRegister Reg,
                                 llvm::Type *RegType = nullptr);

  /// Materialize the value of \p Reg in <tt>BB</tt>'s register value map;
  /// The value will be materialized at the end of the IR basic block, before
  /// the first terminator instruction
  /// If \p OutRegType is given, appropriately converts the returned
  /// value to match the <tt>OutRegType</tt>; Otherwise, the returned type will
  /// have an integer type.
  /// \note \p OutType's scalar type must be either integer, floating point,
  /// or pointer
  /// \note the scalar type of \p OutRegType must either be an int, float,
  /// or a pointer
  llvm::Value &getOperandAsValue(const llvm::BasicBlock &BB,
                                 llvm::MCRegister Reg,
                                 llvm::Type *OutRegType = nullptr);

  /// Materializes the value of the register identified by \p Key in the
  /// <tt>BB</tt>'s register value map. The value will be materialized using
  /// the \p Builder
  /// \see getOperandAsValue
  llvm::Value &getOperandAsValue(const llvm::BasicBlock &BB,
                                 const RegFileKey &Key,
                                 llvm::IRBuilderBase &Builder,
                                 llvm::Type *OutRegType = nullptr);

  /// Materializes the value identified by \p KeyReg by looking at all
  /// overlapping entries in \p State and optimally composing the value from
  /// available defined registers. If a sub-value is not defined, emits a PHI
  /// when \p BB has IR predecessors, otherwise emits a \c freeze(poison)
  /// instruction in its place
  llvm::Value *materializeFromOverlapping(RegValueMap &State,
                                          const llvm::BasicBlock &BB,
                                          const RegFileKey &ReadKeyReg,
                                          llvm::IRBuilderBase &Builder,
                                          llvm::Type &RegType);

  /// Helper: Extract a chunk from a source entry's ValueTypeMap
  /// \param State the type to value map to materialize the value from
  /// \param RegKey
  /// \param VecChunkSize the size of scalar type to break the initial value
  /// to
  /// \param Idx Index to extract the final element from the value
  /// \param NumChunks Number of elements to extract from the value
  /// \param Builder the \c llvm::IRBuilderBase to use to emit any intermediate
  /// instructions for materializing the value
  /// \return the extracted value, as a scalar (the chunks will be merged
  /// together and bitcasted to a scalar value)
  llvm::Value *extractChunkFromSource(RegValueMap &State,
                                      const RegFileKey &RegKey,
                                      unsigned VecChunkSize, unsigned Idx,
                                      unsigned NumChunks,
                                      llvm::IRBuilderBase &Builder);

  /// Retrieves the register associated with the named destination operand
  /// \p OpName in \p MI and sets its associated value in the register value
  /// map of \p MI's basic block to \p Val
  void setRegOperandValue(const llvm::MachineInstr &MI,
                          llvm::AMDGPU::OpName OpName, llvm::Value *Val);

  /// Sets the register's associated value in \p MI's basic block register
  /// value map to \p Val
  void setRegOperandValue(const llvm::MachineInstr &MI, llvm::MCRegister Reg,
                          llvm::Value *Val);

  /// Sets the destination operand's associated value in \p MI's basic block
  /// register value map to \p Val
  void setRegOperandValue(const llvm::MachineOperand &Op, llvm::Value *Val);

  /// Sets the value associated with the \p Key in the register value map
  /// associated with \p BB to \p Val
  /// \p Builder is used for invalidating stale values overlapping with
  /// the \p Key
  void setRegOperandValue(const llvm::BasicBlock &BB,
                          const RegFileKey &Key, llvm::IRBuilderBase &Builder,
                          llvm::Value *Val);

  /// Invalidates register value entries overlapping with \p RegKey before
  /// a write happens by \c setRegOperandValue
  ///
  /// For each stored entry whose \c [Offset, Offset+NumHalves) range overlaps:
  ///  - Stored ⊂ Reg (fully covered): erase.
  ///  - Reg ⊂ Stored (partial overwrite of super-reg): bitcast to
  ///    \c <NumHalves x i16>, extract the non-overlapping halves, and
  ///    preserve them as new sub-register entries.
  ///  - Stored = Reg (exact match): skip, since the \c setRegOperandValue
  ///  is going to overwrite the entry anyway
  void invalidateOverlaps(RegValueMap &State, const RegFileKey &WrittenRegKey,
                          llvm::IRBuilderBase &Builder);

  /// Goes over the unresolved PHIs generated when materializing values during
  /// the instruction translation period, and fixes them up by looking up the
  /// PHI's value in its predecessor blocks
  /// \note If a value is not found in a predecessor block, another PHI is
  /// generated at the beginning of the predecessor block, and the generated
  /// PHI is then processed in the same loop
  /// \note If a PHI value does not have a predecessor block, it will be
  /// changed to \c freeze(poision)
  /// \note After PHIs are processed, if it is determined that the PHI node only
  /// has a single predecessor, it is changed to a direct value use
  void fixupPhis();

  /// Materializes the value associated with the register file of \p Reg
  /// in the value register map of \p BB
  /// \p Builder is used to materialize needed instructions
  /// \p LaneTy specifies the scalar type used to divide the register file  ///
  /// value in the returned vector type
  llvm::Value *getRegisterFile(const llvm::BasicBlock &BB,
                               llvm::MCRegister Reg,
                               llvm::IRBuilderBase &Builder,
                               llvm::Type *LaneTy = nullptr);

  /// Materializes the value associated with the register file of \p Register
  /// at \p MI's translation point
  /// \p LaneTy specifies the scalar type used in the returned register
  /// file vector
  llvm::Value *getRegisterFile(const llvm::MachineInstr &MI,
                               llvm::MCRegister Register,
                               llvm::Type *LaneTy = nullptr);

  /// Same as \c getRegisterFile but takes the register from \p MI's named
  /// operand \p OpName. Used by the \c GetRegisterFile DSL primitive to
  /// index into the register file starting at a base named operand.
  llvm::Value *getRegisterFile(const llvm::MachineInstr &MI,
                               llvm::AMDGPU::OpName OpName,
                               llvm::Type *LaneTy = nullptr);

  /// Sets the entire register file associated with \p Reg at \p MI's
  /// translation point to the new value \p NewVec
  void setRegisterFile(const llvm::MachineInstr &MI, llvm::MCRegister Reg,
                       llvm::Value *NewVec);

  /// Same as \c setRegisterFile but takes the register from \p MI's named
  /// operand \p OpName.
  void setRegisterFile(const llvm::MachineInstr &MI,
                       llvm::AMDGPU::OpName OpName, llvm::Value *NewVec);

  /// Sets the entire register file associated with \p Reg in \p BB's
  /// register value map to \p Val
  /// \p Builder is used to create any intermediate instructions
  /// needed for the write
  void setRegisterFile(const llvm::BasicBlock &BB, llvm::MCRegister Reg,
                       llvm::IRBuilderBase &Builder, llvm::Value *Val);

  /// Emits a direct tail call. \p InstAddr is the instruction's PC-value
  /// (used to compute the absolute call target when \p Target is still a raw
  /// displacement). \p Target is either an \c llvm::Function* (post-callgraph
  /// fixup) or a \c ConstantInt displacement (pre-fixup); the implementation
  /// dispatches on the runtime type. \p Builder is the per-MI IRBuilder
  void emitDirectTailCall(const llvm::MachineInstr &MI,
                          llvm::IRBuilderBase &Builder, llvm::Value *InstAddr,
                          llvm::Value *Target);

  /// Like \c emitIndirectCall but the call is a tail call followed by
  /// \c ret of the call's return value. \p Builder is the per-MI IRBuilder
  void emitIndirectTailCall(const llvm::MachineInstr &MI,
                            llvm::IRBuilderBase &Builder, llvm::Value *Target);

public:
  TraceFunctionTranslator(llvm::MachineFunction &MF, llvm::Error &Err);

  /// Provides the function type used for trace device functions; Useful to
  /// know when creating new device function for translation in the same
  /// module
  llvm::FunctionType *getStandardDeviceFunctionType() const;

  /// Stateless version of \c getStandardDeviceFunctionType that builds the
  /// canonical device-function \c FunctionType used for trace functions.
  /// Can be used to obtain the estimated prototype for a trace function before
  /// discovering and populating it with machine instructions.
  /// Returns an \c llvm::Error if \p NumSGPRs or \p NumVGPRs is zero.
  static llvm::Expected<llvm::FunctionType *>
  computeStandardDeviceFunctionType(llvm::LLVMContext &Ctx,
                                    const llvm::GCNSubtarget &ST,
                                    unsigned NumSGPRs, unsigned NumVGPRs);

  /// Main function for performing the IR translation
  void translate();

  /// Re-translates the body of \p MBB in place after its MIR was modified.
  /// The MBB's BodyBB identity (and, for vector MBBs, its Check/Skip
  /// scaffolding) is preserved; only the body instructions are regenerated.
  /// Cross-block dataflow is repaired by re-materializing every register
  /// value the old body exported and replacing its external uses.
  /// \pre \p MBB has been translated (its \c getBasicBlock is non-null) and
  /// belongs to the translated function; the MBB's CFG edges are unchanged.
  /// Call \c runPostTranslateCleanup after the last re-translation.
  /// \returns true if an old value escaped the block without a register-file
  /// key, in which case in-place repair is impossible: the caller must do a
  /// full re-translation and afterwards delete the orphaned instructions
  /// parked in \p PendingDeadInsts
  llvm::Expected<bool>
  retranslateMBB(const llvm::MachineBasicBlock &MBB,
                 llvm::SmallVectorImpl<llvm::Instruction *> &PendingDeadInsts);

  /// Re-runs the post-translation cleanup (simplification and dead non-trace
  /// instruction removal) after one or more \c retranslateMBB calls
  void runPostTranslateCleanup() { optimizeNonTraceInsts(); }

  /// \returns true if \p MBB's IR terminator targets are consistent with its
  /// current MIR successor list (a vector successor is reached through its
  /// CheckBB). False indicates the MIR CFG changed shape since translation,
  /// which is beyond \c retranslateMBB's in-place repair
  bool irSuccessorsMatchMIR(const llvm::MachineBasicBlock &MBB) const;
};

} // namespace luthier

#endif