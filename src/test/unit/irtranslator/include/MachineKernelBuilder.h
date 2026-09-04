//===-- MachineKernelBuilder.h ----------------------------------*- C++ -*-===//
//
// Reference path of the instruction semantic fuzzer.
//
// Builds the instruction under test as a real MachineInstr, wraps it in an
// AMDGPU kernel MachineFunction (seed inputs -> run the MI -> capture outputs),
// and emits that MachineFunction to a loadable code object.
//
// The prolog/epilog are not per-opcode: they decompose into a small closed set
// of per-operand "binding steps" (RegBinding) that are dialect-neutral (no MIR
// opcodes), so the same step list can later drive the IR path. buildBindings()
// turns an InstrProfile into ordered Prolog/Epilog step lists; the MIR emitter
// in the .cpp renders each step.
//
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_MACHINE_KERNEL_BUILDER_H
#define LUTHIER_TEST_MACHINE_KERNEL_BUILDER_H

#include "InstrDescriptor.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luthier::test {

//===----------------------------------------------------------------------===//
// Shared kernarg / output-buffer layout (single-sourced here, consumed
// unchanged by the host driver and, later, the IR path).
//===----------------------------------------------------------------------===//
struct KernargLayout {
  uint32_t TotalSize = 0;       ///< Total kernarg buffer size in bytes.
  uint32_t OutputPtrOffset = 0; ///< Offset of the output buffer pointer.

  struct Field {
    uint32_t Offset;
    uint32_t SizeBytes;
    std::string Name;
  };
  std::vector<Field> Inputs;  ///< Input value fields in the kernarg buffer.

  uint32_t OutputBufSize = 0;
  struct OutputField {
    uint32_t Offset;
    uint32_t SizeBytes;
    std::string Name;
  };
  std::vector<OutputField> Outputs; ///< Result fields in the output buffer.

  //=== Dispatch geometry / memory segments ============================//
  uint32_t GridSizeX = 1;         ///< Total work-items (full wave for DS).
  uint32_t WorkgroupSizeX = 1;
  uint32_t GroupSegmentSize = 0;  ///< LDS bytes for the AQL packet (DS).
  uint32_t PrivateSegmentSize = 0;///< Scratch bytes for the AQL packet.

  //=== Host-visible global scratch buffer (FLAT/GLOBAL) ================//
  /// If set, the host allocates a \c DataBufSizeBytes fine-grained buffer and
  /// writes its 64-bit device pointer at kernarg[DataBufPtrOffset]. The kernel
  /// initializes and reads it back into the output buffer itself, so no host
  /// pre-fill or post-compare is required.
  uint32_t DataBufPtrOffset = UINT32_MAX;
  uint32_t DataBufSizeBytes = 0;
};

//===----------------------------------------------------------------------===//
// Dialect-neutral binding steps.
//
// These describe WHAT to seed/capture around the instruction under test, not
// HOW (no MIR opcodes). The MIR emitter maps each Role (+Special) to concrete
// instructions; an IR emitter could map the same list to loads/stores/values.
//===----------------------------------------------------------------------===//

/// A hardware special register a Set/CaptureSpecial step touches.
enum class SpecialReg { None, VCC, EXEC, M0, MODE, SCC };

struct RegBinding {
  enum RoleKind {
    LoadInputSGPR,  ///< P1: kernarg dword(s) -> SGPR operand.
    LoadInputVGPR,  ///< P2: kernarg dword -> VGPR operand (broadcast to lanes).
    SetSpecial,     ///< P4: seed VCC/EXEC/M0/MODE/SCC.
    SeedTiedDef,    ///< P5: seed a def register the MI also reads (accumulate).
    StoreOutput,    ///< E1/E2: operand register -> output buffer.
    CaptureSpecial, ///< E3: VCC/MODE/SCC -> output buffer.
    MemReadback,    ///< E5: LDS words -> output buffer (DS memory-effect ops).
  } Role;

  llvm::MCRegister Reg;         ///< Operand/def reg (Set/Capture: unused).
  unsigned Dwords = 1;          ///< Register width in dwords.
  bool IsVGPR = false;
  bool PerLane = false;         ///< P3/E4: per-lane addressing (DS/global).
  uint32_t KernargOffset = 0;   ///< Source field (Load/Set-from-kernarg/Seed).
  uint32_t OutputOffset = 0;    ///< Destination field (Store/Capture/Readback).

  SpecialReg Special = SpecialReg::None; ///< For Set/CaptureSpecial.
  bool FromKernarg = false;     ///< SetSpecial: seed from kernarg vs fixed.
  uint64_t SpecialValue = 0;    ///< SetSpecial fixed value / MemReadback bytes.

  std::string Note;             ///< Human-readable description for the log.
};

/// One resolved operand of the instruction under test.
struct OperandBinding {
  enum Kind { DefReg, UseReg, TiedUse, Imm } Kind;
  llvm::MCRegister Reg;   ///< DefReg/UseReg/TiedUse.
  int64_t ImmValue = 0;   ///< Imm.
  unsigned MIOperandIdx;  ///< Slot in the MI operand list.
};

/// The VOP2 subgroup an opcode falls into (auditable classification).
enum class VOP2Subgroup {
  NotVOP2,
  Sg1_DstSrc01,     ///< DST0, SRC0, SRC1 only.
  Sg2_Literal,      ///< + an encoded simm32 literal (madak/fmaak/madmk/fmamk).
  Sg2_Cndmask,      ///< + implicit VCC use, no VCC def (v_cndmask_b32).
  Sg3_CarryInOut,   ///< implicit VCC use AND def (v_addc/v_subb).
  CarryOutOnly,     ///< implicit VCC def only (v_add_co_u32).
  Tied,             ///< src2 tied to dst (v_mac_f32/v_fmac_f32).
};

const char *toString(VOP2Subgroup S);

/// The DS (LDS) shape an opcode falls into (auditable classification).
enum class DSKind {
  NotDS,
  Load,        ///< LDS[addr] -> vdst (DS_1A_RET, e.g. ds_read_b32).
  Store,       ///< data0 -> LDS[addr] (write members).
  AtomicNoRet, ///< LDS[addr] op= data0, no return (DS_1A1D_NORET atomics).
  AtomicRet,   ///< vdst = LDS[addr]; LDS[addr] op= data0 (DS_1A1D_RET).
  Paired,      ///< read2/write2/cmpst/mskor (2 data or 2 offsets) -- TODO.
  Permute,     ///< ds_permute/bpermute/swizzle: cross-lane, no LDS -- TODO.
  Unsupported, ///< gws/append/consume/ordered/addtid: not data-testable.
};

const char *toString(DSKind K);

//===----------------------------------------------------------------------===//
// Kernel MachineFunction context.
//===----------------------------------------------------------------------===//
struct KernelMFContext {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Mod;
  std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP;

  llvm::MachineFunction *MF = nullptr;
  llvm::MachineInstr *MI = nullptr; ///< The single instruction under test.

  llvm::SmallVector<OperandBinding, 6> Operands;
  llvm::SmallVector<RegBinding, 8> Prolog;
  llvm::SmallVector<RegBinding, 8> Epilog;

  std::string KernelName;
  std::string BindingLog; ///< opcode -> subgroup + binding list (auditable).
};

//===----------------------------------------------------------------------===//
// Translation-path diagnostic artifacts.
//===----------------------------------------------------------------------===//

/// Textual snapshots taken while the reference kernel is raised back to LLVM
/// IR, so a failing opcode can be inspected after the fact.
///
/// These are captured eagerly rather than on failure because the two paths are
/// compared only after both kernels have run, and a translated kernel that
/// faults on the device aborts the whole process from inside the HSA runtime —
/// there is no point after the dispatch at which a report could still be
/// produced. See DiagnosticDump.h.
struct TranslationArtifacts {
  /// The reference kernel MachineFunction, printed before it is raised. This
  /// is the "MI side" of the comparison: the exact machine instructions the
  /// translator was asked to model.
  std::string ReferenceMIR;
  /// The reference kernel's IR shell as it looks before translate() replaces
  /// the body (signature, attributes, and the kernarg ABI both paths share).
  std::string ReferenceIR;
  /// The lifted IR function, after the identity llvm.ssa.copy calls are
  /// stripped — i.e. exactly what gets handed to codegen.
  std::string RaisedIR;
  /// Facts about the reference MachineFunction that do not survive the round
  /// trip through IR (frame objects, ABI-shaping attributes). A translated
  /// kernel is rebuilt from IR alone, so anything recorded here that the raised
  /// IR does not encode is silently lost.
  std::string ReferenceFrameFacts;
};

//===----------------------------------------------------------------------===//
// Builder.
//===----------------------------------------------------------------------===//
class MachineKernelBuilder {
public:
  explicit MachineKernelBuilder(llvm::TargetMachine &TM) : TM(TM) {}

  /// Build a reference kernel MachineFunction for \p Profile and compute the
  /// shared \p Layout. Emits the binding-step analysis into the returned
  /// context (Prolog/Epilog/BindingLog).
  llvm::Expected<KernelMFContext> build(const InstrProfile &Profile,
                                        KernargLayout &Layout);

  /// Emit \p KCtx's MachineFunction to an in-memory loadable code object.
  /// Consumes KCtx.MMIWP, so call at most once per context.
  llvm::Expected<llvm::SmallVector<char, 0>> emitToELF(KernelMFContext &KCtx);

  /// Translation path: raise \p KCtx's MachineFunction back to LLVM IR via
  /// luthier::TraceFunctionTranslator (which replaces the kernel's IR body with the
  /// raised semantics), strip the identity register-tracking copies, then
  /// codegen the module from IR to a loadable code object. The result is a
  /// kernel with the same kernarg layout as the reference, so both paths can be
  /// dispatched with identical inputs and compared. Uses a fresh MachineModule
  /// pipeline (does not consume KCtx.MMIWP), so build a distinct context for it.
  ///
  /// When \p Artifacts is non-null, the reference MIR and the raised IR are
  /// captured into it as text for diagnostics.
  llvm::Expected<llvm::SmallVector<char, 0>>
  emitTranslatedToELF(KernelMFContext &KCtx,
                      TranslationArtifacts *Artifacts = nullptr);

  static std::string getKernelName(const InstrProfile &Profile) {
    return ("test_ref_" + Profile.Name).str();
  }

private:
  llvm::TargetMachine &TM;
};

} // namespace luthier::test

#endif // LUTHIER_TEST_MACHINE_KERNEL_BUILDER_H
