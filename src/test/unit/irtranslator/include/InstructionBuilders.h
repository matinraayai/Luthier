//===-- InstructionBuilders.h -----------------------------------*- C++ -*-===//
//
// Registry of per-instruction-class reference-kernel builders. Each class
// (VOP2/ALU, DS, and later SMEM/MUBUF/SCRATCH/...) provides a matcher over MC
// facts and a build function; MachineKernelBuilder::build() picks the first
// matching entry. Adding a class is one entry here plus one instructions/*.cpp.
//
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_INSTRUCTION_BUILDERS_H
#define LUTHIER_TEST_INSTRUCTION_BUILDERS_H

#include "MachineKernelBuilder.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/Support/Error.h>

namespace luthier::test {

/// VOP family (VOP1 / VOP2 / VOPC, no-modifier VALU): seed VGPR (and VCC/tied)
/// inputs, run the MI, capture register (and/or VCC mask) outputs. Defined in
/// instructions/VOP.cpp.
llvm::Expected<KernelMFContext> buildVOP(llvm::TargetMachine &TM,
                                         const InstrProfile &Profile,
                                         KernargLayout &Layout);

/// DS (LDS): seed M0/address, initialize LDS, run the MI, read LDS back, and
/// capture register + LDS results. Defined in instructions/DS.cpp.
llvm::Expected<KernelMFContext> buildDS(llvm::TargetMachine &TM,
                                        const InstrProfile &Profile,
                                        KernargLayout &Layout);

/// Scalar ALU (SOP1 / SOP2 / SOPK / SOPC): scalar (SGPR) inputs and output,
/// SCC as the scalar condition register (seeded and captured like VCC for
/// VALU), and SOPK's 16-bit inline literal. Defined in instructions/SOP.cpp.
llvm::Expected<KernelMFContext> buildSOP(llvm::TargetMachine &TM,
                                         const InstrProfile &Profile,
                                         KernargLayout &Layout);

/// FLAT family (GLOBAL / FLAT / SCRATCH): a host-visible global buffer stands
/// in for device memory; the kernel initializes it, runs the load/store/atomic
/// under test, and reads it back into the output buffer. Currently the GLOBAL
/// `_SADDR` forms (saddr base pointer + voffset). Defined in instructions/FLAT.cpp.
llvm::Expected<KernelMFContext> buildFLAT(llvm::TargetMachine &TM,
                                          const InstrProfile &Profile,
                                          KernargLayout &Layout);

/// SMEM (scalar memory) loads: S_LOAD (64-bit base pointer) and S_BUFFER_LOAD
/// (128-bit V# resource descriptor built in-kernel from the data-buffer
/// pointer), all widths (DWORD..DWORDX16) and offset forms (IMM/SGPR/SGPR_IMM).
/// A host-visible global buffer stands in for memory; the kernel initializes it,
/// invalidates the scalar cache, runs the load under test, and captures the
/// sdst SGPR tuple. Defined in instructions/SMEM.cpp.
llvm::Expected<KernelMFContext> buildSMEM(llvm::TargetMachine &TM,
                                          const InstrProfile &Profile,
                                          KernargLayout &Layout);

/// MUBUF (untyped buffer): OFFSET-form buffer load/store/atomic addressed
/// through a 128-bit V# resource descriptor (built in-kernel from the data
/// buffer pointer) + a scalar offset. DWORD / wide / sub-dword. Defined in
/// instructions/MUBUF.cpp.
llvm::Expected<KernelMFContext> buildMUBUF(llvm::TargetMachine &TM,
                                           const InstrProfile &Profile,
                                           KernargLayout &Layout);

/// SCRATCH (flat-scratch) family: private-segment load/store/atomic. The kernel
/// enables the flat-scratch ABI (flat_scratch_init + wave offset), initializes a
/// scratch slot via a helper SCRATCH_STORE, runs the instruction under test, and
/// reads the slot back into the output buffer. Defined in instructions/Scratch.cpp.
llvm::Expected<KernelMFContext> buildScratch(llvm::TargetMachine &TM,
                                             const InstrProfile &Profile,
                                             KernargLayout &Layout);

/// A pluggable instruction-class builder.
struct InstrClassBuilder {
  llvm::StringRef Name;
  /// Does this class handle \p Desc / \p Profile?
  bool (*Matches)(const llvm::MCInstrDesc &Desc, const InstrProfile &Profile);
  /// Build the reference kernel for it.
  llvm::Expected<KernelMFContext> (*Build)(llvm::TargetMachine &TM,
                                           const InstrProfile &Profile,
                                           KernargLayout &Layout);
};

/// The ordered builder table (first match wins).
llvm::ArrayRef<InstrClassBuilder> getInstrClassBuilders();

} // namespace luthier::test

#endif // LUTHIER_TEST_INSTRUCTION_BUILDERS_H
