//===-- Intrinsics.h - Luthier's built-in Intrinsics ------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes utilities to write device code bindings to Luthier
/// intrinsics, as well as a set of bindings to Luthier's built-in intrinsics.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSICS_H
#define LUTHIER_INTRINSIC_INTRINSICS_H
#include <llvm/MC/MCRegister.h>

namespace luthier {

/// All bindings to Luthier intrinsics must be annotated using this macro; It
/// defines the binding as a device function, adds a noinline attribute as
/// well as a \c LUTHIER_INTRINSIC_ATTRIBUTE attribute
#define LUTHIER_INTRINSIC_ANNOTATE                                             \
  __attribute__((device, noinline, annotate("luthier_intrinsic")))

#if defined(__HIPCC__)

/// \brief Macro to use to prevent the compiler from optimizing a code region
/// away
/// \details This macro places an empty volatile inline assembly with "memory"
/// side-effects to prevent the compiler from dead-code eliminating a basic
/// block
#define LUTHIER_DONT_OPTIMIZE __asm__ __volatile__("" : : : "memory");

/// \brief Macro to use on all values involved in an intrinsic device
/// binding, to prevent their elimination from the binding prototype by the
/// compiler
/// \details This macro places an empty volatile inline assembly with
/// the arbitrary constraint on the passed <tt>Value</t>>, which will prevent
/// the compiler from optimizing it away
/// \note These operations will not show up in the IR of a Luthier device
/// module, as the body of Luthier intrinsic bindings are removed at the end of
/// the LLVM IR pipeline
/// \param Value the L-value to prevent optimization on
template <typename T>
__attribute__((device, always_inline)) void doNotOptimize(T const &Value) {
  __asm__ __volatile__("" : : "X"(Value) : "memory");
}

/// \brief Intrinsic to read the value of a register
/// \details The readReg intrinsic reads the value of the \p Reg and returns it
/// For now, VGPRs, SGPRs, and the EXEC mask are supported;
/// \tparam T the return type of the output; Must be of integral type and be
/// compatible with the size of \p Reg; For example reading \c
/// llvm::AMDGPU::SGPR4_SGPR5 must return a <tt>uint64_t</tt>
/// \param Reg the ID of the register to be read; It will be removed during
/// the IR processing stage from the IR; Must be a constant value,
/// and the register must be at most 64-bit wide
/// \returns the value of the read register
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
LUTHIER_INTRINSIC_ANNOTATE T readReg(llvm::MCRegister Reg) {
  T Out;
  doNotOptimize(Reg);
  doNotOptimize(Out);
  return Out;
}

/// \brief Intrinsic to write the value of a register
/// \details The writeReg intrinsic writes \p Val into the register named \p Reg
/// For now, VGPRs, SGPRs, and the EXEC mask are supported;
/// \tparam T the type of value to be written output; Must be of integral type
/// and be compatible with the size of \p Reg; For example writing to
// \c llvm::AMDGPU::SGPR4_SGPR5 requires a <tt>uint64_t</tt> \p Val
/// \param Reg the ID of the register to be read; It will be removed during
/// the IR processing stage from the IR; Must be a constant value,
/// and the register must be at most 64-bit wide
/// \param Val the value to write into the register
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
LUTHIER_INTRINSIC_ANNOTATE void writeReg(llvm::MCRegister Reg, T Val) {
  doNotOptimize(Reg);
  doNotOptimize(Val);
}

#endif

} // namespace luthier
#endif