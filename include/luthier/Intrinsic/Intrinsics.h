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

/// All bindings to Luthier intrinsics must have this attribute
#define LUTHIER_INTRINSIC_ATTRIBUTE "luthier_intrinsic"


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

//TODO: Write docs for the bindings

template <typename T>
LUTHIER_INTRINSIC_ANNOTATE T readReg(llvm::MCRegister Reg) {
  T Out;
  doNotOptimize(Reg);
  doNotOptimize(Out);
  return Out;
}

template <typename T>
LUTHIER_INTRINSIC_ANNOTATE void writeReg(llvm::MCRegister Reg, T Val) {
  doNotOptimize(Reg);
  doNotOptimize(Val);
}

#endif