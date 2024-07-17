//===-- llvm_dense_map_info.h - DenseMapInfo Implementation  -----*- C++-*-===//
//
//===----------------------------------------------------------------------===//
///
/// \file Includes the <tt>DenseMapInfo</tt> for primitives used by Luthier,
/// so that they can be stored in LLVM-based maps.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LLVM_DENSE_MAP_INFO
#define LUTHIER_LLVM_DENSE_MAP_INFO
#include <hsa/hsa.h>
#include <llvm/ADT/DenseMapInfo.h>

namespace llvm {

#define HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(T)                                 \
  template <> struct DenseMapInfo<T> {                                         \
    static inline T getEmptyKey() {                                            \
      return (T){DenseMapInfo<decltype(T::handle)>::getEmptyKey()};            \
    }                                                                          \
                                                                               \
    static inline T getTombstoneKey() {                                        \
      return (T){DenseMapInfo<decltype(T::handle)>::getTombstoneKey()};        \
    }                                                                          \
                                                                               \
    static unsigned getHashValue(const T &P) {                                 \
      return DenseMapInfo<decltype(T::handle)>::getHashValue(P.handle);        \
    }                                                                          \
    static bool isEqual(const T &LHS, const T &RHS) {                          \
      return LHS.handle == RHS.handle;                                         \
    }                                                                          \
  };

HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(hsa_executable_t);
HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(hsa_loaded_code_object_t);
HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(hsa_executable_symbol_t);

#undef HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE
} // namespace llvm

#endif