//===-- HandleType.hpp - Wrapper Around HSA Handle Types ------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// This file defines the \c HandleType class under the \c luthier::hsa
/// namespace, which represents a wrapper around handle type objects defined and
/// used by the HSA library.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HANDLE_TYPE_HPP
#define LUTHIER_HSA_HANDLE_TYPE_HPP
#include "Type.hpp"

namespace luthier::hsa {

/// \brief A subclass of <tt>hsa::Type</tt>, used to encapsulate opaque
/// handle type object used by the HSA runtime
/// \details An opaque handle type in HSA has a following definition:
/// \code
/// typedef struct hsa_my_handle_type_s {
///   uint64_t handle;
/// }
/// \endcode
/// the \c handle field generally points to the memory location of the C++
/// version of the handle on the HSA side.
/// \tparam HT the handle type being encapsulated
template <typename HT> class HandleType : public Type<HT> {
protected:
  /// Direct constructor from the handle type being encapsulated
  /// \param Primitive the HSA object being encapsulated; If \p Primitive
  /// requires explicit initialization using an HSA API, then the \p Primitive
  /// must already be initialized and valid
  explicit HandleType(HT Primitive) : Type<HT>(Primitive) {};

public:
  /// \returns the handle field of the wrapped opaque handle type
  [[nodiscard]] decltype(HT::handle) hsaHandle() const {
    return this->asHsaType().handle;
  }

  /// copy constructor
  HandleType(const HandleType &Type) : HandleType(Type.asHsaType()) {};

  /// copy assignment constructor
  HandleType &operator=(const HandleType &Other) {
    Type<HT>::operator=(Other);
    return *this;
  }

  /// move constructor
  HandleType &operator=(HandleType &&Other) noexcept {
    Type<HT>::operator=(Other);
    return *this;
  }
};

} // namespace luthier::hsa

#define DECLARE_LLVM_MAP_INFO_STRUCTS_FOR_HANDLE_TYPE(HandleStruct, HT)        \
  namespace llvm {                                                             \
  template <> struct DenseMapInfo<HandleStruct *> {                            \
    static inline HandleStruct *getEmptyKey() {                                \
      return reinterpret_cast<HandleStruct *>(                                 \
          DenseMapInfo<decltype(HT::handle)>::getEmptyKey());                  \
    }                                                                          \
    static inline HandleStruct *getTombstoneKey() {                            \
      return reinterpret_cast<HandleStruct *>(                                 \
          DenseMapInfo<decltype(HT::handle)>::getTombstoneKey());              \
    }                                                                          \
    static unsigned getHashValue(const HandleStruct *A) {                      \
      if (A)                                                                   \
        return DenseMapInfo<decltype(hsa_agent_t::handle)>::getHashValue(      \
            A->hsaHandle());                                                   \
      else                                                                     \
        return DenseMapInfo<decltype(hsa_agent_t::handle)>::getHashValue(0);   \
    }                                                                          \
    static bool isEqual(const HandleStruct *Lhs, const HandleStruct *Rhs) {    \
      if (Lhs && Rhs)                                                          \
        return Lhs->hsaHandle() == Rhs->hsaHandle();                           \
      else                                                                     \
        return false;                                                          \
    }                                                                          \
  };                                                                           \
  }

#endif