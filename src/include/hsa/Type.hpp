//===-- Type.hpp - HSA Type Wrapper ---------------------------------------===//
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
/// This file defines the \c Type class under the \c luthier::hsa namespace,
/// which represents a wrapper around a type defined and used by the HSA
/// library.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TYPE_HPP
#define LUTHIER_HSA_TYPE_HPP
#include "HsaRuntimeInterceptor.hpp"

namespace luthier::hsa {
/// \brief a wrapper around data types (e.g. structs, handles) defined and
/// used by the HSA namespace
/// \details This class is used to encapsulate operations primarily related to a
/// specific HSA type in order to create an object-oriented programming
/// interface to the HSA library, and also to encapsulate calling the
/// \c HsaRuntimeInterceptor API table done many, many times
/// on the Luthier side to prevent interception of Luthier-related HSA work
/// \tparam HT type of the HSA struct being encapsulated
/// \note \c HT must be trivially copyable
/// \note Usage of these wrapper types depend on the
/// \c hsa::HsaRuntimeInterceptor to be initialized, and have captured
/// the API tables of the HSA runtime

template <typename HT> class Type {
protected:
  HT HsaType; ///< Should be trivially copyable

  /// Direct constructor from the object being encapsulated
  /// \param HsaType the HSA object being encapsulated; If \p HsaType requires
  /// explicit initialization using an HSA API, then the \p HsaType must already
  /// be initialized and valid
  explicit Type(HT HsaType) : HsaType(HsaType){};

  /// Convenience method to get a reference to the HSA API table saved by the
  /// \c hsa::HsaRuntimeInterceptor singleton
  /// \return the \c HsaApiTableContainer
  [[nodiscard]] inline const HsaApiTableContainer &getApiTable() const {
    return hsa::HsaRuntimeInterceptor::instance().getSavedApiTableContainer();
  }

  /// Convenience method to get a reference to the AMD vendor API table saved
  /// by the \c hsa::HsaRuntimeInterceptor singleton
  /// \return the \c hsa_ven_amd_loader_1_03_pfn_t
  [[nodiscard]] inline const hsa_ven_amd_loader_1_03_pfn_t &
  getLoaderTable() const {
    return hsa::HsaRuntimeInterceptor::instance().getHsaVenAmdLoaderTable();
  }

public:
  /// \return the encapsulated HSA type represented by this wrapper
  HT asHsaType() const { return HsaType; }

  /// copy constructor
  Type(const Type &Type) : HsaType(Type.asHsaType()){};

  /// copy assignment constructor
  Type &operator=(const Type &Other) {
    this->HsaType = Other.HsaType;
    return *this;
  }

  /// move constructor
  Type &operator=(Type &&Other) noexcept {
    this->HsaType = Other.HsaType;
    return *this;
  }
};

} // namespace luthier::hsa

#endif