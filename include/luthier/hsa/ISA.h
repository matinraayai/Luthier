//===-- ISA.h ---------------------------------------------------*- C++ -*-===//
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
/// Defines commonly used functionality around the \c hsa_isa_t type in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_ISA_H
#define LUTHIER_HSA_ISA_H
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/HsaError.h"
#include <hsa/hsa.h>
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

namespace luthier::hsa {
/// Queries the \c ISA handle associated with the \p FullIsaName
/// \param HsaISAFromNameFn The underlying \c hsa_isa_from_name function used
/// to carry out the operation
/// \param FullIsaName the full name of the ISA
/// \note HSA for the most part follows LLVM's convention for naming ISAs,
/// with the only minor difference being the location of +/- coming after
/// subtarget features, not after. For more details, refer to <tt>isa.cpp</tt>
/// source file in the ROCr runtime project
/// \returns \c the \c hsa_isa_t handle of the queried ISA on success,
/// \c llvm::Error on failure
inline llvm::Expected<hsa_isa_t>
isaFromName(const decltype(hsa_isa_from_name) &HsaISAFromNameFn,
            llvm::StringRef FullIsaName) {
  hsa_isa_t Isa;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaISAFromNameFn(FullIsaName.data(), &Isa)));
  return Isa;
}

/// Queries the \c ISA handle associated with the given <tt>TT</tt>,
/// <tt>CPU</tt>, and <tt>Features</tt> from LLVM
/// \param HsaISAFromNameFn The underlying \c hsa_isa_from_name function used
/// to carry out the operation
/// \param TT the \c llvm::Triple of the ISA
/// \param GPUName the name of the GPU
/// \param Features the subtarget features of the target; (e.g. <tt>xnack</tt>,
/// <tt>sramecc</tt>)
/// \note refer to the LLVM AMDGPU backend documentation for more details
/// on the supported ISA and their names
/// \returns \c the \c hsa_isa_t handle of the queried ISA on success,
/// \c llvm::Error on failure
llvm::Expected<hsa_isa_t>
isaFromLlvm(const decltype(hsa_isa_from_name) &HsaISAFromNameFn,
            const llvm::Triple &TT, llvm::StringRef GPUName,
            const llvm::SubtargetFeatures &Features);

/// \returns the full name of the ISA on success, an \c llvm::Error on failure
llvm::Expected<std::string>
getISAName(hsa_isa_t ISA,
           const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns the architecture field of the LLVM target triple of the ISA on
/// success, \c llvm::Error on failure
[[nodiscard]] llvm::Expected<std::string>
getISAArchitecture(hsa_isa_t ISA,
                   const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns the vendor field of the LLVM target triple of the \p ISA on
/// success, \c llvm::Error on failure
[[nodiscard]] llvm::Expected<std::string>
getISAVendor(hsa_isa_t ISA,
             const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns the OS field of the LLVM target triple of the \p ISA on success,
/// or \c llvm::Error on failure
[[nodiscard]] llvm::Expected<std::string>
getISAOperatingSystem(hsa_isa_t ISA,
                      const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns the environment field of the LLVM target triple of the \p ISA
/// on success, \c llvm::Error on failure
[[nodiscard]] llvm::Expected<std::string>
getISAEnvironment(hsa_isa_t ISA,
                  const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns the name of the GPU processor on success,
/// \c llvm::Error on failure
[[nodiscard]] llvm::Expected<std::string>
getISAGPUName(hsa_isa_t ISA,
              const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns on success, returns \c true if the ISA enables XNACK, \c false
/// otherwise; On failure, \c llvm::Error is returned
[[nodiscard]] llvm::Expected<bool>
doesISASupportXNACK(hsa_isa_t ISA,
                    const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns on success, returns \c true if the \p ISA has SRAMECC, \c false
/// otherwise; On failure, an \c llvm::Error
[[nodiscard]] llvm::Expected<bool>
doesISASupportSRAMECC(hsa_isa_t ISA,
                      const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns the \c llvm::Triple of the \p ISA on success,
/// \c llvm::Error on failure
[[nodiscard]] llvm::Expected<llvm::Triple>
getISATargetTriple(hsa_isa_t ISA,
                   const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

/// \returns the LLVM subtarget feature string of the \p ISA on success,
/// \c llvm::Error on failure
[[nodiscard]] llvm::Expected<llvm::SubtargetFeatures>
getISASubTargetFeatures(hsa_isa_t ISA,
                        const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn);

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<hsa_isa_t> {
  static inline hsa_isa_t getEmptyKey() {
    return hsa_isa_t(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getEmptyKey()});
  }

  static inline hsa_isa_t getTombstoneKey() {
    return hsa_isa_t(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const hsa_isa_t &ISA) {
    return DenseMapInfo<decltype(hsa_isa_t::handle)>::getHashValue(ISA.handle);
  }

  static bool isEqual(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_isa_t> {
  size_t operator()(const hsa_isa_t &Obj) const {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct less<hsa_isa_t> {
  bool operator()(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) const {
    return Lhs.handle < Rhs.handle;
  }
};

template <> struct less_equal<hsa_isa_t> {
  bool operator()(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) const {
    return Lhs.handle <= Rhs.handle;
  }
};

template <> struct equal_to<hsa_isa_t> {
  bool operator()(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

template <> struct not_equal_to<hsa_isa_t> {
  bool operator()(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) const {
    return Lhs.handle != Rhs.handle;
  }
};

template <> struct greater<hsa_isa_t> {
  bool operator()(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) const {
    return Lhs.handle > Rhs.handle;
  }
};

template <> struct greater_equal<hsa_isa_t> {
  bool operator()(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) const {
    return Lhs.handle >= Rhs.handle;
  }
};

} // namespace std

#endif