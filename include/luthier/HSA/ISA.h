//===-- ISA.h ---------------------------------------------------*- C++ -*-===//
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
/// \file ISA.h
/// Defines commonly used functionality around the \c hsa_isa_t type in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_ISA_H
#define LUTHIER_HSA_ISA_H
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/ApiTable.h"
#include "luthier/HSA/HsaError.h"
#include <array>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

namespace luthier::hsa {
/// Queries the \c ISA handle associated with the \p FullIsaName
/// \param CoreApi The \c ::CoreApiTable used to dispatch HSA  \c
/// hsa_isa_from_name function
/// \param FullIsaName the full name of the ISA
/// \note HSA for the most part follows LLVM's convention for naming ISAs,
/// with the only minor difference being the location of +/- coming after
/// subtarget features, not after. For more details, refer to <tt>isa.cpp</tt>
/// source file in the ROCr runtime project
/// \returns Expects the \c hsa_isa_t handle of the queried ISA
inline llvm::Expected<hsa_isa_t>
isaFromName(const ApiTableContainer<::CoreApiTable> &CoreApi,
            llvm::StringRef FullIsaName) {
  hsa_isa_t Isa;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_isa_from_name_fn>(
          FullIsaName.data(), &Isa),
      llvm::formatv("Failed to get the ISA {0} using its name from HSA.",
                    FullIsaName)));
  return Isa;
}

/// Queries the \c ISA handle associated with the given <tt>TT</tt>,
/// <tt>CPU</tt>, and <tt>Features</tt> from LLVM
/// \note refer to the LLVM AMDGPU backend documentation for more details
/// on the supported ISA and their names
/// \returns Expects \c hsa_isa_t handle of the queried ISA
llvm::Expected<hsa_isa_t>
isaFromLLVM(const ApiTableContainer<::CoreApiTable> &CoreApi,
            const llvm::Triple &TT, llvm::StringRef GPUName,
            const llvm::SubtargetFeatures &Features);

/// \returns Expects the full name of the ISA
llvm::Expected<std::string>
isaGetName(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA);

/// \returns Expects the architecture field of the LLVM target triple of the
/// \p ISA
[[nodiscard]] llvm::Expected<std::string>
isaGetArchitecture(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_isa_t ISA);

/// \returns Expects the vendor field of the LLVM target triple of the \p ISA
[[nodiscard]] llvm::Expected<std::string>
isaGetVendor(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA);

/// \returns Expects the OS field of the LLVM target triple of the \p ISA
[[nodiscard]] llvm::Expected<std::string>
isaGetOperatingSystem(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_isa_t ISA);

/// \returns the environment field of the LLVM target triple of the \p ISA
[[nodiscard]] llvm::Expected<std::string>
isaGetEnvironment(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_isa_t ISA);

/// \returns the name of the GPU processor of the \p ISA
[[nodiscard]] llvm::Expected<std::string>
isaGetGPUName(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA);

/// \returns Expects \c true if the \p ISA enables XNACK, \c false
/// otherwise
[[nodiscard]] llvm::Expected<bool>
isaGetXnackSupport(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_isa_t ISA);

/// \returns Expects \c true if the \p ISA has SRAM error correction, \c false
/// otherwise
[[nodiscard]] llvm::Expected<bool>
isaGetSramEcc(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA);

/// \returns Expects the \c llvm::Triple of the \p ISA
[[nodiscard]] llvm::Expected<llvm::Triple>
isaGetTargetTriple(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_isa_t ISA);

/// \returns Expects the LLVM subtarget feature string of the \p ISA
[[nodiscard]] llvm::Expected<llvm::SubtargetFeatures>
isaGetSubTargetFeatures(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        hsa_isa_t ISA);

/// Queries which machine models are supported by the \p ISA.
/// Returns a two-element array indexed by \c hsa_machine_model_t:
/// element 0 = SMALL, element 1 = LARGE.
/// \sa hsa_isa_get_info_alt, HSA_ISA_INFO_MACHINE_MODELS
[[nodiscard]] llvm::Expected<std::array<bool, 2>>
isaGetMachineModels(const ApiTableContainer<::CoreApiTable> &CoreApi,
                    hsa_isa_t ISA);

/// Queries which profiles are supported by the \p ISA.
/// Returns a two-element array indexed by \c hsa_profile_t:
/// element 0 = BASE, element 1 = FULL.
/// \sa hsa_isa_get_info_alt, HSA_ISA_INFO_PROFILES
[[nodiscard]] llvm::Expected<std::array<bool, 2>>
isaGetProfiles(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA);

/// Queries the supported default float rounding modes of the \p ISA.
/// Returns a three-element array indexed by
/// \c hsa_default_float_rounding_mode_t.
/// \sa hsa_isa_get_info_alt with HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES
[[nodiscard]] llvm::Expected<std::array<bool, 3>>
isaGetDefaultFloatRoundingModes(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA);

/// Queries whether the \p ISA supports fast half-precision operations.
/// \sa hsa_isa_get_info_alt with HSA_ISA_INFO_FAST_F16_OPERATION
[[nodiscard]] llvm::Expected<bool>
isaGetFastF16(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA);

/// Queries the maximum workgroup dimensions of the \p ISA.
/// Returns a 3-element array [x, y, z].
/// \sa hsa_isa_get_info_alt with HSA_ISA_INFO_WORKGROUP_MAX_DIM
[[nodiscard]] llvm::Expected<std::array<uint16_t, 3>>
isaGetWorkgroupMaxDim(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_isa_t ISA);

/// Queries the maximum total workgroup size of the \p ISA.
/// \sa hsa_isa_get_info_alt with HSA_ISA_INFO_WORKGROUP_MAX_SIZE
[[nodiscard]] llvm::Expected<uint32_t>
isaGetWorkgroupMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       hsa_isa_t ISA);

/// Queries the maximum grid dimensions of the \p ISA.
/// Returns an \c hsa_dim3_t.
/// \sa hsa_isa_get_info_alt with HSA_ISA_INFO_GRID_MAX_DIM
[[nodiscard]] llvm::Expected<hsa_dim3_t>
isaGetGridMaxDim(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_isa_t ISA);

/// Queries the maximum total grid size of the \p ISA.
/// \sa hsa_isa_get_info_alt with HSA_ISA_INFO_GRID_MAX_SIZE
[[nodiscard]] llvm::Expected<uint64_t>
isaGetGridMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_isa_t ISA);

/// Queries the maximum number of fbarriers per workgroup for the \p ISA.
/// \sa hsa_isa_get_info_alt with HSA_ISA_INFO_FBARRIER_MAX_SIZE
[[nodiscard]] llvm::Expected<uint32_t>
isaGetFbarrierMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_isa_t ISA);

/// Queries the exception policies for the given \p ISA and \p Profile.
/// Returns a bitmask of \c hsa_exception_policy_t.
/// \sa hsa_isa_get_exception_policies
[[nodiscard]] llvm::Expected<uint16_t>
isaGetExceptionPolicies(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        hsa_isa_t ISA, hsa_profile_t Profile);

/// Queries the round method for the given \p ISA, floating-point type,
/// and flush mode.
/// \sa hsa_isa_get_round_method
[[nodiscard]] llvm::Expected<hsa_round_method_t>
isaGetRoundMethod(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_isa_t ISA, hsa_fp_type_t FpType,
                  hsa_flush_mode_t FlushMode);

/// Iterates over the wavefronts supported by the \p ISA.
/// \sa hsa_isa_iterate_wavefronts
llvm::Error isaIterateWavefronts(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA,
    const std::function<llvm::Error(hsa_wavefront_t)> &Callback);

/// Queries all wavefronts supported by the \p ISA.
llvm::Error
isaGetWavefronts(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_isa_t ISA,
                 llvm::SmallVectorImpl<hsa_wavefront_t> &Wavefronts);

/// Queries the size (number of work-items) of the \p Wavefront.
/// \sa hsa_wavefront_get_info, HSA_WAVEFRONT_INFO_SIZE
[[nodiscard]] llvm::Expected<uint32_t>
wavefrontGetSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_wavefront_t Wavefront);

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_isa_t> {
  static hsa_isa_t getEmptyKey() {
    return hsa_isa_t(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getEmptyKey()});
  }

  static hsa_isa_t getTombstoneKey() {
    return hsa_isa_t(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const hsa_isa_t &ISA) {
    return DenseMapInfo<decltype(hsa_isa_t::handle)>::getHashValue(ISA.handle);
  }

  static bool isEqual(const hsa_isa_t &Lhs, const hsa_isa_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
}; // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_isa_t> {
  size_t operator()(const hsa_isa_t &Obj) const noexcept {
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