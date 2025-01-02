//===-- ISA.hpp - HSA ISA Wrapper -----------------------------------------===//
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
/// This file defines the \c ISA class under the \c luthier::hsa
/// namespace, which is a wrapper around the \c hsa_isa_t HSA opaque handle
/// type.
//===----------------------------------------------------------------------===//
#ifndef HSA_ISA_HPP
#define HSA_ISA_HPP
#include "HandleType.hpp"
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

namespace luthier::hsa {

/// \brief wrapper around \c hsa_isa_t
/// \details Besides implementing typical HSA functionality, this wrapper also
/// implements methods for interoperability between \c hsa_isa_t and LLVM
/// ISA names
class ISA : public HandleType<hsa_isa_t> {
public:
  /// Constructor
  /// \param Isa the \c hsa_isa_t handle being encapsulated
  explicit ISA(hsa_isa_t Isa) : HandleType<hsa_isa_t>(Isa){};

  /// Queries the \c ISA handle associated with the \p FullIsaName
  /// \param FullIsaName the full name of the ISA
  /// \note HSA for the most part follows LLVM's convention for naming ISAs,
  /// with the only minor difference being the location of +/- coming after
  /// subtarget features, not after. For more details, refer to <tt>isa.cpp</tt>
  /// source file in the ROCr runtime project
  /// \returns on success, the \c hsa::ISA handle of the queried ISA; On failure,
  /// a \c luthier::HsaError indicating issues encountered by HSA
  static llvm::Expected<ISA> fromName(llvm::StringRef FullIsaName);

  /// Queries the \c ISA handle associated with the given <tt>TT</tt>,
  /// <tt>CPU</tt>, and <tt>Features</tt> from LLVM
  /// \param TT the \c llvm::Triple of the ISA
  /// \param GPUName the name of the GPU
  /// \param Features the subtarget features of the target; (e.g. <tt>xnack</tt>,
  /// <tt>sramecc</tt>)
  /// \note refer to the LLVM AMDGPU backend documentation for more details
  /// on the supported ISA and their names
  /// \returns on success, the \c hsa::ISA handle of the queried ISA; On failure,
  /// a \c luthier::HsaError indicating issues encountered by HSA
  static llvm::Expected<ISA> fromLLVM(const llvm::Triple &TT,
                                      llvm::StringRef GPUName,
                                      const llvm::SubtargetFeatures &Features);

  /// \returns the full name of the ISA on success, a \c luthier::HsaError
  /// if an HSA error was encountered
  [[nodiscard]] llvm::Expected<std::string> getName() const;

  /// \returns the architecture field of the LLVM target triple of the ISA, or a
  /// \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] llvm::Expected<std::string> getArchitecture() const;

  /// \returns the vendor field of the LLVM target triple of the ISA, or a
  /// \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] llvm::Expected<std::string> getVendor() const;

  /// \returns the OS field of the LLVM target triple of the ISA, or
  /// a \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] llvm::Expected<std::string> getOS() const;

  /// \returns the environment field of the LLVM target triple of the ISA,
  /// or a \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] llvm::Expected<std::string> getEnvironment() const;

  /// \returns the name of the GPU processor on success,
  /// or a \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] llvm::Expected<std::string> getGPUName() const;

  /// \returns on success, returns true if the ISA enables XNACK, false
  /// otherwise; On failure, a \c luthier::HsaError if any HSA error
  /// was encountered
  [[nodiscard]] llvm::Expected<bool> isXNACKSupported() const;

  /// \returns on success, returns true if the ISA enables SRAMECC, false
  /// otherwise; On failure, a \c luthier::HsaError if any HSA error
  /// was encountered
  [[nodiscard]] llvm::Expected<bool> isSRAMECCSupported() const;

  /// \returns the entire \c llvm::Triple of the ISA on success, or an
  /// \c llvm::Error if any issues are encountered in the process
  [[nodiscard]] llvm::Expected<llvm::Triple> getTargetTriple() const;

  /// \returns the LLVM subtraget feature string of the ISA on success,
  /// or an \c llvm::HsaError if an HSA issue was encountered in
  /// the process
  [[nodiscard]] llvm::Expected<llvm::SubtargetFeatures>
  getSubTargetFeatures() const;
};

} // namespace luthier::hsa


//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::ISA> {
  static inline luthier::hsa::ISA getEmptyKey() {
    return luthier::hsa::ISA(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::ISA getTombstoneKey() {
    return luthier::hsa::ISA(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::ISA &ISA) {
    return DenseMapInfo<decltype(hsa_isa_t::handle)>::getHashValue(
        ISA.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::ISA &Lhs,
                      const luthier::hsa::ISA &Rhs) {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace llvm


//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//


namespace std {

template <> struct hash<luthier::hsa::ISA> {
  size_t operator()(const luthier::hsa::ISA &Obj) const {
    return hash<unsigned long>()(Obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() < Rhs.hsaHandle();
  }
};

template <> struct less_equal<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() <= Rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

template <> struct not_equal_to<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() != Rhs.hsaHandle();
  }
};

template <> struct greater<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() > Rhs.hsaHandle();
  }
};

template <> struct greater_equal<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() >= Rhs.hsaHandle();
  }
};

} // namespace std

#endif