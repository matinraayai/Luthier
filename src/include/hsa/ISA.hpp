//===-- ISA.hpp -----------------------------------------------------------===//
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
/// This file defines the \c hsa::ISA interface, a wrapper around the
/// \c hsa_isa_t HSA opaque handle type.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_ISA_HPP
#define LUTHIER_HSA_ISA_HPP
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
  explicit ISA(hsa_isa_t Isa) : HandleType<hsa_isa_t>(Isa) {};

  virtual ~ISA() = default;

  /// Queries the \c ISA handle associated with the \p FullIsaName and
  /// assigns it to the current object's handle
  /// \param FullIsaName the full name of the ISA
  /// \note HSA for the most part follows LLVM's convention for naming ISAs,
  /// with the only minor difference being the location of +/- coming after
  /// subtarget features, not after. For more details, refer to <tt>isa.cpp</tt>
  /// source file in the ROCr runtime project
  /// \returns a \c luthier::HsaError indicating issues encountered by HSA
  virtual llvm::Error
  fromName(llvm::StringRef FullIsaName) = 0;

  /// Queries the \c ISA handle associated with the given <tt>TT</tt>,
  /// <tt>CPU</tt>, and <tt>Features</tt> from LLVM, and assigns its handle
  /// to the current object
  /// \param TT the \c llvm::Triple of the ISA
  /// \param GPUName the name of the GPU
  /// \param Features the subtarget features of the target; (e.g.
  /// <tt>xnack</tt>, <tt>sramecc</tt>)
  /// \note refer to the LLVM AMDGPU backend documentation for more details
  /// on the supported ISA and their names
  /// \returns a \c luthier::HsaError indicating issues encountered by HSA
  virtual llvm::Error
  fromLLVM(const llvm::Triple &TT, llvm::StringRef GPUName,
           const llvm::SubtargetFeatures &Features) = 0;

  /// \returns the full name of the ISA on success, a \c luthier::HsaError
  /// if an HSA error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getName() const = 0;

  /// \returns the architecture field of the LLVM target triple of the ISA, or a
  /// \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getArchitecture() const = 0;

  /// \returns the vendor field of the LLVM target triple of the ISA, or a
  /// \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getVendor() const = 0;

  /// \returns the OS field of the LLVM target triple of the ISA, or
  /// a \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getOS() const = 0;

  /// \returns the environment field of the LLVM target triple of the ISA,
  /// or a \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getEnvironment() const = 0;

  /// \returns the name of the GPU processor on success,
  /// or a \c luthier::HsaError if an HSA error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getGPUName() const = 0;

  /// \returns on success, returns true if the ISA enables XNACK, false
  /// otherwise; On failure, a \c luthier::HsaError if any HSA error
  /// was encountered
  [[nodiscard]] virtual llvm::Expected<bool> isXNACKSupported() const = 0;

  /// \returns on success, returns true if the ISA enables SRAMECC, false
  /// otherwise; On failure, a \c luthier::HsaError if any HSA error
  /// was encountered
  [[nodiscard]] virtual llvm::Expected<bool> isSRAMECCSupported() const = 0;

  /// \returns the entire \c llvm::Triple of the ISA on success, or an
  /// \c llvm::Error if any issues are encountered in the process
  [[nodiscard]] virtual llvm::Expected<llvm::Triple>
  getTargetTriple() const = 0;

  /// \returns the LLVM subtraget feature string of the ISA on success,
  /// or an \c llvm::HsaError if an HSA issue was encountered in
  /// the process
  [[nodiscard]] virtual llvm::Expected<llvm::SubtargetFeatures>
  getSubTargetFeatures() const = 0;
};

} // namespace luthier::hsa

DECLARE_LLVM_MAP_INFO_STRUCTS_FOR_HANDLE_TYPE(luthier::hsa::ISA, hsa_isa_t)

#endif