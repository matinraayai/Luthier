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
/// This file defines the \c hsa::ISA interface, representing an ISA in the
/// HSA runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_ISA_HPP
#define LUTHIER_HSA_ISA_HPP
#include <llvm/Support/Error.h>
#include <llvm/Support/ExtensibleRTTI.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

namespace luthier::hsa {

/// \brief wrapper around HSA ISAs
/// \details Besides defining typical HSA functionality, this wrapper also
/// defines methods for interoperability between HSA ISA and LLVM
/// ISA names
class ISA : public llvm::RTTIExtends<ISA, llvm::RTTIRoot> {
public:
  static char ID;
  /// Queries the \c ISA handle associated with the \p FullIsaName and
  /// assigns it to this \c ISA object
  /// \param FullIsaName the full name of the ISA
  /// \note HSA for the most part follows LLVM's convention for naming ISAs,
  /// with the only minor difference being the location of +/- coming after
  /// subtarget features, not after. For more details, refer to <tt>isa.cpp</tt>
  /// source file in the ROCr runtime project
  /// \returns an \c llvm::Error indicating the success or failure of the
  /// operation
  virtual llvm::Error fromName(llvm::StringRef FullIsaName) = 0;

  /// Queries the \c ISA handle associated with the given <tt>TT</tt>,
  /// <tt>CPU</tt>, and <tt>Features</tt> from LLVM and assigns it to this
  /// \c ISA object
  /// \param TT the \c llvm::Triple of the ISA
  /// \param GPUName the name of the GPU
  /// \param Features the subtarget features of the target; (e.g.
  /// <tt>xnack</tt>, <tt>sramecc</tt>)
  /// \note refer to the LLVM AMDGPU backend documentation for more details
  /// on the supported ISA and their names
  /// \returns an \c llvm::Error indicating the success or failure of the
  /// operation
  virtual llvm::Error fromLLVM(const llvm::Triple &TT, llvm::StringRef GPUName,
                               const llvm::SubtargetFeatures &Features) = 0;

  /// \returns the full name of the ISA on success, an \c llvm::Error
  /// if an error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getName() const = 0;

  /// \returns the architecture field of the LLVM target triple of the ISA, or a
  /// \c llvm::Error if an error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getArchitecture() const = 0;

  /// \returns the vendor field of the LLVM target triple of the ISA, or a
  /// \c llvm::Error if an error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getVendor() const = 0;

  /// \returns the OS field of the LLVM target triple of the ISA, or
  /// a \c llvm::Error if an error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getOS() const = 0;

  /// \returns the environment field of the LLVM target triple of the ISA,
  /// or a \c llvm::Error if an error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getEnvironment() const = 0;

  /// \returns the name of the GPU processor on success,
  /// or a \c llvm::Error if an error was encountered
  [[nodiscard]] virtual llvm::Expected<std::string> getGPUName() const = 0;

  /// \returns on success, returns true if the ISA enables XNACK, false
  /// otherwise; On failure, a \c llvm::Error if any error
  /// was encountered
  [[nodiscard]] virtual llvm::Expected<bool> isXNACKSupported() const = 0;

  /// \returns on success, returns true if the ISA enables SRAMECC, false
  /// otherwise; On failure, a \c llvm::Error if any error
  /// was encountered
  [[nodiscard]] virtual llvm::Expected<bool> isSRAMECCSupported() const = 0;

  /// \returns the \c llvm::Triple of the ISA on success, or an
  /// \c llvm::Error if any issues are encountered in the process
  [[nodiscard]] virtual llvm::Expected<llvm::Triple>
  getTargetTriple() const = 0;

  /// \returns the LLVM subtraget feature string of the ISA on success,
  /// or an \c llvm::Error if an issue was encountered in
  /// the process
  [[nodiscard]] virtual llvm::Expected<llvm::SubtargetFeatures>
  getSubTargetFeatures() const = 0;
};

} // namespace luthier::hsa

#endif