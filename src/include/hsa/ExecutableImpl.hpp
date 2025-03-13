//===-- ExecutableImpl.hpp ------------------------------------------------===//
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
/// This file defines the \c hsa::ExecutableImpl class, which is the concrete
/// implementation of the \c hsa::Executable interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_IMPL_HPP
#define LUTHIER_HSA_EXECUTABLE_IMPL_HPP
#include "hsa/Executable.hpp"
#include "hsa/HandleType.hpp"
#include <hsa/hsa.h>

namespace luthier::hsa {

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

/// \brief Concrete implementation of the \c Executable interface
class ExecutableImpl : public Executable, public HandleType<hsa_executable_t> {
public:
  /// Default constructor
  ExecutableImpl() : HandleType<hsa_executable_t>({0}) {};

  /// Constructor using an already created \c hsa_executable_t \p Exec handle
  /// \warning This constructor must only be used with handles already created
  /// by HSA. To create executables from scratch, use \c create instead.
  explicit ExecutableImpl(hsa_executable_t Exec);

  llvm::Error
  create(hsa_profile_t Profile,
         hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) override;

  llvm::Expected<hsa::LoadedCodeObject>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent,
                      llvm::StringRef LoaderOptions) override;

  llvm::Error defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                                llvm::StringRef SymbolName,
                                                const void *Address) override;

  llvm::Error freeze() override;

  llvm::Expected<bool> validate() override;

  llvm::Error destroy() override;

  llvm::Expected<hsa_profile_t> getProfile() override;

  [[nodiscard]] llvm::Expected<hsa_executable_state_t>
  getState() const override;

  llvm::Expected<hsa_default_float_rounding_mode_t> getRoundingMode() override;

  llvm::Error getLoadedCodeObjects(
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObject>> &LCOs)
      const override;

  /// Looks up the \c ExecutableSymbol by its \p Name in the Executable
  /// \param Name Name of the symbol being looked up; If the queried symbol
  /// is a kernel then it must have ".kd" as a suffix in the name
  /// \return on success, the \c ExecutableSymbol with the given \p Name if
  /// found, otherwise <tt>std::nullopt</tt>; Otherwise a \c luthier::HsaError
  /// indicating any issue encountered during the process
  llvm::Expected<std::optional<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name, const hsa::GpuAgent &Agent);
};

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::Executable> {
  static inline luthier::hsa::Executable getEmptyKey() {
    return luthier::hsa::Executable(
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::Executable getTombstoneKey() {
    return luthier::hsa::Executable(
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::Executable &Executable) {
    return DenseMapInfo<decltype(hsa_executable_t::handle)>::getHashValue(
        Executable.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::Executable &lhs,
                      const luthier::hsa::Executable &rhs) {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<luthier::hsa::Executable> {
  size_t operator()(const luthier::hsa::Executable &Obj) const {
    return hash<unsigned long>()(Obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::Executable> {
  bool operator()(const luthier::hsa::Executable &Lhs,
                  const luthier::hsa::Executable &Rhs) const {
    return Lhs.hsaHandle() < Rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::Executable> {
  bool operator()(const luthier::hsa::Executable &Lhs,
                  const luthier::hsa::Executable &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace std

#endif