//===-- Executable.hpp - HSA Executable Wrapper ---------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file defines the \c Executable class under the \c luthier::hsa
/// namespace, representing a wrapper around the \c hsa_executable_t.
//===----------------------------------------------------------------------===//
#ifndef HSA_EXECUTABLE_HPP
#define HSA_EXECUTABLE_HPP
#include <optional>
#include <vector>

#include <llvm/ADT/DenseMapInfo.h>

#include "hsa/CodeObjectReader.hpp"
#include "hsa/HandleType.hpp"

namespace luthier::hsa {

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

/// \brief wrapper around the \c hsa_executable_t handle
class Executable final : public HandleType<hsa_executable_t> {
public:
  /// Creates a new, empty \c hsa_executable_t handle and wraps it inside
  /// an \c Executable object
  /// \param Profile \c hsa_profile_t of the executable; set to
  /// \c HSA_PROFILE_FULL by default
  /// \param DefaultFloatRoundingMode \c hsa_default_float_rounding_mode_t
  /// of the executable; set to \c HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT
  /// by default
  /// \note the HSA function \c hsa_executable_create_alt used by this
  /// method to create executables also takes in an \c options argument;
  /// However, this argument does not get used by
  /// <tt>hsa_executable_create_alt</tt>, which is why it is absent from this
  /// method's arguments.
  /// \return on success, a new <tt>Executable</tt>, on failure, an
  /// \c luthier::HsaError
  /// \sa hsa_executable_create_alt
  /// \sa hsa_profile_t
  /// \sa hsa_default_float_rounding_mode_t
  static llvm::Expected<Executable>
  create(hsa_profile_t Profile = HSA_PROFILE_FULL,
         hsa_default_float_rounding_mode_t DefaultFloatRoundingMode =
             HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);

  /// Loads the code object read by the \p Reader onto the <tt>Agent</tt>'s
  /// memory. Results in creation of a \c LoadedCodeObject which will be
  /// managed by this executable.
  /// \param Reader the \c CodeObjectReader encapsulating a code object to
  /// be read into the executable
  /// \param Agent the \c GpuAgent the code object will be loaded onto
  /// \param LoaderOptions the loader options passed; See \c
  /// rocr::amd::hsa::loader::LoaderOptions::LoaderOptions in the HSA
  /// runtime source code for a the complete list of options
  /// \return on success, a newly created <tt>LoadedCodeObject</tt>. On
  /// failure, a \c luthier::HsaError
  /// \sa hsa_executable_load_agent_code_object
  llvm::Expected<hsa::LoadedCodeObject>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent,
                      llvm::StringRef LoaderOptions = "");

  /// Creates and defines a new external \c hsa_executable_symbol_t with the
  /// given \p SymbolName inside the executable. The new symbol will reside in
  /// the passed \p Address which must be accessible to the given \p Agent
  /// \param Agent the \c GpuAgent the newly created symbol will be defined
  /// for
  /// \param SymbolName the name of the newly created symbol
  /// \param Address the device address accessible by the \c Agent where
  /// this symbol will reside in
  /// \note This function must be called before the \c LoadedCodeObject that
  /// uses it
  /// \return an \c llvm::ErrorSuccess if the operation was successful, or a
  /// \c luthier::HsaError if the operation fails
  /// \sa hsa_executable_agent_global_variable_define
  llvm::Error defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                                llvm::StringRef SymbolName,
                                                const void *Address);

  /// Freezes the wrapped <tt>hsa_executable_t</tt>, which prevents it from
  /// being modified.
  /// \note the HSA function \c hsa_executable_freeze called by this method
  /// takes in an extra <tt>options</tt> argument which goes unused; Hence why
  /// it is not present in this method's arguments
  /// \return on success, an \c llvm::ErrorSuccess and on failure,
  /// a \c luthier::HsaError
  /// \sa hsa_executable_freeze
  llvm::Error freeze();

  /// Validates the executable
  /// \note the HSA function \c hsa_executable_validate_alt called by this
  /// method takes in an extra <tt>options</tt> argument which goes unused;
  /// Hence why it is not present in this method's arguments
  /// \return on success, true if the executable is valid, false otherwise; On
  /// error, returns a \c luthier::HsaError indicating the HSA issue encountered
  /// during the process
  /// \sa hsa_executable_validate_alt
  llvm::Expected<bool> validate();

  /// Destroys the executable handle
  /// \return an \c llvm::ErrorSuccess if successful, or a \c luthier::HsaError
  /// indicating any issues encountered by HSA
  /// \sa hsa_executable_destroy
  llvm::Error destroy();

  /// Constructor using a \c hsa_executable_t handle
  /// \warning This constructor must only be used with handles already created
  /// by HSA. To create executables from scratch, use \c create instead.
  /// \param Exec
  explicit Executable(hsa_executable_t Exec);

  /// Queries the \c hsa_profile_t of the wrapped \c hsa_executable_t
  /// \return the profile of the executable on success, and an
  /// \c luthier::HsaError on failure
  llvm::Expected<hsa_profile_t> getProfile();

  /// Queries the \c hsa_executable_state_t of the executable
  /// \return on success the state of the executable (i.e. frozen or not), on
  /// failure a \c luthier::HsaError indicating any issue encountered with the
  /// HSA runtime
  [[nodiscard]] llvm::Expected<hsa_executable_state_t> getState() const;

  /// Queries the default rounding mode of the executable
  /// \return on success the \c hsa_default_float_rounding_mode_t of the
  /// executable, and on failure an \c luthier::HsaError indicating any issue
  /// encountered with the HSA runtime
  llvm::Expected<hsa_default_float_rounding_mode_t> getRoundingMode();

  /// Queries the loaded code objects managed by this executable
  /// \param [out] LCOs the list of <tt>LoadedCodeObject</tt>s managed by this
  /// executable
  /// \return \c llvm::ErrorSuccess if the operation was successful, or a
  /// \c luthier::HsaError on failure
  llvm::Error
  getLoadedCodeObjects(llvm::SmallVectorImpl<LoadedCodeObject> &LCOs) const;

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