//===-- Executable.hpp - HSA Executable Wrapper ---------------------------===//
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
/// This file defines the \c hsa::Executable interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_HPP
#define LUTHIER_HSA_EXECUTABLE_HPP
#include <optional>

namespace luthier::hsa {

class CodeObjectReader;

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

/// \brief an interface representing the concept of an executable inside
/// the HSA standard
class Executable {
public:
  /// Creates a new executable and assigns it to be managed by this
  /// \c Executable object
  /// \param Profile \c hsa_profile_t of the executable
  /// \param DefaultFloatRoundingMode \c hsa_default_float_rounding_mode_t
  /// of the executable
  /// \note the HSA function \c hsa_executable_create_alt used by this
  /// method to create executables also takes in an \c options argument;
  /// However, this argument does not get used by
  /// <tt>hsa_executable_create_alt</tt>, which is why it is absent from this
  /// method's arguments
  /// \returns an error if the underlying handle is not zero
  /// \return \c llvm::Error indicating the success or failure of the operation
  /// \sa hsa_executable_create_alt
  /// \sa hsa_profile_t
  /// \sa hsa_default_float_rounding_mode_t
  virtual llvm::Error
  create(hsa_profile_t Profile,
         hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) = 0;

  llvm::Error create();

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
  /// failure, an \c llvm::Error
  /// \sa hsa_executable_load_agent_code_object
  virtual llvm::Expected<hsa::LoadedCodeObject>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent,
                      llvm::StringRef LoaderOptions) = 0;

  llvm::Expected<hsa::LoadedCodeObject>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent);

  /// Creates and defines a new external \c hsa_executable_symbol_t with the
  /// given \p SymbolName inside the executable. The new symbol will reside in
  /// the passed \p Address which must be accessible by the given \p Agent
  /// \param Agent the \c GpuAgent the newly created symbol will be defined
  /// for
  /// \param SymbolName the name of the newly created symbol
  /// \param Address the device address accessible by the \c Agent where
  /// this symbol will reside in
  /// \note This function must be called before the \c LoadedCodeObject that
  /// uses it
  /// \return an \c llvm::Error indicating the success of failure of the
  /// operation
  /// \sa hsa_executable_agent_global_variable_define
  virtual llvm::Error
  defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                    llvm::StringRef SymbolName,
                                    const void *Address) = 0;

  /// Freezes the wrapped <tt>hsa_executable_t</tt>, which prevents it from
  /// being modified.
  /// \note the HSA function \c hsa_executable_freeze called by this method
  /// takes in an extra <tt>options</tt> argument which goes unused; Hence why
  /// it is not present in this method's arguments
  /// \return an \c llvm::Error indicating the success or failure of
  /// the operation
  /// \sa hsa_executable_freeze
  virtual llvm::Error freeze() = 0;

  /// Validates the executable
  /// \note the HSA function \c hsa_executable_validate_alt called by this
  /// method takes in an extra <tt>options</tt> argument which goes unused;
  /// Hence why it is not present in this method's arguments
  /// \return an \c llvm::Error indicating the success or failure of
  /// the operation
  /// \sa hsa_executable_validate_alt
  virtual llvm::Expected<bool> validate() = 0;

  /// Destroys the executable handle
  /// \return an \c llvm::Error indicating the success or failure of
  /// the operation
  /// \sa hsa_executable_destroy
  virtual llvm::Error destroy() = 0;

  /// Queries the \c hsa_profile_t of the wrapped \c hsa_executable_t
  /// \return the profile of the executable on success, and an
  /// \c luthier::HsaError on failure
  virtual llvm::Expected<hsa_profile_t> getProfile() = 0;

  /// Queries the \c hsa_executable_state_t of the executable
  /// \return on success the state of the executable (i.e. frozen or not), on
  /// failure a \c llvm::Error
  [[nodiscard]] virtual llvm::Expected<hsa_executable_state_t>
  getState() const = 0;

  /// Queries the default rounding mode of the executable
  /// \return on success the \c hsa_default_float_rounding_mode_t of the
  /// executable, and on failure an \c llvm::Error
  virtual llvm::Expected<hsa_default_float_rounding_mode_t>
  getRoundingMode() = 0;

  /// Queries the loaded code objects managed by this executable
  /// \param [out] LCOs the list of <tt>LoadedCodeObject</tt>s managed by this
  /// executable
  /// \return \c llvm::ErrorSuccess if the operation was successful, or an
  /// \c llvm::Error on failure
  [[nodiscard]] virtual llvm::Error getLoadedCodeObjects(
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObject>> &LCOs) const = 0;

  /// Looks up the \c ExecutableSymbol by its \p Name in the Executable
  /// \param Name Name of the symbol being looked up; If the queried symbol
  /// is a kernel then it must have ".kd" as a suffix in the name
  /// \return on success, the \c ExecutableSymbol with the given \p Name if
  /// found, otherwise <tt>std::nullopt</tt>; On failure an \c llvm::Error
  /// describing the issue encountered during the process
  virtual llvm::Expected<std::optional<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name,
                            const hsa::GpuAgent &Agent) = 0;
};

} // namespace luthier::hsa

#endif