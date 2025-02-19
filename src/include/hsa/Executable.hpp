//===-- Executable.hpp - HSA Executable Interface -------------------------===//
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
/// This file defines the \c Executable interface under the \c luthier::hsa
/// namespace, an abstraction around the \c hsa_executable_t handle operations
/// in ROCr.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_HPP
#define LUTHIER_HSA_EXECUTABLE_HPP
#include "hsa/CodeObjectReader.hpp"
#include "hsa/HandleType.hpp"
#include <llvm/ADT/DenseMapInfo.h>

namespace luthier::hsa {

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

/// \brief wrapper around the \c hsa_executable_t handle
class Executable : public HandleType<hsa_executable_t> {
public:
  /// Constructor using a \c hsa_executable_t handle
  /// \warning This constructor must only be used with handles already created
  /// by HSA. To create executables from scratch, use \c create instead.
  /// \param Exec
  explicit Executable(hsa_executable_t Exec)
      : HandleType<hsa_executable_t>(Exec) {};

  virtual ~Executable() = default;

  /// Creates a new \c hsa_executable_t handle
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
  [[nodiscard]] virtual llvm::Error
  create(hsa_profile_t Profile,
         hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) = 0;

  llvm::Error create() {
    return create(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);
  }

  /// \return a new cloned version of this executable
  [[nodiscard]] virtual std::unique_ptr<Executable> clone() const = 0;

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
  virtual llvm::Expected<std::unique_ptr<hsa::LoadedCodeObject>>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent,
                      llvm::StringRef LoaderOptions) = 0;

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
  virtual llvm::Error
  defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                    llvm::StringRef SymbolName,
                                    const void *Address) = 0;

  /// Freezes the wrapped <tt>hsa_executable_t</tt>, which prevents it from
  /// being modified.
  /// \note the HSA function \c hsa_executable_freeze called by this method
  /// takes in an extra <tt>options</tt> argument which goes unused; Hence why
  /// it is not present in this method's arguments
  /// \return on success, an \c llvm::ErrorSuccess and on failure,
  /// a \c luthier::HsaError
  /// \sa hsa_executable_freeze
  virtual llvm::Error freeze() = 0;

  /// Destroys the executable handle
  /// \return an \c llvm::ErrorSuccess if successful, or a \c luthier::HsaError
  /// indicating any issues encountered by HSA
  /// \sa hsa_executable_destroy
  virtual llvm::Error destroy() = 0;

  /// Queries the \c hsa_profile_t of the wrapped \c hsa_executable_t
  /// \return the profile of the executable on success, and an
  /// \c luthier::HsaError on failure
  virtual llvm::Expected<hsa_profile_t> getProfile() = 0;

  /// Queries the \c hsa_executable_state_t of the executable
  /// \return on success the state of the executable (i.e. frozen or not), on
  /// failure a \c luthier::HsaError indicating any issue encountered with the
  /// HSA runtime
  [[nodiscard]] virtual llvm::Expected<hsa_executable_state_t>
  getState() const = 0;

  /// Queries the default rounding mode of the executable
  /// \return on success the \c hsa_default_float_rounding_mode_t of the
  /// executable, and on failure an \c luthier::HsaError indicating any issue
  /// encountered with the HSA runtime
  virtual llvm::Expected<hsa_default_float_rounding_mode_t>
  getRoundingMode() = 0;

  /// Queries the loaded code objects managed by this executable
  /// \param [out] LCOs the list of <tt>LoadedCodeObject</tt>s managed by this
  /// executable
  /// \return \c llvm::ErrorSuccess if the operation was successful, or a
  /// \c luthier::HsaError on failure
  virtual llvm::Error getLoadedCodeObjects(
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObject>> &LCOs) const = 0;

  /// Looks up the \c ExecutableSymbol by its \p Name in the Executable
  /// \param Name Name of the symbol being looked up; If the queried symbol
  /// is a kernel then it must have ".kd" as a suffix in the name
  /// \return on success, the \c ExecutableSymbol with the given \p Name if
  /// found, otherwise <tt>std::nullopt</tt>; Otherwise a \c luthier::HsaError
  /// indicating any issue encountered during the process
  virtual llvm::Expected<std::unique_ptr<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name,
                            const hsa::GpuAgent &Agent) = 0;
};

} // namespace luthier::hsa

DECLARE_LLVM_MAP_INFO_STRUCTS_FOR_HANDLE_TYPE(luthier::hsa::Executable,
                                              hsa_executable_t)

#endif