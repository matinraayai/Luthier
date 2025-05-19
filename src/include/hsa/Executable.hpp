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
/// Defines the \c Executable class, a wrapper around the \c hsa_executable_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_HPP
#define LUTHIER_HSA_EXECUTABLE_HPP
#include "hsa/HandleType.hpp"
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/Support/Error.h>
#include <optional>

namespace luthier::hsa {

class CodeObjectReader;

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

/// \brief wrapper around the \c hsa_executable_t handle
class Executable final : public HandleType<hsa_executable_t> {
public:
  /// Constructor using a \c hsa_executable_t handle
  /// \note This constructor must only be used with handles already created
  /// by HSA. To create executables from scratch, use \c create instead.
  /// \param Exec an already-created HSA executable
  explicit Executable(hsa_executable_t Exec);

  /// Creates a new, empty \c hsa_executable_t handle and wraps it inside
  /// an \c Executable object
  /// \param HsaCreateExecutableCreateAltFn the \c hsa_executable_create_alt
  /// function used to carry out the operation
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
  /// \return Expects a newly created \c Executable
  /// \sa hsa_executable_create_alt
  /// \sa hsa_profile_t
  /// \sa hsa_default_float_rounding_mode_t
  static llvm::Expected<Executable>
  create(decltype(hsa_executable_create_alt) *HsaCreateExecutableCreateAltFn,
         hsa_profile_t Profile = HSA_PROFILE_FULL,
         hsa_default_float_rounding_mode_t DefaultFloatRoundingMode =
             HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);

  /// Loads the code object read by the \p Reader onto the <tt>Agent</tt>'s
  /// memory. Results in creation of a \c LoadedCodeObject which will be
  /// managed by this executable.
  /// \param HsaExecutableLoadAgentCodeObjectFn the \c
  /// hsa_executable_load_agent_code_object used to carry out the operation
  /// \param Reader the \c CodeObjectReader encapsulating a code object to
  /// be read into the executable
  /// \param Agent the \c GpuAgent the code object will be loaded onto
  /// \param LoaderOptions the loader options passed; See \c
  /// rocr::amd::hsa::loader::LoaderOptions::LoaderOptions in the HSA
  /// runtime source code for a complete list of options
  /// \return on success, a newly created <tt>LoadedCodeObject</tt>. On
  /// failure, a \c luthier::HsaError
  /// \sa hsa_executable_load_agent_code_object
  llvm::Expected<LoadedCodeObject>
  loadAgentCodeObject(const decltype(hsa_executable_load_agent_code_object)
                          *HsaExecutableLoadAgentCodeObjectFn,
                      const CodeObjectReader &Reader, const GpuAgent &Agent,
                      llvm::StringRef LoaderOptions = "");

  /// Creates and defines a new external HSA executable symbol with the
  /// given \p SymbolName inside the executable
  /// \param HsaExecutableAgentGlobalVariableDefineFn the \c
  /// hsa_executable_agent_global_variable_define_fn function used to carry out
  /// this operation
  /// \param Agent the \c GpuAgent the symbol will be defined for
  /// \param SymbolName the name of the created symbol
  /// \param Address the device address of the symbol; Must accessible by
  /// the \c Agent
  /// \return \c llvm::Error indicating the success or failure of the operation
  /// \sa hsa_executable_agent_global_variable_define
  llvm::Error defineExternalAgentGlobalVariable(
      const decltype(hsa_executable_agent_global_variable_define)
          *HsaExecutableAgentGlobalVariableDefineFn,
      const GpuAgent &Agent, llvm::StringRef SymbolName, const void *Address);

  /// Freezes the wrapped <tt>hsa_executable_t</tt>
  /// \param HsaExecutableFreezeFn The \c hsa_executable_freeze function
  /// used to carry out the operation
  /// \note the HSA function \c hsa_executable_freeze called by this method
  /// takes in an extra <tt>options</tt> argument which goes unused; Hence why
  /// it is not present in this method's arguments
  /// \return \c llvm::Error indicating the success or failure of the operation
  /// \sa hsa_executable_freeze
  llvm::Error
  freeze(const decltype(hsa_executable_freeze) *HsaExecutableFreezeFn);

  /// Destroys the executable handle
  /// \param HsaExecutableDestroyFn The \c hsa_executable_destroy function
  /// used to carry out the operation
  /// \return \c llvm::Error indicating the success of failure of the operation
  /// \sa hsa_executable_destroy
  llvm::Error
  destroy(const decltype(hsa_executable_destroy) *HsaExecutableDestroyFn);

  /// Queries the \c hsa_profile_t of the wrapped \c hsa_executable_t
  /// \param HsaExecutableGetInfoFn the \c hsa_executable_get_info function
  /// used to carry out the operation
  /// \return \c llvm::Error indicating the success or failure of the operation
  llvm::Expected<hsa_profile_t> getProfile(
      const decltype(hsa_executable_get_info) *HsaExecutableGetInfoFn) const;

  /// Queries the \c hsa_executable_state_t of the executable
  /// \param HsaExecutableGetInfoFn the \c hsa_executable_get_info function
  /// used to carry out the operation
  /// \return on success the state of the executable (i.e. frozen or not), on
  /// failure a \c luthier::HsaError indicating any issue encountered with the
  /// HSA runtime
  [[nodiscard]] llvm::Expected<hsa_executable_state_t> getState(
      const decltype(hsa_executable_get_info) *HsaExecutableGetInfoFn) const;

  /// Queries the loaded code objects managed by this executable
  /// \param HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn the
  /// \c hsa_ven_amd_loader_executable_iterate_loaded_code_objects function
  /// used to complete the operation
  /// \param [out] LCOs the list of <tt>LoadedCodeObject</tt>s managed by this
  /// executable
  /// \return \c llvm::Error indicating the success or failure of the operation
  llvm::Error getLoadedCodeObjects(
      const decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
          *HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn,
      llvm::SmallVectorImpl<LoadedCodeObject> &LCOs) const;

  /// Looks up the \c ExecutableSymbol by its \p Name in the Executable
  /// \param HsaExecutableGetSymbolByNameFn the
  /// \c hsa_executable_get_symbol_by_name function used to complete the
  /// operation
  /// \param Name Name of the symbol being looked up; If the queried symbol
  /// is a kernel then it must have ".kd" as a suffix in the name
  /// \param Agent the \c GpuAgent the symbol belongs to
  /// \return on success, the \c ExecutableSymbol with the given \p Name if
  /// found, otherwise <tt>std::nullopt</tt>; Otherwise a \c llvm::Error
  /// indicating any issue encountered during the process
  llvm::Expected<std::optional<ExecutableSymbol>>
  getExecutableSymbolByName(const decltype(hsa_executable_get_symbol_by_name)
                                *HsaExecutableGetSymbolByNameFn,
                            llvm::StringRef Name, const GpuAgent &Agent) const;
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