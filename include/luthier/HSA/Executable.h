//===-- Executable.h ---------------------------------------------*- C++-*-===//
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
/// Defines a set of commonly used functionality for the \c hsa_executable_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_H
#define LUTHIER_HSA_EXECUTABLE_H
#include "luthier/HSA/ApiTable.h"
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <optional>

namespace luthier::hsa {

/// Creates a new \c hsa_executable_t handle
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
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
/// \return Expects a newly created \c hsa_executable_t
/// \sa hsa_executable_create_alt
/// \sa hsa_profile_t
/// \sa hsa_default_float_rounding_mode_t
llvm::Expected<hsa_executable_t>
executableCreate(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_profile_t Profile = HSA_PROFILE_FULL,
                 hsa_default_float_rounding_mode_t DefaultFloatRoundingMode =
                     HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);

/// Loads the code object read by the \p Reader into the <tt>Agent</tt>'s
/// memory. Results in creation of a \c hsa_loaded_code_object_t which will be
/// managed by this executable.
/// \param Exec the \c hsa_executable_t where the code object will be loaded
/// into
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Reader the \c hsa_code_object_reader_t encapsulating a code object
/// to be read into the executable
/// \param Agent the \c hsa_agent_t the code object will be loaded onto
/// \param LoaderOptions the loader options passed; See \c
/// rocr::amd::hsa::loader::LoaderOptions::LoaderOptions in the HSA
/// runtime source code for a complete list of options
/// \return Expects the newly created \c hsa_loaded_code_object_t on success
/// \sa hsa_executable_load_agent_code_object
llvm::Expected<hsa_loaded_code_object_t> executableLoadAgentCodeObject(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_executable_t Exec,
    hsa_code_object_reader_t Reader, hsa_agent_t Agent,
    llvm::StringRef LoaderOptions = "");

/// Creates and defines a new external HSA executable symbol with the
/// given \p SymbolName inside the executable
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the \c hsa_executable_t where the symbol will be defined in
/// \param Agent the \c hsa_agent_t the symbol will be defined for
/// \param SymbolName the name of the created symbol
/// \param Address the device address of the symbol; Must accessible by
/// the \p Agent
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_executable_agent_global_variable_define
llvm::Error executableDefineExternalAgentGlobalVariable(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_executable_t Exec,
    hsa_agent_t Agent, llvm::StringRef SymbolName, const void *Address);

/// Freezes the \p Exec handle
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the \c hsa_executable_t being frozen
/// \note the HSA function \c hsa_executable_freeze called by this method
/// takes in an extra <tt>options</tt> argument which goes unused; Hence why
/// it is not present in this method's arguments
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_executable_freeze
llvm::Error executableFreeze(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_executable_t Exec);

/// Destroys the executable handle
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the \c hsa_executable_t being destroyed
/// \return \c llvm::Error indicating the success of failure of the operation
/// \sa hsa_executable_destroy
llvm::Error executableDestroy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                              hsa_executable_t Exec);

/// Queries the \c hsa_profile_t of the wrapped \c hsa_executable_t
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the \c hsa_executable_t being queried
/// \return \c llvm::Error indicating the success or failure of the operation
llvm::Expected<hsa_profile_t>
executableGetProfile(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_executable_t Exec);

/// Queries the \c hsa_executable_state_t of the executable
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the \c hsa_executable_t being queried
/// \return Expects the state of the executable (i.e. frozen or not) on success
[[nodiscard]] llvm::Expected<hsa_executable_state_t>
executableGetState(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_executable_t Exec);

/// Queries the loaded code objects managed by this executable
/// \param LoaderApi the HSA loader API table container used to perform HSA
/// loader calls
/// \param Exec the \c hsa_executable_t being queried
/// \param [out] LCOs the list of <tt>hsa_loaded_code_object</tt>s managed by
/// this executable
/// \return \c llvm::Error indicating the success or failure of the operation
template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
llvm::Error executableGetLoadedCodeObjects(
    const ExtensionTableContainer<HSA_EXTENSION_AMD_LOADER, LoaderTableType>
        &LoaderApi,
    hsa_executable_t Exec,
    llvm::SmallVectorImpl<hsa_loaded_code_object_t> &LCOs) {
  auto Iterator = [](hsa_executable_t, hsa_loaded_code_object_t LCO,
                     void *Data) -> hsa_status_t {
    auto Out =
        static_cast<llvm::SmallVectorImpl<hsa_loaded_code_object_t> *>(Data);
    if (!Out)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    Out->emplace_back(LCO);
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApi.callFunction<
          &LoaderTableType::
              hsa_ven_amd_loader_executable_iterate_loaded_code_objects>(
          Exec, Iterator, &LCOs),
      llvm::formatv(
          "Failed to iterate over the code objects of executable {0:x}", Exec));
}

/// Looks up the \c hsa_executable_symbol_t by its \p Name in the Executable
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the \c hsa_executable_t being queried
/// \param Name Name of the symbol being looked up; If the queried symbol
/// is a kernel then it must have ".kd" as a suffix in the name
/// \param Agent the \c GpuAgent the symbol belongs to
/// \return Expects the \c hsa_executable_symbol_t with the given \p Name if
/// found, otherwise <tt>std::nullopt</tt> on success
llvm::Expected<std::optional<hsa_executable_symbol_t>>
executableGetSymbolByName(const ApiTableContainer<::CoreApiTable> &CoreApi,
                          hsa_executable_t Exec, llvm::StringRef Name,
                          hsa_agent_t Agent);

/// Iterates over the symbols inside the \p Exec that belong to \p Agent and
/// invokes the provided \p Callback
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the executable being inspected
/// \param Agent the GPU agent for which the symbols are going to be iterated
/// over
/// \param Callback a generic \c std::function that gets invoked during the
/// iteration for the current \c hsa_executable_symbol_t of the iteration.
/// If it returns a \c llvm::Error in a failure state, the function halts
/// the iteration and returns the failure error instead
/// \return a \c llvm::Error indicating the success or the failure of the
/// operation
llvm::Error executableIterateAgentSymbols(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_executable_t Exec,
    hsa_agent_t Agent,
    const std::function<llvm::Error(hsa_executable_symbol_t)> &Callback);

/// Iterates over the symbols inside the \p Exec that belong to \p Agent and
/// finds the first \c hsa_executable_symbol_t inside \p Exec that returns
/// true for the given \p Predicate
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Exec the executable being inspected
/// \param Agent the GPU agent for which the symbols are going to be iterated
/// over
/// \param Predicate a generic \c std::function that gets invoked during the
/// iteration. It takes in the current iteration's \c hsa_executable_symbol_t
/// as its parameter and expects a boolean value. If the predicate returns
/// an error, the iteration halts and the error is returned
/// \return Expects a \c hsa_executable_symbol_t if a symbol was found using
/// the \p Predicate; \c std::nullopt is expected if no symbol is found
llvm::Expected<std::optional<hsa_executable_symbol_t>>
executableFindFirstAgentSymbol(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_executable_t Exec,
    hsa_agent_t Agent,
    const std::function<llvm::Expected<bool>(hsa_executable_symbol_t)>
        &Predicate);

} // namespace luthier::hsa

inline bool operator==(const hsa_executable_t &Lhs,
                       const hsa_executable_t &Rhs) {
  return Lhs.handle == Rhs.handle;
}

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_executable_t> {
  static hsa_executable_t getEmptyKey() {
    return hsa_executable_t(
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getEmptyKey()});
  }

  static hsa_executable_t getTombstoneKey() {
    return hsa_executable_t(
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const hsa_executable_t &Executable) {
    return DenseMapInfo<decltype(hsa_executable_t::handle)>::getHashValue(
        Executable.handle);
  }

  static bool isEqual(const hsa_executable_t &Lhs,
                      const hsa_executable_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
}; // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_executable_t> {
  size_t operator()(const hsa_executable_t &Obj) const noexcept {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct equal_to<hsa_executable_t> {
  bool operator()(const hsa_executable_t &Lhs,
                  const hsa_executable_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace std

#endif