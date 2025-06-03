//===-- LoadedInstrumentationModule.h ---------------------------*- C++ -*-===//
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
/// Describes the \c hsa::LoadedInstrumentationModule class, representing an
/// \c InstrumentationModule that has been loaded onto a GPU device in the HSA
/// runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_INSTRUMENTATION_MODULE_H
#define LUTHIER_HSA_LOADED_INSTRUMENTATION_MODULE_H
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/Support/Error.h>
#include <luthier/tooling/InstrumentationModule.h>

namespace luthier::hsa {

class ToolExecutableLoader;

/// \brief encapsulates an \c InstrumentationModule that has been loaded onto
/// a device using a \c hsa_loaded_code_object_t
class LoadedInstrumentationModule {
protected:
  /// The executable of this loaded instrumentation module
  const hsa_executable_t Exec;

  /// The loaded code object of the loaded instrumentation module
  const hsa_loaded_code_object_t LCO;

  /// The instrumentation module loaded
  std::unique_ptr<InstrumentationModule> IModule;

  //===--------------------------------------------------------------------===//
  // A set of underlying functions that the \c ToolExecutableLoader will
  // supply to the Loaded IModule to carry out its query operations.
  //===--------------------------------------------------------------------===//

  const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn;

  const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
      &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn;

  LoadedInstrumentationModule(
      hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
      std::unique_ptr<InstrumentationModule> IModule,
      const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn)
      : Exec(Exec), LCO(LCO), IModule(std::move(IModule)),
        HsaExecutableGetInfoFn(HsaExecutableGetInfoFn),
        HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
            HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {};

  virtual ~LoadedInstrumentationModule() = default;

public:
  /// \returns On success, expects \c true if the instrumentation module has
  /// been loaded onto the device (i.e. its executable has been frozen), \c
  /// false otherwise;
  llvm::Expected<bool> isLoaded() const;

  /// \returns the \c InstrumentationModule of this loaded module
  [[nodiscard]] const InstrumentationModule &getIModule() const {
    return *IModule;
  }

  /// \returns the \c hsa_loaded_code_object_t this module is loaded on
  [[nodiscard]] hsa_loaded_code_object_t getLCO() const { return LCO; }

  /// \return Expects the mapping between all symbols inside this loaded
  /// instrumentation module and their load address on the GPU
  llvm::Expected<llvm::StringMap<uint64_t>> getSymbolLoadAddressesMap() const;
};

/// \brief Keeps track of instrumentation modules loaded via HIP FAT
/// binary mechanism
class HipLoadedInstrumentationModule final
    : public LoadedInstrumentationModule {
  friend ToolExecutableLoader;

  HipLoadedInstrumentationModule(
      const hsa_executable_t Exec, const hsa_loaded_code_object_t LCO,
      std::unique_ptr<InstrumentationModule> IModule,
      const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn)
      : LoadedInstrumentationModule(
            Exec, LCO, std::move(IModule), HsaExecutableGetInfoFn,
            HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {};

public:
  ~HipLoadedInstrumentationModule() override = default;
};

class DynamicallyLoadedInstrumentationModule final
    : public LoadedInstrumentationModule {
  friend ToolExecutableLoader;

  /// Needed to destroy the underlying executable of the module in the
  /// destructor
  const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn;

  DynamicallyLoadedInstrumentationModule(
      hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
      std::unique_ptr<InstrumentationModule> IModule,
      const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn,
      const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn)
      : LoadedInstrumentationModule(Exec, LCO, std::move(IModule),
                                    HsaExecutableGetInfoFn,
                                    HsaVenAmdLoaderLoadedCodeObjectGetInfoFn),
        HsaExecutableDestroyFn(HsaExecutableDestroyFn) {};

public:
  ~DynamicallyLoadedInstrumentationModule() override;
};

} // namespace luthier::hsa
#endif