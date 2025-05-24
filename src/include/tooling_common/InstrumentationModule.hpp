//===-- InstrumentationModule.hpp - Luthier Instrumentation Module --------===//
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
/// This file describes Luthier's Instrumentation Module, and its Loaded version
/// when it is loaded on the GPU.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INSTRUMENTATION_MODULE_HPP
#define LUTHIER_TOOLING_COMMON_INSTRUMENTATION_MODULE_HPP
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h>
#include <luthier/object/ELFObjectUtils.h>
#include <luthier/object/ObjectUtils.h>

namespace luthier {

class ToolExecutableLoader;

/// \brief Encapsulates a Luthier instrumentation module, consisting of
/// a shared object file with its LLVM bitcode embedded in one of its sections
class InstrumentationModule {
  /// A host copy of the code object used to load the module
  const std::vector<uint8_t> CodeObject;

  /// Parsed representation of the \c CodeObject
  const std::unique_ptr<llvm::object::ObjectFile> ObjectFile;

  /// The LLVM bitcode buffer inside the \c CodeObject
  const std::unique_ptr<llvm::MemoryBuffer> BCBuffer;

  InstrumentationModule(std::vector<uint8_t> CodeObject,
                        std::unique_ptr<llvm::object::ObjectFile> ObjectFile,
                        std::unique_ptr<llvm::MemoryBuffer> BCBuffer)
      : CodeObject(std::move(CodeObject)), ObjectFile(std::move(ObjectFile)),
        BCBuffer(std::move(BCBuffer)) {}

public:
  static llvm::Expected<std::unique_ptr<InstrumentationModule>>
  create(std::vector<uint8_t> CodeObject);

  /// \returns the parsed object file representation of the instrumentation
  /// module
  [[nodiscard]] const llvm::object::ObjectFile &getObject() const {
    return *ObjectFile;
  }

  /// Reads the bitcode of this InstrumentationModule into a new
  /// \c llvm::Module backed by the \p Ctx
  /// \param Ctx an \c LLVMContext of the returned Module
  /// \return the \c llvm::Module of the loaded instrumentation module on
  /// success
  llvm::Expected<std::unique_ptr<llvm::Module>>
  readBitcodeIntoContext(llvm::LLVMContext &Ctx) const {
    return llvm::parseBitcodeFile(*BCBuffer, Ctx);
  }

  /// \param [out] Manifest the manifest of the instrumentation module
  /// (i.e. the offsets of each object symbol of the module
  /// \return \c llvm::Error indicating the success or failure of the operation
  llvm::Error readManifest(llvm::StringMap<uint64_t> &Manifest) const {
    return object::readManifest(*ObjectFile, Manifest);
  }
};

/// \brief encapsulates a Luthier tool
class InstrumentationModule {
protected:
  /// The loaded code object of the loaded instrumentation module
  const hsa_loaded_code_object_t LCO;

  /// Stored to get the load manifest of the
  const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
      &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn;

  std::unique_ptr<InstrumentationModule> IModule;

  InstrumentationModule(
      hsa_loaded_code_object_t LCO,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn,
      std::unique_ptr<InstrumentationModule> IModule)
      : LCO(LCO), HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
                      HsaVenAmdLoaderLoadedCodeObjectGetInfoFn),
        IModule(std::move(IModule)) {};

  virtual ~InstrumentationModule() = default;

public:
  [[nodiscard]] const InstrumentationModule &getIModule() const {
    return *IModule;
  }

  [[nodiscard]] hsa_loaded_code_object_t getLCO() const { return LCO; }

  /// \param [out] Manifest the manifest of the instrumentation module
  /// (i.e. the addresses of each object symbol of the module is loaded on the
  /// device)
  /// \return \c llvm::Error indicating the success or failure of the operation
  llvm::Error readManifest(llvm::StringMap<uint64_t> &Manifest) const;
};

//===----------------------------------------------------------------------===//
// Static Instrumentation Module
//===----------------------------------------------------------------------===//

/// \brief Keeps track of instrumentation code loaded via a static HIP FAT
/// binary
/// \details an implementation of \c InstrumentationModule which keeps track of
/// <b>the</b> static HIP FAT binary embedded in the shared object of a Luthier
/// tool.\n
/// For now we anticipate that only a single Luthier tool will be loaded at any
/// given time; i.e. we don't think there is a case to instrument an already
/// instrumented GPU device code. \c ToolExecutableManager
/// enforces this by keeping a single instance of this variable, as
/// well as keeping its constructor private to itself. \n
/// Furthermore, If two or more Luthier tools are loaded then
/// \c StaticInstrumentationModule will detect it by checking the compile unit
/// ID of each executable passed to it.\n
/// For each GPU Agent, the HIP runtime extracts an ISA-compatible
/// code object from the static FAT binary and loads it into a single
/// executable. This is done in a lazy fashion if deferred loading is enabled,
/// meaning the loading only occurs on a device if the app starts using it. \n
/// \c StaticInstrumentationModule gets notified when a new \c hsa::Executable
/// of the FAT binary gets loaded onto each device. On the first occurrence,
/// it will record the CUID of the module, and creates a list of global
/// variables in the module, as well as their associated \c
/// hsa::ExecutableSymbol on the loaded \c hsa::GpuAgent. On subsequent
/// executable loads, it only updates the global variable list. It should be
/// clear by now that \c StaticInstrumentationModule does not do any GPU memory
/// management and relies solely on HIP for loading.\n A similar mechanism is in
/// place to detect unloading of the instrumentation module's executables; As
/// they get destroyed, the affected \c hsa::ExecutableSymbols get invalidated
/// as well.\n
/// \c StaticInstrumentationModule also gets notified of the kernel shadow host
/// pointers of each hook, and converts them to the correct hook name to
/// be found in the module later on.
/// \sa llvm::BitcodeBuffer, LUTHIER_HOOK_ANNOTATE,
/// LUTHIER_EXPORT_HOOK_HANDLE
class HipLoadedInstrumentationModule final
    : public InstrumentationModule {
  HipLoadedInstrumentationModule(
      hsa_loaded_code_object_t LCO,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn,
      std::unique_ptr<InstrumentationModule> IModule)
      : InstrumentationModule(LCO,
                                    HsaVenAmdLoaderLoadedCodeObjectGetInfoFn,
                                    std::move(IModule)) {};

public:
  static llvm::Expected<std::unique_ptr<HipLoadedInstrumentationModule>>
  getIfHipLoadedIModule(
      hsa_loaded_code_object_t LCO,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &LoadedCodeObjectGetInfoFun);

  ~HipLoadedInstrumentationModule() override = default;
};

class DynamicallyLoadedInstrumentationModule final
    : public InstrumentationModule {

  /// The executable used to load this module
  hsa_executable_t Exec;

  const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn;

  DynamicallyLoadedInstrumentationModule(
      hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn,
      std::unique_ptr<InstrumentationModule> IModule,
      const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn)
      : InstrumentationModule(
            LCO, HsaVenAmdLoaderLoadedCodeObjectGetInfoFn, std::move(IModule)),
        Exec(Exec), HsaExecutableDestroyFn(HsaExecutableDestroyFn) {};

public:
  static llvm::Expected<std::unique_ptr<DynamicallyLoadedInstrumentationModule>>
  loadInstrumentationModule(
      std::vector<uint8_t> CodeObject, hsa_agent_t Agent,
      const decltype(hsa_executable_create_alt) &HsaExecutableCreateAltFn,
      const decltype(hsa_code_object_reader_create_from_memory)
          &HsaCodeObjectReaderCreateFromMemory,
      const decltype(hsa_executable_load_agent_code_object)
          &HsaExecutableLoadAgentCodeObjectFn,
      const decltype(hsa_executable_freeze) &HsaExecutableFreezeFn,
      const decltype(hsa_code_object_reader_destroy)
          &HsaCodeObjectReaderDestroyFn,
      const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &LoadedCodeObjectGetInfoFun);

  ~DynamicallyLoadedInstrumentationModule() override;
};

} // namespace luthier
#endif