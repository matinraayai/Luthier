//===-- MockAMDGPULoader.h --------------------------------------*- C++ -*-===//
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
/// \file Defines the \c MockAMDGPULoader and \c MockLoadedCodeObject
/// classes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_MOCK_AMDGPU_LOADER_H
#define LUTHIER_MOCK_AMDGPU_LOADER_H

#include "luthier/object/AMDGCNObjectFile.h"
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Error.h>
#include <string>

namespace luthier {

class MockAMDGPULoader;

class MockLoadedCodeObject {
  friend class MockAMDGPULoader;

  /// The executable this loaded code object belongs to
  MockAMDGPULoader &Parent;

  /// The AMDGCN object file of the ELF loaded; The code object itself is
  /// externally managed
  std::unique_ptr<object::AMDGCNObjectFile> Elf{};

  /// A contiguous memory region where the code object is loaded
  llvm::MutableArrayRef<std::byte> Segment{};

  /// List of load segments of the ELF, sorted w.r.t their virtual address
  llvm::SmallVector<std::reference_wrapper<const llvm::object::ELF64LE::Phdr>,
                    4>
      PTLoadSegments{};

  MockLoadedCodeObject(MockAMDGPULoader &Owner, llvm::ArrayRef<std::byte> Elf,
                       llvm::Error &Err);

public:
  MockLoadedCodeObject(const MockLoadedCodeObject &) = delete;

  MockLoadedCodeObject &operator=(const MockLoadedCodeObject &) = delete;

  ~MockLoadedCodeObject() {
    if (Segment.data())
      delete Segment.data();
  }

  [[nodiscard]] const object::AMDGCNObjectFile &getCodeObject() const {
    return *Elf;
  }

  llvm::ArrayRef<std::byte> getLoadedSegment() const { return Segment; }

  [[nodiscard]] MockAMDGPULoader &getExecutable() const { return Parent; }

private:
  llvm::Error finalize();

  llvm::Error applyStaticRelocation(llvm::object::ELFRelocationRef Rel);

  llvm::Error applyDynamicRelocation(llvm::object::ELFRelocationRef Rel);
};

/// \brief a code loader for AMD GPU code objects that loads and dynamically
/// links multiple code objects onto host memory instead of device memory
class MockAMDGPULoader {
private:
  typedef llvm::StringMap<void *> ExternalSymbolMap;

  /// If true, loading of the code objects has been finalized
  bool IsFinalized{false};

  /// Mapping between the name of externally defined symbols
  ExternalSymbolMap ExternalSymbols;

  /// Loaded code objects managed by the executable
  llvm::SmallVector<std::unique_ptr<MockLoadedCodeObject>> LoadedCodeObjects;

public:
  MockAMDGPULoader() = default;

  ~MockAMDGPULoader() = default;

  /// Disallowed copy constructor
  MockAMDGPULoader(const MockAMDGPULoader &Loader) = delete;

  /// Disallowed copy assignment
  MockAMDGPULoader &operator=(const MockAMDGPULoader &Loader) = delete;

  /// \returns \c true if the loader has finalized loading all the code objects,
  /// \c false otherwise
  bool isFinalized() const { return IsFinalized; }

  /// Defines an externally defined symbol with a public binding at the
  /// specified <tt>Address</tt>; The defined symbol will be used to resolve
  /// the dynamic relocations of any undefined symbol in the loaded code objects
  /// managed by the loader during finalization
  /// \param Name name of the symbol being defined
  /// \param Address the address where the symbol resides
  /// \return \c llvm::ErrorSuccess on success
  /// \return \c llvm::Error if the symbol with the same name is already defined
  llvm::Error defineExternalSymbol(llvm::StringRef Name, void *Address);

  [[nodiscard]] ExternalSymbolMap::const_iterator
  external_symbol_begin() const {
    return ExternalSymbols.begin();
  }

  [[nodiscard]] ExternalSymbolMap::const_iterator external_symbol_end() const {
    return ExternalSymbols.end();
  }

  [[nodiscard]] llvm::iterator_range<ExternalSymbolMap::const_iterator>
  external_symbols() const {
    return llvm::make_range(external_symbol_begin(), external_symbol_end());
  }

  [[nodiscard]] unsigned external_symbol_size() const {
    return ExternalSymbols.size();
  }

  [[nodiscard]] ExternalSymbolMap::const_iterator
  findExternalSymbol(llvm::StringRef Name) const {
    return ExternalSymbols.find(Name);
  }

  /// Loads the \c CodeObject onto the host memory and creates an instance
  /// of \c MockHsaLoadedCodeObject managed by the loader
  /// \return the loaded code object handle on success; \c llvm::Error if an
  /// issue was encountered in the process
  llvm::Expected<const MockLoadedCodeObject &>
  loadCodeObject(llvm::ArrayRef<std::byte> CodeObject);

  /// Finalizes the loading of all the code objects loaded by this loader
  /// instance by resolving the dynamic relocations of all loaded code objects
  /// \returns \c llvm::Error indicating the success or failure of the
  /// operation
  llvm::Error finalize();

  /// Non-const function to iterate over the loaded code objects managed by
  /// the loader and invoke a \c C
  /// \tparam Callable a callable type that takes a non-const \c
  /// MockHsaLoadedCodeObject as the first parameter, and the \c Args as the
  /// subsequent parameters, and returns a \c llvm::Error indicating it success
  /// or failure
  /// \tparam ExtraArgs the additional argument types required to invoke \c
  /// Callable
  /// \param C the \c Callable to be invoked on every loaded code object managed
  /// by the loader
  /// \param Args the additional arguments of \p C
  /// \return \c llvm::Error indicating the success or failure of the iteration
  template <typename Callable, typename... ExtraArgs>
  llvm::Error iterateLoadedCodeObjects(Callable C, ExtraArgs... Args) {
    for (auto &LCO : LoadedCodeObjects) {
      LUTHIER_RETURN_ON_ERROR(C(*LCO, Args...));
    }
    return llvm::Error::success();
  }

  /// Const iterator function over the loaded code objects managed by the loader
  /// \see iterateLoadedCodeObjects
  template <typename Callable, typename... ExtraArgs>
  llvm::Error iterateLoadedCodeObjects(Callable C, ExtraArgs... Args) const {
    for (const auto &LCO : LoadedCodeObjects) {
      LUTHIER_RETURN_ON_ERROR(C(*LCO, Args...));
    }
    return llvm::Error::success();
  }

  /// Iterator function over the loaded code objects managed by the loader with
  /// no extra arguments
  /// \see iterateLoadedCodeObjects
  template <typename Callable>
  llvm::Error iterateLoadedCodeObjects(Callable C) {
    for (auto &LCO : LoadedCodeObjects) {
      LUTHIER_RETURN_ON_ERROR(C(*LCO));
    }
    return llvm::Error::success();
  }

  /// Const iterator function over the loaded code objects managed by the loader
  /// with no extra arguments
  /// \see iterateLoadedCodeObjects
  template <typename Callable>
  llvm::Error iterateLoadedCodeObjects(Callable C) const {
    for (const auto &LCO : LoadedCodeObjects) {
      LUTHIER_RETURN_ON_ERROR(C(*LCO));
    }
    return llvm::Error::success();
  }
};

} // namespace luthier

#endif
