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
#ifndef LUTHIER_TOOLING_MOCK_AMDGPU_LOADER_H
#define LUTHIER_TOOLING_MOCK_AMDGPU_LOADER_H

#include "luthier/Object/AMDGCNObjectFile.h"
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/PassManager.h>
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
  llvm::MutableArrayRef<std::byte> LoadedRegion{};

  /// List of load segments of the ELF, sorted w.r.t their virtual address
  llvm::SmallVector<std::reference_wrapper<const llvm::object::ELF64LE::Phdr>,
                    4>
      PTLoadSegments{};

  MockLoadedCodeObject(MockAMDGPULoader &Owner, const llvm::MemoryBuffer &Elf,
                       llvm::Error &Err);

public:
  MockLoadedCodeObject(const MockLoadedCodeObject &) = delete;

  MockLoadedCodeObject &operator=(const MockLoadedCodeObject &) = delete;

  ~MockLoadedCodeObject() {
    if (LoadedRegion.data())
      delete LoadedRegion.data();
  }

  [[nodiscard]] const object::AMDGCNObjectFile &getCodeObject() const {
    return *Elf;
  }

  llvm::ArrayRef<std::byte> getLoadedRegion() const { return LoadedRegion; }

  [[nodiscard]] MockAMDGPULoader &getExecutable() const { return Parent; }

  llvm::ArrayRef<std::reference_wrapper<const llvm::object::ELF64LE::Phdr>>
  getLoadSegments() const {
    return PTLoadSegments;
  }

private:
  llvm::Error finalize();

  llvm::Error applyRelocation(llvm::object::ELFRelocationRef Rel);
};

/// \brief a code loader for AMD GPU code objects that loads and dynamically
/// links multiple code objects onto host memory instead of device memory
class MockAMDGPULoader {
private:
  typedef llvm::StringMap<void *> ExternalSymbolMap;

  typedef llvm::SmallVector<std::unique_ptr<MockLoadedCodeObject>> LCOVector;

  /// If true, loading of the code objects has been finalized
  bool IsFinalized{false};

  /// Mapping between the name of externally defined symbols
  ExternalSymbolMap ExternalSymbols;

  /// Loaded code objects managed by the executable
  LCOVector LoadedCodeObjects;

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
  loadCodeObject(const llvm::MemoryBuffer &CodeObject);

  /// Finalizes the loading of all the code objects loaded by this loader
  /// instance by resolving the dynamic relocations of all loaded code objects
  /// \returns \c llvm::Error indicating the success or failure of the
  /// operation
  llvm::Error finalize();

  class loaded_code_object_iterator {
    LCOVector::iterator It;

  public:
    explicit loaded_code_object_iterator(LCOVector::iterator It) : It(It) {}

    MockLoadedCodeObject &operator*() const { return **It; }

    loaded_code_object_iterator &operator++() {
      ++It;
      return *this;
    }

    loaded_code_object_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const loaded_code_object_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const loaded_code_object_iterator &Other) const {
      return !(*this == Other);
    }
  };

  class const_loaded_code_object_iterator {
    LCOVector::const_iterator It;

  public:
    explicit const_loaded_code_object_iterator(LCOVector::const_iterator It)
        : It(It) {}

    const MockLoadedCodeObject &operator*() const { return **It; }

    const_loaded_code_object_iterator &operator++() {
      ++It;
      return *this;
    }

    const_loaded_code_object_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const const_loaded_code_object_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const const_loaded_code_object_iterator &Other) const {
      return !(*this == Other);
    }
  };

  loaded_code_object_iterator loaded_code_objects_begin() {
    return loaded_code_object_iterator(LoadedCodeObjects.begin());
  }

  loaded_code_object_iterator loaded_code_objects_end() {
    return loaded_code_object_iterator(LoadedCodeObjects.begin());
  }

  llvm::iterator_range<loaded_code_object_iterator> loaded_code_objects() {
    return llvm::make_range(loaded_code_objects_begin(),
                            loaded_code_objects_end());
  }

  [[nodiscard]] const_loaded_code_object_iterator
  loaded_code_objects_begin() const {
    return const_loaded_code_object_iterator(LoadedCodeObjects.begin());
  }

  [[nodiscard]] const_loaded_code_object_iterator
  loaded_code_objects_end() const {
    return const_loaded_code_object_iterator(LoadedCodeObjects.end());
  }

  [[nodiscard]] llvm::iterator_range<const_loaded_code_object_iterator>
  loaded_code_objects() const {
    return llvm::make_range(loaded_code_objects_begin(),
                            loaded_code_objects_end());
  }

  [[nodiscard]] unsigned loaded_code_objects_size() const {
    return LoadedCodeObjects.size();
  }
};

class MockAMDGPULoaderAnalysis
    : public llvm::AnalysisInfoMixin<MockAMDGPULoaderAnalysis> {
  friend llvm::AnalysisInfoMixin<MockAMDGPULoaderAnalysis>;

  MockAMDGPULoader &Loader;

public:
  static llvm::AnalysisKey Key;

  class Result {
    friend class MockAMDGPULoaderAnalysis;

    MockAMDGPULoader &Loader;

    explicit Result(MockAMDGPULoader &Loader) : Loader(Loader) {}

  public:
    /// Never invalidate the loader
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    const MockAMDGPULoader &getLoader() const { return Loader; }

    MockAMDGPULoader &getLoader() { return Loader; }
  };

  explicit MockAMDGPULoaderAnalysis(MockAMDGPULoader &Loader)
      : Loader(Loader) {}

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result(Loader);
  }
};

} // namespace luthier

#endif
