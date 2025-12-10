//===-- HsaMockLoader.h -----------------------------------------*- C++ -*-===//
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
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file Contains definitions used to implement the HSA mock code loader used
/// to test the lifting passes. Adapted from the HSA runtime source code.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_MOCK_LOADER_HSA_MOCK_LOADER_H
#define LUTHIER_MOCK_LOADER_HSA_MOCK_LOADER_H

#include "amd_hsa_locks.hpp"
#include "luthier/mock-loader/amd_hsa_code.hpp"
#include "luthier/mock-loader/amd_hsa_loader.hpp"
#include "luthier/object/AMDGCNObjectFile.h"
#include <cassert>
#include <cstring>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <list>
#include <llvm/Support/Error.h>
#include <llvm/Support/ExtensibleRTTI.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace luthier {

class MockHsaLoadedCodeObject;
class MockHsaExecutable;

//===----------------------------------------------------------------------===//
// Mock HSA Loaded Symbol                                                     //
//===----------------------------------------------------------------------===//

/// \brief Represents a symbol definition defined and loaded by an HSA mock
/// executable
class MockHsaDefinedSymbol
    : public llvm::RTTIExtends<MockHsaDefinedSymbol, llvm::RTTIRoot> {

protected:
  MockHsaDefinedSymbol() = default;

public:
  static char ID;

  MockHsaDefinedSymbol(const MockHsaDefinedSymbol &S) = delete;

  MockHsaDefinedSymbol &operator=(const MockHsaDefinedSymbol &S) = delete;

  [[nodiscard]] virtual uint8_t getType() const = 0;

  [[nodiscard]] virtual llvm::StringRef getName() const = 0;

  [[nodiscard]] virtual uint64_t getAddress() const = 0;
};

/// \brief Represents a symbol in an executable that has been defined by a
/// loaded code object
class MockHsaLoadedCodeObjectSymbol
    : public llvm::RTTIExtends<MockHsaLoadedCodeObjectSymbol,
                               MockHsaDefinedSymbol> {
private:
  friend MockHsaLoadedCodeObject;

  /// The loaded code object this symbol belongs to
  MockHsaLoadedCodeObject &Parent;

  /// The symbol ref that defines this symbol
  object::AMDGCNElfSymbolRef DynSymRef;

  MockHsaLoadedCodeObjectSymbol(MockHsaLoadedCodeObject &Parent,
                                object::AMDGCNElfSymbolRef DynSymRef)
      : Parent(Parent), DynSymRef(DynSymRef) {}

public:
  static char ID;

  [[nodiscard]] uint8_t getType() const override {
    return DynSymRef.getELFType();
  }

  [[nodiscard]] llvm::StringRef getName() const override {
    return llvm::cantFail(DynSymRef.getName());
  }

  [[nodiscard]] uint64_t getAddress() const override {
    return llvm::cantFail(DynSymRef.getAddress());
  }

  [[nodiscard]] object::AMDGCNElfSymbolRef getSymbolRef() const {
    return DynSymRef;
  }
};

class MockExternalDefinitionSymbol
    : public llvm::RTTIExtends<MockExternalDefinitionSymbol,
                               MockHsaDefinedSymbol> {
private:
  std::string SymbolName;

  uint64_t Address;

public:
protected:
  MockExternalDefinitionSymbol(llvm::StringRef SymbolName, uint64_t Address)
      : SymbolName(SymbolName), Address(Address) {}

public:
  [[nodiscard]] llvm::StringRef getName() const override { return SymbolName; }

  [[nodiscard]] uint64_t getAddress() const override { return Address; }
};

//===----------------------------------------------------------------------===//
// Executable.                                                                //
//===----------------------------------------------------------------------===//

class MockHsaLoadedRegion;

class MockHsaLoadedCodeObject {
  friend class MockHsaExecutable;

private:
  /// The executable this loaded code object belongs to
  MockHsaExecutable &Parent;

  /// The AMDGCN object file of the ELF loaded; The code object itself is
  /// externally managed
  std::unique_ptr<object::AMDGCNObjectFile> Elf;

  /// A contiguous memory region where the code object is loaded
  std::unique_ptr<MockHsaLoadedRegion> LoadedRegion;

  MockHsaLoadedCodeObject(MockHsaExecutable &Owner,
                          llvm::ArrayRef<std::byte> Elf, llvm::Error Err);

public:
  MockHsaLoadedCodeObject(const MockHsaLoadedCodeObject &) = delete;

  MockHsaLoadedCodeObject &operator=(const MockHsaLoadedCodeObject &) = delete;

  object::AMDGCNObjectFile &getCodeObject() { return *Elf; }

  const MockHsaLoadedRegion &getLoadedSegments() const {
    return *this->LoadedRegion;
  }

  MockHsaExecutable getExecutable() const;

  llvm::ArrayRef<std::byte> getELF() const;

  llvm::ArrayRef<std::byte> getLoadedRegion() const;

  uint64_t getStorageOffset() const;

  int64_t getDelta() const;

private:
  llvm::Error LoadSegments();

  hsa_status_t LoadSegmentV2(const code::Segment *data_segment,
                             loader::Segment *load_segment);

  hsa_status_t LoadSymbol(amd::hsa::code::Symbol *sym, uint32_t majorVersion);

  hsa_status_t LoadDefinitionSymbol(amd::hsa::code::Symbol *sym,
                                    uint32_t majorVersion);

  hsa_status_t LoadDeclarationSymbol(rocr::amd::hsa::code::Symbol *sym,
                                     uint32_t majorVersion);

  /// Applies relocations to the loaded code object
  llvm::Error ApplyRelocations();

  llvm::Error ApplyStaticRelocationSection(llvm::object::ELFRelocationRef sec);

  llvm::Error ApplyStaticRelocation(llvm::object::ELFRelocationRef Rel);
  hsa_status_t
  ApplyDynamicRelocationSection(rocr::amd::hsa::code::RelocationSection *sec);

  hsa_status_t ApplyDynamicRelocation(rocr::amd::hsa::code::Relocation *rel);

  MockHsaLoadedRegion *VirtualAddressSegment(uint64_t vaddr);

  uint64_t SymbolAddress(rocr::amd::hsa::code::Symbol *sym);

  uint64_t SymbolAddress(rocr::amd::elf::Symbol *sym);

  llvm::Error freeze();
};

/// \brief a contiguous region of memory where a code object is loaded by the
/// mock loaded code object
class MockHsaLoadedRegion {
private:
  friend MockHsaLoadedCodeObject;

  /// The loaded code object this region belongs to
  MockHsaLoadedCodeObject &Parent;

  /// Base pointer of the allocation
  std::byte *Ptr;

  /// Size of the allocation
  size_t Size;

  /// The base virtual address of the region
  uint64_t BaseVAddr;

  ///
  size_t storage_offset;

public:
  MockHsaLoadedRegion(MockHsaLoadedCodeObject &Parent, size_t Size,
                      size_t Alignment, uint64_t VAddr, size_t Offset,
                      llvm::Error &Err, bool Zero = true);

  ~MockHsaLoadedRegion();

  void *getPtr() const { return Ptr; }

  size_t getSize() const { return Size; }

  uint64_t VAddr() const { return vaddr; }

  size_t StorageOffset() const { return storage_offset; }

  uint64_t Offset(uint64_t addr); // Offset within segment. Used together with
                                  // ptr with loader context functions.

  void *Address(uint64_t addr); // Address in segment. Used for relocations and
                                // valid on agent.

  bool Freeze();

  bool IsAddressInSegment(uint64_t addr);

  void Copy(uint64_t addr, const void *src, size_t size);
};

/// \brief Object managing the memory segments used by the HSA mock loader
class HsaMockSegmentMemory {

  ///
  std::byte *ptr_{nullptr};

  size_t size_{0};

public:
  HsaMockSegmentMemory() = default;

  ~HsaMockSegmentMemory() override = default;

  [[nodiscard]] std::byte *getAddress(size_t Offset) const override {
    assert(this->Allocated());
    return &ptr_[Offset];
  }

  void *HostAddress(size_t offset = 0) const override {
    return this->getAddress(offset);
  }

  bool Allocated() const override { return nullptr != ptr_; }

  llvm::Error Allocate(size_t size, size_t align, bool zero) override;

  llvm::Error Copy(size_t offset, const void *src, size_t size) override;

  void Free() override;

  bool Freeze() override;

private:
  HsaMockSegmentMemory(const HsaMockSegmentMemory &);
  HsaMockSegmentMemory &operator=(const HsaMockSegmentMemory &);
};

class MockHsaExecutable {
  friend class MockHsaLoader;

private:
  typedef std::unordered_map<std::string, std::unique_ptr<MockHsaDefinedSymbol>>
      AgentSymbolMap;

  rocr::amd::hsa::common::ReaderWriterLock rw_lock_;

  hsa_executable_state_t state_{HSA_EXECUTABLE_STATE_UNFROZEN};

  AgentSymbolMap agent_symbols_;

  /// The AMDGPU ELF Machine code of the device being mocked; It can either be
  /// a generic or a specific ISA
  unsigned DeviceMach;

  /// Subtarget features supported by the mock device
  llvm::SubtargetFeatures Features;

  /// Loaded code objects managed by the executable
  llvm::SmallVector<std::unique_ptr<MockHsaLoadedCodeObject>> LoadedCodeObjects;

public:
  MockHsaExecutable(unsigned AMDGPUElfMach, llvm::SubtargetFeatures Features)
      : DeviceMach(AMDGPUElfMach), Features(std::move(Features)) {}

  ~MockHsaExecutable();

  const hsa_executable_state_t &state() const { return state_; }

  unsigned getEMach() const { return DeviceMach; }

  hsa_status_t defineAgentExternalVariable(llvm::StringRef name,
                                           hsa_agent_t agent,
                                           hsa_variable_segment_t segment,
                                           void *address);

  llvm::Expected<const MockHsaLoadedCodeObject &>
  loadCodeObject(llvm::ArrayRef<std::byte> CodeObject, const std::string &uri);

  llvm::Error freeze();

  Symbol *GetSymbol(const char *symbol_name, const hsa_agent_t *agent) override;

  hsa_status_t IterateSymbols(iterate_symbols_f callback, void *data) override;

  /// @since hsa v1.1.
  hsa_status_t IterateAgentSymbols(
      hsa_agent_t agent,
      hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent,
                               hsa_executable_symbol_t symbol, void *data),
      void *data) override;

  hsa_status_t IterateLoadedCodeObjects(
      hsa_status_t (*callback)(hsa_executable_t executable,
                               hsa_loaded_code_object_t loaded_code_object,
                               void *data),
      void *data) override;

  size_t GetNumSegmentDescriptors() override;

  size_t QuerySegmentDescriptors(
      hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
      size_t total_num_segment_descriptors,
      size_t first_empty_segment_descriptor) override;

  uint64_t FindHostAddress(uint64_t device_address) override;

  void EnableReadOnlyMode();

  void DisableReadOnlyMode();

  MockHsaExecutable(const MockHsaExecutable &e) = delete;

  MockHsaExecutable &operator=(const MockHsaExecutable &e) = delete;

private:
  Symbol *GetSymbolInternal(const char *symbol_name, const hsa_agent_t *agent);
};

class MockHsaLoader {
private:
  MockHsaLoaderContext &Ctx;
  std::vector<std::unique_ptr<MockHsaExecutable>> executables;
  amd::hsa::common::ReaderWriterLock rw_lock_;

public:
  explicit MockHsaLoader(MockHsaLoaderContext &Ctx) : Ctx(Ctx) {}

  MockHsaLoaderContext &GetContext() const { return Ctx; }

  std::unique_ptr<MockHsaExecutable> CreateExecutable();

  void DestroyExecutable(Executable *executable) override;

  hsa_status_t IterateExecutables(
      hsa_status_t (*callback)(hsa_executable_t executable, void *data),
      void *data) override;

  hsa_status_t QuerySegmentDescriptors(
      hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
      size_t *num_segment_descriptors) override;

  MockHsaExecutable *FindExecutable(uint64_t device_address);

  uint64_t FindHostAddress(uint64_t device_address) override;

  void EnableReadOnlyMode();
  void DisableReadOnlyMode();
};

} // namespace luthier

#endif // HSA_RUNTIME_CORE_LOADER_EXECUTABLE_HPP_
