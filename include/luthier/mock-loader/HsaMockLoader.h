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
///
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

class MockHsaKernelSymbol;
class VariableSymbol;

class MockHsaLoadedCodeObject;
class MockHsaExecutable;

/// \brief Represents a symbol loaded by the HSA mock loader
class MockHsaLoadedSymbol
    : public llvm::RTTIExtends<MockHsaLoadedSymbol, llvm::RTTIRoot> {

protected:
  MockHsaLoadedSymbol() = default;

public:
  static char ID;

  virtual ~MockHsaLoadedSymbol() = default;

  MockHsaLoadedSymbol(const MockHsaLoadedSymbol &S) = delete;

  MockHsaLoadedSymbol &operator=(const MockHsaLoadedSymbol &S) = delete;

  [[nodiscard]] virtual uint8_t getType() const = 0;

  [[nodiscard]] virtual llvm::StringRef getName() const = 0;

  [[nodiscard]] virtual std::byte *getAddress() const = 0;
};

class MockHsaLoadedCodeObjectSymbol
    : public llvm::RTTIExtends<MockHsaLoadedCodeObjectSymbol,
                               MockHsaLoadedSymbol> {
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

  [[nodiscard]] std::byte *getAddress() const override;

  [[nodiscard]] object::AMDGCNElfSymbolRef getSymbolRef() const {
    return DynSymRef;
  }
};

class MockExternalDefinitionSymbol
    : public llvm::RTTIExtends<MockExternalDefinitionSymbol,
                               MockHsaLoadedSymbol> {
private:
  std::string SymbolName;

  bool IsDefinition;

  uint64_t Address;

public:
protected:
  MockHsaLoadedSymbol(const hsa_symbol_kind_t &_kind,
                      llvm::StringRef SymbolName, bool IsDefinition,
                      uint64_t Address = 0)
      : Kind(_kind), SymbolName(SymbolName), IsDefinition(IsDefinition),
        Address(Address) {}

public:
  [[nodiscard]] llvm::StringRef getName() const { return SymbolName; }

  [[nodiscard]] bool isDefinition() const { return IsDefinition; }

  [[nodiscard]] uint64_t getAddress() const { return Address; }
};

//===----------------------------------------------------------------------===//
// Executable.                                                                //
//===----------------------------------------------------------------------===//

class MockHsaLoadedSegment;

class MockExecutableObject {
protected:
  MockHsaExecutable &Parent;

public:
  explicit MockExecutableObject(MockHsaExecutable &Parent) : Parent(Parent) {}

  MockHsaExecutable &Parent() const { return Parent; }

  virtual ~MockExecutableObject() = default;
};

class MockHsaLoadedCodeObject : public MockExecutableObject {
  friend class AmdHsaCodeLoader;

private:
  llvm::ArrayRef<std::byte> Elf;
  llvm::SmallVector<std::unique_ptr<MockHsaLoadedSegment>, 1> LoadedSegments;

public:
  MockHsaLoadedCodeObject(MockHsaExecutable *owner_, hsa_agent_t agent_,
                          const void *elf_data_, size_t elf_size_)
      : MockExecutableObject(owner_, agent_), elf_data(elf_data_),
        elf_size(elf_size_) {
    memset(&r_debug_info, 0, sizeof(r_debug_info));
  }

  MockHsaLoadedCodeObject(const MockHsaLoadedCodeObject &) = delete;

  MockHsaLoadedCodeObject &operator=(const MockHsaLoadedCodeObject &) = delete;

  llvm::ArrayRef<std::byte> getElf() const { return Elf; }

  std::vector<MockHsaLoadedSegment *> &LoadedSegments() {
    return loaded_segments;
  }

  bool GetInfo(amd_loaded_code_object_info_t attribute, void *value) override;

  hsa_executable_t getExecutable() const override;

  uint64_t getElfData() const override;

  uint64_t getElfSize() const override;

  uint64_t getStorageOffset() const override;

  uint64_t getLoadBase() const override;

  uint64_t getLoadSize() const override;

  int64_t getDelta() const override;

  std::string getUri() const override;

  link_map r_debug_info;
};

class MockHsaLoadedSegment : public MockExecutableObject {
private:
  amdgpu_hsa_elf_segment_t segment;
  void *ptr;
  size_t size;
  uint64_t vaddr;
  bool frozen;
  size_t storage_offset;

public:
  MockHsaLoadedSegment(MockHsaExecutable *owner_, hsa_agent_t agent_,
                       amdgpu_hsa_elf_segment_t segment_, void *ptr_,
                       size_t size_, uint64_t vaddr_, size_t storage_offset_)
      : MockExecutableObject(owner_, agent_), segment(segment_), ptr(ptr_),
        size(size_), vaddr(vaddr_), frozen(false),
        storage_offset(storage_offset_) {}

  amdgpu_hsa_elf_segment_t ElfSegment() const { return segment; }

  void *Ptr() const { return ptr; }

  size_t Size() const { return size; }

  uint64_t VAddr() const { return vaddr; }

  size_t StorageOffset() const { return storage_offset; }

  bool GetInfo(amd_loaded_segment_info_t attribute, void *value) override;

  uint64_t Offset(uint64_t addr); // Offset within segment. Used together with
                                  // ptr with loader context functions.

  void *Address(uint64_t addr); // Address in segment. Used for relocations and
                                // valid on agent.

  bool Freeze();

  bool IsAddressInSegment(uint64_t addr);

  void Copy(uint64_t addr, const void *src, size_t size);

  void Print(std::ostream &out) override;

  void Destroy() override;
};

class MockHsaSegmentMemory {
public:
  MockHsaSegmentMemory(const MockHsaSegmentMemory &) = delete;

  MockHsaSegmentMemory &operator=(const MockHsaSegmentMemory &) = delete;

  virtual ~MockHsaSegmentMemory() = default;

  [[nodiscard]] virtual std::byte *Address(size_t Offset = 0) const = 0;

  [[nodiscard]] virtual std::byte *HostAddress(size_t Offset = 0) const = 0;

  [[nodiscard]] virtual bool Allocated() const = 0;

  virtual llvm::Error Allocate(size_t size, size_t align, bool zero) = 0;

  virtual llvm::Error Copy(size_t offset, const void *src, size_t size) = 0;

  virtual void Free() = 0;

  virtual bool Freeze() = 0;

protected:
  MockHsaSegmentMemory() = default;
};

class MallocedMemory final : public MockHsaSegmentMemory {
public:
  MallocedMemory() = default;

  ~MallocedMemory() override = default;

  [[nodiscard]] std::byte *Address(size_t Offset) const override {
    assert(this->Allocated());
    return &ptr_[Offset];
  }

  void *HostAddress(size_t offset = 0) const override {
    return this->Address(offset);
  }

  bool Allocated() const override { return nullptr != ptr_; }

  llvm::Error Allocate(size_t size, size_t align, bool zero) override;

  llvm::Error Copy(size_t offset, const void *src, size_t size) override;

  void Free() override;

  bool Freeze() override;

private:
  MallocedMemory(const MallocedMemory &);
  MallocedMemory &operator=(const MallocedMemory &);

  std::byte *ptr_{nullptr};

  size_t size_{0};
};

class MockHsaLoaderContext {
public:
  MockHsaLoaderContext() = default;

  ~MockHsaLoaderContext() = default;

  hsa_isa_t IsaFromName(const char *name) override;

  bool IsaSupportedByAgent(hsa_agent_t agent, hsa_isa_t code_object_isa,
                           unsigned codeGenericVersion) override;

  void *SegmentAlloc(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                     size_t size, size_t align, bool zero) override;

  bool SegmentCopy(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                   void *dst, size_t offset, const void *src,
                   size_t size) override;

  void SegmentFree(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                   void *seg, size_t size = 0) override;

  void *SegmentAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                       void *seg, size_t offset) override;

  void *SegmentHostAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                           void *seg, size_t offset) override;

  bool SegmentFreeze(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                     void *seg, size_t size) override;

private:
  LoaderContext(const LoaderContext &);
  LoaderContext &operator=(const LoaderContext &);
};

class MockHsaExecutable {
  friend class AmdHsaCodeLoader;

public:
  typedef std::unordered_map<std::string, std::unique_ptr<MockHsaLoadedSymbol>>
      AgentSymbolMap;

  const hsa_executable_state_t &state() const { return state_; }

  MockHsaExecutable(
      Context *context, size_t id,
      hsa_default_float_rounding_mode_t default_float_rounding_mode);

  MockHsaExecutable(
      std::unique_ptr<Context> unique_context, size_t id,
      hsa_default_float_rounding_mode_t default_float_rounding_mode);

  ~MockHsaExecutable() override;

  hsa_status_t GetInfo(hsa_executable_info_t executable_info,
                       void *value) override;

  hsa_status_t defineAgentExternalVariable(llvm::StringRef name,
                                           hsa_agent_t agent,
                                           hsa_variable_segment_t segment,
                                           void *address) override;

  llvm::Error
  LoadCodeObject(hsa_agent_t agent, hsa_code_object_t code_object,
                 const char *options, const std::string &uri,
                 hsa_loaded_code_object_t *loaded_code_object) override;

  llvm::Expected<const MockHsaLoadedCodeObject &>
  loadCodeObject(llvm::ArrayRef<std::byte> CodeObject,
                 const std::string &uri) override;

  llvm::Error freeze() override;

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

  Context *context() { return context_; }

  size_t id() { return id_; }

  MockHsaExecutable(const MockHsaExecutable &e) = delete;

  MockHsaExecutable &operator=(const MockHsaExecutable &e) = delete;

private:
  std::unique_ptr<amd::hsa::code::AmdHsaCode> code;

  Symbol *GetSymbolInternal(const char *symbol_name, const hsa_agent_t *agent);

  hsa_status_t LoadSegments(hsa_agent_t agent, const code::AmdHsaCode *c,
                            uint32_t majorVersion);

  hsa_status_t LoadSegmentsV2(hsa_agent_t agent, const code::AmdHsaCode *c);

  hsa_status_t LoadSegmentV2(const code::Segment *data_segment,
                             loader::Segment *load_segment);

  hsa_status_t LoadSymbol(hsa_agent_t agent, amd::hsa::code::Symbol *sym,
                          uint32_t majorVersion);

  hsa_status_t LoadDefinitionSymbol(hsa_agent_t agent,
                                    amd::hsa::code::Symbol *sym,
                                    uint32_t majorVersion);

  hsa_status_t LoadDeclarationSymbol(hsa_agent_t agent,
                                     amd::hsa::code::Symbol *sym,
                                     uint32_t majorVersion);

  hsa_status_t ApplyRelocations(hsa_agent_t agent,
                                amd::hsa::code::AmdHsaCode *c);
  hsa_status_t
  ApplyStaticRelocationSection(hsa_agent_t agent,
                               amd::hsa::code::RelocationSection *sec);
  hsa_status_t ApplyStaticRelocation(hsa_agent_t agent,
                                     amd::hsa::code::Relocation *rel);
  hsa_status_t
  ApplyDynamicRelocationSection(hsa_agent_t agent,
                                amd::hsa::code::RelocationSection *sec);

  hsa_status_t ApplyDynamicRelocation(hsa_agent_t agent,
                                      amd::hsa::code::Relocation *rel);

  MockHsaLoadedSegment *VirtualAddressSegment(uint64_t vaddr);

  uint64_t SymbolAddress(amd::hsa::code::Symbol *sym);

  uint64_t SymbolAddress(amd::elf::Symbol *sym);

  MockHsaLoadedSegment *SymbolSegment(amd::hsa::code::Symbol *sym);

  MockHsaLoadedSegment *SectionSegment(amd::hsa::code::Section *sec);

  amd::hsa::common::ReaderWriterLock rw_lock_;
  MockHsaLoaderContext &Context;
  hsa_executable_state_t state_;

  AgentSymbolMap agent_symbols_;
  std::vector<MockExecutableObject *> objects;
  std::vector<std::unique_ptr<MockHsaLoadedCodeObject>> loaded_code_objects;
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
