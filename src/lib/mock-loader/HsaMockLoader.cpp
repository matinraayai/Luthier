////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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
////////////////////////////////////////////////////////////////////////////////

#include <libelf.h>
#include <limits.h>
#include <llvm/IR/InlineAsm.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/GenericLuthierError.h>
#include <luthier/hsa/HsaError.h>
#include <luthier/hsa/ISA.h>
#include <luthier/mock-loader/amd_hsa_code_util.hpp>
#if defined(__linux__)
#include <link.h>
#include <unistd.h>
#else
#include <cstdint>
#endif

#include "amd_hsa_code_util.hpp"
#include "amd_options.hpp"
#include "core/inc/amd_hsa_code.hpp"
#include "core/util/utils.h"
#include "hsa/amd_hsa_elf.h"
#include "hsa/amd_hsa_kernel_code.h"
#include "luthier/mock-loader/HsaMockLoader.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llvm/Support/AMDHSAKernelDescriptor.h>

using namespace rocr::amd::hsa;
using namespace rocr::amd::hsa::common;

namespace luthier {

MockHsaLoadedCodeObject::MockHsaLoadedCodeObject(MockHsaExecutable &Owner,
                                                 llvm::ArrayRef<std::byte> Elf,
                                                 llvm::Error Err)
    : Parent(Owner) {
  llvm::ErrorAsOutParameter EAO(Err);

  /// Parse the code object
  Err = object::AMDGCNObjectFile::createAMDGCNObjectFile(
            llvm::StringRef(reinterpret_cast<const char *>(Elf.data()),
                            Elf.size()))
            .moveInto(this->Elf);
  if (Err)
    return;

  /// Cast to object::ELFObjectFileBase since for some reason methods for
  /// querying the ELF EMachine and the ABI versions are private in the
  /// little endian 64-bit sub-class version
  auto &ElfBase = llvm::cast<llvm::object::ELFObjectFileBase>(*this->Elf);

  unsigned CodeObjectMach = ElfBase.getEMachine();

  /// Check if the ISA of the code object and the executable are compatible
  /// TODO: check XNACK and SRAMECC fields in the AMDGPU object file
  if (CodeObjectMach != Owner.getEMach()) {
    unsigned GenericCodeObjectMach =
        object::getGenericAMDGPUMach(CodeObjectMach);
    unsigned GenericExecutableMach =
        object::getGenericAMDGPUMach(CodeObjectMach);
    if (GenericCodeObjectMach != GenericExecutableMach) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(
          "The code object's e-machine is not compatible with the executable's "
          "e-machine");
      return;
    }
  }

  uint8_t CodeObjectVersion = ElfBase.getEIdentABIVersion();

  if (CodeObjectVersion < llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V2 ||
      CodeObjectVersion > llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V6) {
    Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Unsupported code object version {0}", CodeObjectVersion + 2));
    return;
  }

  Err = LoadSegments();
  if (Err)
    return;

  /// Add the

  for (size_t i = 0; i < code->SymbolCount(); ++i) {
    if (majorVersion >= 2 &&
        code->GetSymbol(i)->elfSym()->type() != STT_AMDGPU_HSA_KERNEL &&
        code->GetSymbol(i)->elfSym()->binding() == STB_LOCAL)
      continue;

    status = LoadSymbol(code->GetSymbol(i), majorVersion);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  status = ApplyRelocations(code.get());
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  code.reset();

  return loaded_code_objects.back();
}

llvm::Error HsaMockSegmentMemory::Allocate(size_t Size, size_t align,
                                           bool zero) {
  ptr_ = new (std::align_val_t{align}, std::nothrow) std::byte[Size];
  if (nullptr == ptr_) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Failed to allocate {0} bytes of memory", Size));
  }
  if (zero) {
    std::memset(ptr_, 0, Size);
  }
  size_ = Size;
  return llvm::Error::success();
}

llvm::Error HsaMockSegmentMemory::Copy(size_t offset, const void *src,
                                       size_t size) {
  assert(this->Allocated());
  assert(nullptr != src);
  assert(0 < size);
  std::memcpy(this->getAddress(offset), src, size);
  return llvm::Error::success();
}

void HsaMockSegmentMemory::Free() {
  assert(this->Allocated());
  delete[] ptr_;
  ptr_ = nullptr;
  size_ = 0;
}

bool HsaMockSegmentMemory::Freeze() {
  assert(this->Allocated());
  return true;
}

void *MockHsaLoaderContext::SegmentAlloc(amdgpu_hsa_elf_segment_t segment,
                                         hsa_agent_t agent, size_t size,
                                         size_t align, bool zero) {
  assert(0 < size);
  assert(0 < align && 0 == (align & (align - 1)));

  HsaMockSegmentMemory *Mem = new (std::nothrow) HsaMockSegmentMemory();

  if (nullptr == Mem) {
    return nullptr;
  }

  if (!Mem->Allocate(size, align, zero)) {
    delete Mem;
    return nullptr;
  }

  return Mem;
}

bool MockHsaLoaderContext::SegmentCopy(
    amdgpu_hsa_elf_segment_t segment, // not used.
    hsa_agent_t agent,                // not used.
    void *dst, size_t offset, const void *src, size_t size) {
  assert(nullptr != dst);
  return ((HsaMockSegmentMemory *)dst)->Copy(offset, src, size);
}

void *MockHsaLoaderContext::SegmentAddress(
    amdgpu_hsa_elf_segment_t segment, // not used.
    hsa_agent_t agent,                // not used.
    void *seg, size_t offset) {
  assert(nullptr != seg);
  return ((HsaMockSegmentMemory *)seg)->getAddress(offset);
}

void *MockHsaLoaderContext::SegmentHostAddress(
    amdgpu_hsa_elf_segment_t segment, // not used.
    hsa_agent_t agent,                // not used.
    void *seg, size_t offset) {
  assert(nullptr != seg);
  return ((HsaMockSegmentMemory *)seg)->HostAddress(offset);
}

bool MockHsaLoaderContext::SegmentFreeze(
    amdgpu_hsa_elf_segment_t segment, // not used.
    hsa_agent_t agent,                // not used.
    void *seg,
    size_t size) // not used.
{
  assert(nullptr != seg);
  return ((HsaMockSegmentMemory *)seg)->Freeze();
}

Loader *Loader::Create(Context *context) { return new MockHsaLoader(context); }

void Loader::Destroy(Loader *loader) { delete loader; }

std::unique_ptr<MockHsaExecutable> MockHsaLoader::CreateExecutable() {
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);

  executables.push_back(new ExecutableImpl(profile, context, executables.size(),
                                           default_float_rounding_mode));
  return executables.back();
}

Executable *MockHsaLoader::CreateExecutable(
    std::unique_ptr<Ctx> isolated_context, hsa_profile_t profile,
    const char *options,
    hsa_default_float_rounding_mode_t default_float_rounding_mode) {
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);

  executables.push_back(new ExecutableImpl(profile, std::move(isolated_context),
                                           executables.size(),
                                           default_float_rounding_mode));
  return executables.back();
}

hsa_status_t MockHsaLoader::FreezeExecutable(Executable *executable,
                                             const char *options) {
  hsa_status_t status = executable->Freeze(options);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  return HSA_STATUS_SUCCESS;
}

void MockHsaLoader::DestroyExecutable(Executable *executable) {

  executables[((ExecutableImpl *)executable)->id()] = nullptr;
  delete executable;
}

hsa_status_t MockHsaLoader::IterateExecutables(
    hsa_status_t (*callback)(hsa_executable_t executable, void *data),
    void *data) {
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);
  assert(callback);

  for (auto &exec : executables) {
    if (exec != nullptr) {
      hsa_status_t status = callback(Executable::Handle(exec), data);
      if (status != HSA_STATUS_SUCCESS) {
        return status;
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MockHsaLoader::QuerySegmentDescriptors(
    hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
    size_t *num_segment_descriptors) {
  if (!num_segment_descriptors) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (*num_segment_descriptors == 0 && segment_descriptors) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (*num_segment_descriptors != 0 && !segment_descriptors) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  this->EnableReadOnlyMode();

  size_t actual_num_segment_descriptors = 0;
  for (auto &executable : executables) {
    if (executable) {
      actual_num_segment_descriptors += executable->GetNumSegmentDescriptors();
    }
  }

  if (*num_segment_descriptors == 0) {
    *num_segment_descriptors = actual_num_segment_descriptors;
    this->DisableReadOnlyMode();
    return HSA_STATUS_SUCCESS;
  }
  if (*num_segment_descriptors != actual_num_segment_descriptors) {
    this->DisableReadOnlyMode();
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }

  size_t i = 0;
  for (auto &executable : executables) {
    if (executable) {
      i += executable->QuerySegmentDescriptors(
          segment_descriptors, actual_num_segment_descriptors, i);
    }
  }

  this->DisableReadOnlyMode();
  return HSA_STATUS_SUCCESS;
}

uint64_t MockHsaLoader::FindHostAddress(uint64_t device_address) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  if (device_address == 0) {
    return 0;
  }

  for (auto &exec : executables) {
    if (exec != nullptr) {
      uint64_t host_address = exec->FindHostAddress(device_address);
      if (host_address != 0) {
        return host_address;
      }
    }
  }
  return 0;
}

void MockHsaLoader::EnableReadOnlyMode() {
  rw_lock_.ReaderLock();
  for (auto &executable : executables) {
    if (executable) {
      ((ExecutableImpl *)executable)->EnableReadOnlyMode();
    }
  }
}

void MockHsaLoader::DisableReadOnlyMode() {
  rw_lock_.ReaderUnlock();
  for (auto &executable : executables) {
    if (executable) {
      ((ExecutableImpl *)executable)->DisableReadOnlyMode();
    }
  }
}

uint64_t MockHsaLoadedRegion::Offset(uint64_t addr) {
  assert(IsAddressInSegment(addr));
  return addr - vaddr;
}

void *MockHsaLoadedRegion::Address(uint64_t addr) {
  return owner->context()->SegmentAddress(segment, agent, ptr, Offset(addr));
}

bool MockHsaLoadedRegion::Freeze() {
  return !frozen ? (frozen = owner->context()->SegmentFreeze(segment, agent,
                                                             ptr, size))
                 : true;
}

bool MockHsaLoadedRegion::IsAddressInSegment(uint64_t addr) {
  return vaddr <= addr && addr < vaddr + size;
}

void MockHsaLoadedRegion::Copy(uint64_t addr, const void *src, size_t size) {
  // loader must do copies before freezing.

  if (size > 0) {
    owner->context()->SegmentCopy(segment, agent, ptr, Offset(addr), src, size);
    std::memcpy(this->getAddress(Offset(addr), src, size);
  }
}

//===----------------------------------------------------------------------===//
// ExecutableImpl. //
//===----------------------------------------------------------------------===//

ExecutableImpl::ExecutableImpl(
    Context *context, size_t id,
    hsa_default_float_rounding_mode_t default_float_rounding_mode)
    : Executable(), context_(context), id_(id),
      default_float_rounding_mode_(default_float_rounding_mode),
      state_(HSA_EXECUTABLE_STATE_UNFROZEN),
      program_allocation_segment(nullptr) {}

ExecutableImpl::ExecutableImpl(
    std::unique_ptr<Context> unique_context, size_t id,
    hsa_default_float_rounding_mode_t default_float_rounding_mode)
    : Executable(), unique_context_(std::move(unique_context)), id_(id),
      default_float_rounding_mode_(default_float_rounding_mode),
      state_(HSA_EXECUTABLE_STATE_UNFROZEN),
      program_allocation_segment(nullptr) {
  context_ = unique_context_.get();
}

ExecutableImpl::~ExecutableImpl() {

  for (auto &symbol_entry : agent_symbols_) {
    delete symbol_entry.second;
  }
}

hsa_status_t ExecutableImpl::defineAgentExternalVariable(
    llvm::StringRef name, hsa_agent_t agent, hsa_variable_segment_t segment,
    void *address) {
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);

  if (HSA_EXECUTABLE_STATE_FROZEN == state_) {
    return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;
  }

  auto symbol_entry =
      agent_symbols_.find(std::make_pair(std::string(name), agent));
  if (symbol_entry != agent_symbols_.end()) {
    return HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED;
  }

  auto insert_status = agent_symbols_.insert(std::make_pair(
      std::make_pair(std::string(name), agent),
      new VariableSymbol(true,
                         "", // Only program linkage symbols can be
                             // defined.
                         std::string(name), HSA_SYMBOL_LINKAGE_PROGRAM, true,
                         HSA_VARIABLE_ALLOCATION_AGENT, segment,
                         0,     // TODO: size.
                         0,     // TODO: align.
                         false, // TODO: const.
                         true, reinterpret_cast<uint64_t>(address))));
  assert(insert_status.second);
  insert_status.first->second->agent = agent;

  return HSA_STATUS_SUCCESS;
}

Symbol *ExecutableImpl::GetSymbol(const char *symbol_name,
                                  const hsa_agent_t *agent) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  return this->GetSymbolInternal(symbol_name, agent);
}

Symbol *ExecutableImpl::GetSymbolInternal(const char *symbol_name,
                                          const hsa_agent_t *agent) {
  assert(symbol_name);

  std::string mangled_name = std::string(symbol_name);
  if (mangled_name.empty()) {
    return nullptr;
  }

  auto agent_symbol = agent_symbols_.find(std::make_pair(mangled_name, *agent));
  if (agent_symbol != agent_symbols_.end()) {
    return agent_symbol->second;
  }
  return nullptr;
}

hsa_status_t ExecutableImpl::IterateSymbols(iterate_symbols_f callback,
                                            void *data) {
  ReaderLockGuard<ReaderWriterLock> ReaderLock(rw_lock_);
  assert(callback);

  for (auto &symbol_entry : agent_symbols_) {
    hsa_status_t hsc = callback(Executable::Handle(this),
                                Symbol::Handle(symbol_entry.second), data);
    if (HSA_STATUS_SUCCESS != hsc) {
      return hsc;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::IterateAgentSymbols(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent,
                             hsa_executable_symbol_t symbol, void *data),
    void *data) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  assert(callback);

  for (auto &symbol_entry : agent_symbols_) {
    if (symbol_entry.second->GetAgent().handle != agent.handle) {
      continue;
    }

    hsa_status_t status = callback(Executable::Handle(this), agent,
                                   Symbol::Handle(symbol_entry.second), data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::IterateLoadedCodeObjects(
    hsa_status_t (*callback)(hsa_executable_t executable,
                             hsa_loaded_code_object_t loaded_code_object,
                             void *data),
    void *data) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  assert(callback);

  for (auto &loaded_code_object : loaded_code_objects) {
    hsa_status_t status =
        callback(Executable::Handle(this),
                 LoadedCodeObject::Handle(loaded_code_object), data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  return HSA_STATUS_SUCCESS;
}

size_t ExecutableImpl::GetNumSegmentDescriptors() {
  // assuming we are in readonly mode.
  size_t actual_num_segment_descriptors = 0;
  for (auto &obj : loaded_code_objects) {
    actual_num_segment_descriptors += obj->LoadedSegments().size();
  }
  return actual_num_segment_descriptors;
}

size_t ExecutableImpl::QuerySegmentDescriptors(
    hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
    size_t total_num_segment_descriptors,
    size_t first_empty_segment_descriptor) {
  // assuming we are in readonly mode.
  assert(segment_descriptors);
  assert(first_empty_segment_descriptor < total_num_segment_descriptors);

  size_t i = first_empty_segment_descriptor;
  for (auto &obj : loaded_code_objects) {
    assert(i < total_num_segment_descriptors);
    for (auto &seg : obj->LoadedSegments()) {
      segment_descriptors[i].agent = seg->Agent();
      segment_descriptors[i].executable = Executable::Handle(seg->Owner());
      segment_descriptors[i].code_object_storage_type =
          HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
      segment_descriptors[i].code_object_storage_base = obj->ElfData();
      segment_descriptors[i].code_object_storage_size = obj->ElfSize();
      segment_descriptors[i].code_object_storage_offset = seg->StorageOffset();
      segment_descriptors[i].segment_base = seg->Address(seg->VAddr());
      segment_descriptors[i].segment_size = seg->Size();
      ++i;
    }
  }

  return i - first_empty_segment_descriptor;
}

hsa_agent_t MockHsaLoadedCodeObject::getAgent() const {
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return loaded_segments.front()->Agent();
}

hsa_executable_t LoadedCodeObjectImpl::getExecutable() const {
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return Executable::Handle(loaded_segments.front()->Owner());
}
uint64_t LoadedCodeObjectImpl::getElfData() const {
  return reinterpret_cast<uint64_t>(elf_data);
}
uint64_t LoadedCodeObjectImpl::getElfSize() const { return (uint64_t)elf_size; }
uint64_t LoadedCodeObjectImpl::getStorageOffset() const {
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return (uint64_t)loaded_segments.front()->StorageOffset();
}
uint64_t LoadedCodeObjectImpl::getLoadBase() const {
  // TODO Add support for code objects with 0 segments.
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return reinterpret_cast<uint64_t>(loaded_segments.front()->Address(0));
}
uint64_t LoadedCodeObjectImpl::getLoadSize() const {
  // TODO Add support for code objects with 0 or >1 segments.
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return (uint64_t)loaded_segments.front()->Size();
}
int64_t LoadedCodeObjectImpl::getDelta() const {
  // TODO Add support for code objects with 0 segments.
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return getLoadBase() - loaded_segments.front()->VAddr();
}

std::string MockHsaLoadedCodeObject::getUri() const {
  return std::string(r_debug_info.l_name);
}

MockHsaExecutable *MockHsaLoader::FindExecutable(uint64_t device_address) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  if (device_address == 0) {
    return nullptr;
  }

  for (auto &exec : executables) {
    if (exec != nullptr) {
      uint64_t host_address = exec->FindHostAddress(device_address);
      if (host_address != 0) {

        return exec;
      }
    }
  }
  return nullptr;
}

uint64_t MockHsaExecutable::FindHostAddress(uint64_t device_address) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  for (auto &obj : LoadedCodeObjects) {
    assert(obj);
    for (auto &seg : obj->getLoadedSegments()) {
      assert(seg);
      uint64_t paddr = (uint64_t)(uintptr_t)seg->Address(seg->VAddr());
      if (paddr <= device_address && device_address < paddr + seg->Size()) {
        void *haddr =
            context_->SegmentHostAddress(seg->ElfSegment(), seg->Agent(),
                                         seg->Ptr(), device_address - paddr);
        return nullptr == haddr ? 0 : (uint64_t)(uintptr_t)haddr;
      }
    }
  }
  return 0;
}

void MockHsaExecutable::EnableReadOnlyMode() { rw_lock_.ReaderLock(); }

void MockHsaExecutable::DisableReadOnlyMode() { rw_lock_.ReaderUnlock(); }

llvm::Expected<const MockHsaLoadedCodeObject &>
MockHsaExecutable::loadCodeObject(llvm::ArrayRef<std::byte> CodeObject,
                                  const std::string &uri) {
  WriterLockGuard<ReaderWriterLock> WriterLock(rw_lock_);

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(HSA_EXECUTABLE_STATE_FROZEN != state_,
                                  "LoaderError: executable is already frozen"));

  code.reset(new code::AmdHsaCode());

  std::string codeIsa;
  unsigned genericVersion;
  if (!code->GetIsa(codeIsa, &genericVersion)) {
    logger_ << "LoaderError: failed to determine code object's ISA\n";
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  uint32_t majorVersion, minorVersion;
  if (!code->GetCodeObjectVersion(&majorVersion, &minorVersion)) {
    logger_ << "LoaderError: failed to determine code object's version\n";
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  if (majorVersion < 1 || majorVersion > 6) {
    logger_ << "LoaderError: unsupported code object version: " << majorVersion
            << "\n";
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  if (agent.handle == 0 && majorVersion == 1) {
    logger_ << "LoaderError: code object v1 requires non-null agent\n";
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  hsa_isa_t objectsIsa = context_->IsaFromName(codeIsa.c_str());
  if (!objectsIsa.handle) {
    logger_ << "LoaderError: code object's ISA (" << codeIsa.c_str()
            << ") is invalid\n";
    return HSA_STATUS_ERROR_INVALID_ISA_NAME;
  }

  if (agent.handle != 0 &&
      !context_->IsaSupportedByAgent(agent, objectsIsa, genericVersion)) {
    logger_ << "LoaderError: code object's ISA (" << codeIsa.c_str()
            << ") is not supported by the agent\n";
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }

  objects.push_back(
      new LoadedCodeObjectImpl(this, agent, code->ElfData(), code->ElfSize()));
  loaded_code_objects.push_back((LoadedCodeObjectImpl *)objects.back());

  hsa_status_t status = LoadSegments(agent, code.get(), majorVersion);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  for (size_t i = 0; i < code->SymbolCount(); ++i) {
    if (majorVersion >= 2 &&
        code->GetSymbol(i)->elfSym()->type() != STT_AMDGPU_HSA_KERNEL &&
        code->GetSymbol(i)->elfSym()->binding() == STB_LOCAL)
      continue;

    status = LoadSymbol(code->GetSymbol(i), majorVersion);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  status = ApplyRelocations(code.get());
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  code.reset();

  loaded_code_objects.back()->r_debug_info.l_addr =
      loaded_code_objects.back()->getDelta();
  loaded_code_objects.back()->r_debug_info.l_name = strdup(uri.c_str());
  loaded_code_objects.back()->r_debug_info.l_prev = nullptr;
  loaded_code_objects.back()->r_debug_info.l_next = nullptr;

  return loaded_code_objects.back();
}

hsa_status_t MockHsaExecutable::LoadSegments(hsa_agent_t agent,
                                             const code::AmdHsaCode *c) {
  return LoadSegmentsV2(agent, c);
}

llvm::Error MockHsaLoadedCodeObject::LoadSegments() {

  const auto &CodeObjectELFFile = Elf->getELFFile();

  /// Get the PT_LOAD segments of the ELF
  auto ProgramHeadersOrErr = CodeObjectELFFile.program_headers();
  LUTHIER_RETURN_ON_ERROR(ProgramHeadersOrErr.takeError());

  llvm::SmallVector<std::reference_wrapper<llvm::object::ELF64LE::Phdr>, 4>
      PTLoadSegments;

  for (auto Phdr : *ProgramHeadersOrErr) {
    if (Phdr.p_type == PT_LOAD) {
      PTLoadSegments.push_back(Phdr);
    }
  }

  if (PTLoadSegments.empty()) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "The code object has no PT_LOAD sections");
  }

  /// PT_LOAD segments are sorted w.r.t their virtual address but we sort them
  /// anyway just to be sure. We then use the sorted list to calculate the
  /// starting virtual address of the loaded region and its size
  llvm::sort(PTLoadSegments,
             [](std::reference_wrapper<llvm::object::ELF64LE::Phdr> LHS,
                std::reference_wrapper<llvm::object::ELF64LE::Phdr> RHS) {
               return LHS.get().p_vaddr < RHS.get().p_vaddr;
             });

  const auto &FirstLoadSegment = PTLoadSegments.begin()->get();
  const auto &LastLoadSegment = PTLoadSegments.rbegin()->get();
  uint64_t VAddrBegin = FirstLoadSegment.p_vaddr;
  uint64_t Size = LastLoadSegment.p_vaddr + LastLoadSegment.p_memsz;

  /// Allocate the region
  llvm::Error Err = llvm::Error::success();

  LoadedRegion = std::make_unique<MockHsaLoadedRegion>(
      *this, Size, AMD_ISA_ALIGN_BYTES, VAddrBegin, FirstLoadSegment.p_offset,
      Err);

  LUTHIER_RETURN_ON_ERROR(Err);

  /// If region allocation was successful, load the PT_LOAD segments
  const char *ElfStart = Elf->getMemoryBufferRef().getBufferStart();

  for (auto PTLoadSegment : PTLoadSegments) {

    Err = LoadedRegion->Copy(PTLoadSegment.get().p_vaddr,
                             ElfStart + PTLoadSegment.get().p_offset,
                             PTLoadSegment.get().p_filesz);
    LUTHIER_RETURN_ON_ERROR(Err);
  }

  return llvm::Error::success();
}

hsa_status_t MockHsaExecutable::LoadDeclarationSymbol(hsa_agent_t agent,
                                                      code::Symbol *sym,
                                                      uint32_t majorVersion) {
  auto agent_symbol = agent_symbols_.find(std::make_pair(sym->Name(), agent));
  if (agent_symbol == agent_symbols_.end()) {
    logger_ << "LoaderError: symbol \"" << sym->Name() << "\" is undefined\n";

    // TODO(spec): this is not spec compliant.
    return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
  }
  return HSA_STATUS_SUCCESS;
}

Segment *MockHsaExecutable::VirtualAddressSegment(uint64_t vaddr) {
  for (auto &seg : loaded_code_objects.back()->LoadedSegments()) {
    if (seg->IsAddressInSegment(vaddr)) {
      return seg;
    }
  }
  return 0;
}

uint64_t MockHsaExecutable::SymbolAddress(code::Symbol *sym) {
  code::Section *sec = sym->GetSection();
  Segment *seg = SectionSegment(sec);
  return nullptr == seg ? 0 : (uint64_t)(uintptr_t)seg->Address(sym->VAddr());
}

uint64_t MockHsaExecutable::SymbolAddress(elf::Symbol *sym) {
  elf::Section *sec = sym->section();
  if (!sec) {
    return NULL;
  }

  Segment *seg = SectionSegment(sec);
  uint64_t vaddr = sec->addr() + sym->value();
  return nullptr == seg ? 0 : (uint64_t)(uintptr_t)seg->Address(vaddr);
}

Segment *MockHsaExecutable::SymbolSegment(code::Symbol *sym) {
  return SectionSegment(sym->GetSection());
}

Segment *MockHsaExecutable::SectionSegment(code::Section *sec) {
  for (Segment *seg : loaded_code_objects.back()->LoadedSegments()) {
    if (seg->IsAddressInSegment(sec->addr())) {
      return seg;
    }
  }
  return nullptr;
}

llvm::Error MockHsaLoadedCodeObject::ApplyRelocations() {
  /// Apply static relocations
  for (const llvm::object::SectionRef Section : Elf->sections()) {
    for (const llvm::object::ELFRelocationRef Reloc : Section.relocations()) {
      /// In ROCr static relocations are only applied to code object v1; Here
      /// we allow it to be applied since we support device functions
      LUTHIER_RETURN_ON_ERROR(ApplyStaticRelocationSection(Reloc));
    }
  }

  /// Apply dynamic relocations
  for (const llvm::object::SectionRef DynRelocSection :
       llvm::cast<llvm::object::ObjectFile>(Elf)
           ->dynamic_relocation_sections()) {
    for (const llvm::object::ELFRelocationRef Reloc : DynRelocSection) {

      LUTHIER_RETURN_ON_ERROR(ApplyDynamicRelocationSection(Reloc));
    }
  }
  return llvm::Error::success();
}

llvm::Error MockHsaLoadedCodeObject::ApplyStaticRelocationSection(
    llvm::object::ELFRelocationRef sec) {
  // Skip link-time relocations (if any).
  if (!(sec->targetSection()->flags() & SHF_ALLOC)) {
    return HSA_STATUS_SUCCESS;
  }
  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < sec->relocationCount(); ++i) {
    status = ApplyStaticRelocation(sec->relocation(i));
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
}

llvm::Error MockHsaLoadedCodeObject::ApplyStaticRelocation(
    llvm::object::ELFRelocationRef Rel) {

  llvm::object::elf_symbol_iterator Sym = Rel.getSymbol();
  if (Sym == Rel.getObject()->symbol_end()) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Relocation section doesn't have a symbol");
  }

  uint64_t RelOffset = Rel.getOffset();
  switch (Rel.getType()) {
  case llvm::ELF::R_AMDGPU_ABS32_LO:
  case llvm::ELF::R_AMDGPU_ABS32_HI:
  case llvm::ELF::R_AMDGPU_ABS64: {
    uint64_t Addr;
    switch (Sym->getELFType()) {
    case llvm::ELF::STT_OBJECT:
    case llvm::ELF::STT_SECTION:
    case llvm::ELF::STT_FUNC:
    case llvm::ELF::STT_GNU_IFUNC:
      LUTHIER_RETURN_ON_ERROR(Sym->getAddress().moveInto(Addr));
      break;
    default:
      return LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Invalid symbol type {0}", Sym->getELFType()));
    }
    llvm::Expected<uint64_t> AddendOrErr = Rel.getAddend();
    LUTHIER_RETURN_ON_ERROR(AddendOrErr.takeError());

    Addr += *AddendOrErr;

    uint32_t addr32 = 0;
    switch (Rel.getType()) {
    case llvm::ELF::R_AMDGPU_ABS32_HI:
      addr32 = static_cast<uint32_t>((Addr >> 32) & 0xFFFFFFFF);
      LoadedRegion->Copy(RelOffset, &addr32, sizeof(addr32));
      break;
    case llvm::ELF::R_AMDGPU_ABS32_LO:
      addr32 = static_cast<uint32_t>(Addr & 0xFFFFFFFF);
      LoadedRegion->Copy(RelOffset, &addr32, sizeof(addr32));
      break;
    case llvm::ELF::R_AMDGPU_ABS64:
      LoadedRegion->Copy(RelOffset, &Addr, sizeof(Addr));
      break;
    default:
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Encountered invalid relocation type {0}", Rel.getType()));
    }
    break;
  }

  default:
    // Ignore.
    break;
  }
  return llvm::Error::success();
}

MockHsaLoadedRegion::MockHsaLoadedRegion(MockHsaLoadedCodeObject &Parent,
                                         size_t Size, size_t Alignment,
                                         uint64_t VAddr, size_t Offset,
                                         llvm::Error &Err, bool Zero)
    : Parent(Parent), Size(Size), BaseVAddr(VAddr), storage_offset(Offset) {
  llvm::ErrorAsOutParameter EAO(Err);

  Ptr = new (std::align_val_t{Alignment}, std::nothrow) std::byte[Size];
  if (nullptr == Ptr) {
    Err = LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Failed to allocate {0} bytes of memory", Size));
    return;
  }
  if (Zero) {
    std::memset(Ptr, 0, Size);
  }
}

MockHsaLoadedRegion::~MockHsaLoadedRegion() { delete[] Ptr; }

hsa_status_t MockHsaLoadedCodeObject::ApplyDynamicRelocation(
    llvm::object::ELFRelocationRef Rel) {
  Segment *relSeg = VirtualAddressSegment(rel->offset());
  uint64_t symAddr = 0;
  switch (rel->symbol()->type()) {
  case llvm::ELF::STT_OBJECT:
  case llvm::ELF::STT_FUNC: {
    Segment *symSeg = VirtualAddressSegment(rel->symbol()->value());
    symAddr =
        reinterpret_cast<uint64_t>(symSeg->Address(rel->symbol()->value()));
    break;
  }

    // External symbols, they must be defined prior loading.
  case llvm::ELF::STT_NOTYPE: {
    // TODO: Only agent allocation variables are supported in v2.1. How will
    // we distinguish between program allocation and agent allocation
    // variables?
    auto agent_symbol =
        agent_symbols_.find(std::make_pair(rel->symbol()->name(), agent));
    if (agent_symbol != agent_symbols_.end())
      symAddr = agent_symbol->second->address;
    break;
  }

  default:
    // Only objects and kernels are supported in v2.1.
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  symAddr += rel->addend();

  switch (rel->type()) {
  case ELF::R_AMDGPU_ABS32_HI: {
    if (!symAddr) {
      logger_ << "LoaderError: symbol \"" << rel->symbol()->name()
              << "\" is undefined\n";
      return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
    }

    uint32_t symAddr32 = uint32_t((symAddr >> 32) & 0xFFFFFFFF);
    relSeg->Copy(rel->offset(), &symAddr32, sizeof(symAddr32));
    break;
  }

  case ELF::R_AMDGPU_ABS32_LO: {
    if (!symAddr) {
      logger_ << "LoaderError: symbol \"" << rel->symbol()->name()
              << "\" is undefined\n";
      return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
    }

    uint32_t symAddr32 = uint32_t(symAddr & 0xFFFFFFFF);
    relSeg->Copy(rel->offset(), &symAddr32, sizeof(symAddr32));
    break;
  }

  case ELF::R_AMDGPU_ABS32: {
    if (!symAddr) {
      logger_ << "LoaderError: symbol \"" << rel->symbol()->name()
              << "\" is undefined\n";
      return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
    }

    uint32_t symAddr32 = uint32_t(symAddr);
    relSeg->Copy(rel->offset(), &symAddr32, sizeof(symAddr32));
    break;
  }

  case ELF::R_AMDGPU_ABS64: {
    if (!symAddr) {
      logger_ << "LoaderError: symbol \"" << rel->symbol()->name()
              << "\" is undefined\n";
      return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
    }

    relSeg->Copy(rel->offset(), &symAddr, sizeof(symAddr));
    break;
  }

  case ELF::R_AMDGPU_RELATIVE64: {
    int64_t baseDelta =
        reinterpret_cast<uint64_t>(relSeg->Address(0)) - relSeg->VAddr();
    uint64_t relocatedAddr = baseDelta + rel->addend();
    relSeg->Copy(rel->offset(), &relocatedAddr, sizeof(relocatedAddr));
    break;
  }

  default:
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  return HSA_STATUS_SUCCESS;
}

llvm::Error MockHsaExecutable::freeze() {
  amd::hsa::common::WriterLockGuard<amd::hsa::common::ReaderWriterLock>
      writer_lock(rw_lock_);
  if (HSA_EXECUTABLE_STATE_FROZEN == state_) {
    return LUTHIER_MAKE_GENERIC_ERROR("The executable is already frozen");
  }

  for (auto &lco : loaded_code_objects) {
    for (auto &ls : lco->LoadedSegments()) {
      ls->Freeze();
    }
  }

  state_ = HSA_EXECUTABLE_STATE_FROZEN;
  return llvm::Error::success();
}

} // namespace luthier
