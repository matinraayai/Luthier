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
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/GenericLuthierError.h>
#include <luthier/hsa/HsaError.h>
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

llvm::Error MallocedMemory::Allocate(size_t Size, size_t align, bool zero) {
  assert(!this->Allocated());
  assert(0 < Size);
  assert(0 < align && 0 == (align & (align - 1)));
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

llvm::Error MallocedMemory::Copy(size_t offset, const void *src, size_t size) {
  assert(this->Allocated());
  assert(nullptr != src);
  assert(0 < size);
  memcpy(this->Address(offset), src, size);
  return llvm::Error::success();
}

void MallocedMemory::Free() {
  assert(this->Allocated());
  delete[] ptr_;
  ptr_ = nullptr;
  size_ = 0;
}

bool MallocedMemory::Freeze() {
  assert(this->Allocated());
  return true;
}

hsa_isa_t MockHsaLoaderContext::IsaFromName(const char *name) {
  assert(name);

  hsa_status_t HsaStatus = HSA_STATUS_SUCCESS;
  hsa_isa_t isa_handle;
  isa_handle.handle = 0;

  hsa_status = HSA::hsa_isa_from_name(name, &isa_handle);
  if (HSA_STATUS_SUCCESS != HsaStatus) {
    isa_handle.handle = 0;
    return isa_handle;
  }

  return isa_handle;
}

bool MockHsaLoaderContext::IsaSupportedByAgent(hsa_agent_t agent,
                                               hsa_isa_t code_object_isa,
                                               unsigned codeGenericVersion) {
  struct callBackData {
    std::pair<hsa_isa_t, bool> comparison_data;
    const unsigned int codeGenericV;
  } cbData = {{code_object_isa, false}, codeGenericVersion};

  auto IsIsaEquivalent = [](hsa_isa_t agent_isa_h, void *data) {
    assert(data);

    struct callBackData *inOutCB = reinterpret_cast<decltype(&cbData)>(data);

    std::pair<hsa_isa_t, bool> *data_pair = &inOutCB->comparison_data;
    const unsigned int codeGenericV = inOutCB->codeGenericV;

    assert(data_pair);
    assert(!data_pair->second);

    const core::Isa *agent_isa = core::Isa::Object(agent_isa_h);
    assert(agent_isa);
    const core::Isa *code_object_isa = core::Isa::Object(data_pair->first);
    assert(code_object_isa);

    data_pair->second =
        core::Isa::IsCompatible(*code_object_isa, *agent_isa, codeGenericV);
    return data_pair->second ? HSA_STATUS_INFO_BREAK : HSA_STATUS_SUCCESS;
  };

  hsa_status_t status =
      HSA::hsa_agent_iterate_isas(agent, IsIsaEquivalent, &cbData);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    return false;
  }
  return cbData.comparison_data.second;
}

void *MockHsaLoaderContext::SegmentAlloc(amdgpu_hsa_elf_segment_t segment,
                                         hsa_agent_t agent, size_t size,
                                         size_t align, bool zero) {
  assert(0 < size);
  assert(0 < align && 0 == (align & (align - 1)));

  SegmentMemory *Mem = new (std::nothrow) MallocedMemory();

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
  return ((SegmentMemory *)dst)->Copy(offset, src, size);
}

void MockHsaLoaderContext::SegmentFree(
    amdgpu_hsa_elf_segment_t segment, // not used.
    hsa_agent_t agent,                // not used.
    void *seg,
    size_t size) // not used.
{
  assert(nullptr != seg);
  SegmentMemory *mem = (SegmentMemory *)seg;
  mem->Free();
  delete mem;
  mem = nullptr;
}

void *
LoaderContext::SegmentAddress(amdgpu_hsa_elf_segment_t segment, // not used.
                              hsa_agent_t agent,                // not used.
                              void *seg, size_t offset) {
  assert(nullptr != seg);
  return ((SegmentMemory *)seg)->Address(offset);
}

void *
LoaderContext::SegmentHostAddress(amdgpu_hsa_elf_segment_t segment, // not used.
                                  hsa_agent_t agent,                // not used.
                                  void *seg, size_t offset) {
  assert(nullptr != seg);
  return ((SegmentMemory *)seg)->HostAddress(offset);
}

bool LoaderContext::SegmentFreeze(amdgpu_hsa_elf_segment_t segment, // not used.
                                  hsa_agent_t agent,                // not used.
                                  void *seg,
                                  size_t size) // not used.
{
  assert(nullptr != seg);
  return ((SegmentMemory *)seg)->Freeze();
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

//===----------------------------------------------------------------------===//
// SymbolImpl. //
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// KernelSymbol.                                                              //
//===----------------------------------------------------------------------===//

bool KernelSymbol::GetInfo(hsa_symbol_info32_t symbol_info, void *value) {
  static_assert(
      (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE) ==
       symbol_attribute32_t(
           HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE)),
      "attributes are not compatible");
  static_assert(
      (symbol_attribute32_t(
           HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT) ==
       symbol_attribute32_t(
           HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT)),
      "attributes are not compatible");
  static_assert(
      (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) ==
       symbol_attribute32_t(
           HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE)),
      "attributes are not compatible");
  static_assert(
      (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) ==
       symbol_attribute32_t(
           HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE)),
      "attributes are not compatible");
  static_assert(
      (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK) ==
       symbol_attribute32_t(
           HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK)),
      "attributes are not compatible");

  assert(value);

  switch (symbol_info) {
  case HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: {
    *((uint32_t *)value) = kernarg_segment_size;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT: {
    *((uint32_t *)value) = kernarg_segment_alignment;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: {
    *((uint32_t *)value) = group_segment_size;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: {
    *((uint32_t *)value) = private_segment_size;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK: {
    *((bool *)value) = is_dynamic_callstack;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE: {
    *((uint32_t *)value) = wavefront_size;
    break;
  }
  case HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_SIZE: {
    *((uint32_t *)value) = size;
    break;
  }
  case HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_ALIGN: {
    *((uint32_t *)value) = alignment;
    break;
  }
  default: {
    return SymbolImpl::GetInfo(symbol_info, value);
  }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// VariableSymbol.                                                            //
//===----------------------------------------------------------------------===//

bool VariableSymbol::GetInfo(hsa_symbol_info32_t symbol_info, void *value) {

  switch (symbol_info) {
  case HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION: {
    *((hsa_variable_allocation_t *)value) = allocation;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT: {
    *((hsa_variable_segment_t *)value) = segment;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT: {
    *((uint32_t *)value) = alignment;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE: {
    *((uint32_t *)value) = size;
    break;
  }
  case HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST: {
    *((bool *)value) = is_constant;
    break;
  }
  default: {
    return SymbolImpl::GetInfo(symbol_info, value);
  }
  }

  return true;
}

bool LoadedCodeObjectImpl::GetInfo(amd_loaded_code_object_info_t attribute,
                                   void *value) {
  assert(value);

  switch (attribute) {
  case AMD_LOADED_CODE_OBJECT_INFO_ELF_IMAGE:
    ((hsa_code_object_t *)value)->handle = reinterpret_cast<uint64_t>(elf_data);
    break;
  case AMD_LOADED_CODE_OBJECT_INFO_ELF_IMAGE_SIZE:
    *((size_t *)value) = elf_size;
    break;
  default: {
    return false;
  }
  }

  return true;
}

void LoadedCodeObjectImpl::Print(std::ostream &out) {
  out << "Code Object" << std::endl;
}

bool Segment::GetInfo(amd_loaded_segment_info_t attribute, void *value) {
  assert(value);

  switch (attribute) {
  case AMD_LOADED_SEGMENT_INFO_TYPE: {
    *((amdgpu_hsa_elf_segment_t *)value) = segment;
    break;
  }
  case AMD_LOADED_SEGMENT_INFO_ELF_BASE_ADDRESS: {
    *((uint64_t *)value) = vaddr;
    break;
  }
  case AMD_LOADED_SEGMENT_INFO_LOAD_BASE_ADDRESS: {
    *((uint64_t *)value) =
        reinterpret_cast<uint64_t>(this->Address(this->VAddr()));
    break;
  }
  case AMD_LOADED_SEGMENT_INFO_SIZE: {
    *((size_t *)value) = size;
    break;
  }
  default: {
    return false;
  }
  }

  return true;
}

uint64_t Segment::Offset(uint64_t addr) {
  assert(IsAddressInSegment(addr));
  return addr - vaddr;
}

void *Segment::Address(uint64_t addr) {
  return owner->context()->SegmentAddress(segment, agent, ptr, Offset(addr));
}

bool Segment::Freeze() {
  return !frozen ? (frozen = owner->context()->SegmentFreeze(segment, agent,
                                                             ptr, size))
                 : true;
}

bool Segment::IsAddressInSegment(uint64_t addr) {
  return vaddr <= addr && addr < vaddr + size;
}

void Segment::Copy(uint64_t addr, const void *src, size_t size) {
  // loader must do copies before freezing.
  assert(!frozen);

  if (size > 0) {
    owner->context()->SegmentCopy(segment, agent, ptr, Offset(addr), src, size);
  }
}

void Segment::Print(std::ostream &out) {
  out << "Segment" << std::endl
      << "    Type: " << AmdHsaElfSegmentToString(segment)
      << "    Size: " << size << "    VAddr: " << vaddr << std::endl
      << "    Ptr: " << std::hex << ptr << std::dec << std::endl;
}

void Segment::Destroy() {
  owner->context()->SegmentFree(segment, agent, ptr, size);
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
  for (ExecutableObject *o : objects) {
    delete o;
  }
  objects.clear();

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

  if (!agent) {
    auto program_symbol = program_symbols_.find(mangled_name);
    if (program_symbol != program_symbols_.end()) {
      return program_symbol->second;
    }
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

std::string LoadedCodeObjectImpl::getUri() const {
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

uint64_t ExecutableImpl::FindHostAddress(uint64_t device_address) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  for (auto &obj : loaded_code_objects) {
    assert(obj);
    for (auto &seg : obj->LoadedSegments()) {
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

void ExecutableImpl::EnableReadOnlyMode() { rw_lock_.ReaderLock(); }

void ExecutableImpl::DisableReadOnlyMode() { rw_lock_.ReaderUnlock(); }

hsa_status_t ExecutableImpl::GetInfo(hsa_executable_info_t executable_info,
                                     void *value) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);

  assert(value);

  switch (executable_info) {
  case HSA_EXECUTABLE_INFO_STATE: {
    *((hsa_executable_state_t *)value) = state_;
    break;
  }
  case HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE: {
    *((hsa_default_float_rounding_mode_t *)value) =
        default_float_rounding_mode_;
    break;
  }
  default: {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  }

  return HSA_STATUS_SUCCESS;
}

llvm::Error
ExecutableImpl::LoadCodeObject(hsa_agent_t agent, hsa_code_object_t code_object,
                               const char *options, const std::string &uri,
                               hsa_loaded_code_object_t *loaded_code_object) {
  return LoadCodeObject(agent, code_object, 0, options, uri,
                        loaded_code_object);
}

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

    status = LoadSymbol(agent, code->GetSymbol(i), majorVersion);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  status = ApplyRelocations(agent, code.get());
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
                                             const code::AmdHsaCode *c,
                                             uint32_t majorVersion) {
  return LoadSegmentsV2(agent, c);
}

hsa_status_t MockHsaExecutable::LoadSegmentsV2(hsa_agent_t agent,
                                               const code::AmdHsaCode *c) {
  assert(c->Machine() == ELF::EM_AMDGPU &&
         "Program code objects are not supported");

  if (!c->DataSegmentCount())
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;

  uint64_t vaddr = c->DataSegment(0)->vaddr();
  uint64_t size = c->DataSegment(c->DataSegmentCount() - 1)->vaddr() +
                  c->DataSegment(c->DataSegmentCount() - 1)->memSize();

  void *ptr = context_->SegmentAlloc(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, size,
                                     AMD_ISA_ALIGN_BYTES, true);
  if (!ptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto *LoadSegment =
      new (std::nothrow) Segment(this, agent, AMDGPU_HSA_SEGMENT_CODE_AGENT,
                                 ptr, size, vaddr, c->DataSegment(0)->offset());
  if (!LoadSegment)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < c->DataSegmentCount(); ++i) {
    status = LoadSegmentV2(c->DataSegment(i), LoadSegment);
    if (status != HSA_STATUS_SUCCESS)
      return status;
  }

  objects.push_back(LoadSegment);
  loaded_code_objects.back()->LoadedSegments().push_back(LoadSegment);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MockHsaExecutable::LoadSegmentV2(const code::Segment *data_segment,
                                              loader::Segment *load_segment) {
  assert(data_segment && load_segment);
  load_segment->Copy(data_segment->vaddr(), data_segment->data(),
                     data_segment->imageSize());

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MockHsaExecutable::LoadSymbol(hsa_agent_t agent, code::Symbol *sym,
                                           uint32_t majorVersion) {
  if (sym->IsDeclaration()) {
    return LoadDeclarationSymbol(agent, sym, majorVersion);
  } else {
    return LoadDefinitionSymbol(agent, sym, majorVersion);
  }
}

hsa_status_t MockHsaExecutable::LoadDefinitionSymbol(hsa_agent_t agent,
                                                     code::Symbol *sym,
                                                     uint32_t majorVersion) {
  auto agent_symbol = agent_symbols_.find(std::make_pair(sym->Name(), agent));
  if (agent_symbol != agent_symbols_.end()) {
    // TODO(spec): this is not spec compliant.
    return HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED;
  }

  uint64_t address = SymbolAddress(sym);
  SymbolImpl *symbol = nullptr;
  if (llvm::StringRef(sym->GetSymbolName()).ends_with(".kd")) {
    // V3.
    llvm::amdhsa::kernel_descriptor_t kd;
    sym->GetSection()->getData(sym->SectionOffset(), &kd, sizeof(kd));

    uint32_t kernarg_segment_size =
        kd.kernarg_size; // FIXME: If 0 then the compiler is not specifying the
    // size.
    uint32_t kernarg_segment_alignment =
        16; // FIXME: Use the minumum HSA required alignment.
    uint32_t group_segment_size = kd.group_segment_fixed_size;
    uint32_t private_segment_size = kd.private_segment_fixed_size;
    bool is_dynamic_callstack =
        AMDHSA_BITS_GET(kd.kernel_code_properties,
                        llvm::amdhsa::KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK);
    bool uses_wave32 = AMDHSA_BITS_GET(
        kd.kernel_code_properties,
        llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);

    uint64_t size = sym->Size();

    KernelSymbol *kernel_symbol = new KernelSymbol(
        true, sym->GetModuleName(), sym->GetSymbolName(), sym->Linkage(),
        true, // sym->IsDefinition()
        kernarg_segment_size, kernarg_segment_alignment, group_segment_size,
        private_segment_size, is_dynamic_callstack, size, 64,
        uses_wave32 ? 32 : 64, address);
    symbol = kernel_symbol;
  } else if (sym->IsVariableSymbol()) {
    symbol = new VariableSymbol(
        true, sym->GetModuleName(), sym->GetSymbolName(), sym->Linkage(),
        true, // sym->IsDefinition()
        sym->Allocation(), sym->Segment(), sym->Size(), sym->Alignment(),
        sym->IsConst(), false, address);
  } else if (sym->IsKernelSymbol()) {
    amd_kernel_code_t akc;
    sym->GetSection()->getData(sym->SectionOffset(), &akc, sizeof(akc));

    uint32_t kernarg_segment_size = uint32_t(akc.kernarg_segment_byte_size);
    uint32_t kernarg_segment_alignment =
        uint32_t(1 << akc.kernarg_segment_alignment);
    uint32_t group_segment_size =
        uint32_t(akc.workgroup_group_segment_byte_size);
    uint32_t private_segment_size =
        uint32_t(akc.workitem_private_segment_byte_size);
    bool is_dynamic_callstack =
        AMD_HSA_BITS_GET(akc.kernel_code_properties,
                         AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK)
            ? true
            : false;
    bool uses_wave32 = akc.wavefront_size == AMD_POWERTWO_32;

    uint64_t size = sym->Size();

    if (!size && sym->SectionOffset() < sym->GetSection()->size()) {
      // ORCA Runtime relies on symbol size equal to size of kernel ISA. If
      // symbol size is 0 in ELF, calculate end of segment - symbol value.
      size = sym->GetSection()->size() - sym->SectionOffset();
    }
    KernelSymbol *kernel_symbol = new KernelSymbol(
        true, sym->GetModuleName(), sym->GetSymbolName(), sym->Linkage(),
        true, // sym->IsDefinition()
        kernarg_segment_size, kernarg_segment_alignment, group_segment_size,
        private_segment_size, is_dynamic_callstack, size, 256,
        uses_wave32 ? 32 : 64, address);
    kernel_symbol->debug_info.elf_raw = code->ElfData();
    kernel_symbol->debug_info.elf_size = code->ElfSize();
    kernel_symbol->debug_info.kernel_name = kernel_symbol->full_name.c_str();
    kernel_symbol->debug_info.owning_segment =
        (void *)SymbolSegment(sym)->Address(sym->GetSection()->addr());
    symbol = kernel_symbol;

    // \todo kzhuravl 10/15/15 This is a debugger backdoor: needs to be
    // removed.
    uint64_t target_address =
        sym->GetSection()->addr() + sym->SectionOffset() +
        ((size_t)(&((amd_kernel_code_t *)0)->runtime_loader_kernel_symbol));
    uint64_t source_value = (uint64_t)(uintptr_t)&kernel_symbol->debug_info;
    SymbolSegment(sym)->Copy(target_address, &source_value,
                             sizeof(source_value));
  } else {
    assert(!"Unexpected symbol type in LoadDefinitionSymbol");
    return HSA_STATUS_ERROR;
  }

  assert(symbol);
  symbol->agent = agent;
  agent_symbols_.insert(
      std::make_pair(std::make_pair(sym->Name(), agent), symbol));
  return HSA_STATUS_SUCCESS;
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

hsa_status_t
MockHsaExecutable::ApplyRelocations(hsa_agent_t agent,
                                    amd::hsa::code::AmdHsaCode *c) {
  hsa_status_t status = HSA_STATUS_SUCCESS;

  uint32_t majorVersion, minorVersion;
  if (!c->GetCodeObjectVersion(&majorVersion, &minorVersion)) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  for (size_t i = 0; i < c->RelocationSectionCount(); ++i) {
    if (c->GetRelocationSection(i)->targetSection()) {
      // Static relocations may be present if --emit-relocs
      // option was passed to lld, but they cannot be applied
      // again, so skip it for code object v2 and up.
      if (majorVersion >= 2) {
        continue;
      }

      status = ApplyStaticRelocationSection(agent, c->GetRelocationSection(i));
    } else {
      // Dynamic relocations are supported starting code object v2.1.
      if (majorVersion < 2) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if (majorVersion == 2 && minorVersion < 1) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      status = ApplyDynamicRelocationSection(agent, c->GetRelocationSection(i));
    }
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MockHsaExecutable::ApplyStaticRelocationSection(
    hsa_agent_t agent, amd::hsa::code::RelocationSection *sec) {
  // Skip link-time relocations (if any).
  if (!(sec->targetSection()->flags() & SHF_ALLOC)) {
    return HSA_STATUS_SUCCESS;
  }
  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < sec->relocationCount(); ++i) {
    status = ApplyStaticRelocation(agent, sec->relocation(i));
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t
MockHsaExecutable::ApplyStaticRelocation(hsa_agent_t agent,
                                         amd::hsa::code::Relocation *rel) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  amd::elf::Symbol *sym = rel->symbol();
  code::RelocationSection *rsec = rel->section();
  code::Section *sec = rsec->targetSection();
  Segment *rseg = SectionSegment(sec);
  size_t reladdr = sec->addr() + rel->offset();
  switch (rel->type()) {
  case R_AMDGPU_V1_32_LOW:
  case R_AMDGPU_V1_32_HIGH:
  case R_AMDGPU_V1_64: {
    uint64_t addr;
    switch (sym->type()) {
    case STT_OBJECT:
    case STT_SECTION:
    case STT_AMDGPU_HSA_KERNEL:
    case STT_AMDGPU_HSA_INDIRECT_FUNCTION:
      addr = SymbolAddress(sym);
      if (!addr) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      break;
    case STT_COMMON: {
      hsa_agent_t *sagent = &agent;
      if (STA_AMDGPU_HSA_GLOBAL_PROGRAM ==
          ELF64_ST_AMDGPU_ALLOCATION(sym->other())) {
        sagent = nullptr;
      }
      SymbolImpl *esym =
          (SymbolImpl *)GetSymbolInternal(sym->name().c_str(), sagent);
      if (!esym) {
        logger_ << "LoaderError: symbol \"" << sym->name()
                << "\" is undefined\n";
        return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
      }
      addr = esym->address;
      break;
    }
    default:
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    addr += rel->addend();

    uint32_t addr32 = 0;
    switch (rel->type()) {
    case R_AMDGPU_V1_32_HIGH:
      addr32 = uint32_t((addr >> 32) & 0xFFFFFFFF);
      rseg->Copy(reladdr, &addr32, sizeof(addr32));
      break;
    case R_AMDGPU_V1_32_LOW:
      addr32 = uint32_t(addr & 0xFFFFFFFF);
      rseg->Copy(reladdr, &addr32, sizeof(addr32));
      break;
    case R_AMDGPU_V1_64:
      rseg->Copy(reladdr, &addr, sizeof(addr));
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    break;
  }

  default:
    // Ignore.
    break;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MockHsaExecutable::ApplyDynamicRelocationSection(
    hsa_agent_t agent, amd::hsa::code::RelocationSection *sec) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < sec->relocationCount(); ++i) {
    status = ApplyDynamicRelocation(agent, sec->relocation(i));
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t
MockHsaExecutable::ApplyDynamicRelocation(hsa_agent_t agent,
                                          amd::hsa::code::Relocation *rel) {
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
  case STT_NOTYPE: {
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
