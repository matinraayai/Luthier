//===-- Memory.h ------------------------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file Memory.h
/// Defines wrappers for HSA core memory allocation and management APIs.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_MEMORY_H
#define LUTHIER_HSA_MEMORY_H
#include "luthier/HSA/ApiTable.h"
#include <llvm/Support/Error.h>

namespace luthier::hsa {

//===----------------------------------------------------------------------===//
// Core memory allocation (hsa_memory_*)
//===----------------------------------------------------------------------===//

/// Allocate \p Size bytes from the given \p Region.
/// \returns The pointer to the allocated memory, or an error.
/// \sa hsa_memory_allocate
llvm::Expected<void *>
memoryAllocate(const ApiTableContainer<::CoreApiTable> &CoreApi,
               hsa_region_t Region, size_t Size);

/// Free a block of memory previously allocated with \c memoryAllocate.
/// \sa hsa_memory_free
llvm::Error memoryFree(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       void *Ptr);

/// Copy \p Size bytes from \p Src to \p Dst.
/// Both \p Src and \p Dst may be GPU-accessible memory.
/// \sa hsa_memory_copy
llvm::Error memoryCopy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       void *Dst, const void *Src, size_t Size);

/// Assign \p Ptr (previously allocated by \c memoryAllocate) to \p Agent
/// with the given \p Access permission.
/// \sa hsa_memory_assign_agent
llvm::Error memoryAssignAgent(const ApiTableContainer<::CoreApiTable> &CoreApi,
                              void *Ptr, hsa_agent_t Agent,
                              hsa_access_permission_t Access);

/// Register a host memory block [\p Ptr, \p Ptr + \p Size) so that it can
/// be accessed by GPU agents.
/// \sa hsa_memory_register
llvm::Error memoryRegister(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           void *Ptr, size_t Size);

/// Deregister a previously registered host memory block.
/// \sa hsa_memory_deregister
llvm::Error memoryDeregister(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             void *Ptr, size_t Size);

} // namespace luthier::hsa

#endif // LUTHIER_HSA_MEMORY_H
