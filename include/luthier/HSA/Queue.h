//===-- Queue.h -------------------------------------------------*- C++ -*-===//
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
/// \file
/// Defines a set of commonly used functionality for the \c hsa_queue_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_QUEUE_H
#define LUTHIER_HSA_QUEUE_H
#include "luthier/HSA/ApiTable.h"
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// Creates a new user-mode queue on \p Agent with \p Size AQL packet slots.
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Agent the \c hsa_agent_t the queue will be created on
/// \param Size number of packet slots in the queue; must be a power of two
/// within <tt>[HSA_AGENT_INFO_QUEUE_MIN_SIZE,
/// HSA_AGENT_INFO_QUEUE_MAX_SIZE]</tt>
/// \param Type the \c hsa_queue_type32_t requested for the queue; set to
/// \c HSA_QUEUE_TYPE_SINGLE by default
/// \return Expects a pointer to the newly created \c hsa_queue_t on success
/// \sa hsa_queue_create
llvm::Expected<hsa_queue_t *>
queueCreate(const ApiTableContainer<::CoreApiTable> &CoreApi,
            hsa_agent_t Agent, uint32_t Size,
            hsa_queue_type32_t Type = HSA_QUEUE_TYPE_SINGLE);

/// Destroys a queue created by \c queueCreate.
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Queue the \c hsa_queue_t being destroyed
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_queue_destroy
llvm::Error queueDestroy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         hsa_queue_t *Queue);

} // namespace luthier::hsa

#endif // LUTHIER_HSA_QUEUE_H
