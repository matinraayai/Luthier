//===-- Signal.h ------------------------------------------------*- C++ -*-===//
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
/// Defines a set of commonly used functionality for the \c hsa_signal_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_SIGNAL_H
#define LUTHIER_HSA_SIGNAL_H
#include "luthier/HSA/ApiTable.h"
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// Creates a new HSA signal with the given \p InitialValue, visible to all
/// agents.
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param InitialValue the initial value of the signal
/// \return Expects the newly created \c hsa_signal_t on success
/// \sa hsa_signal_create
llvm::Expected<hsa_signal_t>
signalCreate(const ApiTableContainer<::CoreApiTable> &CoreApi,
            hsa_signal_value_t InitialValue);

/// Destroys a signal created by \c signalCreate.
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Signal the \c hsa_signal_t being destroyed
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_signal_destroy
llvm::Error signalDestroy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         hsa_signal_t Signal);

/// Blocks the calling thread, using a blocking OS wait state, until
/// \p Signal's value compares true against \p CompareValue under
/// \p Condition.
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Signal the \c hsa_signal_t being waited on
/// \param Condition the comparison operator used to evaluate the wait
/// condition
/// \param CompareValue the value \p Signal is compared against
/// \return the observed value of \p Signal that satisfied the wait condition
/// \sa hsa_signal_wait_scacquire
hsa_signal_value_t
signalWait(const ApiTableContainer<::CoreApiTable> &CoreApi,
          hsa_signal_t Signal, hsa_signal_condition_t Condition,
          hsa_signal_value_t CompareValue);

/// Blocks the calling thread until \p Signal 's value compares true against
/// \p CompareValue under \p Condition, or until roughly \p TimeoutHint
/// system-clock ticks have elapsed — whichever happens first.
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Signal the \c hsa_signal_t being waited on
/// \param Condition the comparison operator used to evaluate the wait
/// condition
/// \param CompareValue the value \p Signal is compared against
/// \param TimeoutHint maximum duration of the wait, in
/// \c HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY ticks; treated as a hint by HSA
/// \param WaitState whether the thread should spin or block while waiting
/// \return the value of \p Signal observed when the wait returned; when the
/// wait timed out this need not satisfy the wait condition
/// \sa hsa_signal_wait_scacquire
hsa_signal_value_t
signalWaitTimeout(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_signal_t Signal, hsa_signal_condition_t Condition,
                  hsa_signal_value_t CompareValue, uint64_t TimeoutHint,
                  hsa_wait_state_t WaitState);

/// Stores \p Value into \p Signal with release memory ordering.
/// \param CoreApi the HSA ::CoreApi table container used to perform HSA calls
/// \param Signal the \c hsa_signal_t being written
/// \param Value the value to store
/// \sa hsa_signal_store_screlease
void signalStore(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_signal_t Signal, hsa_signal_value_t Value);

} // namespace luthier::hsa

#endif // LUTHIER_HSA_SIGNAL_H
