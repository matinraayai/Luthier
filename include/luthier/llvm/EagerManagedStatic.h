//===-- EagerManagedStatic.h ------------------------------------*- C++ -*-===//
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
/// \file This file defines the \c luthier::EagerManagedStatic class, a variant
/// of \c llvm::ManagedStatic that is eagerly initialized but is still managed
/// by LLVM and requires the explicit call to \c llvm_shutdown to be freed.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_EAGER_MANAGED_STATIC_H
#define LUTHIER_EAGER_MANAGED_STATIC_H
#include <llvm/Support/ManagedStatic.h>

namespace luthier {

/// \brief a variant of \c llvm::ManagedStatic which takes constructor
/// parameters of \c C and eagerly initializes them.
/// \details Unlike <tt>llvm::ManagedStatic<tt>, this class doesn't aim to reduce
/// initialization time of the library; Instead it was implemented to give
/// control over when the destructor of its managed object is called. When
/// using <tt>LD_PRELOAD</tt>, it is possible that the destructors of static
/// global variables are called prematurely before rocprofiler can invoke the
/// finalizer of the Luthier tool. A good example is using
/// <tt>llvm::outs</tt> inside <tt>luthier::atToolFini</tt>, where the exit
/// handler might first invoke the destructor of the
/// static \c llvm::raw_fd_ostream constructed inside \c llvm::outs before
/// calling \c luthier::atToolFini. A similar issue was seen when using the
/// Meyers singleton inside the Luthier tooling library, which was remedied
/// by instead constructing them using explicit \c new and \c delete functions.
/// The \c EagerManagedStatic objects will remain valid until \c llvm_shutdown
/// is called, which happens after \c luthier::atToolFini is invoked after
/// the tooling library has been finalized.
/// \note It is not safe to use this class for managing the lifetime of
/// static variables inside a HIP-based Luthier tool that contains device code.
/// Use explicit \c new and \c delete inside \c luthier::atToolInit and \c
/// luthier::atToolFini instead
/// \tparam C the underlying class of the object being managed
/// \tparam Deleter the function to be called to free the underlying object
template <class C, class Deleter = llvm::object_deleter<C>>
class EagerManagedStatic : public llvm::ManagedStaticBase {
public:
  template <typename... Args>
  explicit EagerManagedStatic(Args &&...VarArgs) : llvm::ManagedStaticBase() {
    RegisterManagedStatic(
        static_cast<void *(*)()>([]() -> void * { return nullptr; }),
        Deleter::call);
    Ptr = new C(std::forward<Args>(VarArgs)...);
  }
  // Accessors.
  C &operator*() {
    return *static_cast<C *>(Ptr.load(std::memory_order_relaxed));
  }

  C *operator->() { return &**this; }

  const C &operator*() const {
    return *static_cast<C *>(Ptr.load(std::memory_order_relaxed));
  }

  const C *operator->() const { return &**this; }

  // Extract the instance, leaving the ManagedStatic uninitialized. The
  // user is then responsible for the lifetime of the returned instance.
  C *claim() { return static_cast<C *>(Ptr.exchange(nullptr)); }
};
} // namespace luthier
#endif