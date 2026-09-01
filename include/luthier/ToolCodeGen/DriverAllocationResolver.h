//===-- DriverAllocationResolver.h ------------------------------*- C++ -*-===//
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
///
/// \file
/// A source of memory allocations observed \e below the GPU runtime.
///
/// \par Why this exists as its own interface
/// \c MemoryAllocationAccessor answers the question Luthier's passes ask. This
/// answers a strictly smaller one -- "which driver-level allocation contains
/// this address, and where can the host read it" -- and it exists so that
/// \c HsaMemoryAllocationAccessor can consult a driver-level source without
/// depending on the KFD module, and so that source can be faked in a test with
/// no GPU and no preloaded wrapper.
///
/// It is deliberately \b not a \c MemoryAllocationAccessor, even though
/// \c DriverOnlyMemoryAllocationAccessor adapts one into exactly that. The
/// distinction is between a leaf and a list: a named component with one job is
/// fine, whereas an accessor holding a \e list of accessors was the composite
/// arrangement this replaced, and that invited a specific mistake -- combining
/// one source's allocation base with another's parsed code object, which makes
/// every symbol offset computed from the pair meaningless. A resolver cannot
/// express a code object at all, so through this interface the mistake is not
/// available.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_DRIVER_ALLOCATION_RESOLVER_H
#define LUTHIER_TOOL_CODE_GEN_DRIVER_ALLOCATION_RESOLVER_H
#include <llvm/Support/Error.h>

#include <cstddef>
#include <cstdint>

namespace luthier {

class DriverAllocationResolver {
public:
  /// \brief One allocation as the driver described it, plus a way to read it.
  struct Allocation {
    /// Start of the allocation in the application's virtual address space.
    const std::byte *DeviceBase{nullptr};

    /// A host-readable view of the same bytes. Obtaining this is the expensive
    /// part of resolving, and how it is obtained is the implementation's
    /// business -- below HSA there is no runtime to copy through.
    const std::byte *HostBase{nullptr};

    size_t Size{0};

    [[nodiscard]] bool empty() const { return Size == 0; }
  };

  /// \brief Find the allocation containing \p DeviceAddr.
  ///
  /// Three outcomes, and the difference between the last two is load-bearing
  /// because a caller placed behind another source has to know whether to keep
  /// looking:
  /// \li a non-empty \c Allocation -- found, and readable;
  /// \li an \b empty \c Allocation -- this resolver has never seen the address.
  ///     Normal, not a failure: memory imported from another process or managed
  ///     through paths this resolver does not watch lands here;
  /// \li an \c llvm::Error -- the address \e is this resolver's and something
  ///     went wrong obtaining a host view of it, e.g. a mapping the hardware
  ///     refused.
  [[nodiscard]] virtual llvm::Expected<Allocation>
  resolve(uint64_t DeviceAddr) const = 0;

  /// \brief Whether this resolver has a source of records at all.
  ///
  /// Distinct from "resolved nothing". A resolver whose records live in a
  /// library that was never loaded can answer no question, and reporting that as
  /// "no allocation here" would be indistinguishable from an application that
  /// allocated nothing -- so callers check this instead of inferring it.
  [[nodiscard]] virtual bool isAvailable() const = 0;

  virtual ~DriverAllocationResolver() = default;
};

} // namespace luthier

#endif // LUTHIER_TOOL_CODE_GEN_DRIVER_ALLOCATION_RESOLVER_H
