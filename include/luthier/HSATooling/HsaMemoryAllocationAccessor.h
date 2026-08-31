//===-- HsaMemoryAllocationAccessor.h ---------------------------*- C++ -*-===//
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
/// Describes the \c HsaMemoryAllocationAccessor class which implements the
/// \c MemoryAllocationAccessor interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_HSA_MEMORY_ALLOCATION_ACCESSOR_H
#define LUTHIER_HSA_TOOLING_HSA_MEMORY_ALLOCATION_ACCESSOR_H
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/ToolCodeGen/DriverAllocationResolver.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"

#include <memory>

namespace luthier {

/// \brief The \c MemoryAllocationAccessor for Luthier's instrumentation
/// pipeline: answers from HSA when HSA knows, and from a driver-level resolver
/// when it does not.
///
/// \par The three sources, in the order they are asked
/// \li The HSA \b loader. Names the exact loaded code object and hands back its
///     parsed ELF. The most precise answer available.
/// \li \c hsa_amd_pointer_info. Names an HSA-managed allocation with no parsed
///     code object.
/// \li The \c DriverAllocationResolver, if one was supplied and is available.
///     Names the driver-level allocation, which is coarser than either of the
///     above -- measured, a \c kernel_object at \c 0x5202400003c0 resolves to a
///     2 MB suballocation arena.
///
/// \par Why the order is precise-first rather than complete-first
/// Sitting lower in the software stack makes a source see \e more allocations
/// and describe each of them \e more coarsely, because the driver cannot see how
/// a runtime subdivides what it handed out. \c InstructionTraces disassembles
/// forward until it reaches the end of the reported allocation
/// (\c InstructionTracesAnalysis.cpp:105-106), so a coarse answer sends it
/// running for megabytes past the end of the kernel, through other kernels and
/// data. Asking the complete source first would mean it always answers, and
/// permanently shadows the precise one.
///
/// The general form is worth remembering: \b what \b we \b intercept
/// \b determines \b what \b we \b can \b observe, \b not \b who \b has \b the
/// \b best \b answer.
///
/// \par A non-empty HSA answer is final, even without a code object
/// \c hsa_amd_pointer_info reports a real allocation but never a parsed code
/// object, and it is tempting to treat that as a half-answer worth improving on.
/// It is not, for two reasons. The resolver's answer would be \e coarser, per
/// the paragraph above. And downstream, \c CodeDiscoveryPass treats "code object
/// present but no symbol at that offset" as a hard error
/// (\c CodeDiscoveryPass.cpp:743-745) while falling back gracefully to a
/// synthetic \c kernel-<addr> name when there is no code object at all
/// (\c :761) -- so a missing code object is a supported outcome, not a defect to
/// route around.
///
/// \par Why HSA may be entirely unavailable, and why that is not an error
/// An application that drives the KFD driver itself holds the DRM virtual
/// address space for its GPUs, and the kernel permits only one such VM per GPU
/// per process. So \c hsa_init inside such a process fails -- measured, both
/// orderings: the application's \c ACQUIRE_VM then \c hsa_init gives
/// \c HSA_STATUS_ERROR_OUT_OF_RESOURCES, and the reverse makes the
/// application's \c ACQUIRE_VM fail with \c EBUSY. Linking HSA is therefore not
/// the same as having HSA, and in such a process the resolver is the \e only
/// source.
///
/// This matters mechanically, not just conceptually. The accessor reaches HSA
/// through rocprofiler-captured API tables, and those are filled by a callback
/// that fires when HSA initializes. Reading a snapshot that was never filled is
/// a \b fatal error by design (\c HsaApiTableSnapshot.h:125-128), so the
/// snapshots are held here rather than their tables, and
/// \c wasRegistrationCallbackInvoked is checked before any of them is read.
class HsaMemoryAllocationAccessor : public MemoryAllocationAccessor {

  const LoadedCodeObjectCache &COC;

  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreTable;

  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtTable;

  /// Held as the snapshot, not as the table it wraps: reading the table of an
  /// uninitialized snapshot is fatal, and in a KFD-only process it never
  /// initializes. See the class comment.
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &VenLoaderSnapshot;

  /// The last source, consulted only when HSA does not recognise an address.
  /// May be null, which simply means there is no driver-level source in this
  /// process. Owned, because its lifetime is exactly this accessor's -- it
  /// caches host mappings that must be released with it.
  std::unique_ptr<DriverAllocationResolver> DriverResolver;

  /// Cache for holding on to the host copy of memory allocations in HSA that
  /// don't have a host-accessible copy
  mutable llvm::SmallDenseMap<void *, std::vector<std::byte>>
      CachedAllocationsHostCopy;

  /// \brief Whether the HSA sources can be read at all.
  ///
  /// False in a process where HSA was never initialized. Checked rather than
  /// assumed because the alternative is a fatal error -- see the class comment.
  [[nodiscard]] bool isHsaUsable() const {
    return CoreTable.wasRegistrationCallbackInvoked() &&
           AmdExtTable.wasRegistrationCallbackInvoked() &&
           VenLoaderSnapshot.wasRegistrationCallbackInvoked();
  }

  /// \brief Ask the driver-level resolver, if there is one to ask.
  /// \returns an empty descriptor when there is no resolver, none is available,
  /// or it does not know the address.
  [[nodiscard]] llvm::Expected<AllocationDescriptor>
  askDriverResolver(uint64_t DeviceAddr) const;

public:
  [[nodiscard]] llvm::Expected<AllocationDescriptor>
  getAllocationDescriptor(uint64_t DeviceAddr) const override;

  /// \param DriverResolver an optional last source for addresses HSA does not
  /// manage. Passing \c nullptr gives an HSA-only accessor.
  HsaMemoryAllocationAccessor(
      const LoadedCodeObjectCache &COC,
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreTable,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtTable,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &VenLoaderSnapshot,
      std::unique_ptr<DriverAllocationResolver> DriverResolver = nullptr)
      : COC(COC), CoreTable(CoreTable), AmdExtTable(AmdExtTable),
        VenLoaderSnapshot(VenLoaderSnapshot),
        DriverResolver(std::move(DriverResolver)) {};

  ~HsaMemoryAllocationAccessor() override = default;
};

} // namespace luthier

#endif
