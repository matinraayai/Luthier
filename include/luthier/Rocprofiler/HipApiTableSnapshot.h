//===-- HipApiTableSnapshot.h -----------------------------------*- C++ -*-===//
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
/// Defines the HIP API table snapshot class that captures a snapshot of a HIP
/// API table when registered with rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_HIP_API_TABLE_SNAPSHOT_H
#define LUTHIER_ROCPROFILER_HIP_API_TABLE_SNAPSHOT_H
#include "luthier/HIP/ApiTable.h"
#include "luthier/Rocprofiler/ApiTableRegistrationCallbackProvider.h"
#include "luthier/Rocprofiler/HipApiTableEnumInfo.h"
#include <algorithm>
#include <cstring>

namespace luthier::rocprofiler {

/// \brief Provides a snapshot of the HIP API table to other components
/// using rocprofiler-sdk
template <rocprofiler_intercept_table_t TableType>
class HipApiTableSnapshot final
    : public ApiTableRegistrationCallbackProvider<TableType> {
private:
  /// Where the snapshot of the HIP API Table is stored
  typename ApiTableEnumInfo<TableType>::ApiTableType ApiTable{};

public:
  explicit HipApiTableSnapshot(const DynamicLibrary &RocprofilerLib,
                               llvm::Error &Err)
      : ApiTableRegistrationCallbackProvider<TableType>(
            [this](llvm::ArrayRef<
                       typename ApiTableEnumInfo<TableType>::ApiTableType *>
                       Tables,
                   uint64_t LibVersion, uint64_t LibInstance) {
              /// Capture only the first registration. When the application
              /// finalizes and re-initializes the runtime, rocprofiler-sdk
              /// invokes this callback again with \c LibInstance > 0, but the
              /// runtime's function-pointer addresses are stable across the
              /// cycle, so the first snapshot remains valid; ignore re-fires.
              if (this->wasRegistrationCallbackInvoked())
                return;
              if (Tables.empty())
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
                    "No Api tables were passed to the callback"));
              /// Bound the copy by the destination struct size: \c
              /// Tables[0]->size is the runtime table's size, which can exceed
              /// \c sizeof(ApiTable) when the application runs against a newer
              /// ROCm than Luthier was compiled against. Copying the full
              /// runtime size would overflow \c ApiTable. \c ApiTable is
              /// zero-initialized, so a smaller source leaves the tail zeroed.
              std::memcpy(&ApiTable, Tables[0],
                          std::min(sizeof(ApiTable),
                                   static_cast<size_t>(Tables[0]->size)));
            },
            RocprofilerLib, Err) {};

  ~HipApiTableSnapshot() override = default;

  hip::ApiTableContainer<TableType> getTable() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        this->wasRegistrationCallbackInvoked(), "Snapshot is not initialized"));
    return hip::ApiTableContainer<TableType>(ApiTable);
  }
};

} // namespace luthier::rocprofiler

#endif
