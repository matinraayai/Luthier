//===-- ApiTableEnumInfo.h --------------------------------------*- C++ -*-===//
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
/// Defines the \c ApiTableEnumInfo customization point, mapping a
/// \c rocprofiler_intercept_table_t to its concrete API table type and
/// associated metadata.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_ENUM_INFO_H
#define LUTHIER_ROCPROFILER_API_TABLE_ENUM_INFO_H
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::rocprofiler {

/// \brief Primary template (customization point) providing static information
/// about a \c rocprofiler_intercept_table_t: its concrete C API table type
/// (\c ApiTableType), the number of tables rocprofiler-sdk registers for it
/// (\c NumApiTables), a human-readable \c ApiTableName
///
/// \details This is the single seam carrying all HSA/HIP-specific knowledge.
/// The base \c ApiTableRegistrationCallbackProvider depends only on this
/// declaration, so it remains free of HSA and HIP headers. Each library
/// provides its own specialization:
///   - \c HsaApiTableEnumInfo.h specializes \c ROCPROFILER_HSA_TABLE
///   - \c HipApiTableEnumInfo.h specializes the \c ROCPROFILER_HIP_*_TABLE
///     entries
/// Include the relevant specialization header (transitively, via the matching
/// snapshot/installer header) before instantiating a provider for that table.
template <rocprofiler_intercept_table_t TableType> struct ApiTableEnumInfo;

} // namespace luthier::rocprofiler

#endif
