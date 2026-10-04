//===-- HsaApiTableEnumInfo.h -----------------------------------*- C++ -*-===//
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
/// Provides the \c ApiTableEnumInfo specialization for the HSA API table.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_HSA_API_TABLE_ENUM_INFO_H
#define LUTHIER_ROCPROFILER_HSA_API_TABLE_ENUM_INFO_H
#include "luthier/Rocprofiler/ApiTableEnumInfo.h"
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

namespace luthier::rocprofiler {

template <> struct ApiTableEnumInfo<ROCPROFILER_HSA_TABLE> {
  using ApiTableType = ::HsaApiTable;
  constexpr static auto NumApiTables = 1;
  constexpr static auto ApiTableName = "HSA";
};

} // namespace luthier::rocprofiler

#endif
