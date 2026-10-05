//===-- HipApiTableEnumInfo.h -----------------------------------*- C++ -*-===//
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
/// Provides the \c ApiTableEnumInfo specializations for the HIP API tables.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_HIP_API_TABLE_ENUM_INFO_H
#define LUTHIER_ROCPROFILER_HIP_API_TABLE_ENUM_INFO_H
#include "luthier/Rocprofiler/ApiTableEnumInfo.h"
/// hip_api_trace.hpp declares typedefs for the *entire* HIP API surface —
/// including the R0000-suffixed deprecated entries and the GL interop
/// entries — but doesn't itself include the headers that define those
/// types; Pull them in here
#include <hip/hip_runtime.h>
#include <hip/hip_deprecated.h>
#include <hip/hip_gl_interop.h>
#include <hip/amd_detail/hip_api_trace.hpp>

namespace luthier::rocprofiler {

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE> {
  using ApiTableType = ::HipCompilerDispatchTable;
  constexpr static auto NumApiTables = 1;
  constexpr static auto ApiTableName = "HIP Compiler";
};

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE> {
  using ApiTableType = ::HipDispatchTable;
  constexpr static auto NumApiTables = 1;
  constexpr static auto ApiTableName = "HIP Runtime";
};

} // namespace luthier::rocprofiler

#endif
