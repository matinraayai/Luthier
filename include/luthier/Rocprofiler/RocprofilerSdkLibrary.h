//===-- RocprofilerSdkLibrary.h ---------------------------------*- C++ -*-===//
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
/// Declares utilities for obtaining a \c luthier::DynamicLibrary handle to
/// the rocprofiler-sdk library used by the target application.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_ROCPROFILER_SDK_LIBRARY_H
#define LUTHIER_ROCPROFILER_ROCPROFILER_SDK_LIBRARY_H
#include "luthier/Common/DynamicLibrary.h"
#include "luthier/Rocprofiler/ApiTable.h"
#include <llvm/Support/Error.h>

namespace luthier::rocprofiler {

/// \brief Opens the rocprofiler-sdk library inside the linker namespace
/// \p Lmid, which by default is the namespace of the target application
/// \param Lmid the linker namespace to open rocprofiler-sdk in
/// \return an owning \c DynamicLibrary of rocprofiler-sdk, or an
/// \c llvm::Error if the library failed to open
llvm::Expected<DynamicLibrary>
openRocprofilerSdkLibrary(Lmid_t Lmid = LM_ID_BASE);

} // namespace luthier::rocprofiler

#endif
