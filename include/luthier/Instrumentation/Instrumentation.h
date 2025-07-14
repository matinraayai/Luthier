//===-- Instrumentation.h ---------------------------------------*- C++ -*-===//
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
/// \file
/// Provides a set of high-level instrumentation functions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_INSTRUMENTATION_H
#define LUTHIER_INSTRUMENTATION_INSTRUMENTATION_H

namespace luthier {

/// Initializes the LLVM AMDGPU backend; Must be called before the client
/// tool starts generating any instrumented code
void initializeAMDGPULLVMBackend();

}

#endif


