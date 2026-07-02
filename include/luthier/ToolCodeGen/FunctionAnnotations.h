//===-- FunctionAnnotations.h -------------------------------------*-C++-*-===//
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
/// Defines a set of annotations, prefixes and suffixes used throughout
/// the code generation process, as well as methods to set/extract information
/// related to them from the IR function.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_FUNCTION_ANNOTATIONS_H
#define LUTHIER_TOOL_CODE_GEN_FUNCTION_ANNOTATIONS_H
#include "luthier/ToolCodeGen/EntryPoint.h"
#include <llvm/ADT/StringRef.h>
#include <optional>

namespace llvm {

class Function;

}

namespace luthier {

//===----------------------------------------------------------------------===//
// Utility macros (see https://github.com/pfultz2/Cloak)
//===----------------------------------------------------------------------===//
#define LUTHIER_STRINGIFY(S) LUTHIER_PRIMITIVE_STR(S)
#define LUTHIER_PRIMITIVE_STR(S) #S

#define LUTHIER_CAT(a, ...) LUTHIER_PRIMITIVE_CAT(a, __VA_ARGS__)
#define LUTHIER_PRIMITIVE_CAT(a, ...) a##__VA_ARGS__

/// All bindings to Luthier intrinsics must have this attribute. The
/// value of this attribute must be the base name of the intrinsic e.g.
/// \c luthier::readReg
#define LUTHIER_INTRINSIC_ATTRIBUTE luthier.intrinsic

/// All injected payload functions during instrumentation (i.e. functions that
/// their machine code will be inserted before an instrumentation point) must
/// have this attribute
#define LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE luthier.function.injected_payload

static constexpr llvm::StringLiteral IntrinsicAttribute{
    LUTHIER_STRINGIFY(LUTHIER_INTRINSIC_ATTRIBUTE)};

static constexpr llvm::StringLiteral InjectedPayloadAttribute{
    LUTHIER_STRINGIFY(LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE)};

static constexpr llvm::StringLiteral EntryPointAddrAttr{
    "luthier.function.entrypoint.addr"};

static constexpr llvm::StringLiteral InitialEntryPointAttr{
    "luthier.function.initial_entrypoint"};

static constexpr llvm::StringLiteral InitialExecutionPointAttr =
    "luthier.function.initial_execution_point";

static constexpr llvm::StringLiteral TargetInstrPointAttr =
    "luthier.target_instr_point";

/// Annotation string attached to every \c __host__ function serving as a
/// handle for its \c __device__ overload inside the host code.
inline constexpr llvm::StringLiteral ExportFunctionHandleMarker =
    "luthier.function.export_device_handle";

/// TODO: Move these to the entry point file.
void setFunctionEntryPoint(llvm::Function &F, EntryPoint EP);

std::optional<EntryPoint> getFunctionEntryPoint(llvm::Function &F);

} // namespace luthier

#endif