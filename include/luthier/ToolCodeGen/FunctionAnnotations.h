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
/// \file FunctionAnnotations.h
/// Defines a set of function annotations, prefixes and suffixes used throughout
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

/// Prefix appended to all host-accessible device functions' handle kernels
#define LUTHIER_DEVICE_FUNCTION_HANDLE_PREFIX __luthier_builtin_dev_func_handle_

/// All hooks in instrumentation modules must have this attribute
#define LUTHIER_HOOK_ATTRIBUTE luthier.function.hook

/// All bindings to Luthier intrinsics must have this attribute. The
/// value of this attribute must be the base name of the intrinsic e.g.
/// \c luthier::readReg
#define LUTHIER_INTRINSIC_ATTRIBUTE luthier.intrinsic

/// Prefix of the CUID symbol inside a HIP module
#define LUTHIER_HIP_CUID_PREFIX __hip_cuid_

/// All injected payload functions during instrumentation (i.e. functions that
/// their machine code will be inserted before an instrumentation point) must
/// have this attribute
#define LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE luthier.function.injected_payload

static constexpr llvm::StringLiteral DevFuncHandlePrefix{
    LUTHIER_STRINGIFY(LUTHIER_DEVICE_FUNCTION_HANDLE_PREFIX)};

static constexpr llvm::StringLiteral HipCUIDPrefix{
    LUTHIER_STRINGIFY(LUTHIER_HIP_CUID_PREFIX)};

static constexpr llvm::StringLiteral IntrinsicAttribute{
    LUTHIER_STRINGIFY(LUTHIER_INTRINSIC_ATTRIBUTE)};

static constexpr llvm::StringLiteral InjectedPayloadAttribute{
    LUTHIER_STRINGIFY(LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE)};

#define EntryPointAddrAttr "luthier.function.entrypoint.addr"

#define InitialEntryPointAttr "luthier.function.initial_entrypoint"

#define ElfSymbolAttr           "luthier.entryfuncion.elf.symbol"

/// Annotation strings attached to the inline-static \c llvm::ArrayRef slots
/// on \c DeviceToolCodeFatBinaryLoader. \c LoadHIPFATBinaryInfoPass matches
/// by string and rewrites each slot's initializer to the appropriate
/// constant-array view.
///
/// Each annotation has two forms: a bare-token macro consumed by
/// \c LUTHIER_ANNOTATE_VARIABLE for the \c __attribute__((annotate(...)))
/// site, and a \c static \c constexpr \c StringLiteral for runtime
/// comparisons in the IR pass.
#define LUTHIER_HIP_FAT_BINARIES_ATTR luthier.loader.hip_fat_binaries
#define LUTHIER_HIP_KERNELS_ATTR luthier.loader.hip_kernels
#define LUTHIER_HIP_DEVICE_FUNCTIONS_ATTR luthier.loader.hip_device_functions
#define LUTHIER_HIP_DEVICE_VARS_ATTR luthier.loader.hip_device_vars
#define LUTHIER_HIP_MANAGED_VARS_ATTR luthier.loader.hip_managed_vars
#define LUTHIER_HIP_TEXTURE_VARS_ATTR luthier.loader.hip_texture_vars
#define LUTHIER_HIP_SURFACE_VARS_ATTR luthier.loader.hip_surface_vars

static constexpr llvm::StringLiteral HipFatBinariesAttr{
    LUTHIER_STRINGIFY(LUTHIER_HIP_FAT_BINARIES_ATTR)};
static constexpr llvm::StringLiteral HipKernelsAttr{
    LUTHIER_STRINGIFY(LUTHIER_HIP_KERNELS_ATTR)};
static constexpr llvm::StringLiteral HipDeviceFunctionsAttr{
    LUTHIER_STRINGIFY(LUTHIER_HIP_DEVICE_FUNCTIONS_ATTR)};
static constexpr llvm::StringLiteral HipDeviceVarsAttr{
    LUTHIER_STRINGIFY(LUTHIER_HIP_DEVICE_VARS_ATTR)};
static constexpr llvm::StringLiteral HipManagedVarsAttr{
    LUTHIER_STRINGIFY(LUTHIER_HIP_MANAGED_VARS_ATTR)};
static constexpr llvm::StringLiteral HipTextureVarsAttr{
    LUTHIER_STRINGIFY(LUTHIER_HIP_TEXTURE_VARS_ATTR)};
static constexpr llvm::StringLiteral HipSurfaceVarsAttr{
    LUTHIER_STRINGIFY(LUTHIER_HIP_SURFACE_VARS_ATTR)};

static constexpr const char *InitialExecutionPointAttr =
    "luthier.function.initial_execution_point";

static constexpr const char *TargetInstrPointAttr =
    "luthier.target_instr_point";

/// Annotation string attached to every \c __host__ function serving as a
/// handle for its \c __device__ overload inside the host code.
inline constexpr llvm::StringLiteral ExportFunctionHandleMarker =
    "luthier.function.export_device_handle";

/// Annotation string attached only to \c __host__ handles that were
/// \e synthesized automatically (as opposed to pre-existing
/// \c __host__ overloads that get annotated in place).
inline constexpr llvm::StringLiteral SynthesizedExportFunctionHandleMarker =
    "luthier.function.synthesized_export_device_handle";

/// Tag a variable declaration with a Clang \c annotate attribute. \p Sym
/// is a bare-token macro (e.g. \c LUTHIER_HIP_FAT_BINARIES_ATTR) that
/// expands to a dotted symbol; the preprocessor stringifies it for the
/// attribute.
#if defined(__clang__)
#define LUTHIER_ANNOTATE_VARIABLE(Sym)                                         \
  __attribute__((annotate(LUTHIER_STRINGIFY(Sym))))
#else
#define LUTHIER_ANNOTATE_VARIABLE(Sym)
#endif

void setFunctionEntryPoint(llvm::Function &F, EntryPoint EP);

std::optional<EntryPoint> getFunctionEntryPoint(llvm::Function &F);

} // namespace luthier

#endif