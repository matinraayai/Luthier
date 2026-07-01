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

static constexpr llvm::StringLiteral HipCUIDPrefix{
    LUTHIER_STRINGIFY(LUTHIER_HIP_CUID_PREFIX)};

static constexpr llvm::StringLiteral IntrinsicAttribute{
    LUTHIER_STRINGIFY(LUTHIER_INTRINSIC_ATTRIBUTE)};

static constexpr llvm::StringLiteral InjectedPayloadAttribute{
    LUTHIER_STRINGIFY(LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE)};

#define EntryPointAddrAttr "luthier.function.entrypoint.addr"

#define InitialEntryPointAttr "luthier.function.initial_entrypoint"

static constexpr const char *InitialExecutionPointAttr =
    "luthier.function.initial_execution_point";

static constexpr const char *TargetInstrPointAttr =
    "luthier.target_instr_point";

/// \brief Tags a \c __device__ function to be accessed by the tool's
/// host code
#define LUTHIER_EXPORT_FUNCTION_HANDLE_ATTR                                    \
  __attribute__((luthier_export_function_handle))

#define LUTHIER_HOOK_ANNOTATE                                                  \
  __attribute__((device, used,                                                 \
                 annotate(LUTHIER_STRINGIFY(LUTHIER_HOOK_ATTRIBUTE))))         \
  LUTHIER_EXPORT_FUNCTION_HANDLE_ATTR extern "C" void

/// Attribute pack for hooks declared as \c static members of a tool
/// class. The base \c LUTHIER_HOOK_ANNOTATE macro embeds an
/// \c extern \c "C" linkage specifier plus a \c void return type, which
/// is incompatible with class-scope (the linkage spec is illegal there)
/// and with the out-of-line definition syntax for static member
/// functions (where \c static is not repeated). This macro expands to
/// just the device + hook-tag + export-handle attribute block so the
/// caller writes \c static + the return type themselves at the
/// declaration, and writes the same attributes at the definition. The
/// host-shadow handle for a static member hook is \c &MyTool::myHook
/// — HIP-Clang generates an \c __hipRegisterFunction entry for it the
/// same way it does for free \c __device__ functions.
#define LUTHIER_HOOK_MEMBER_ATTR                                               \
  __attribute__((                                                              \
      device, used,                                                            \
      annotate(LUTHIER_STRINGIFY(                                              \
          LUTHIER_HOOK_ATTRIBUTE)))) LUTHIER_EXPORT_FUNCTION_HANDLE_ATTR

/// Marks a non-hook \c __device__ function as host-addressable. Use this
/// when host code needs the address of a device function that is not a
/// Luthier hook (e.g. a helper invoked indirectly).
#define LUTHIER_HOST_VISIBLE_DEVICE_FN                                         \
  __attribute__((device)) LUTHIER_EXPORT_FUNCTION_HANDLE_ATTR

/// Annotation string the plugin attaches to every \c FunctionDecl carrying
/// the \c [[luthier::export_function_handle]] attribute. External tools
/// can query this via \c AnnotateAttr to decide whether a device function
/// is host-addressable.
inline constexpr llvm::StringLiteral ExportFunctionHandleMarker =
    "luthier.export_function_handle";

/// Annotation added exclusively to \c __host__ sibling declarations that were
/// synthesized by \c EmitHostHandleAttrInfo — as opposed to
/// \c __host__ overloads written by the user. The
/// \c EmitHostSiblingForDevFuncConsumer uses this tag to find synthesized
/// siblings that still need an empty body and/or an access-specifier
/// correction, while leaving user-defined overloads untouched.
inline constexpr llvm::StringLiteral ExportFunctionHandleAutoGenMarker =
    "luthier.export_function_handle.autogen";

/// Prefix for the synthesized host-side handle function names for tagged
/// \c __device__ function templates. The full handle name is
/// \c BuiltinDevFuncHandlePrefix followed by the Itanium-mangled name of the
/// original specialization. The prefix is chosen to be a valid C++ identifier
/// that Clang can Itanium-mangle, producing a demangleable symbol.
inline constexpr llvm::StringLiteral BuiltinDevFuncHandlePrefix =
    "__luthier_builtin_dev_func_handle__";

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