//===-- Attributes.h ---------------------------------------------*- C++-*-===//
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
/// Defines attribute parsers used in Luthier's CXX tool compilation process.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CXX_COMPILATION_ATTRIBUTES_H
#define LUTHIER_TOOL_CXX_COMPILATION_ATTRIBUTES_H
#include "clang/Sema/ParsedAttr.h"

namespace luthier {

/// \brief Handles the \c [[luthier::export_function_handle]] attribute.
///
/// If applied to a \c __device__ function (concrete or templated), this
/// attribute (with the help of the \c ExportDevFuncHostHandleConsumer) ensures
/// that a sibling \c __host__ overload of it is present for use within the host
/// portion of the translation unit.
///
/// The attribute parser performs the following actions after encountering an
/// annotated \c __device__ function:
///   - If the (templated) function's declaration is not \c __host__ and
///   does not have an associated \c __host__ declaration or definition yet
///   defined, the declaration is first cloned. The cloned declaration is then
///   given the \c __host__ attribute and annotated with both
///   \c ExportFunctionHandleAutoGenMarker and \c ExportFunctionHandleMarker.
///   The \c ExportFunctionHandleAutoGenMarker indicates to
///   \c ExportDevFuncHostHandleConsumer that this declaration was synthesized
///   and should be treated accordingly.
///   - If the (templated) function's declaration is also \c __host__, the
///   function itself will only be annotated with \c ExportFunctionHandleMarker.
///   - If the (templated) function's declaration already has a \c __host__
///   version declared, both the \c __device__ and \c __host__ declarations will
///   be marked with the \c ExportFunctionHandleMarker annotation.
///
/// All other usage of this annotation is ignored, and no diagnostics will be
/// emitted when this happens.
/// \note Presence of this attribute to a \c __device__ function does not
/// imply that the exported host function handle will not be
/// Dead-Code Eliminated. An explicit \c used attribute must be used with the
/// \c __device__ function and its \c __host__ handle to prevent that from
/// happening.
/// \sa ExportDevFuncHostHandleConsumer
struct LuthierExportFunctionHandleAttrInfo : public clang::ParsedAttrInfo {
  LuthierExportFunctionHandleAttrInfo();

  bool diagAppertainsToDecl(clang::Sema &S, const clang::ParsedAttr &Attr,
                            const clang::Decl *D) const override;

  AttrHandling
  handleDeclAttribute(clang::Sema &S, clang::Decl *D,
                      const clang::ParsedAttr &Attr) const override;
};

} // namespace luthier

#endif
