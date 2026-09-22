//===-- DynamicLibraryFunctionEntry.h ---------------------------*- C++ -*-===//
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
///
/// \file
/// Defines \c luthier::DynamicLibraryFunctionEntry, the customization point
/// mapping an API entry to the symbol name, used by
/// \c luthier::DynamicLibrary for looked up.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_DYNAMIC_LIBRARY_FUNCTION_ENTRY_H
#define LUTHIER_COMMON_DYNAMIC_LIBRARY_FUNCTION_ENTRY_H

namespace luthier {
/// \brief Primary template (customization point) giving the symbol name and
/// the function type an API entry is dynamically resolved under.
template <auto Func> struct DynamicLibraryFunctionEntry;

/// \brief Registers \p NAME as a dynamically-resolvable API entry, mapping it
/// to its own name as a string and to its declared type.
#define LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(NAME)                           \
  template <> struct ::luthier::DynamicLibraryFunctionEntry<NAME> {            \
    using ApiType = decltype(&(NAME));                                         \
    static constexpr auto ApiName = #NAME;                                     \
  };
} // namespace luthier

#endif
