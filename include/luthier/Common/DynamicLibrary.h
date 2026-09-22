//===-- DynamicLibrary.h ----------------------------------------*- C++ -*-===//
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
/// Defines \c luthier::DynamicLibrary, an owning wrapper around the \c void*
/// handle returned by \c dlopen / \c dlmopen, and
/// \c luthier::DynamicLibraryFunctionEntry, the customization point mapping an
/// API entry to the symbol name it is looked up under.
///
/// Together they make a dynamically-resolved call look like a direct one: with
/// an entry registered for \c hsa_init, a caller writes
/// \code
///   Lib.callFunction<hsa_init>();
/// \endcode
/// instead of repeating the symbol name and the function type at every call
/// site.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_DYNAMIC_LIBRARY_H
#define LUTHIER_COMMON_DYNAMIC_LIBRARY_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "luthier/Common/GenericLuthierError.h"
#include <dlfcn.h>
#include <llvm/Support/Error.h>
#include <utility>

namespace luthier {

/// \brief Primary template (customization point) giving the symbol name and
/// the function type an API entry is dynamically resolved under.
template <auto Func> struct DynamicLibraryFunctionEntry;

/// \brief Registers \p NAME as a dynamically-resolvable API entry, mapping it
/// to its own name as a string and to its declared type.
#define LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(NAME)                           \
  template <> struct ::luthier::DynamicLibraryFunctionEntry<&NAME> {           \
    using FunctionType = decltype(NAME);                                       \
    static constexpr const char *FunctionName = #NAME;                         \
  };

/// \brief A wrapper around a dynamically loaded library.
class DynamicLibrary {
  /// The \c dlopen / \c dlmopen handle, or \c nullptr when this instance holds
  /// nothing.
  void *Handle{nullptr};

  /// Whether the wrapper owns the handle or not.
  bool Owned{false};

  DynamicLibrary(void *Handle, bool Owned) : Handle(Handle), Owned(Owned) {}

  /// \brief Returns the most recent dynamic loading error.
  static llvm::Error getLastError();

public:
  /// \brief Loads the library at \p Path into the caller's own linker
  /// namespace, via \c dlopen.
  /// \param Path the library name/path as resolved by the dlfcn search rules
  /// \param Flags \c dlopen flags
  /// \return the loaded library, or an \c llvm::Error
  static llvm::Expected<DynamicLibrary>
  open(const char *Path, int Flags = RTLD_NOW | RTLD_GLOBAL);

  /// \brief Loads the library at \p Path into the linker namespace \p Lmid,
  /// via \c dlmopen.
  ///
  /// \param Lmid the target namespace
  /// \param Path the library name/path as resolved by the dlfcn search rules
  /// \param Flags \c dlmopen flags
  /// \return the loaded library, or an \c llvm::Error
  static llvm::Expected<DynamicLibrary>
  openInNamespace(Lmid_t Lmid, const char *Path,
                  int Flags = RTLD_NOW | RTLD_LOCAL);

  /// \brief Wraps an existing handle \b without taking ownership of it.
  static DynamicLibrary borrow(void *Handle) {
    return {Handle, /*Owned=*/false};
  }

  DynamicLibrary(const DynamicLibrary &) = delete;
  DynamicLibrary &operator=(const DynamicLibrary &) = delete;

  DynamicLibrary(DynamicLibrary &&Other) noexcept
      : Handle(std::exchange(Other.Handle, nullptr)),
        Owned(std::exchange(Other.Owned, false)) {}
  DynamicLibrary &operator=(DynamicLibrary &&Other) noexcept;

  ~DynamicLibrary();

  /// \brief Drops this instance's reference to the library, if it owns one.
  void close();

  /// \brief Gives up ownership without closing, returning the raw handle.
  void *release();

  /// \return the underlying \c dlopen / \c dlmopen handle
  [[nodiscard]] void *getHandle() const { return Handle; }

  /// \return whether this instance holds a library it will close
  [[nodiscard]] bool isOwning() const { return Owned && Handle != nullptr; }

  /// \return whether this instance holds a valid handle.
  [[nodiscard]] explicit operator bool() const {
    return Handle != nullptr || Owned;
  }

  /// \brief Returns the id of the linker namespace this library was loaded
  /// into, or \c -1 if it cannot be determined.
  [[nodiscard]] Lmid_t getNamespace() const;

  /// \brief Resolves the symbol \p Name as an object of type \p T.
  /// \tparam T the type of the symbol's \e referent
  /// \return the symbol's address, or \c nullptr if it is not found
  template <typename T> [[nodiscard]] T *getSymbol(const char *Name) const {
    return static_cast<T *>(::dlsym(Handle, Name));
  }

  /// \brief Resolves the API entry \p Func under the name its
  /// \c DynamicLibraryFunctionEntry gives it.
  /// \tparam Func a function registered with
  /// \c LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY
  /// \return a pointer to the library's definition, or \c nullptr if the
  /// library does not export it
  template <auto Func> [[nodiscard]] auto *getFunction() const {
    using Entry = DynamicLibraryFunctionEntry<Func>;
    return reinterpret_cast<typename Entry::FunctionType *>(
        ::dlsym(Handle, Entry::FunctionName));
  }

  /// \return whether this library exports the API entry \p Func
  template <auto Func> [[nodiscard]] bool hasFunction() const {
    return getFunction<Func>() != nullptr;
  }

  /// \brief Resolves the API entry \p Func and calls it with \p Arguments.
  /// \tparam Func a function registered with
  /// \c LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY
  /// \param Arguments the arguments to forward to \p Func
  /// \return whatever \p Func returns
  template <auto Func, typename... ArgsT>
  decltype(auto) callFunction(ArgsT &&...Arguments) const {
    auto F = getFunction<Func>();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        F != nullptr,
        llvm::formatv("Failed to find the function {0} in the dynamic library",
                      DynamicLibraryFunctionEntry<Func>::FunctionName)));
    return getFunction<Func>()(std::forward<ArgsT>(Arguments)...);
  }
};

} // namespace luthier

#endif // LUTHIER_COMMON_DYNAMIC_LIBRARY_H
