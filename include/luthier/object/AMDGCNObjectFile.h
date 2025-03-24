#ifndef LUTHIER_COMMON_AMDGCN_OBJECT_FILE_H
#define LUTHIER_COMMON_AMDGCN_OBJECT_FILE_H
#include "ELFObjectUtils.h"

namespace luthier {

class AMDGCNObjectFile;

/// \brief Wrapper around a \c ELFSymbolRefWrapper representing a
/// symbol inside a \c AMDGCNObjectFile
class AMDGCNElfSymbolRef : public llvm::object::ELFSymbolRef {
  /// \c AMDGCNObjectFile can skip checks on \c ELFSymbolRefWrapper
  friend AMDGCNObjectFile;

protected:
  // NOLINTBEGIN(google-explicit-constructor)
  /*implicit*/ AMDGCNElfSymbolRef(const llvm::object::SymbolRef &S)
      : llvm::object::ELFSymbolRef(S) {};
  // NOLINTEND(google-explicit-constructor)

public:
  /// \returns If \p S is a \c AMDGCNElfSymbolRef returns \p S as one;
  /// Otherwise, returns \c std::nullopt
  static std::optional<AMDGCNElfSymbolRef>
  getIfAMDGCNSymbolRef(const llvm::object::SymbolRef &S) {
    auto *Obj = S.getObject();
    if (Obj && llvm::isa<AMDGCNObjectFile>(*Obj))
      return AMDGCNElfSymbolRef{S};
    else
      return std::nullopt;
  }

  const luthier::AMDGCNObjectFile *getObject() const {
    return llvm::cast<luthier::AMDGCNObjectFile>(
        llvm::object::ELFSymbolRef::getObject());
  }

  /// \returns \c true if the symbol is a kernel descriptor, \c false if not,
  /// \c llvm::Error if an error was encountered
  [[nodiscard]] llvm::Expected<bool> isKernelDescriptor() const;

  /// \returns \c true if the symbol is a global or local variable, \c false
  /// if not, \c llvm::Error if an error was encountered
  [[nodiscard]] llvm::Expected<bool> isVariable() const;

  /// \returns \c true if the symbol contains the device code of a kernel
  /// descriptor, \c false if not, \c llvm::Error if an error was encountered
  [[nodiscard]] llvm::Expected<bool> isKernelFunction() const;

  /// \returns \c true if the symbol is a function that can only be invoked on
  /// the device side, \c false if not, \c llvm::Error if an error was
  /// encountered
  [[nodiscard]] llvm::Expected<bool> isDeviceFunction() const;
};

class amdgcn_elf_symbol_iterator : public llvm::object::elf_symbol_iterator {
private:
  /// This is so that the implicit constructor from
  /// \c llvm::object::basic_symbol_iterator works without hassle
  friend AMDGCNObjectFile;

  // NOLINTBEGIN(google-explicit-constructor)
  /* implicit */ amdgcn_elf_symbol_iterator(
      const llvm::object::basic_symbol_iterator &B)
      : llvm::object::elf_symbol_iterator(B) {}
  // NOLINTEND(google-explicit-constructor)

public:
  // NOLINTBEGIN(google-explicit-constructor)
  /* implicit */ amdgcn_elf_symbol_iterator(
      const luthier::AMDGCNElfSymbolRef &S)
      : llvm::object::elf_symbol_iterator(S) {}
  // NOLINTEND(google-explicit-constructor)

  const AMDGCNElfSymbolRef *operator->() const {
    return static_cast<const AMDGCNElfSymbolRef *>(
        elf_symbol_iterator::operator->());
  }

  const AMDGCNElfSymbolRef &operator*() const {
    return static_cast<const AMDGCNElfSymbolRef &>(
        elf_symbol_iterator::operator*());
  }
};

#define CREATE_AMDGCN_SYMBOL_ITERATOR(IteratorName, SymbolType, SymbolChecker) \
  class IteratorName : public amdgcn_elf_symbol_iterator {                     \
  private:                                                                     \
    /** Reference to an external error that needs to be checked **/            \
    /** every time the iterator is incremented**/                              \
    llvm::Error &Err;                                                          \
                                                                               \
  public:                                                                      \
    IteratorName(amdgcn_elf_symbol_iterator Iter, llvm::Error &Err)            \
        : amdgcn_elf_symbol_iterator(Iter), Err(Err) {}                        \
                                                                               \
    IteratorName(const SymbolType &S, llvm::Error &Err)                        \
        : amdgcn_elf_symbol_iterator(S), Err(Err) {}                           \
                                                                               \
    const SymbolType *operator->() const {                                     \
      return static_cast<const SymbolType *>(                                  \
          amdgcn_elf_symbol_iterator::operator->());                           \
    }                                                                          \
                                                                               \
    const SymbolType &operator*() const {                                      \
      return static_cast<const SymbolType &>(                                  \
          amdgcn_elf_symbol_iterator::operator*());                            \
    }                                                                          \
    content_iterator &operator++();                                            \
  }

#define IMPLEMENT_AMDGCN_ITERATOR_INCREMENT_OPERATOR(IteratorName, SymbolType, \
                                                     SymbolChecker)            \
  IteratorName::content_iterator IteratorName::&operator++() {                 \
    auto EndSymbolIter = (*this)->getObject()->symbol_end();                   \
                                                                               \
    while (*this != EndSymbolIter) {                                           \
      llvm::Expected<bool> IsSymbolKindOrErr = (*this)->SymbolChecker();       \
      if (auto Error = IsSymbolKindOrErr.takeError()) {                        \
        this->Err = std::move(Error);                                          \
        return *this;                                                          \
      }                                                                        \
      if (*IsSymbolKindOrErr)                                                  \
        return *this;                                                          \
      amdgcn_elf_symbol_iterator::operator++();                                \
    }                                                                          \
    return *this;                                                              \
  }

class AMDGCNVariableSymbolRef : public AMDGCNElfSymbolRef {
private:
  // NOLINTBEGIN(google-explicit-constructor)
  /*implicit*/ AMDGCNVariableSymbolRef(llvm::object::SymbolRef &S)
      : AMDGCNElfSymbolRef(S) {};
  // NOLINTEND(google-explicit-constructor)

public:
  static llvm::Expected<std::optional<AMDGCNVariableSymbolRef>>
  getAsAMDGCNVariableSymbol(const llvm::object::SymbolRef &S);
};

CREATE_AMDGCN_SYMBOL_ITERATOR(amdgcn_variable_symbol_iterator,
                              AMDGCNVariableSymbolRef, isVariable);

class AMDGCNKernelFuncSymbolRef;

/// \brief Wrapper around a \c ELFSymbolRefWrapper which represents a
/// symbol inside a \c AMDGCNObjectFile
class AMDGCNKernelDescSymbolRef : public AMDGCNElfSymbolRef {
private:
  // NOLINTBEGIN(google-explicit-constructor)
  /*implicit*/ AMDGCNKernelDescSymbolRef(llvm::object::SymbolRef &S)
      : AMDGCNElfSymbolRef(S) {};
  // NOLINTEND(google-explicit-constructor)

public:
  static llvm::Expected<std::optional<AMDGCNKernelDescSymbolRef>>
  getAsAMDGCNKernelDescSymbol(const llvm::object::SymbolRef &S);

  static llvm::Expected<AMDGCNKernelDescSymbolRef>
  fromKernelFunction(const AMDGCNKernelFuncSymbolRef &S);

  [[nodiscard]] llvm::Expected<AMDGCNKernelFuncSymbolRef>
  getKernelFunctionSymbol() const;
};

CREATE_AMDGCN_SYMBOL_ITERATOR(amdgcn_kernel_descriptor_iterator,
                              AMDGCNKernelDescSymbolRef, isKernelDescriptor);

class AMDGCNKernelFuncSymbolRef : public AMDGCNElfSymbolRef {
private:
  // NOLINTBEGIN(google-explicit-constructor)
  /*implicit*/ AMDGCNKernelFuncSymbolRef(llvm::object::SymbolRef &S)
      : AMDGCNElfSymbolRef(S) {};
  // NOLINTEND(google-explicit-constructor)

public:
  static llvm::Expected<std::optional<AMDGCNKernelFuncSymbolRef>>
  getAsAMDGCNKernelFuncSymbol(const llvm::object::SymbolRef &S);

  static llvm::Expected<AMDGCNKernelFuncSymbolRef>
  fromKernelDescriptor(const AMDGCNKernelDescSymbolRef &S);

  [[nodiscard]] llvm::Expected<AMDGCNKernelDescSymbolRef>
  getKernelDescriptorSymbol() const {
    return AMDGCNKernelDescSymbolRef::fromKernelFunction(*this);
  }
};

CREATE_AMDGCN_SYMBOL_ITERATOR(amdgcn_kernel_function_iterator,
                              AMDGCNKernelFuncSymbolRef, isKernelFunction);

class AMDGCNDeviceFuncSymbolRef : public AMDGCNElfSymbolRef {
private:
  // NOLINTBEGIN(google-explicit-constructor)
  /*implicit*/ AMDGCNDeviceFuncSymbolRef(llvm::object::SymbolRef &S)
      : AMDGCNElfSymbolRef(S) {};
  // NOLINTEND(google-explicit-constructor)

public:
  static llvm::Expected<std::optional<AMDGCNDeviceFuncSymbolRef>>
  getAsAMDGCNDeviceFuncSymbol(const llvm::object::SymbolRef &S);
};

CREATE_AMDGCN_SYMBOL_ITERATOR(amdgcn_device_function_iterator,
                              AMDGCNDeviceFuncSymbolRef, isDeviceFunction);
#undef CREATE_AMDGCN_SYMBOL_ITERATOR

using amdgcn_elf_symbol_iterator_range =
    llvm::iterator_range<amdgcn_elf_symbol_iterator>;

using amdgcn_variable_symbol_iterator_range =
    llvm::iterator_range<amdgcn_variable_symbol_iterator>;

using amdgcn_kernel_descriptor_iterator_range =
    llvm::iterator_range<amdgcn_kernel_descriptor_iterator>;

using amdgcn_kernel_function_iterator_range =
    llvm::iterator_range<amdgcn_kernel_function_iterator>;

using amdgcn_device_function_iterator_range =
    llvm::iterator_range<amdgcn_device_function_iterator>;

/// \brief Adds additional methods to the \c llvm::object::ELF64LEObjectFile
/// class specific to amdgcn code object files
/// As per <a href="https://llvm.org/docs/AMDGPUUsage.html#elf-code-object">
/// AMDGPU backend documentation</a>, AMDGCN object files are 64-bit LE.
class AMDGCNObjectFile : public llvm::object::ELF64LEObjectFile {
protected:
  AMDGCNObjectFile(llvm::object::ELF64LEObjectFile &&ObjFile)
      : llvm::object::ELF64LEObjectFile(std::move(ObjFile)) {}

  static llvm::Expected<AMDGCNObjectFile> create(llvm::MemoryBufferRef Object,
                                                 bool InitContent = true) {

    llvm::Expected<llvm::object::ELF64LEObjectFile> Elf64LEObj =
        llvm::object::ELF64LEObjectFile::create(Object, InitContent);
    LUTHIER_RETURN_ON_ERROR(Elf64LEObj.takeError());
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(Elf64LEObj->makeTriple().isAMDGCN(),
                            "Passed ELF object is not for an amdgcn target."));
    return AMDGCNObjectFile{std::move(*Elf64LEObj)};
  }

public:
  /// Parses the ELF file pointed to by \p Elf into a \c AMDGCNObjectFile.
  /// \param ELF \c llvm::StringRef encompassing the ELF file in memory
  /// \return a \c std::unique_ptr<AMDGCNObjectFile> on successful
  /// parsing, an \c llvm::Error on failure
  static llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
  createAMDGCNObjectFile(llvm::MemoryBufferRef Object,
                         bool InitContent = true) {
    llvm::Expected<AMDGCNObjectFile> GCNObjFileOrErr =
        create(Object, InitContent);
    LUTHIER_RETURN_ON_ERROR(GCNObjFileOrErr.takeError());
    return std::make_unique<AMDGCNObjectFile>(std::move(*GCNObjFileOrErr));
  }

  /// Parses the ELF file pointed to by \p Elf into a \c AMDGCNObjectFile.
  /// \param ELF \c llvm::StringRef encompassing the ELF file in memory
  /// \return a \c std::unique_ptr<AMDGCNObjectFile> on successful
  /// parsing, an \c llvm::Error on failure
  static llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
  createAMDGCNObjectFile(llvm::StringRef Elf) {
    std::unique_ptr<llvm::MemoryBuffer> Buffer =
        llvm::MemoryBuffer::getMemBuffer(Elf, "", false);
    return createAMDGCNObjectFile(*Buffer);
  }

  /// Parses the ELF file pointed to by \b Elf into a \b AMDGCNObjectFile.
  /// \param ELF \p llvm::ArrayRef<uint8_t> encompassing the ELF file in memory
  static llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
  createAMDGCNObjectFile(llvm::ArrayRef<uint8_t> Elf) {
    return createAMDGCNObjectFile(llvm::toStringRef(Elf));
  }

  llvm::Expected<std::optional<luthier::AMDGCNElfSymbolRef>>
  lookupSymbol(llvm::StringRef SymbolName) const {
    auto Out = luthier::lookupSymbolByName(*this, SymbolName);
    LUTHIER_RETURN_ON_ERROR(Out.takeError());
    if (Out->has_value())
      return AMDGCNElfSymbolRef{**Out};
    else
      return std::nullopt;
  }

  amdgcn_elf_symbol_iterator_range symbols() const {
    return amdgcn_elf_symbol_iterator_range(symbol_begin(), symbol_end());
  }

  amdgcn_elf_symbol_iterator dynamic_symbol_begin() const {
    return llvm::object::ELF64LEObjectFile::dynamic_symbol_begin();
  }

  amdgcn_elf_symbol_iterator dynamic_symbol_end() const {
    return llvm::object::ELF64LEObjectFile::dynamic_symbol_end();
  }

#define CREATE_AMDGCN_OBJECT_SYMBOL_ITERATOR_FUNCTION(                         \
    IterName, IterType, SymbolCheckFunc, IterRangeType)                        \
  IterType IterName##_begin(llvm::Error &Err) const {                          \
    amdgcn_elf_symbol_iterator Out = symbol_begin();                           \
    for (; Out != symbol_end(); ++Out) {                                       \
      llvm::Expected<bool> IsSymOrErr = Out->SymbolCheckFunc();                \
      if (auto Error = IsSymOrErr.takeError()) {                               \
        Err = std::move(Error);                                                \
        Out = symbol_end();                                                    \
        break;                                                                 \
      }                                                                        \
      if (*IsSymOrErr)                                                         \
        break;                                                                 \
    }                                                                          \
    return IterType(Out, Err);                                                 \
  };                                                                           \
  IterType IterName##_end(llvm::Error &Err) const {                            \
    return IterType(symbol_end(), Err);                                        \
  };                                                                           \
  IterRangeType IterName##s(llvm::Error &Err) const {                          \
    return llvm::make_range(IterName##_begin(Err), IterName##_end(Err));       \
  };

  CREATE_AMDGCN_OBJECT_SYMBOL_ITERATOR_FUNCTION(
      variable, amdgcn_variable_symbol_iterator, isVariable,
      amdgcn_variable_symbol_iterator_range);

  CREATE_AMDGCN_OBJECT_SYMBOL_ITERATOR_FUNCTION(
      kernel_descriptor, amdgcn_kernel_descriptor_iterator, isKernelDescriptor,
      amdgcn_kernel_descriptor_iterator_range);

  CREATE_AMDGCN_OBJECT_SYMBOL_ITERATOR_FUNCTION(
      kernel_function, amdgcn_device_function_iterator, isDeviceFunction,
      amdgcn_device_function_iterator_range);

  CREATE_AMDGCN_OBJECT_SYMBOL_ITERATOR_FUNCTION(
      device_function, amdgcn_device_function_iterator, isDeviceFunction,
      amdgcn_device_function_iterator_range);

#undef CREATE_AMDGCN_OBJECT_SYMBOL_ITERATOR_FUNCTIONS
}; // namespace luthier

//===----------------------------------------------------------------------===//
// Implementation Details
//===----------------------------------------------------------------------===//

llvm::Expected<bool> AMDGCNElfSymbolRef::isKernelDescriptor() const {
  llvm::Expected<llvm::StringRef> SymNameOrErr = getName();
  LUTHIER_RETURN_ON_ERROR(SymNameOrErr.takeError());
  uint8_t Binding = getBinding();
  uint64_t Size = getSize();
  return (Binding == llvm::ELF::STT_OBJECT && SymNameOrErr->ends_with(".kd") &&
          Size == 64) ||
         (Binding == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64);
}

llvm::Expected<bool> AMDGCNElfSymbolRef::isVariable() const {
  llvm::Expected<bool> IsKdOrErr = isKernelDescriptor();
  LUTHIER_RETURN_ON_ERROR(IsKdOrErr.takeError());
  return getBinding() == llvm::ELF::STT_OBJECT && !*IsKdOrErr;
}

llvm::Expected<bool> AMDGCNElfSymbolRef::isKernelFunction() const {
  if (getELFType() != llvm::ELF::STT_FUNC)
    return false;

  llvm::Expected<llvm::StringRef> SymbolNameOrErr = getName();
  LUTHIER_RETURN_ON_ERROR(SymbolNameOrErr.takeError());

  auto KDSymbolIfFoundOrError =
      getObject()->lookupSymbol((*getName() + ".kd").str());
  LUTHIER_RETURN_ON_ERROR(KDSymbolIfFoundOrError.takeError());

  return KDSymbolIfFoundOrError->has_value();
}

llvm::Expected<bool> AMDGCNElfSymbolRef::isDeviceFunction() const {
  if (getELFType() != llvm::ELF::STT_FUNC)
    return false;
  return !*isKernelFunction();
}

llvm::Expected<std::optional<AMDGCNVariableSymbolRef>>
AMDGCNVariableSymbolRef::getAsAMDGCNVariableSymbol(
    const llvm::object::SymbolRef &S) {
  std::optional<AMDGCNElfSymbolRef> AsGCNSymRef =
      AMDGCNElfSymbolRef::getIfAMDGCNSymbolRef(S);
  if (AsGCNSymRef.has_value()) {
    llvm::Expected<bool> IsVarOrErr = AsGCNSymRef->isVariable();
    LUTHIER_REPORT_FATAL_ON_ERROR(IsVarOrErr.takeError());
    if (*IsVarOrErr)
      return AMDGCNVariableSymbolRef{*AsGCNSymRef};
  } else
    return std::nullopt;
}

llvm::Expected<std::optional<AMDGCNKernelDescSymbolRef>>
AMDGCNKernelDescSymbolRef::getAsAMDGCNKernelDescSymbol(
    const llvm::object::SymbolRef &S) {
  std::optional<AMDGCNElfSymbolRef> AsGCNSymRef =
      AMDGCNElfSymbolRef::getIfAMDGCNSymbolRef(S);
  if (AsGCNSymRef.has_value()) {
    llvm::Expected<bool> IsKDOrErr = AsGCNSymRef->isKernelDescriptor();
    LUTHIER_REPORT_FATAL_ON_ERROR(IsKDOrErr.takeError());
    if (*IsKDOrErr)
      return AMDGCNKernelDescSymbolRef{*AsGCNSymRef};
  } else
    return std::nullopt;
}

llvm::Expected<AMDGCNKernelDescSymbolRef>
AMDGCNKernelDescSymbolRef::fromKernelFunction(
    const AMDGCNKernelFuncSymbolRef &S) {
  llvm::Expected<llvm::StringRef> NameOrErr = S.getName();
  LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
  llvm::Expected<std::optional<AMDGCNElfSymbolRef>> ExpectedKernelDescSym =
      S.getObject()->lookupSymbol((*NameOrErr + ".kd").str());
  LUTHIER_RETURN_ON_ERROR(ExpectedKernelDescSym.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      ExpectedKernelDescSym->has_value(),
      "Failed to find the kernel descriptor associated with "
      "kernel function {0}.",
      *NameOrErr));
  return AMDGCNKernelDescSymbolRef{**ExpectedKernelDescSym};
}

llvm::Expected<AMDGCNKernelFuncSymbolRef>
AMDGCNKernelDescSymbolRef::getKernelFunctionSymbol() const {
  return AMDGCNKernelFuncSymbolRef::fromKernelDescriptor(*this);
}

llvm::Expected<std::optional<AMDGCNKernelFuncSymbolRef>>
AMDGCNKernelFuncSymbolRef::getAsAMDGCNKernelFuncSymbol(
    const llvm::object::SymbolRef &S) {
  auto AsGCNSymRef = AMDGCNElfSymbolRef::getIfAMDGCNSymbolRef(S);
  if (AsGCNSymRef.has_value()) {
    llvm::Expected<bool> IsKernelFuncOrErr = AsGCNSymRef->isKernelFunction();
    LUTHIER_RETURN_ON_ERROR(IsKernelFuncOrErr.takeError());
    if (*IsKernelFuncOrErr)
      return AMDGCNKernelFuncSymbolRef{*AsGCNSymRef};
  }
  return std::nullopt;
}

llvm::Expected<AMDGCNKernelFuncSymbolRef>
AMDGCNKernelFuncSymbolRef::fromKernelDescriptor(
    const AMDGCNKernelDescSymbolRef &S) {
  llvm::Expected<llvm::StringRef> NameOrErr = S.getName();
  LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
  llvm::Expected<std::optional<AMDGCNElfSymbolRef>> ExpectedKernelFuncSym =
      S.getObject()->lookupSymbol(
          NameOrErr->substr(0, NameOrErr->rfind(".kd")));
  LUTHIER_RETURN_ON_ERROR(ExpectedKernelFuncSym.takeError());

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(ExpectedKernelFuncSym->has_value(),
                          "Failed to find the kernel function associated with "
                          "kernel descriptor {0}.",
                          *NameOrErr));

  return AMDGCNKernelFuncSymbolRef{**ExpectedKernelFuncSym};
}

llvm::Expected<std::optional<AMDGCNDeviceFuncSymbolRef>>
AMDGCNDeviceFuncSymbolRef::getAsAMDGCNDeviceFuncSymbol(
    const llvm::object::SymbolRef &S) {
  auto AsGCNSymRef = AMDGCNElfSymbolRef::getIfAMDGCNSymbolRef(S);
  if (AsGCNSymRef.has_value()) {
    llvm::Expected<bool> IsDeviceFuncOrErr = AsGCNSymRef->isDeviceFunction();
    LUTHIER_RETURN_ON_ERROR(IsDeviceFuncOrErr.takeError());
    if (*IsDeviceFuncOrErr)
      return AMDGCNDeviceFuncSymbolRef{*AsGCNSymRef};
  }
  return std::nullopt;
}

IMPLEMENT_AMDGCN_ITERATOR_INCREMENT_OPERATOR(amdgcn_variable_symbol_iterator,
                                             AMDGCNVariableSymbolRef,
                                             isVariable);

IMPLEMENT_AMDGCN_ITERATOR_INCREMENT_OPERATOR(amdgcn_kernel_descriptor_iterator,
                                             AMDGCNKernelDescSymbolRef,
                                             isKernelDescriptor);

IMPLEMENT_AMDGCN_ITERATOR_INCREMENT_OPERATOR(amdgcn_kernel_function_iterator,
                                             AMDGCNKernelFuncSymbolRef,
                                             isKernelFunction);

IMPLEMENT_AMDGCN_ITERATOR_INCREMENT_OPERATOR(amdgcn_device_function_iterator,
                                             AMDGCNDeviceFuncSymbolRef,
                                             isDeviceFunction);

#undef IMPLEMENT_AMDGCN_ITERATOR_INCREMENT_OPERATOR

} // namespace luthier

#endif