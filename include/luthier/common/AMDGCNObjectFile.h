
#ifndef LUTHIER_COMMON_AMDGCN_OBJECT_FILE_H
#define LUTHIER_COMMON_AMDGCN_OBJECT_FILE_H
#include <luthier/common/ELFCodeObject.h>

namespace luthier {

class AMDGCNObjectFile;

/// \brief Wrapper around a \c ELFSymbolRefWrapper which represents a
/// symbol inside a \c AMDGCNObjectFile
class AMDGCNElfSymbolRef : public ELFSymbolRefWrapper {
private:
  // NOLINTBEGIN(google-explicit-constructor)
  /*implicit*/ AMDGCNElfSymbolRef(llvm::object::SymbolRef &S)
      : ELFSymbolRefWrapper(S) {};
  // NOLINTEND(google-explicit-constructor)

public:
  static bool classof(const ELFSymbolRefWrapper *S);

  const luthier::AMDGCNObjectFile *getObject() const;

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

class amdgpu_elf_symbol_iterator : public llvm::object::elf_symbol_iterator {
private:
  friend AMDGCNObjectFile;

  // NOLINTBEGIN(google-explicit-constructor)
  /* implicit */ amdgpu_elf_symbol_iterator(
      const llvm::object::basic_symbol_iterator &B)
      : llvm::object::elf_symbol_iterator(B) {}
  // NOLINTEND(google-explicit-constructor)

public:
  // NOLINTBEGIN(google-explicit-constructor)
  /* implicit */ amdgpu_elf_symbol_iterator(
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

/// \brief Adds additional methods to the \c llvm::object::ELF64LEObjectFile
/// class specific to amdgcn code object files
/// As per <a href="https://llvm.org/docs/AMDGPUUsage.html#elf-code-object">
/// AMDGPU backend documentation</a>, AMDGCN object files are 64-bit LE.

using amdgcn_elf_symbol_iterator_range =
    llvm::iterator_range<amdgpu_elf_symbol_iterator>;

class AMDGCNObjectFile : public luthier::ELF64LEObjectFileWrapper {
public:
  static bool classof(const llvm::object::Binary *v);

  /// Parses the ELF file pointed to by \p Elf into a \c AMDGCNObjectFile.
  /// \param ELF \c llvm::StringRef encompassing the ELF file in memory
  /// \return a \c std::unique_ptr<AMDGCNObjectFile> on successful
  /// parsing, an \c llvm::Error on failure
  static llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
  createAMDGCNObjectFile(llvm::StringRef Elf);

  /// Parses the ELF file pointed to by \b Elf into a \b AMDGCNObjectFile.
  /// \param ELF \p llvm::ArrayRef<uint8_t> encompassing the ELF file in memory
  /// \return a \c std::unique_ptr<AMDGCNObjectFile> on successful
  /// parsing, an \c llvm::Error on failure
  static llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
  createAMDGCNObjectFile(llvm::ArrayRef<uint8_t> Elf);

  amdgcn_elf_symbol_iterator_range symbols() const {
    return amdgcn_elf_symbol_iterator_range(symbol_begin(), symbol_end());
  }

  amdgpu_elf_symbol_iterator dynamic_symbol_begin() const {
    return luthier::ELF64LEObjectFileWrapper::dynamic_symbol_begin();
  }

  amdgpu_elf_symbol_iterator dynamic_symbol_end() const {
    return luthier::ELF64LEObjectFileWrapper::dynamic_symbol_end();
  }
};

} // namespace luthier

#endif