#ifndef LUTHIER_TEST_KERNEL_COMPILER_H
#define LUTHIER_TEST_KERNEL_COMPILER_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

namespace luthier::test {

/// Compiles LLVM IR Modules to AMDGPU ELF code objects in memory.
class KernelCompiler {
public:
  /// Compile an LLVM IR Module to an AMDGPU ELF code object.
  ///
  /// \param M         The Module to compile.  Its triple and data layout must
  ///                  already be set for the AMDGPU target.
  /// \param GpuTarget The GPU target CPU string (e.g. "gfx908").
  /// \returns The raw ELF bytes, or an error.
  static llvm::Expected<llvm::SmallVector<char, 0>>
  compileIR(llvm::Module &M, llvm::StringRef GpuTarget);
};

} // namespace luthier::test

#endif // LUTHIER_TEST_KERNEL_COMPILER_H
