//===-- instrumentation_module.hpp - Instrumentation Module declaration ---===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's Instrumentation Module, which contains a
/// single LLVM bitcode containing hooks and static variables it uses.
//===----------------------------------------------------------------------===//

#ifndef INSTRUMENTATION_MODULE_HPP
#define INSTRUMENTATION_MODULE_HPP
#include "hsa_executable_symbol.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <unordered_set>

namespace luthier {

class CodeObjectManager;

class InstrumentationModule {
private:
  /// For now only CodeObjectManager is allowed to create Instrumentation Modules
  friend CodeObjectManager;

  /// A buffer owned by the InstrumentationModule object to save the
  /// processed bitcode
  /// The bitcode is first processed to make all its Global Variables external
  /// Then it is stored again in the bitcode format here. This is so that the
  /// bitcode can be copied over to different LLVM Contexts
  llvm::SmallVector<char> BitcodeBuffer{};
  /// Name of the Module
  std::string ModuleName{};
  /// List of "static" symbols; Static symbols come with Instrumentation Modules
  /// loaded with Loaded Code Objects, which means they are already loaded by the
  /// HSA runtime. This means Instrumentation Module doesn't need to worry about
  /// managing them
  llvm::StringMap<hsa::ExecutableSymbol> StaticSymbols{};

  /// List of "Dynamic" symbols in the Module; Dynamic symbols are not loaded
  /// by HSA, meaning the Instrumentation Module must create and managed them
  /// dynamically during its creation.
  llvm::StringMap<void *> DynamicSymbols{};

  /// Extern symbols are declared extern (i.e. they don't have a definition)
  /// in the bitcode itself.
  /// They are useful to communicate between different instrumentation tasks
  /// InstrumentationModule doesn't manage the lifetime of these symbols
  std::unordered_set<std::string> ExternSymbols{};

  InstrumentationModule() = default;

  /// Creates an instrumentation module given a \c hsa::LoadedCodeObject.
  /// The \p LCO must belong to a Luthier tool and been statically loaded
  /// by the HIP Compiler extension.
  /// \param LCO the Luthier tool Loaded Code Object
  static llvm::Expected<InstrumentationModule>
  create(const hsa::LoadedCodeObject &LCO);

  /// Copies the Module's bitcode into the passed \p Ctx
  /// \param Ctx a thread-safe context to read the bitcode into
  /// \return a thread-safe Module
  llvm::Expected<llvm::orc::ThreadSafeModule>
  readBitcodeIntoModule(llvm::orc::ThreadSafeContext &Ctx);
};

} // namespace luthier

#endif