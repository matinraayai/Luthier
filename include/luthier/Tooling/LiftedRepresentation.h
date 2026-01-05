//===-- LiftedRepresentation.h ----------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This file describes the <tt>LiftedRepresentation</tt>, which encapsulates
/// the representation of a kernel symbol in LLVM MIR, as well as mappings
/// between <tt>hsa::LoadedCodeObjectSymbol</tt>s and their lifted LLVM
/// equivalent.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_LIFTED_REPRESENTATION_H
#define LUTHIER_TOOLING_LIFTED_REPRESENTATION_H
#include "AMDGPUTargetMachine.h"
#include "luthier/HSA/Instr.h"
#include "luthier/HSA/LoadedCodeObjectDeviceFunction.h"
#include "luthier/HSA/LoadedCodeObjectKernel.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

namespace luthier {

class CodeLifter;

namespace hsa {

class LoadedCodeObjectVariable;

class LoadedCodeObjectExternSymbol;

} // namespace hsa

/// \brief Holds information regarding a lifted AMD GPU kernel and the mapping
/// between the HSA and LLVM objects involved in the representation
/// \details "Lifting" in Luthier is the process of inspecting the contents
/// of AMDGPU binaries loaded on a device to recover a valid LLVM Machine IR
/// representation equivalent or very close to what the application's compiler
/// used originally for creating the inspected binaries. The Machine IR allows
/// for flexible modification of the binary's instruction.\n
/// Luthier's \c CodeLifter is the only entity allowed to construct or clone a
/// <tt>LiftedRepresentation</tt>. This allows internal caching and thread-safe
/// access to its instances by other components. The cached copy of the
/// representation gets invalidated when the executable of the kernel gets
/// destroyed. \n Each lifted kernel has an independent \c
/// llvm::orc::ThreadSafeContext for independent processing and synchronization
/// by multiple threads. Subsequent clones of the lifted
/// representation use the same thread-safe context.
class LiftedRepresentation {
  /// Only Luthier's CodeLifter is able to create <tt>LiftedRepresentation</tt>s
  friend luthier::CodeLifter;

private:
  /// Target machine of the \c MMIWP
  std::unique_ptr<llvm::GCNTargetMachine> TM{};

  /// A thread-safe context that owns all the thread-safe modules;
  /// Each LiftedRepresentation is given its own context to allow for
  /// independent processing from others\n
  llvm::orc::ThreadSafeContext Context{};

  /// Loaded code object of the lifted kernel
  hsa_loaded_code_object_t LCO{};

  /// Module of the lifted kernel
  std::unique_ptr<llvm::Module> Module{};

  /// MMIWP of the lifted kernel
  std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP{};

  /// The symbol of the lifted kernel
  std::unique_ptr<hsa::LoadedCodeObjectKernel> Kernel{};

  /// MF of the lifted kernel
  llvm::MachineFunction *KernelMF{};

  /// Mapping between the potentially called device function
  /// symbols and their \c llvm::MachineFunction
  std::unordered_map<
      std::unique_ptr<hsa::LoadedCodeObjectDeviceFunction>,
      llvm::MachineFunction *,
      hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectDeviceFunction>,
      hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectDeviceFunction>>
      Functions{};

  /// Mapping between static variables potentially used by the kernel and
  /// their \c llvm::GlobalVariable \n
  /// This map also includes other kernels inside the \c LCO of the
  /// lifted kernel as well
  std::unordered_map<
      std::unique_ptr<hsa::LoadedCodeObjectSymbol>, llvm::GlobalVariable *,
      hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectSymbol>,
      hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectSymbol>>
      Variables{};

  /// A mapping between an \c llvm::MachineInstr in one of the MMIs and
  /// its HSA representation, \c hsa::Instr. This is useful to have in case
  /// the user wants to peak at the original \c llvm::MCInst of the machine
  /// instruction or any other information about where the instruction is loaded
  /// during runtime. \n
  /// This mapping is only valid before any LLVM pass is run over the MMIs;
  /// After that pointers of each machine instruction gets changed by the
  /// underlying allocator, and this map becomes invalid
  llvm::DenseMap<llvm::MachineInstr *, hsa::Instr *> MachineInstrToMCMap{};

  LiftedRepresentation();

public:
  /// Destructor
  ~LiftedRepresentation();

  /// Disallowed copy construction
  LiftedRepresentation(const LiftedRepresentation &) = delete;

  /// Disallowed assignment operation
  LiftedRepresentation &operator=(const LiftedRepresentation &) = delete;

  /// \return the Target Machine of the lifted representation's machine module
  /// info
  [[nodiscard]] const llvm::GCNTargetMachine &getTM() const { return *TM; }

  /// \return the Target Machine of the lifted representation's machine module
  /// info
  [[nodiscard]] llvm::GCNTargetMachine &getTM() { return *TM; }

  /// \return a reference to the \c LLVMContext of this Lifted Representation
  llvm::LLVMContext &getContext() {
    return Context.withContextDo(
        [](llvm::LLVMContext *Ctx) -> llvm::LLVMContext & { return *Ctx; });
  }

  /// \return a const reference to the \c LLVMContext of this
  /// Lifted Representation
  [[nodiscard]] const llvm::LLVMContext &getContext() const {
    return Context.withContextDo(
        [](const llvm::LLVMContext *Ctx) -> const llvm::LLVMContext & {
          return *Ctx;
        });
  }

  /// \return the loaded code object of the lifted kernel
  hsa_loaded_code_object_t getLoadedCodeObject() const { return LCO; }

  /// \return the \c llvm::Module of the lifted representation
  [[nodiscard]] const llvm::Module &getModule() const { return *Module; }

  /// \return the \c llvm::Module of the lifted representation
  [[nodiscard]] llvm::Module &getModule() { return *Module; }

  /// \return the \c llvm::MachineModuleInfo of the lifted representation
  [[nodiscard]] const llvm::MachineModuleInfo &getMMI() const {
    return MMIWP->getMMI();
  }

  /// \return the \c llvm::MachineModuleInfo of the lifted representation
  [[nodiscard]] llvm::MachineModuleInfo &getMMI() { return MMIWP->getMMI(); }

  /// \return the \c llvm::MachineModuleInfoWrapperPass containing the
  /// MIR of the lifted representation
  [[nodiscard]] const llvm::MachineModuleInfoWrapperPass &getMMIWP() const {
    return *MMIWP;
  }

  /// \return the \c llvm::MachineModuleInfoWrapperPass containing the
  /// MIR of the lifted representation
  /// \note the MMIWP will be deleted after running legacy codegen passes on
  /// it, effectively invalidating the entire lifted representation
  [[nodiscard]] std::unique_ptr<llvm::MachineModuleInfoWrapperPass> &
  getMMIWP() {
    return MMIWP;
  }

  /// \return the symbol of the lifted kernel
  const hsa::LoadedCodeObjectKernel &getKernel() const { return *Kernel; }

  /// \return the \c llvm::MachineFunction containing the
  /// <tt>llvm::MachineInstr</tt>s of the lifted kernel
  [[nodiscard]] const llvm::MachineFunction &getKernelMF() const {
    return *KernelMF;
  }

  /// \return the \c llvm::MachineFunction containing the
  /// <tt>llvm::MachineInstr</tt>s of the lifted kernel
  [[nodiscard]] llvm::MachineFunction &getKernelMF() { return *KernelMF; }

  /// Related function iterator
  using function_iterator = decltype(Functions)::iterator;
  /// Related function constant iterator
  using const_function_iterator = decltype(Functions)::const_iterator;
  /// Related Global Variable iterator.
  using global_iterator = decltype(Variables)::iterator;
  /// The Global Variable constant iterator.
  using const_global_iterator = decltype(Variables)::const_iterator;

  /// Function iteration
  function_iterator function_begin() { return Functions.begin(); }
  [[nodiscard]] const_function_iterator function_begin() const {
    return Functions.begin();
  }

  function_iterator function_end() { return Functions.end(); }
  [[nodiscard]] const_function_iterator function_end() const {
    return Functions.end();
  }

  [[nodiscard]] size_t function_size() const { return Functions.size(); };

  [[nodiscard]] bool function_empty() const { return Functions.empty(); };

  llvm::iterator_range<function_iterator> functions() {
    return llvm::make_range(function_begin(), function_end());
  }

  [[nodiscard]] llvm::iterator_range<const_function_iterator>
  functions() const {
    return llvm::make_range(function_begin(), function_end());
  }

  /// Global Variable iteration
  global_iterator global_begin() { return Variables.begin(); }
  [[nodiscard]] const_global_iterator global_begin() const {
    return Variables.begin();
  }

  global_iterator global_end() { return Variables.end(); }
  [[nodiscard]] const_global_iterator global_end() const {
    return Variables.end();
  }

  [[nodiscard]] size_t global_size() const { return Variables.size(); };

  [[nodiscard]] bool global_empty() const { return Variables.empty(); };

  llvm::iterator_range<global_iterator> globals() {
    return llvm::make_range(global_begin(), global_end());
  }

  [[nodiscard]] llvm::iterator_range<const_global_iterator> globals() const {
    return llvm::make_range(global_begin(), global_end());
  }

  /// Iterates over all defined functions in the lifted representation
  /// and applies the \p Lambda function on all of them
  /// Defined functions include the lifted kernel,
  /// as well as all device functions included in the kernel's loaded code
  /// object
  llvm::Error iterateAllDefinedFunctionTypes(
      const std::function<llvm::Error(const hsa::LoadedCodeObjectSymbol &,
                                      llvm::MachineFunction &)> &Lambda);

  /// \return the \c llvm::GlobalVariable associated with
  /// \p VariableSymbol if exists \c nullptr otherwise
  [[nodiscard]] llvm::GlobalVariable *
  getLiftedEquivalent(const hsa::LoadedCodeObjectVariable &VariableSymbol);

  /// \return the \c llvm::GlobalVariable associated with
  /// \p VariableSymbol if exists \c nullptr otherwise
  [[nodiscard]] const llvm::GlobalVariable *getLiftedEquivalent(
      const hsa::LoadedCodeObjectVariable &VariableSymbol) const;

  /// \return the \c llvm::GlobalVariable associated with
  /// \p VariableSymbol if exists \c nullptr otherwise
  [[nodiscard]] const llvm::GlobalVariable *getLiftedEquivalent(
      const hsa::LoadedCodeObjectExternSymbol &ExternSymbol) const;

  /// \return the \c llvm::GlobalVariable associated with
  /// \p VariableSymbol if exists \c nullptr otherwise
  [[nodiscard]] llvm::GlobalVariable *
  getLiftedEquivalent(const hsa::LoadedCodeObjectExternSymbol &ExternSymbol);

  /// \return the \c llvm::GlobalVariable associated with
  /// \p VariableSymbol if exists \c nullptr otherwise
  [[nodiscard]] const llvm::GlobalValue *
  getLiftedEquivalent(const hsa::LoadedCodeObjectKernel &KernelSymbol) const;

  /// \return the \c llvm::GlobalVariable associated with
  /// \p VariableSymbol if exists \c nullptr otherwise
  [[nodiscard]] llvm::GlobalValue *
  getLiftedEquivalent(const hsa::LoadedCodeObjectKernel &KernelSymbol);

  /// \return the \c llvm::Function associated with
  /// \p DevFunc if exists \c nullptr otherwise
  [[nodiscard]] const llvm::Function *
  getLiftedEquivalent(const hsa::LoadedCodeObjectDeviceFunction &DevFunc) const;

  /// \return the \c llvm::Function associated with
  /// \p DevFunc if exists \c nullptr otherwise
  [[nodiscard]] llvm::Function *
  getLiftedEquivalent(const hsa::LoadedCodeObjectDeviceFunction &DevFunc);

  /// \return the \c llvm::GlobalValue associated with \p Symbol if exists;
  /// \c nullptr otherwise
  [[nodiscard]] const llvm::GlobalValue *
  getLiftedEquivalent(const hsa::LoadedCodeObjectSymbol &Symbol) const;

  /// \return the \c llvm::GlobalValue associated with \p Symbol if exists;
  /// \c nullptr otherwise
  [[nodiscard]] llvm::GlobalValue *
  getLiftedEquivalent(const hsa::LoadedCodeObjectSymbol &Symbol);

  /// \returns the \c hsa::Instr that the \p MI was lifted from; If
  /// the \p MI was not part of the lifted code, returns <tt>nullptr</tt>
  [[nodiscard]] const hsa::Instr *
  getLiftedEquivalent(const llvm::MachineInstr &MI) const;
};

} // namespace luthier

#endif