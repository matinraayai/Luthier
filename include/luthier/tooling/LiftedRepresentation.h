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
/// This file describes Lifted Representation, which provides
/// over that can be inspected the
/// \c llvm::Module and \c llvm::MachineModuleInfo of a HSA primitive (a
/// kernel or an executable) disassembled and lifted to LLVM Machine IR,
/// as well as a mapping between the HSA primitives and LLVM IR objects
/// involved.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_LIFTED_REPRESENTATION_H
#define LUTHIER_LIFTED_REPRESENTATION_H
#include "AMDGPUTargetMachine.h"
#include "luthier/hsa/DenseMapInfo.h"
#include "luthier/hsa/Instr.h"
#include <hsa/hsa.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LegacyPassManager.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>

namespace luthier {

class CodeLifter;

class CodeGenerator;

/// \brief contains information regarding a lifted HSA primitive
/// \details "Lifting" in Luthier is the process of inspecting the contents
/// of AMDGPU binaries loaded onto a device to recover a valid LLVM Machine IR
/// representation equivalent or very close to what the clang compiler used (or
/// would have used) to create the inspected binaries.
/// The scope of the lift can be either an \c hsa_executable_t or a
/// <tt>hsa::LoadedCodeObjectKernel</tt>. Luthier's \c CodeLifter is
/// the only entity allowed to construct or clone a
/// <tt>LiftedRepresentation</tt>. This allows internal caching and thread-safe
/// access to its instances by other components. It also allows invalidation of
/// the representations when the executable backing the lifted primitive gets
/// destroyed. \n Each lifted primitive has an independent \c
/// llvm::orc::ThreadSafeContext created for it internally by Luthier's
/// <tt>CodeLifter</tt> to let independent processing of different primitives by
/// multiple threads, and proper synchronization when multiple threads need to
/// instrument the same primitive. Subsequent clones of the lifted
/// representation use the same thread-safe context. The following mappings are
/// also retained internally in the representation:
/// - For each \c hsa_loaded_code_object_t involved,
/// an \c llvm::orc::ThreadSafeModule and a \c llvm::MachineModuleInfo is
/// created and stored.
/// - For each \c hsa_executable_symbol_t of type \c KERNEL or \c
/// DEVICE_FUNCTION involved, a mapping to the \c llvm::MachineFunction it was
/// lifted to is retained. The machine function is owned by one of the machine
/// module info created for its defining <tt>hsa_loaded_code_object_t</tt>.
/// - For each \c hsa_executable_symbol_t of type <tt>VARIABLE</tt>, a mapping
/// to the \c llvm::GlobalVariable it was lifted to is retained.
/// - For each \c llvm::MachineInstr in the lifted functions, a mapping to
/// the disassembled \c hsa::Instr is retained. This is so that the tool writer
/// can track the original MC representation of the instruction as well as
/// its runtime load attributes (i.e. the address it was loaded, its size, etc).
class LiftedRepresentation {
  /// Only Luthier's CodeLifter is able to create <tt>LiftedRepresentation</tt>s
  friend luthier::CodeLifter;

private:
  std::unique_ptr<llvm::GCNTargetMachine> TM{};

  /// A thread-safe context that owns all the thread-safe modules;
  /// Each LiftedRepresentation is given its own context to allow for
  /// independent processing from others\n
  /// During instrumentation all bitcode is loaded into this Context\n
  /// This ThreadSafeContext retains the underlying \c llvm::LLVMContext
  /// unique pointer's ownership
  llvm::orc::ThreadSafeContext Context;

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

  /// Mapping between potentially called device function symbols and their \c
  /// llvm::MachineFunction
  std::unordered_map<
      std::unique_ptr<hsa::LoadedCodeObjectDeviceFunction>,
      llvm::MachineFunction *,
      hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectDeviceFunction>,
      hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectDeviceFunction>>
      RelatedFunctions{};

  /// Mapping between an \c hsa_executable_symbol_t of type variable
  /// and its \c llvm::GlobalVariable
  std::unordered_map<
      std::unique_ptr<hsa::LoadedCodeObjectSymbol>, llvm::GlobalVariable *,
      hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectSymbol>,
      hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectSymbol>>
      RelatedGlobalVariables{};

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

  /// \return a reference to the \c LLVMContext of this Lifted Representation
  llvm::LLVMContext &getContext() { return *Context.getContext(); }

  /// \return a const reference to the \c LLVMContext of this
  /// Lifted Representation
  [[nodiscard]] const llvm::LLVMContext &getContext() const {
    return *Context.getContext();
  }

  /// \return a scoped lock protecting the Context and the TargetMachine of this
  /// \c LiftedRepresentation
  llvm::orc::ThreadSafeContext::Lock getLock() const {
    return Context.getLock();
  }

  /// Related function iterator
  using function_iterator = decltype(RelatedFunctions)::iterator;
  /// Related function constant iterator
  using const_function_iterator = decltype(RelatedFunctions)::const_iterator;
  /// Related Global Variable iterator.
  using global_iterator = decltype(RelatedGlobalVariables)::iterator;
  /// The Global Variable constant iterator.
  using const_global_iterator =
      decltype(RelatedGlobalVariables)::const_iterator;

  /// Function iteration
  function_iterator function_begin() { return RelatedFunctions.begin(); }
  [[nodiscard]] const_function_iterator function_begin() const {
    return RelatedFunctions.begin();
  }

  function_iterator function_end() { return RelatedFunctions.end(); }
  [[nodiscard]] const_function_iterator function_end() const {
    return RelatedFunctions.end();
  }

  [[nodiscard]] size_t function_size() const {
    return RelatedFunctions.size();
  };

  [[nodiscard]] bool function_empty() const {
    return RelatedFunctions.empty();
  };

  llvm::iterator_range<function_iterator> functions() {
    return llvm::make_range(function_begin(), function_end());
  }
  [[nodiscard]] llvm::iterator_range<const_function_iterator>
  functions() const {
    return llvm::make_range(function_begin(), function_end());
  }

  /// Global Variable iteration
  global_iterator global_begin() { return RelatedGlobalVariables.begin(); }
  [[nodiscard]] const_global_iterator global_begin() const {
    return RelatedGlobalVariables.begin();
  }

  global_iterator global_end() { return RelatedGlobalVariables.end(); }
  [[nodiscard]] const_global_iterator global_end() const {
    return RelatedGlobalVariables.end();
  }

  [[nodiscard]] size_t global_size() const {
    return RelatedGlobalVariables.size();
  };

  [[nodiscard]] bool global_empty() const {
    return RelatedGlobalVariables.empty();
  };

  llvm::iterator_range<global_iterator> globals() {
    return llvm::make_range(global_begin(), global_end());
  }
  [[nodiscard]] llvm::iterator_range<const_global_iterator> globals() const {
    return llvm::make_range(global_begin(), global_end());
  }

  [[nodiscard]] const llvm::Module &getModule() const { return *Module; }

  [[nodiscard]] llvm::Module &getModule() { return *Module; }

  [[nodiscard]] const llvm::MachineModuleInfo &getMMI() const {
    return MMIWP->getMMI();
  }

  [[nodiscard]] llvm::MachineModuleInfo &getMMI() { return MMIWP->getMMI(); }

  [[nodiscard]] const llvm::MachineModuleInfoWrapperPass &getMMIWP() const {
    return *MMIWP;
  }

  [[nodiscard]] std::unique_ptr<llvm::MachineModuleInfoWrapperPass> &
  getMMIWP() {
    return MMIWP;
  }

  [[nodiscard]] const llvm::GCNTargetMachine &getTM() const { return *TM; }

  [[nodiscard]] llvm::GCNTargetMachine &getTM() { return *TM; }

  [[nodiscard]] const llvm::MachineFunction &getKernelMF() const {
    return *KernelMF;
  }

  [[nodiscard]] llvm::MachineFunction &getKernelMF() { return *KernelMF; }

  hsa_loaded_code_object_t getLoadedCodeObject() const { return LCO; }

  const hsa::LoadedCodeObjectKernel &getKernel() const { return *Kernel; }

  /// \return the \c llvm::GlobalVariable associated with \p GV if exists;
  /// \c nullptr otherwise
  [[nodiscard]] const llvm::GlobalVariable *
  getGV(const hsa::LoadedCodeObjectSymbol &GV) const {
    auto It = RelatedGlobalVariables.find(&GV);
    if (It == RelatedGlobalVariables.end())
      return nullptr;
    else
      return It->second;
  }

  [[nodiscard]] const llvm::MachineFunction *
  getGV(const hsa::LoadedCodeObjectDeviceFunction &Func) const {
    auto It = RelatedFunctions.find(&Func);
    if (It == RelatedFunctions.end())
      return nullptr;
    else
      return It->second;
  }

  /// \returns the \c hsa::Instr that the \p MI was lifted from; If
  /// the \p MI was not part of the lifted code, returns <tt>nullptr</tt>
  [[nodiscard]] const hsa::Instr *
  getHSAInstr(const llvm::MachineInstr &MI) const {
    auto It = MachineInstrToMCMap.find(&MI);
    if (It == MachineInstrToMCMap.end())
      return nullptr;
    else
      return It->second;
  }
};

} // namespace luthier

#endif