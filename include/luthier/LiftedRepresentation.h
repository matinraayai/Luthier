//===-- LiftedRepresentation.h ----------------------------------*- C++ -*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file describes Luthier's Lifted Representation, which contains the
/// \c llvm::Module and \c llvm::MachineModuleInfo of a HSA primitive (a
/// kernel or an executable) disassembled and lifted to LLVM Machine IR,
/// as well as a mapping between the HSA primitives and LLVM IR objects
/// involved.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_LIFTED_REPRESENTATION_H
#define LUTHIER_LIFTED_REPRESENTATION_H
#include <hsa/hsa.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LegacyPassManager.h>
#include <luthier/hsa/DenseMapInfo.h>
#include <luthier/hsa/Instr.h>

namespace llvm {
class GCNTargetMachine;
} // namespace llvm

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
  /// This is a mapping between the LCO and the \c llvm::GCNTargetMachine used
  /// for its \c llvm::MachineModuleInfo
  /// It also acts as all the TM's storage
  llvm::SmallDenseMap<hsa_loaded_code_object_t,
                      std::unique_ptr<llvm::GCNTargetMachine>, 1>
      TMs{};

  /// A thread-safe context that owns all the thread-safe modules;
  /// Each LiftedRepresentation is given its own context to allow for
  /// independent processing from others\n
  /// During instrumentation all bitcode is loaded into this Context\n
  /// This ThreadSafeContext retains the underlying \c llvm::LLVMContext
  /// unique pointer's ownership
  llvm::orc::ThreadSafeContext Context;

  /// \brief primary storage of the Module and the Machine Module Info of the
  /// lifted loaded code objects
  llvm::SmallDenseMap<
      hsa_loaded_code_object_t,
      std::pair<std::unique_ptr<llvm::Module>,
                std::unique_ptr<llvm::MachineModuleInfoWrapperPass>>,
      1>
      Modules{};

  /// \brief Mapping between an \c hsa_loaded_code_object_t and the
  /// thread-safe Module and MMI representing it in LLVM
  /// \details Modules only hold global object definition/declarations; MMI
  /// and its \c llvm::MachineFunction list contains the \c llvm::MachineInstr
  /// of each function and therefore, do the heavy lifting for Luthier.\n
  /// The modules/MMIs have a one-to-one correspondence with a single
  /// \c hsa_loaded_code_object. This is because an \c hsa_executable_t can
  /// contain multiple loaded code objects. Each module/MMI will be compiled
  /// separately from the other and then passed to Luthier to be loaded
  /// into a single <tt>hsa_executable_t</tt>. In practice, an \c
  /// hsa_executable_t almost always contains a single
  /// <tt>hsa_loaded_code_object_t</tt>
  llvm::SmallDenseMap<hsa_loaded_code_object_t,
                      std::pair<llvm::Module &, llvm::MachineModuleInfo &>, 1>
      RelatedLCOs{};

  /// Mapping between an \c hsa_executable_symbol_t (of type kernel and
  /// device function) and its \c llvm::MachineFunction
  llvm::DenseMap<const hsa::LoadedCodeObjectSymbol *, llvm::MachineFunction *>
      RelatedFunctions{};

  /// Mapping between an \c hsa_executable_symbol_t of type variable
  /// and its \c llvm::GlobalVariable
  llvm::DenseMap<const hsa::LoadedCodeObjectSymbol *, llvm::GlobalVariable *>
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

  /// A mapping between a \c llvm::GlobalValue
  /// (either \c llvm::GlobalVariable or a \c llvm::Function)
  /// and its machine instruction users
  llvm::DenseMap<llvm::GlobalValue *, llvm::SmallVector<llvm::MachineInstr *>>
      GlobalValueMIUses{};

  LiftedRepresentation();

public:
  /// Destructor
  ~LiftedRepresentation() = default;

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

  /// Module iterator
  using module_iterator = decltype(Modules)::iterator;
  /// Module constant iterator
  using const_module_iterator = decltype(Modules)::const_iterator;
  /// Related Loaded Code Object iterator
  using iterator = decltype(RelatedLCOs)::iterator;
  /// Related Loaded Code Object constant iterator
  using const_iterator = decltype(RelatedLCOs)::const_iterator;
  /// Related function iterator
  using function_iterator = decltype(RelatedFunctions)::iterator;
  /// Related function constant iterator
  using const_function_iterator = decltype(RelatedFunctions)::const_iterator;
  /// Related Global Variable iterator.
  using global_iterator = decltype(RelatedGlobalVariables)::iterator;
  /// The Global Variable constant iterator.
  using const_global_iterator =
      decltype(RelatedGlobalVariables)::const_iterator;

  /// Module iteration
  module_iterator module_begin() { return Modules.begin(); }
  [[nodiscard]] const_module_iterator module_begin() const {
    return Modules.begin();
  }

  module_iterator module_end() { return Modules.end(); }
  [[nodiscard]] const_module_iterator module_end() const {
    return Modules.end();
  }

  [[nodiscard]] size_t module_size() const { return Modules.size(); };

  [[nodiscard]] bool module_empty() const { return Modules.empty(); };

  llvm::iterator_range<module_iterator> modules() {
    return make_range(module_begin(), module_end());
  }
  [[nodiscard]] llvm::iterator_range<const_module_iterator> modules() const {
    return make_range(module_begin(), module_end());
  }

  /// LCO iteration
  iterator begin() { return RelatedLCOs.begin(); }
  [[nodiscard]] const_iterator begin() const { return RelatedLCOs.begin(); }

  iterator end() { return RelatedLCOs.end(); }
  [[nodiscard]] const_iterator end() const { return RelatedLCOs.end(); }

  [[nodiscard]] size_t size() const { return RelatedLCOs.size(); }

  [[nodiscard]] bool empty() const { return RelatedLCOs.empty(); }

  llvm::iterator_range<iterator> loaded_code_objects() {
    return make_range(begin(), end());
  }
  [[nodiscard]] llvm::iterator_range<const_iterator>
  loaded_code_objects() const {
    return make_range(begin(), end());
  }

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
    return make_range(function_begin(), function_end());
  }
  [[nodiscard]] llvm::iterator_range<const_function_iterator>
  functions() const {
    return make_range(function_begin(), function_end());
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
    return make_range(global_begin(), global_end());
  }
  [[nodiscard]] llvm::iterator_range<const_global_iterator> globals() const {
    return make_range(global_begin(), global_end());
  }

  /// \return the \c llvm::Module of the lifted \p LCO if \p LCO is
  /// included in the Lifted Representation;
  /// Otherwise, returns <tt>nullptr</tt>
  [[nodiscard]] llvm::Module *getModule(hsa_loaded_code_object_t LCO) const {
    auto It = Modules.find(LCO);
    if (It == Modules.end())
      return nullptr;
    else
      return It->second.first.get();
  }

  /// \return the \c llvm::MachineModuleInfo of the
  /// lifted \p LCO if \p LCO is included in the Lifted Representation;
  /// Otherwise, returns <tt>nullptr</tt>
  [[nodiscard]] llvm::MachineModuleInfo *
  getMMI(hsa_loaded_code_object_t LCO) const {
    auto It = Modules.find(LCO);
    if (It == Modules.end())
      return nullptr;
    else
      return &It->second.second->getMMI();
  }

  /// \return the \c llvm::GCNTargetMachine used to construct the
  /// \c llvm::MachineModuleInfo of the \p LCO if \p LCO is included
  /// in the Lifted Representation; Otherwise, returns <tt>nullptr</tt>
  [[nodiscard]] llvm::GCNTargetMachine *
  getTM(hsa_loaded_code_object_t LCO) const {
    auto It = TMs.find(LCO);
    if (It == TMs.end()) {
      return nullptr;
    } else
      return It->second.get();
  }

  /// \return the \c llvm::MachineFunction associated with \p Func if exists;
  /// \c nullptr otherwise
  [[nodiscard]] const llvm::MachineFunction *
  getMF(const hsa::LoadedCodeObjectSymbol &Func) const {
    auto It = RelatedFunctions.find(&Func);
    if (It == RelatedFunctions.end())
      return nullptr;
    else
      return It->second;
  }

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

  llvm::ArrayRef<llvm::MachineInstr *>
  getUsesOfGlobalValue(const hsa::LoadedCodeObjectSymbol &GV) const;

  llvm::ArrayRef<llvm::MachineInstr *>
  getUsesOfGlobalValue(const llvm::GlobalValue &GV) const {
    auto UsesIt = GlobalValueMIUses.find(&GV);
    if (UsesIt == GlobalValueMIUses.end())
      return {};
    else
      return UsesIt->getSecond();
  }
};

} // namespace luthier

#endif