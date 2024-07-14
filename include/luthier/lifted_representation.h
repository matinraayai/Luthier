//===-- lifted_representation.h - Lifted Representation  --------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's Lifted Representation, which contains the
/// \c llvm::Module and \c llvm::MachineModuleInfo of a lifted HSA primitive (a
/// kernel or an executable), as well as a mapping between the HSA primitives
/// and LLVM IR primitives involved.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_LIFTED_REPRESENTATION_H
#define LUTHIER_LIFTED_REPRESENTATION_H
#include "llvm_dense_map_info.h"
#include <hsa/hsa.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <luthier/instr.h>

namespace llvm {
class GCNTargetMachine;
}

namespace luthier {

class CodeLifter;

/// \brief contains HSA and LLVM information regarding a lifted AMD HSA
/// primitive.
/// \details The primitive can be either an \c hsa_executable_t or a
/// \c hsa_executable_symbol_t of type kernel. Luthier's \c CodeLifter is
/// the only entity allowed to construct or clone a
/// <tt>LiftedRepresentation</tt>. This allows internal caching and thread-safe
/// access to them by other components. It also allows invalidation of the
/// representations when the executable backing the lifted primitive gets
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

  std::shared_ptr<llvm::GCNTargetMachine> TM;

  /// A thread-safe context that owns all the thread-safe modules;
  /// Each LiftedRepresentation is given its own context to allow for
  /// independent processing from others\n
  /// During instrumentation all bitcode is loaded into this Context\n
  /// This ThreadSafeContext retains the underlying \c llvm::LLVMContext
  /// unique pointer's ownership
  llvm::orc::ThreadSafeContext Context;

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
                      std::pair<llvm::orc::ThreadSafeModule,
                                std::unique_ptr<llvm::MachineModuleInfo>>,
                      1>
      RelatedLCOs{};

  /// Mapping between an \c hsa_executable_symbol_t (of type kernel and
  /// device function) and its \c llvm::MachineFunction
  llvm::DenseMap<hsa_executable_symbol_t, llvm::MachineFunction *>
      RelatedFunctions{};

  /// Mapping between an \c hsa_executable_symbol_t of type variable
  /// and its \c llvm::GlobalVariable
  llvm::DenseMap<hsa_executable_symbol_t, llvm::GlobalVariable *>
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

  LiftedRepresentation() = default;

public:
  /// Disallowed copy construction
  LiftedRepresentation(const LiftedRepresentation &) = delete;

  /// Disallowed assignment operation
  LiftedRepresentation &operator=(const LiftedRepresentation &) = delete;

  /// \return a reference to the thread-safe \c LLVMContext of this object
  llvm::orc::ThreadSafeContext &getContext() { return Context; }

  /// \return a const reference to the the thread-safe \c LLVMContext of this
  /// object
  [[nodiscard]] const llvm::orc::ThreadSafeContext &getContext() const {
    return Context;
  }

  template <typename TMT>
  const TMT& getTargetMachine() const {
    return *reinterpret_cast<TMT*>(TM.get());
  }

  template <typename TMT>
  TMT& getTargetMachine() {
    return *reinterpret_cast<TMT*>(TM.get());
  }

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

  const llvm::MachineFunction &getMF(hsa_executable_symbol_t Func) {
    return *RelatedFunctions.at(Func);
  }

  const llvm::GlobalVariable &getGV(hsa_executable_symbol_t GV) {
    return *RelatedGlobalVariables.at(GV);
  }

  [[nodiscard]] const hsa::Instr &
  getHSAInstrOfMachineInstr(const llvm::MachineInstr &MI) const {
    return *MachineInstrToMCMap.at(const_cast<llvm::MachineInstr *>(&MI));
  }
};

} // namespace luthier

#endif