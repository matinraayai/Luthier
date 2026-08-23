//===-- TargetModule.h ------------------------------------------*- C++ -*-===//
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
/// \file
/// Defines \c TargetModule, an owning handle around the \c llvm::Module that
/// holds the code of the application being instrumented and paired with a
/// reference to the \c Prototype that owns it, together with
/// \c TraceFunction, a non-owning handle around an \c llvm::Function inside
/// such a module.
///
/// \c TraceFunction is declared first so that \c TraceFunctionIterator can
/// construct one in-class, and so that \c TargetModule 's function accessors
/// can return it by value.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_H
#define LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_H
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <optional>

namespace luthier {

class Prototype;
class TargetModule;

/// \brief Non-owning handle around an \c llvm::Function belonging to a
/// \c TargetModule, plus a link back to that module.
///
/// \details Mirrors \c TargetModule 's access pattern — \c operator* and
/// \c operator-> reach the wrapped entity, \c getParent reaches the owner —
/// but deliberately does \e not own the function it wraps. An
/// \c llvm::Function lives in its module's \c FunctionList , which is a
/// \c SymbolTableList\<Function\> (\c llvm/IR/Module.h:74 ) and deletes its
/// elements when the module is destroyed. A \c std::unique_ptr member here
/// would therefore double-free. The wrapped function must outlive the handle,
/// which it does as long as the parent \c TargetModule is alive and the
/// function is not erased from it.
class TraceFunction {
  /// The wrapped function. Owned by the parent module, not by this handle.
  llvm::Function &F;

  /// The target module the wrapped function belongs to.
  TargetModule &Parent;

public:
  /// Wraps \p F, which must belong to \p Parent and outlive this handle.
  TraceFunction(llvm::Function &F, TargetModule &Parent)
      : F(F), Parent(Parent) {}

  //===--------------------------------------------------------------------===//
  // Wrapped function access
  //===--------------------------------------------------------------------===//

  llvm::Function &operator*() { return F; }

  [[nodiscard]] const llvm::Function &operator*() const { return F; }

  llvm::Function *operator->() { return &F; }

  [[nodiscard]] const llvm::Function *operator->() const { return &F; }

  /// \return a raw pointer to the wrapped function, for the many LLVM APIs
  /// that take an <tt>llvm::Function *</tt>. Never null.
  llvm::Function *get() { return &F; }

  [[nodiscard]] const llvm::Function *get() const { return &F; }

  //===--------------------------------------------------------------------===//
  // Parent access
  //===--------------------------------------------------------------------===//

  TargetModule &getParent() { return Parent; }

  [[nodiscard]] const TargetModule &getParent() const { return Parent; }
};

/// \brief Adapts one of \c llvm::Module 's function-list iterators so that
/// dereferencing yields a \c TraceFunction bound to the owning
/// \c TargetModule.
///
/// \details A proxy iterator: \c operator* returns a \c TraceFunction by value
/// rather than a reference into the function list, because no
/// \c TraceFunction is stored anywhere — it is synthesized per dereference
/// from the list element and the parent module. Consequently there is no
/// \c operator-> ; reach the wrapped function with <tt>(*It)-\></tt> or
/// <tt>It-\>get()</tt> spelled as <tt>(*It).get()</tt>.
///
/// \tparam WrappedItT the adapted \c llvm::Module iterator, i.e.
/// \c llvm::Module::iterator or \c llvm::Module::reverse_iterator .
template <typename WrappedItT> class TraceFunctionIterator {
  WrappedItT I{};

  TargetModule *Parent = nullptr;

public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = TraceFunction;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = TraceFunction;

  TraceFunctionIterator() = default;

  TraceFunctionIterator(WrappedItT I, TargetModule &Parent)
      : I(I), Parent(&Parent) {}

  TraceFunction operator*() const { return TraceFunction(*I, *Parent); }

  TraceFunctionIterator &operator++() {
    ++I;
    return *this;
  }

  TraceFunctionIterator operator++(int) {
    TraceFunctionIterator Tmp = *this;
    ++I;
    return Tmp;
  }

  TraceFunctionIterator &operator--() {
    --I;
    return *this;
  }

  TraceFunctionIterator operator--(int) {
    TraceFunctionIterator Tmp = *this;
    --I;
    return Tmp;
  }

  bool operator==(const TraceFunctionIterator &RHS) const { return I == RHS.I; }

  bool operator!=(const TraceFunctionIterator &RHS) const { return I != RHS.I; }

  /// \return the adapted \c llvm::Module iterator, for the LLVM APIs that
  /// take one (e.g. \c llvm::Module::getFunctionList().erase ).
  [[nodiscard]] WrappedItT getWrappedIterator() const { return I; }
};

/// \brief Owning handle around the \c llvm::Module holding the code of the
/// application being instrumented, plus a link back to its parent
/// \c Prototype.
///
/// \details Composition rather than inheritance, which buys two things that
/// subclassing \c llvm::Module cannot offer:
///
/// 1. Ownership is a plain <tt>std::unique_ptr\<llvm::Module\></tt>, so the
///    non-virtual \c ~Module (\c llvm/IR/Module.h:243 ) is never a hazard —
///    the pointee really is an \c llvm::Module .
///
/// 2. A module produced by LLVM can be adopted. \c llvm::Module has neither a
///    copy nor a move constructor, and LLVM's parsers hard-code
///    <tt>std::make_unique\<Module\></tt> with no factory hook
///    (\c llvm/lib/AsmParser/Parser.cpp:55 ,
///    \c llvm/lib/CodeGen/MIRParser/MIRParser.cpp:264 and \c :288 ,
///    \c llvm/lib/Transforms/Utils/CloneModule.cpp:60 ), so a subclass could
///    never wrap their output. The adopting constructor below takes the
///    <tt>std::unique_ptr\<llvm::Module\></tt> that \c llvm::parseIR and
///    \c llvm::MIRParser::parseIRModule already return.
class TargetModule {
  /// The wrapped module. Never null for a non-moved-from handle.
  std::unique_ptr<llvm::Module> M;

  /// The prototype that owns this target module.
  Prototype &Parent;

public:
  /// Creates an empty target module named \p ModuleID owned by \p Parent.
  ///
  /// \p Parent need not be fully constructed yet: \c Prototype may pass
  /// <tt>*this</tt> from its own constructor, since only the reference is
  /// stored.
  TargetModule(llvm::StringRef ModuleID, llvm::LLVMContext &C,
               Prototype &Parent);

  /// Adopts \p M — typically straight out of \c llvm::parseIR or
  /// \c llvm::MIRParser::parseIRModule — as the target module of \p Parent.
  ///
  /// \p M must not be null.
  TargetModule(std::unique_ptr<llvm::Module> M, Prototype &Parent);

  TargetModule(const TargetModule &) = delete;
  TargetModule &operator=(const TargetModule &) = delete;

  /// Move construction transfers the module; the moved-from handle is left
  /// wrapping nothing and must not be dereferenced. Move assignment is not
  /// available because \c Parent is a reference and cannot be reseated.
  TargetModule(TargetModule &&) = default;
  TargetModule &operator=(TargetModule &&) = delete;

  //===--------------------------------------------------------------------===//
  // Inner module access
  //===--------------------------------------------------------------------===//

  llvm::Module &operator*() { return *M; }

  [[nodiscard]] const llvm::Module &operator*() const { return *M; }

  llvm::Module *operator->() { return M.get(); }

  [[nodiscard]] const llvm::Module *operator->() const { return M.get(); }

  /// \return a raw pointer to the wrapped module, for the many LLVM APIs that
  /// take an <tt>llvm::Module *</tt>. Null only on a moved-from handle.
  llvm::Module *get() { return M.get(); }

  [[nodiscard]] const llvm::Module *get() const { return M.get(); }

  //===--------------------------------------------------------------------===//
  // Parent access
  //===--------------------------------------------------------------------===//

  Prototype &getParentPrototype() { return Parent; }

  [[nodiscard]] const Prototype &getParentPrototype() const { return Parent; }

  //===--------------------------------------------------------------------===//
  // Function accessors
  //
  // Mirrors llvm::Module's own function-access surface (Module.h:377 and
  // Module.h:694), with the mutable overloads yielding TraceFunction handles.
  //
  // The const overloads yield plain `const llvm::Function`s rather than
  // handles: a TraceFunction hands out mutable access to both the function and
  // its parent module, so one cannot be manufactured from a `const
  // TargetModule` without casting away constness.
  //===--------------------------------------------------------------------===//

  /// The \c TraceFunction iterators.
  using iterator = TraceFunctionIterator<llvm::Module::iterator>;
  using reverse_iterator = TraceFunctionIterator<llvm::Module::reverse_iterator>;

  /// The constant iterators, over the underlying \c llvm::Function s.
  using const_iterator = llvm::Module::const_iterator;
  using const_reverse_iterator = llvm::Module::const_reverse_iterator;

  /// \return a handle to the function named \p Name, or \c std::nullopt if
  /// this module has no such function.
  std::optional<TraceFunction> getFunction(llvm::StringRef Name) {
    if (llvm::Function *F = M->getFunction(Name))
      return TraceFunction(*F, *this);
    return std::nullopt;
  }

  [[nodiscard]] const llvm::Function *getFunction(llvm::StringRef Name) const {
    return M->getFunction(Name);
  }

  iterator begin() { return {M->begin(), *this}; }
  [[nodiscard]] const_iterator begin() const { return M->begin(); }
  iterator end() { return {M->end(), *this}; }
  [[nodiscard]] const_iterator end() const { return M->end(); }
  reverse_iterator rbegin() { return {M->rbegin(), *this}; }
  [[nodiscard]] const_reverse_iterator rbegin() const { return M->rbegin(); }
  reverse_iterator rend() { return {M->rend(), *this}; }
  [[nodiscard]] const_reverse_iterator rend() const { return M->rend(); }

  [[nodiscard]] size_t size() const { return M->size(); }
  [[nodiscard]] bool empty() const { return M->empty(); }

  llvm::iterator_range<iterator> functions() {
    return llvm::make_range(begin(), end());
  }

  [[nodiscard]] llvm::iterator_range<const_iterator> functions() const {
    return llvm::make_range(begin(), end());
  }

  /// Get an iterator range over all function definitions (excluding
  /// declarations).
  auto getFunctionDefs() {
    return llvm::make_filter_range(
        functions(), [](const TraceFunction &TF) { return !TF->isDeclaration(); });
  }

  auto getFunctionDefs() const {
    return llvm::make_filter_range(
        functions(), [](const llvm::Function &F) { return !F.isDeclaration(); });
  }
};

} // namespace luthier

#endif
