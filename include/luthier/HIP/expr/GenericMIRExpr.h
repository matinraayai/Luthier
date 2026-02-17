//===-- GenericMIRExpr.h ------------------------------------------*-C++-*-===//
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
/// Defines the \c GenericMIRExpr class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXPR_GENERIC_MACHINE_IR_EXPR_H
#define LUTHIER_EXPR_GENERIC_MACHINE_IR_EXPR_H
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

namespace llvm {
class raw_ostream;
}

namespace luthier {
class GConstantMIRExpr;

/// \brief Base expression class representing the output/input expression of
/// a \c llvm::GenericMachineInstr
class GenericMIRExpr : public std::enable_shared_from_this<GenericMIRExpr> {

protected:
  /// Generic machine instruction opcode
  uint16_t Opcode;
  /// Children (operands) of the expression
  llvm::SmallVector<std::shared_ptr<GenericMIRExpr>, 5> Children;

  GenericMIRExpr(
      uint16_t Opcode,
      std::initializer_list<std::shared_ptr<GenericMIRExpr>> Children)
      : Opcode(Opcode), Children(Children) {};

public:
  using child_iterator = decltype(Children)::iterator;
  using const_child_iterator = decltype(Children)::const_iterator;

  virtual ~GenericMIRExpr() = default;

  uint16_t getOpcode() const { return Opcode; }

  /// \returns the bit width of the output of the expression
  [[nodiscard]] virtual unsigned int getWidth() const = 0;

  [[nodiscard]] unsigned int getNumChildren() const { return Children.size(); }

  bool isLeaf() const { return Children.empty(); }

  child_iterator begin() { return Children.begin(); }

  const_child_iterator begin() const { return Children.begin(); }

  child_iterator end() { return Children.end(); }

  const_child_iterator end() const { return Children.end(); }

  llvm::iterator_range<child_iterator> children() {
    return llvm::make_range(Children.begin(), Children.end());
  }

  llvm::iterator_range<const_child_iterator> children() const {
    return llvm::make_range(Children.begin(), Children.end());
  }

  std::shared_ptr<GenericMIRExpr> &getChild(unsigned Idx) {
    return Children[Idx];
  }

  const GenericMIRExpr &getChild(unsigned Idx) const { return *Children[Idx]; }

  virtual void print(llvm::raw_ostream &OS) const;

  void dump() const;

  static bool classof(const GenericMIRExpr *) { return true; }
};

} // namespace luthier

#endif