#ifndef LUTHIER_EXPR_CONSTANT_MIR_EXPR_H
#define LUTHIER_EXPR_CONSTANT_MIR_EXPR_H
#include "luthier/expr/MachineIRExpr.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/CodeGen/TargetOpcodes.h>

namespace luthier {

class ConstantMIRExpr : public MachineIRExpr {

protected:
  explicit ConstantMIRExpr(uint16_t Opcode) : MachineIRExpr(Opcode) {}

public:
  unsigned getNumChildren() const override { return 0; }

  const MachineIRExpr *getChild(unsigned) const override { return nullptr; }

  static bool classof(const MachineIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_CONSTANT ||
           E->getOpcode() == llvm::TargetOpcode::G_FCONSTANT;
  }
};

class GConstantMIRExpr final : public ConstantMIRExpr {

private:
  llvm::APInt Value;

  explicit GConstantMIRExpr(const llvm::APInt &v)
      : ConstantMIRExpr(llvm::TargetOpcode::G_CONSTANT), Value(v) {}

public:
  unsigned int getWidth() const override { return Value.getBitWidth(); }

  /// \returns the arbitrary precision value of the constant
  const llvm::APInt &getAPValue() const { return Value; }

  static bool classof(const MachineIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_CONSTANT;
  }

  static std::shared_ptr<GConstantMIRExpr> create(const llvm::APInt &v) {
    return std::shared_ptr<GConstantMIRExpr>(new GConstantMIRExpr(v));
  }
};

class GFConstantMIRExpr final : public ConstantMIRExpr {

private:
  llvm::APFloat Value;

  explicit GFConstantMIRExpr(const llvm::APFloat &v)
      : ConstantMIRExpr(llvm::TargetOpcode::G_FCONSTANT), Value(v) {}

public:
  unsigned int getWidth() const override {
    return Value.getSizeInBits(Value.getSemantics());
  }

  /// \returns the arbitrary precision value of the constant
  const llvm::APFloat &getAPValue() const { return Value; }

  static bool classof(const MachineIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_FCONSTANT;
  }

  static std::shared_ptr<GFConstantMIRExpr> create(const llvm::APFloat &v) {
    return std::shared_ptr<GFConstantMIRExpr>(new GFConstantMIRExpr(v));
  }
};

} // namespace luthier

#endif