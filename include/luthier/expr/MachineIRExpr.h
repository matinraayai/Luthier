
#ifndef LUTHIER_TOOLING_MACHINE_IR_EXPR_H
#define LUTHIER_TOOLING_MACHINE_IR_EXPR_H
#include <memory>

namespace llvm {
class raw_ostream;
}

namespace luthier {
class GConstantMIRExpr;

/// \brief Base expression class representing the output value of a generic
/// machine instruction
class MachineIRExpr : public std::enable_shared_from_this<MachineIRExpr> {

protected:
  /// Generic machine instruction opcode
  uint16_t Opcode;

  explicit MachineIRExpr(uint16_t Opcode) : Opcode(Opcode) {};

public:
  virtual ~MachineIRExpr() = default;

  uint16_t getOpcode() const { return Opcode; }

  [[nodiscard]] virtual unsigned int getWidth() const = 0;

  [[nodiscard]] virtual unsigned int getNumChildren() const = 0;

  [[nodiscard]] virtual const MachineIRExpr *getChild(unsigned Idx) const = 0;

  virtual void print(llvm::raw_ostream &OS) const;

  void dump() const;

  static bool classof(const MachineIRExpr *) { return true; }
};

} // namespace luthier

#endif