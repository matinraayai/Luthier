
#ifndef LUTHIER_TOOLING_INSTRUCTION_MODEL_H
#define LUTHIER_TOOLING_INSTRUCTION_MODEL_H
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/Register.h>
#include <llvm/IR/Value.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>

namespace llvm {

class SIRegisterInfo;

class MachineInstr;

class Error;

} // namespace llvm

namespace luthier {

class MCRegisterValueMap {
  const llvm::SIRegisterInfo &Info;

  llvm::LLVMContext &Ctx;

  llvm::DenseMap<llvm::MCRegister, std::reference_wrapper<llvm::Value>>
      RegValMap;

public:
  explicit MCRegisterValueMap(llvm::LLVMContext &Ctx,
                              const llvm::SIRegisterInfo &Info)
      : Info(Info), Ctx(Ctx) {};

  llvm::Value *getValue(llvm::MCRegister Reg);

  void setValue(llvm::MCRegister Reg, llvm::Value &Val);

  llvm::LLVMContext &getContext() { return Ctx; }

  const llvm::LLVMContext &getContext() const { return Ctx; }
};

llvm::Expected<llvm::Value &>
raiseMachineInstr(const llvm::MachineInstr &MI,
                  MCRegisterValueMap &RegisterValueMap);

} // namespace luthier

#endif