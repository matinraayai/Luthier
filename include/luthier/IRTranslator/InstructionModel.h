
#ifndef LUTHIER_TOOLING_INSTRUCTION_MODEL_H
#define LUTHIER_TOOLING_INSTRUCTION_MODEL_H
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/Register.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Value.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>

namespace llvm {

class SIRegisterInfo;

class MachineInstr;

class Error;

} // namespace llvm

namespace luthier {

class IRTranslator : llvm::PassInfoMixin<IRTranslator> {
public:
  IRTranslator() = default;

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

llvm::Expected<llvm::Value &>
raiseMachineInstr(const llvm::MachineInstr &MI,
                  BasicBlockMCRegisterValueMap &RegisterValueMap);

} // namespace luthier

#endif