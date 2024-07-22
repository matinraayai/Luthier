#ifndef LUTHIER_TOOLING_COMMON_INTRINSIC_INTRINSIC_LOWERING_INFO_HPP
#define LUTHIER_TOOLING_COMMON_INTRINSIC_INTRINSIC_LOWERING_INFO_HPP

#include <functional>
#include <llvm/ADT/Any.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/Error.h>
#include <string>

namespace llvm {
class Register;

class Value;

class Function;

class CallInst;

class GCNTargetMachine;

class MachineInstrBuilder;
} // namespace llvm

namespace luthier {

struct IntrinsicValueLoweringInfo {
  const llvm::Value *Val;
  std::string Constraint;
};

struct IntrinsicIRLoweringInfo {
private:
  std::string IntrinsicName;
  IntrinsicValueLoweringInfo OutValue{nullptr, ""};
  llvm::SmallVector<IntrinsicValueLoweringInfo, 4> Args{};
  llvm::Any Data{};

public:

  void setIntrinsicName(llvm::StringRef Name) {
    this->IntrinsicName = Name;
  }

  [[nodiscard]] llvm::StringRef getIntrinsicName() const {
    return IntrinsicName;
  }

  void setReturnValueInfo(const llvm::Value *Val, llvm::StringRef Constraint) {
    OutValue.Val = Val;
    OutValue.Constraint = Constraint;
  }

  [[nodiscard]] const IntrinsicValueLoweringInfo &getReturnValueInfo() const {
    return OutValue;
  }

  void addArgInfo(const llvm::Value *Val, llvm::StringRef Constraint) {
    Args.emplace_back(Val, std::string(Constraint));
  }

  llvm::ArrayRef<IntrinsicValueLoweringInfo> getArgsInfo() const {
    return Args;
  }

  template <typename T> void setLoweringData(T D) { Data = D; }

  template <typename T> const T &getLoweringData() const {
    return *llvm::any_cast<T>(&Data);
  }
};

typedef std::function<llvm::Expected<IntrinsicIRLoweringInfo>(
    const llvm::Function &, const llvm::CallInst &,
    const llvm::GCNTargetMachine &)>
    IntrinsicIRProcessorFunc;

typedef std::function<llvm::Error(
    const IntrinsicIRLoweringInfo &,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>>,
    const std::function<llvm::MachineInstrBuilder(int)> &)>
    IntrinsicMIRProcessorFunc;

struct IntrinsicProcessor {
  IntrinsicIRProcessorFunc IRProcessor{};
  IntrinsicMIRProcessorFunc MIRProcessor{};
};

} // namespace luthier

#endif