#ifndef LUTHIER_TOOLING_COMMON_INTRINSIC_INTRINSIC_LOWERING_INFO_HPP
#define LUTHIER_TOOLING_COMMON_INTRINSIC_INTRINSIC_LOWERING_INFO_HPP

#include <functional>
#include <llvm/ADT/Any.h>
#include <llvm/ADT/SmallVector.h>
#include <string>

namespace llvm {
class Value;

class Function;

class CallInst;

class GCNTargetMachine;
} // namespace llvm

namespace luthier {

struct IntrinsicValueLoweringInfo {
  llvm::Value *Val;
  std::string Constraint;
};

struct IntrinsicIRLoweringInfo {
private:
  IntrinsicValueLoweringInfo OutValue{nullptr, ""};
  llvm::SmallVector<IntrinsicValueLoweringInfo, 4> Args{};
  llvm::Any Data{};

public:
  void setReturnValueInfo(llvm::Value *Val, llvm::StringRef Constraint) {
    OutValue.Val = Val;
    OutValue.Constraint = Constraint;
  }

  const IntrinsicValueLoweringInfo &getReturnValueInfo() { return OutValue; }

  void addArgInfo(llvm::Value *Val, llvm::StringRef Constraint) {
    Args.emplace_back(Val, std::string(Constraint));
  }

  llvm::ArrayRef<IntrinsicValueLoweringInfo> getArgsInfo() { return Args; }

  template <typename T> void setLoweringData(T D) { Data = D; }

  template <typename T> T &getLoweringData() {
    return *llvm::any_cast<T>(&Data);
  }
};

typedef std::function<llvm::Expected<IntrinsicIRLoweringInfo>(
    const llvm::Function &, const llvm::CallInst &,
    const llvm::GCNTargetMachine &)>
    IntrinsicIRProcessorFunc;

typedef std::function<llvm::Error(const IntrinsicValueLoweringInfo &)>
    IntrinsicMIRProcessorFunc;

struct IntrinsicProcessor {
  IntrinsicIRProcessorFunc IRProcessor{};
  IntrinsicMIRProcessorFunc MIRProcessor{};
};

} // namespace luthier

#endif