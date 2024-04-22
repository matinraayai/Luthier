#include "hsa_isa.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

namespace luthier::hsa {

llvm::Expected<ISA> ISA::fromName(const char *IsaName) {
  hsa_isa_t Isa;
  const auto &CoreApi =
      luthier::hsa::Interceptor::instance().getSavedHsaTables().core;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(CoreApi.hsa_isa_from_name_fn(IsaName, &Isa)));

  return luthier::hsa::ISA(Isa);
}

llvm::Expected<ISA> ISA::fromLLVM(const llvm::Triple &TT, llvm::StringRef CPU,
                                  const llvm::SubtargetFeatures &Features) {
  llvm::Twine ISAName(TT.getTriple() + CPU + ":");
  for (const auto &Feature : Features.getFeatures()) {
    ISAName.concat(Feature.substr(1) + Feature[0]);
  }
  return fromName(ISAName.str().c_str());
}

llvm::Expected<std::string> ISA::getName() const {
  uint32_t IsaNameSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(
          this->asHsaType(), HSA_ISA_INFO_NAME_LENGTH, &IsaNameSize)));
  std::string IsaName(IsaNameSize - 1, '\0');
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(
          this->asHsaType(), HSA_ISA_INFO_NAME, &IsaName.front())));
  return IsaName;
}

inline llvm::Error parseIsaName(llvm::StringRef IsaName,
                                llvm::SmallVectorImpl<llvm::StringRef> &Out) {
  IsaName.split(Out, '-', 4);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Out.size() == 5));
  return llvm::Error::success();
}

llvm::Expected<std::string> ISA::getArchitecture() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrError, IsaNameComponents));
  return IsaNameComponents[0].str();
}

llvm::Expected<std::string> ISA::getVendor() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrError, IsaNameComponents));
  return IsaNameComponents[1].str();
}

llvm::Expected<std::string> ISA::getOS() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrError, IsaNameComponents));
  return std::move(IsaNameComponents[2].str());
}
llvm::Expected<std::string> ISA::getEnvironment() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrError, IsaNameComponents));
  return IsaNameComponents[3].str();
}

llvm::Expected<std::string> ISA::getProcessor() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrError, IsaNameComponents));

  llvm::SmallVector<llvm::StringRef> Features;
  Features.clear();
  IsaNameComponents[4].split(Features, ':');

  return Features[0].str();
}
llvm::Expected<bool> ISA::isXNACSupported() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  auto XNack = IsaNameOrError->find("xnack");
  if (XNack == std::string::npos)
    return false;
  else {
    return (*IsaNameOrError)[XNack + strlen("xnack")] == '+';
  }
}

llvm::Expected<bool> ISA::isSRAMECCSupported() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  auto SRamECC = IsaNameOrError->find("sramecc");
  if (SRamECC == std::string::npos)
    return false;
  else {
    return (*IsaNameOrError)[SRamECC + strlen("sramecc")] == '+';
  }
}

llvm::Expected<llvm::Triple> ISA::getTargetTriple() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrError, IsaNameComponents));
  return llvm::Triple(llvm::Twine(IsaNameComponents[0]) + "-" +
                      llvm::Twine(IsaNameComponents[1]) + "-" +
                      llvm::Twine(IsaNameComponents[2]) + "-" +
                      llvm::Twine(IsaNameComponents[3]));
}

llvm::Expected<llvm::SubtargetFeatures> ISA::getSubTargetFeatures() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> Features;
  llvm::StringRef(*IsaNameOrError)
      .substr(IsaNameOrError->find_first_of(':'))
      .split(Features, ":");
  // The +/- must be before the feature code for LLVM, not after
  std::vector<std::string> FeaturesOut;
  for (auto &Feat : Features) {
    auto FeatureToggle = Feat.substr(Feat.size() - 1);
    auto FeatureName = Feat.substr(0, Feat.size() - 1);
    FeaturesOut.push_back((FeatureToggle + FeatureName).str());
  }
  return llvm::SubtargetFeatures(llvm::join(FeaturesOut, ","));
}

} // namespace luthier::hsa
