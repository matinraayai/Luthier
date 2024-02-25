#include "hsa_isa.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

namespace luthier::hsa {

llvm::Expected<ISA> ISA::fromName(const char *isaName) {
  hsa_isa_t Isa;
  const auto &coreApi =
      luthier::HsaInterceptor::instance().getSavedHsaTables().core;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(coreApi.hsa_isa_from_name_fn(isaName, &Isa)));

  return luthier::hsa::ISA(Isa);
}
llvm::Expected<std::string> ISA::getName() const {
  uint32_t IsaNameSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(
          this->asHsaType(), HSA_ISA_INFO_NAME_LENGTH, &IsaNameSize)));
  std::string isaName(IsaNameSize - 1, '\0');
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(
          this->asHsaType(), HSA_ISA_INFO_NAME, &isaName.front())));
  return isaName;
}

inline void parseIsaName(llvm::StringRef isaName,
                         llvm::SmallVectorImpl<llvm::StringRef> &out) {
  isaName.split(out, '-', 4);
  LUTHIER_CHECK((out.size() == 5));
}

llvm::Expected<std::string> ISA::getArchitecture() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  parseIsaName(*IsaNameOrError, IsaNameComponents);
  return IsaNameComponents[0].str();
}

llvm::Expected<std::string> ISA::getVendor() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  parseIsaName(*IsaNameOrError, IsaNameComponents);
  return IsaNameComponents[1].str();
}

llvm::Expected<std::string> ISA::getOS() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> isaNameComponents;
  parseIsaName(*IsaNameOrError, isaNameComponents);
  return std::move(isaNameComponents[2].str());
}
llvm::Expected<std::string> ISA::getEnvironment() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> isaNameComponents;
  parseIsaName(*IsaNameOrError, isaNameComponents);
  return isaNameComponents[3].str();
}

llvm::Expected<std::string> ISA::getProcessor() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> isaNameComponents;
  parseIsaName(*IsaNameOrError, isaNameComponents);

  llvm::SmallVector<llvm::StringRef> features;
  features.clear();
  isaNameComponents[4].split(features, ':');

  return features[0].str();
}
llvm::Expected<bool> ISA::isXNACSupported() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  auto sRamECC = IsaNameOrError->find("xnack");
  if (sRamECC == std::string::npos)
    return false;
  else {
    return (*IsaNameOrError)[sRamECC + strlen("xnack")] == '+';
  }
}

llvm::Expected<bool> ISA::isSRAMECCSupported() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  auto sRamECC = IsaNameOrError->find("sramecc");
  if (sRamECC == std::string::npos)
    return false;
  else {
    return (*IsaNameOrError)[sRamECC + strlen("sramecc")] == '+';
  }
}

llvm::Expected<std::string> ISA::getLLVMTarget() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  return IsaNameOrError->substr(0, IsaNameOrError->find_first_of(':'));
}

llvm::Expected<std::string> ISA::getLLVMTargetTriple() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> isaNameComponents;
  parseIsaName(*IsaNameOrError, isaNameComponents);
  return (llvm::Twine(isaNameComponents[0]) + "-" +
          llvm::Twine(isaNameComponents[1]) + "-" +
          llvm::Twine(isaNameComponents[2]) + "-" +
          llvm::Twine(isaNameComponents[3]))
      .str();
}

llvm::Expected<std::string> ISA::getFeatureString() const {
  auto IsaNameOrError = getName();
  LUTHIER_RETURN_ON_ERROR(IsaNameOrError.takeError());
  llvm::SmallVector<llvm::StringRef> features;
  llvm::StringRef(*IsaNameOrError)
      .substr(IsaNameOrError->find_first_of(':'))
      .split(features, ":");
  // The +/- must be before the feature code for LLVM, not after
  std::vector<std::string> featuresOut;
  for (auto &feat : features) {
    auto featureToggle = feat.substr(feat.size() - 1);
    auto featureName = feat.substr(0, feat.size() - 1);
    featuresOut.push_back((featureToggle + featureName).str());
  }
  return llvm::join(featuresOut, ",");
}

} // namespace luthier::hsa
