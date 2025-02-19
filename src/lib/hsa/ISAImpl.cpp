//===-- ISAImpl.cpp -------------------------------------------------------===//
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
/// This file implements the \c hsa::ISAImpl class.
//===----------------------------------------------------------------------===//
#include "hsa/ISAImpl.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

namespace luthier::hsa {

llvm::Error ISAImpl::fromName(llvm::StringRef FullIsaName) {
  const auto &CoreApi = luthier::hsa::HsaRuntimeInterceptor::instance()
                            .getSavedApiTableContainer()
                            .core;
  return LUTHIER_HSA_SUCCESS_CHECK(
      CoreApi.hsa_isa_from_name_fn(FullIsaName.data(), &this->HsaType));
}

llvm::Error
ISA::fromLLVM(const llvm::Triple &TT, llvm::StringRef GPUName,
              const llvm::SubtargetFeatures &Features) {
  std::string ISAName = (TT.getTriple() + llvm::Twine("--") + GPUName).str();
  std::vector<std::string> FeatureStrings = Features.getFeatures();
  if (!FeatureStrings.empty()) {
    ISAName += ":";
    for (const auto &Feature : FeatureStrings) {
      ISAName += (Feature.substr(1) + Feature[0]);
    }
  }
  return fromName(ISAName);
}

llvm::Expected<std::string> ISAImpl::getName() const {
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
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      Out.size() == 5,
      "Failed to split the passed ISA name {0} into 5 different fields.",
      IsaName));
  return llvm::Error::success();
}

llvm::Expected<std::string> ISAImpl::getArchitecture() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return IsaNameComponents[0].str();
}

llvm::Expected<std::string> ISAImpl::getVendor() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return IsaNameComponents[1].str();
}

llvm::Expected<std::string> ISAImpl::getOS() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return std::move(IsaNameComponents[2].str());
}
llvm::Expected<std::string> ISAImpl::getEnvironment() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return IsaNameComponents[3].str();
}

llvm::Expected<std::string> ISAImpl::getGPUName() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));

  llvm::SmallVector<llvm::StringRef> Features;
  Features.clear();
  IsaNameComponents[4].split(Features, ':');

  return Features[0].str();
}
llvm::Expected<bool> ISAImpl::isXNACKSupported() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  auto XNack = IsaName->find("xnack");
  if (XNack == std::string::npos)
    return false;
  else {
    return (*IsaName)[XNack + strlen("xnack")] == '+';
  }
}

llvm::Expected<bool> ISAImpl::isSRAMECCSupported() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  auto SRamECC = IsaName->find("sramecc");
  if (SRamECC == std::string::npos)
    return false;
  else {
    return (*IsaName)[SRamECC + strlen("sramecc")] == '+';
  }
}

llvm::Expected<llvm::Triple> ISAImpl::getTargetTriple() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return llvm::Triple(llvm::Twine(IsaNameComponents[0]) + "-" +
                      llvm::Twine(IsaNameComponents[1]) + "-" +
                      llvm::Twine(IsaNameComponents[2]) + "-" +
                      llvm::Twine(IsaNameComponents[3]));
}

llvm::Expected<llvm::SubtargetFeatures> ISAImpl::getSubTargetFeatures() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> Features;
  llvm::StringRef(*IsaName)
      .substr(IsaName->find_first_of(':'))
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
