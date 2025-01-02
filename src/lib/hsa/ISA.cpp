//===-- ISA.cpp - HSA ISA Wrapper -----------------------------------------===//
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
/// This file implements the \c ISA class under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//

#include "hsa/ISA.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

namespace luthier::hsa {

llvm::Expected<ISA> ISA::fromName(llvm::StringRef FullIsaName) {
  hsa_isa_t Isa;
  const auto &CoreApi = luthier::hsa::HsaRuntimeInterceptor::instance()
                            .getSavedApiTableContainer()
                            .core;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      CoreApi.hsa_isa_from_name_fn(FullIsaName.data(), &Isa)));

  return luthier::hsa::ISA(Isa);
}

llvm::Expected<ISA> ISA::fromLLVM(const llvm::Triple &TT,
                                  llvm::StringRef GPUName,
                                  const llvm::SubtargetFeatures &Features) {
  auto ISAName = (TT.getTriple() + llvm::Twine("--") + GPUName).str();
  auto FeatureStrings = Features.getFeatures();
  if (!FeatureStrings.empty()) {
    ISAName += ":";
    for (const auto &Feature : FeatureStrings) {
      ISAName += (Feature.substr(1) + Feature[0]);
    }
  }
  return fromName(ISAName);
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
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      Out.size() == 5,
      "Failed to split the passed ISA name {0} into 5 different fields.",
      IsaName));
  return llvm::Error::success();
}

llvm::Expected<std::string> ISA::getArchitecture() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return IsaNameComponents[0].str();
}

llvm::Expected<std::string> ISA::getVendor() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return IsaNameComponents[1].str();
}

llvm::Expected<std::string> ISA::getOS() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return std::move(IsaNameComponents[2].str());
}
llvm::Expected<std::string> ISA::getEnvironment() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return IsaNameComponents[3].str();
}

llvm::Expected<std::string> ISA::getGPUName() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));

  llvm::SmallVector<llvm::StringRef> Features;
  Features.clear();
  IsaNameComponents[4].split(Features, ':');

  return Features[0].str();
}
llvm::Expected<bool> ISA::isXNACKSupported() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  auto XNack = IsaName->find("xnack");
  if (XNack == std::string::npos)
    return false;
  else {
    return (*IsaName)[XNack + strlen("xnack")] == '+';
  }
}

llvm::Expected<bool> ISA::isSRAMECCSupported() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  auto SRamECC = IsaName->find("sramecc");
  if (SRamECC == std::string::npos)
    return false;
  else {
    return (*IsaName)[SRamECC + strlen("sramecc")] == '+';
  }
}

llvm::Expected<llvm::Triple> ISA::getTargetTriple() const {
  auto IsaName = getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaName, IsaNameComponents));
  return llvm::Triple(llvm::Twine(IsaNameComponents[0]) + "-" +
                      llvm::Twine(IsaNameComponents[1]) + "-" +
                      llvm::Twine(IsaNameComponents[2]) + "-" +
                      llvm::Twine(IsaNameComponents[3]));
}

llvm::Expected<llvm::SubtargetFeatures> ISA::getSubTargetFeatures() const {
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
