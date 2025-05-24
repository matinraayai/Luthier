//===-- ISA.cpp -----------------------------------------------------------===//
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
/// This file implements functionality related to the \c hsa_isa_t type in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/ISA.h"
#include "luthier/common/LuthierError.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

namespace luthier::hsa {

llvm::Expected<hsa_isa_t>
ISAFromLLVM(const decltype(hsa_isa_from_name) &HsaISAFromNameFn,
            const llvm::Triple &TT, llvm::StringRef GPUName,
            const llvm::SubtargetFeatures &Features) {
  auto ISAName = (TT.getTriple() + llvm::Twine("--") + GPUName).str();
  auto FeatureStrings = Features.getFeatures();
  if (!FeatureStrings.empty()) {
    ISAName += ":";
    for (const auto &Feature : FeatureStrings) {
      ISAName += (Feature.substr(1) + Feature[0]);
    }
  }
  return isaFromName(HsaISAFromNameFn, ISAName);
}

llvm::Expected<std::string>
getISAName(hsa_isa_t ISA,
           const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  uint32_t IsaNameSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaIsaGetInfoFn(ISA, HSA_ISA_INFO_NAME_LENGTH, &IsaNameSize)));
  std::string IsaName(IsaNameSize - 1, '\0');
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaIsaGetInfoFn(ISA, HSA_ISA_INFO_NAME, &IsaName.front())));
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

llvm::Expected<std::string>
getISAArchitecture(hsa_isa_t ISA,
                   const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  llvm::Expected<std::string> IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return IsaNameComponents[0].str();
}

llvm::Expected<std::string>
getISAVendor(hsa_isa_t ISA,
             const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  llvm::Expected<std::string> IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return IsaNameComponents[1].str();
}

llvm::Expected<std::string>
getISAOperatingSystem(hsa_isa_t ISA,
                      const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  llvm::Expected<std::string> IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return std::move(IsaNameComponents[2].str());
}
llvm::Expected<std::string>
getISAEnvironment(hsa_isa_t ISA,
                  const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  auto IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return IsaNameComponents[3].str();
}

llvm::Expected<std::string>
getISAGPUName(hsa_isa_t ISA,
              const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  auto IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));

  llvm::SmallVector<llvm::StringRef> Features;
  Features.clear();
  IsaNameComponents[4].split(Features, ':');

  return Features[0].str();
}

llvm::Expected<bool>
doesISASupportXNACK(hsa_isa_t ISA,
                    const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  auto IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  auto XNack = IsaNameOrErr->find("xnack");
  if (XNack == std::string::npos)
    return false;
  else {
    return (*IsaNameOrErr)[XNack + strlen("xnack")] == '+';
  }
}

llvm::Expected<bool>
doesISASupportSRAMECC(hsa_isa_t ISA,
                      const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  llvm::Expected<std::string> IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  auto SRamECC = IsaNameOrErr->find("sramecc");
  if (SRamECC == std::string::npos)
    return false;
  else {
    return (*IsaNameOrErr)[SRamECC + strlen("sramecc")] == '+';
  }
}

llvm::Expected<llvm::Triple>
getISATargetTriple(hsa_isa_t ISA,
                   const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  llvm::Expected<std::string> IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return llvm::Triple(llvm::Twine(IsaNameComponents[0]) + "-" +
                      llvm::Twine(IsaNameComponents[1]) + "-" +
                      llvm::Twine(IsaNameComponents[2]) + "-" +
                      llvm::Twine(IsaNameComponents[3]));
}

llvm::Expected<llvm::SubtargetFeatures>
getISASubTargetFeatures(hsa_isa_t ISA,
                        const decltype(hsa_isa_get_info_alt) &HsaIsaGetInfoFn) {
  llvm::Expected<std::string> IsaNameOrErr = getISAName(ISA, HsaIsaGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> Features;
  llvm::StringRef(*IsaNameOrErr)
      .substr(IsaNameOrErr->find_first_of(':'))
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
