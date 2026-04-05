//===-- ISA.cpp -----------------------------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file ISA.cpp
/// Implements functionality related to the \c hsa_isa_t type in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/ISA.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSA/HsaError.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

namespace luthier::hsa {

llvm::Expected<hsa_isa_t>
isaFromLLVM(const ApiTableContainer<::CoreApiTable> &CoreApi,
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
  return isaFromName(CoreApi, ISAName);
}

llvm::Expected<std::string>
isaGetName(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA) {
  uint32_t IsaNameSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_isa_get_info_alt_fn>(
          ISA, HSA_ISA_INFO_NAME_LENGTH, &IsaNameSize),
      "Failed to get the length of the ISA name from HSA."));
  std::string IsaName(IsaNameSize - 1, '\0');
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_isa_get_info_alt>(ISA, HSA_ISA_INFO_NAME,
                                                 &IsaName.front()),
      "Failed to get the name of the ISA from HSA."));
  return IsaName;
}

inline llvm::Error parseIsaName(llvm::StringRef IsaName,
                                llvm::SmallVectorImpl<llvm::StringRef> &Out) {
  IsaName.split(Out, '-', 4);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Out.size() == 5,
      llvm::formatv(
          "Failed to split the passed ISA name {0} into 5 different fields.",
          IsaName)));
  return llvm::Error::success();
}

llvm::Expected<std::string>
isaGetArchitecture(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_isa_t ISA) {
  llvm::Expected<std::string> IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return IsaNameComponents[0].str();
}

llvm::Expected<std::string>
isaGetVendor(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA) {
  llvm::Expected<std::string> IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return IsaNameComponents[1].str();
}

llvm::Expected<std::string>
isaGetOperatingSystem(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_isa_t ISA) {
  llvm::Expected<std::string> IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return std::move(IsaNameComponents[2].str());
}
llvm::Expected<std::string>
isaGetEnvironment(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_isa_t ISA) {
  auto IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return IsaNameComponents[3].str();
}

llvm::Expected<std::string>
isaGetGPUName(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA) {
  auto IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));

  llvm::SmallVector<llvm::StringRef> Features;
  Features.clear();
  IsaNameComponents[4].split(Features, ':');

  return Features[0].str();
}

llvm::Expected<bool>
isaGetXnackSupport(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_isa_t ISA) {
  auto IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  auto XNack = IsaNameOrErr->find("xnack");
  if (XNack == std::string::npos)
    return false;
  else {
    return (*IsaNameOrErr)[XNack + strlen("xnack")] == '+';
  }
}

llvm::Expected<bool>
isaGetSramEcc(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA) {
  llvm::Expected<std::string> IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  auto SRamECC = IsaNameOrErr->find("sramecc");
  if (SRamECC == std::string::npos)
    return false;
  else {
    return (*IsaNameOrErr)[SRamECC + strlen("sramecc")] == '+';
  }
}

llvm::Expected<llvm::Triple>
isaGetTargetTriple(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_isa_t ISA) {
  llvm::Expected<std::string> IsaNameOrErr = isaGetName(CoreApi, ISA);
  LUTHIER_RETURN_ON_ERROR(IsaNameOrErr.takeError());
  llvm::SmallVector<llvm::StringRef> IsaNameComponents;
  LUTHIER_RETURN_ON_ERROR(parseIsaName(*IsaNameOrErr, IsaNameComponents));
  return llvm::Triple(llvm::Twine(IsaNameComponents[0]) + "-" +
                      llvm::Twine(IsaNameComponents[1]) + "-" +
                      llvm::Twine(IsaNameComponents[2]) + "-" +
                      llvm::Twine(IsaNameComponents[3]));
}

llvm::Expected<llvm::SubtargetFeatures>
isaGetSubTargetFeatures(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        hsa_isa_t ISA) {
  llvm::Expected<std::string> IsaNameOrErr = isaGetName(CoreApi, ISA);
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

template <typename T>
static llvm::Expected<T>
isaGetInfoT(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA,
            hsa_isa_info_t Attr) {
  T Val{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_isa_get_info_alt>(ISA, Attr, &Val),
      llvm::formatv("hsa_isa_get_info_alt({0}) failed for ISA {1:x}", Attr,
                    ISA.handle)));
  return Val;
}

llvm::Expected<std::array<bool, 2>>
isaGetMachineModels(const ApiTableContainer<::CoreApiTable> &CoreApi,
                    hsa_isa_t ISA) {
  std::array<bool, 2> Models{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_isa_get_info_alt>(
          ISA, HSA_ISA_INFO_MACHINE_MODELS, Models.data()),
      llvm::formatv("Failed to get machine models for ISA {0:x}", ISA.handle)));
  return Models;
}

llvm::Expected<std::array<bool, 2>>
isaGetProfiles(const ApiTableContainer<::CoreApiTable> &CoreApi,
               hsa_isa_t ISA) {
  std::array<bool, 2> Profiles{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_isa_get_info_alt>(ISA, HSA_ISA_INFO_PROFILES,
                                                 Profiles.data()),
      llvm::formatv("Failed to get profiles for ISA {0:x}", ISA.handle)));
  return Profiles;
}

llvm::Expected<std::array<bool, 3>> isaGetDefaultFloatRoundingModes(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA) {
  std::array<bool, 3> Modes{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_isa_get_info_alt>(
          ISA, HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES, Modes.data()),
      llvm::formatv("Failed to get default float rounding modes for ISA {0:x}",
                    ISA.handle)));
  return Modes;
}

llvm::Expected<bool>
isaGetFastF16(const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA) {
  return isaGetInfoT<bool>(CoreApi, ISA, HSA_ISA_INFO_FAST_F16_OPERATION);
}

llvm::Expected<std::array<uint16_t, 3>>
isaGetWorkgroupMaxDim(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_isa_t ISA) {
  std::array<uint16_t, 3> Dim{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_isa_get_info_alt>(
          ISA, HSA_ISA_INFO_WORKGROUP_MAX_DIM, Dim.data()),
      llvm::formatv("Failed to get workgroup max dim for ISA {0:x}",
                    ISA.handle)));
  return Dim;
}

llvm::Expected<uint32_t>
isaGetWorkgroupMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       hsa_isa_t ISA) {
  return isaGetInfoT<uint32_t>(CoreApi, ISA, HSA_ISA_INFO_WORKGROUP_MAX_SIZE);
}

llvm::Expected<hsa_dim3_t>
isaGetGridMaxDim(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_isa_t ISA) {
  return isaGetInfoT<hsa_dim3_t>(CoreApi, ISA, HSA_ISA_INFO_GRID_MAX_DIM);
}

llvm::Expected<uint64_t>
isaGetGridMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_isa_t ISA) {
  return isaGetInfoT<uint64_t>(CoreApi, ISA, HSA_ISA_INFO_GRID_MAX_SIZE);
}

llvm::Expected<uint32_t>
isaGetFbarrierMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_isa_t ISA) {
  return isaGetInfoT<uint32_t>(CoreApi, ISA, HSA_ISA_INFO_FBARRIER_MAX_SIZE);
}

llvm::Expected<uint16_t>
isaGetExceptionPolicies(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        hsa_isa_t ISA, hsa_profile_t Profile) {
  uint16_t Mask = 0;
  hsa_status_t S =
      CoreApi.callFunction<hsa_isa_get_exception_policies>(ISA, Profile, &Mask);
  if (S != HSA_STATUS_SUCCESS)
    return llvm::make_error<HsaError>(llvm::formatv(
        "Failed to get exception policies for ISA {0:x}", ISA.handle));
  return Mask;
}

llvm::Expected<hsa_round_method_t>
isaGetRoundMethod(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_isa_t ISA, hsa_fp_type_t FpType,
                  hsa_flush_mode_t FlushMode) {
  hsa_round_method_t Method{};
  hsa_status_t S = CoreApi.callFunction<hsa_isa_get_round_method>(
      ISA, FpType, FlushMode, &Method);
  if (S != HSA_STATUS_SUCCESS)
    return llvm::make_error<HsaError>(
        llvm::formatv("Failed to get round method for ISA {0:x}", ISA.handle));
  return Method;
}

llvm::Error isaIterateWavefronts(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_isa_t ISA,
    const std::function<llvm::Error(hsa_wavefront_t)> &Callback) {
  struct CBData {
    decltype(Callback) CB;
    llvm::Error Err;
  } Data{Callback, llvm::Error::success()};

  auto Iterator = [](hsa_wavefront_t WF, void *D) -> hsa_status_t {
    auto *Data = static_cast<CBData *>(D);
    Data->Err = Data->CB(WF);
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t S = CoreApi.callFunction<hsa_isa_iterate_wavefronts>(
          ISA, Iterator, &Data);
      S == HSA_STATUS_SUCCESS || S == HSA_STATUS_INFO_BREAK)
    return std::move(Data.Err);

  return llvm::make_error<HsaError>(llvm::formatv(
      "Failed to iterate over wavefronts of ISA {0:x}.", ISA.handle));
}

llvm::Error
isaGetWavefronts(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_isa_t ISA,
                 llvm::SmallVectorImpl<hsa_wavefront_t> &Wavefronts) {
  return isaIterateWavefronts(CoreApi, ISA,
                              [&](hsa_wavefront_t WF) -> llvm::Error {
                                Wavefronts.push_back(WF);
                                return llvm::Error::success();
                              });
}

llvm::Expected<uint32_t>
wavefrontGetSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_wavefront_t Wavefront) {
  uint32_t Size = 0;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_wavefront_get_info>(
          Wavefront, HSA_WAVEFRONT_INFO_SIZE, &Size),
      llvm::formatv("Failed to get wavefront size for wavefront {0:x}",
                    Wavefront.handle)));
  return Size;
}

} // namespace luthier::hsa