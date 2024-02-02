#include "hsa_isa.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

namespace luthier::hsa {

Isa Isa::fromName(const char *isaName) {
    hsa_isa_t isa;
    const auto &coreApi = luthier::HsaInterceptor::instance().getSavedHsaTables().core;
    LUTHIER_HSA_CHECK(
        coreApi.hsa_isa_from_name_fn(
            isaName,
            &isa));
    return Isa(isa);
}
std::string Isa::getName() const {
    uint32_t isaNameSize;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(this->asHsaType(),
                                                                 HSA_ISA_INFO_NAME_LENGTH,
                                                                 &isaNameSize));
    std::string isaName(isaNameSize - 1, '\0');
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(this->asHsaType(),
                                                                 HSA_ISA_INFO_NAME,
                                                                 &isaName.front()));
    return isaName;
}

inline llvm::SmallVector<llvm::StringRef, 5> parseIsaName(llvm::StringRef isaName) {
    llvm::SmallVector<llvm::StringRef, 5> isaNameComponents;
    isaName.split(isaNameComponents, '-', 4);
    LUTHIER_CHECK((isaNameComponents.size() == 5));
    return isaNameComponents;
}

std::string Isa::getArchitecture() const {
    std::string isaName = getName();
    llvm::SmallVector<llvm::StringRef, 5> isaNameComponents = parseIsaName(isaName);
    return isaNameComponents[0].str();
}
std::string Isa::getVendor() const {
    std::string isaName = getName();
    llvm::SmallVector<llvm::StringRef, 5> isaNameComponents = parseIsaName(isaName);
    return isaNameComponents[1].str();
}
std::string Isa::getOS() const {
    std::string isaName = getName();
    llvm::SmallVector<llvm::StringRef, 5> isaNameComponents = parseIsaName(isaName);
    return isaNameComponents[2].str();
}
std::string Isa::getEnvironment() const {
    std::string isaName = getName();
    llvm::SmallVector<llvm::StringRef, 5> isaNameComponents = parseIsaName(isaName);
    return isaNameComponents[3].str();
}

std::string Isa::getProcessor() const {
    std::string isaName = getName();
    llvm::SmallVector<llvm::StringRef, 5> isaNameComponents = parseIsaName(isaName);

    llvm::SmallVector<llvm::StringRef> features;
    features.clear();
    isaNameComponents[4].split(features, ':');

    return features[0].str();
}
bool Isa::isXnacSupported() const {
    std::string isaName = getName();
    auto sRamECC = isaName.find("xnack");
    if (sRamECC == std::string::npos)
        return false;
    else {
        return isaName[sRamECC + strlen("xnack")] == '+';
    }
}
bool Isa::isSRamECCSupported() const {
    std::string isaName = getName();
    auto sRamECC = isaName.find("sramecc");
    if (sRamECC == std::string::npos)
        return false;
    else {
        return isaName[sRamECC + strlen("sramecc")] == '+';
    }
}
std::string Isa::getLLVMTarget() const {
    std::string isaName = getName();
    return isaName.substr(0, isaName.find_first_of(':'));
}

std::string Isa::getLLVMTargetTriple() const {
    std::string isaName = getName();
    llvm::SmallVector<llvm::StringRef, 5> isaNameComponents = parseIsaName(isaName);
    return (llvm::Twine(isaNameComponents[0]) + "-" + llvm::Twine(isaNameComponents[1]) + "-" +
            llvm::Twine(isaNameComponents[2]) + "-" + llvm::Twine(isaNameComponents[3])).str();
}

std::string Isa::getFeatureString() const {
    std::string isaName = getName();
    llvm::SmallVector<llvm::StringRef> features;
    llvm::StringRef(isaName).substr(isaName.find_first_of(':')).split(features, ":");
    // The +/- must be before the feature code for LLVM, not after
    std::vector<std::string> featuresOut;
    for (auto& feat: features) {
        auto featureToggle = feat.substr(feat.size() - 1);
        auto featureName = feat.substr(0, feat.size() - 1);
        featuresOut.push_back((featureToggle + featureName).str());
    }
    return llvm::join(featuresOut, ",");
}

}// namespace luthier::hsa
