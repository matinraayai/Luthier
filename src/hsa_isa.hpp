#ifndef HSA_ISA_HPP
#define HSA_ISA_HPP
#include "hsa_handle_type.hpp"

namespace luthier::hsa {

class Isa : public HandleType<hsa_isa_t> {
 public:
    explicit Isa(hsa_isa_t isa) : HandleType<hsa_isa_t>(isa){};

    static Isa fromName(const char *isaName);

    [[nodiscard]] std::string getName() const;

    [[nodiscard]] std::string getArchitecture() const;

    [[nodiscard]] std::string getVendor() const;

    [[nodiscard]] std::string getOS() const;

    [[nodiscard]] std::string getEnvironment() const;

    [[nodiscard]] std::string getProcessor() const;

    [[nodiscard]] bool isXnacSupported() const;

    [[nodiscard]] bool isSRamECCSupported() const;

    [[nodiscard]] std::string getLLVMTarget() const;

    [[nodiscard]] std::string getLLVMTargetTriple() const;

    [[nodiscard]] std::string getFeatureString() const;
};

}// namespace luthier::hsa

namespace std {

template<>
struct hash<luthier::hsa::Isa> {
    size_t operator()(const luthier::hsa::Isa &obj) const {
        return hash<unsigned long>()(obj.hsaHandle());
    }
};

template<>
struct less<luthier::hsa::Isa> {
    bool operator()(const luthier::hsa::Isa &lhs, const luthier::hsa::Isa &rhs) const {
        return lhs.hsaHandle() < rhs.hsaHandle();
    }
};

template<>
struct less_equal<luthier::hsa::Isa> {
    bool operator()(const luthier::hsa::Isa &lhs, const luthier::hsa::Isa &rhs) const {
        return lhs.hsaHandle() <= rhs.hsaHandle();
    }
};

template<>
struct equal_to<luthier::hsa::Isa> {
    bool operator()(const luthier::hsa::Isa &lhs, const luthier::hsa::Isa &rhs) const {
        return lhs.hsaHandle() == rhs.hsaHandle();
    }
};

template<>
struct not_equal_to<luthier::hsa::Isa> {
    bool operator()(const luthier::hsa::Isa &lhs, const luthier::hsa::Isa &rhs) const {
        return lhs.hsaHandle() != rhs.hsaHandle();
    }
};

template<>
struct greater<luthier::hsa::Isa> {
    bool operator()(const luthier::hsa::Isa &lhs, const luthier::hsa::Isa &rhs) const {
        return lhs.hsaHandle() > rhs.hsaHandle();
    }
};

template<>
struct greater_equal<luthier::hsa::Isa> {
    bool operator()(const luthier::hsa::Isa &lhs, const luthier::hsa::Isa &rhs) const {
        return lhs.hsaHandle() >= rhs.hsaHandle();
    }
};

}// namespace std

#endif