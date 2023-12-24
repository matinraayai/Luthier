#ifndef HSA_EXECUTABLE_SYMBOL_HPP
#define HSA_EXECUTABLE_SYMBOL_HPP
#include "code_object_manipulation.hpp"
#include "hsa_primitive.hpp"
#include "luthier_types.h"
#include <hsa/hsa.h>
#include <optional>
#include <string>

namespace luthier::hsa {

class GpuAgent;

class Executable;

class ExecutableSymbol : public HandleType<hsa_executable_symbol_t> {
 private:
    hsa_agent_t agent_;
    hsa_executable_t executable_;
    std::optional<std::string> indirectFunctionName_{std::nullopt};
    std::optional<luthier::co_manip::code_view_t> indirectFunctionCode_{std::nullopt};

//    ExecutableSymbol(const ExecutableSymbol& other) : HandleType<hsa_executable_symbol_t>(other.asHsaType()) {
//        this->agent_ = other.agent_;
//        this->executable_ = other.executable_;
//        if (other.getType() == HSA_SYMBOL_KIND_INDIRECT_FUNCTION) {
//            *indirectFunctionName_ = std::copy(*other.indirectFunctionName_);
//            *indirectFunctionCode_ = std::move(*other.indirectFunctionCode_);
//        }
//    }

 public:
    ExecutableSymbol(hsa_executable_symbol_t symbol, hsa_agent_t agent, hsa_executable_t executable) : HandleType<hsa_executable_symbol_t>(symbol),
                                                                                                       agent_(agent),
                                                                                                       executable_(executable){};

    ExecutableSymbol(std::string indirectFunctionName, luthier::co_manip::code_view_t indirectFunctionCode, hsa_agent_t agent, hsa_executable_t executable) : HandleType<hsa_executable_symbol_t>({0}),
                                                                                                                                                              agent_(agent),
                                                                                                                                                              executable_(executable),
                                                                                                                                                              indirectFunctionCode_(indirectFunctionCode),
                                                                                                                                                              indirectFunctionName_(std::move(indirectFunctionName)){};

    static ExecutableSymbol fromKernelDescriptor(const kernel_descriptor_t *kd);

    [[nodiscard]] hsa_symbol_kind_t getType() const;

    [[nodiscard]] std::string getName() const;

    [[nodiscard]] hsa_symbol_linkage_t getLinkage() const;

    [[nodiscard]] luthier_address_t getVariableAddress() const;

    [[nodiscard]] const kernel_descriptor_t *getKernelDescriptor() const;

    [[nodiscard]] GpuAgent getAgent() const;

    [[nodiscard]] Executable getExecutable() const;

    [[nodiscard]] const luthier::co_manip::code_view_t getIndirectFunctionCode() const;
};

}// namespace luthier::hsa

namespace std {

template<>
struct hash<luthier::hsa::ExecutableSymbol> {
    size_t operator()(const luthier::hsa::ExecutableSymbol &obj) const {
        if (obj.getType() != HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
            return hash<unsigned long>()(obj.hsaHandle());
        else
            return hash<unsigned long>()(reinterpret_cast<luthier_address_t>(obj.getIndirectFunctionCode().data()));
    }
};

template<>
struct less<luthier::hsa::ExecutableSymbol> {
    bool operator()(const luthier::hsa::ExecutableSymbol &lhs, const luthier::hsa::ExecutableSymbol &rhs) const {
        auto lhsHandle = lhs.getType() == HSA_SYMBOL_KIND_INDIRECT_FUNCTION ? reinterpret_cast<luthier_address_t>(lhs.getIndirectFunctionCode().data()) : lhs.hsaHandle();
        auto rhsHandle = rhs.getType() == HSA_SYMBOL_KIND_INDIRECT_FUNCTION ? reinterpret_cast<luthier_address_t>(rhs.getIndirectFunctionCode().data()) : rhs.hsaHandle();
        return lhsHandle < rhsHandle;
    }
};

template<>
struct equal_to<luthier::hsa::ExecutableSymbol> {
    bool operator()(const luthier::hsa::ExecutableSymbol &lhs, const luthier::hsa::ExecutableSymbol &rhs) const {
        auto lhsHandle = lhs.getType() == HSA_SYMBOL_KIND_INDIRECT_FUNCTION ? reinterpret_cast<luthier_address_t>(lhs.getIndirectFunctionCode().data()) : lhs.hsaHandle();
        auto rhsHandle = rhs.getType() == HSA_SYMBOL_KIND_INDIRECT_FUNCTION ? reinterpret_cast<luthier_address_t>(rhs.getIndirectFunctionCode().data()) : rhs.hsaHandle();
        return lhsHandle == rhsHandle;
    }
};

}// namespace std

#endif