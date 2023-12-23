#ifndef HSA_EXECUTABLE_SYMBOL_HPP
#define HSA_EXECUTABLE_SYMBOL_HPP
#include "hsa_primitive.hpp"
#include "luthier_types.h"
#include <hsa/hsa.h>
#include <string>

namespace luthier::hsa {

class GpuAgent;

class Executable;

class ExecutableSymbol : public HandleType<hsa_executable_symbol_t> {
 private:
    hsa_agent_t agent_;
    hsa_executable_t executable_;

 public:
    explicit ExecutableSymbol(hsa_executable_symbol_t symbol, hsa_agent_t agent, hsa_executable_t executable) : HandleType<hsa_executable_symbol_t>(symbol),
                                                                                                                agent_(agent),
                                                                                                                executable_(executable){};

    static ExecutableSymbol fromKernelDescriptor(const kernel_descriptor_t *kd);

    [[nodiscard]] hsa_symbol_kind_t getType() const;

    [[nodiscard]] std::string getName() const;

    [[nodiscard]] hsa_symbol_linkage_t getLinkage() const;

    [[nodiscard]] luthier_address_t getVariableAddress() const;

    [[nodiscard]] const kernel_descriptor_t *getKernelDescriptor() const;

    [[nodiscard]] GpuAgent getAgent() const;

    [[nodiscard]] Executable getExecutable() const;
};

}// namespace luthier::hsa

#endif