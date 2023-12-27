#ifndef HSA_EXECUTABLE_HPP
#define HSA_EXECUTABLE_HPP
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <optional>
#include <vector>

#include "hsa_handle_type.hpp"

namespace luthier::hsa {

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

class Executable : public HandleType<hsa_executable_t> {
 public:
    explicit Executable(hsa_executable_t executable);

    static Executable create(
        hsa_profile_t profile = HSA_PROFILE_FULL,
        hsa_default_float_rounding_mode_t defaultFloatRoundingMode = HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
        const char *options = "");

    hsa_status_t freeze(const char *options = "");

    hsa_profile_t getProfile();

    hsa_executable_state_t getState();

    hsa_default_float_rounding_mode_t getRoundingMode();

    [[nodiscard]] std::vector<ExecutableSymbol> getSymbols(const luthier::hsa::GpuAgent &agent) const;

    [[nodiscard]] std::optional<ExecutableSymbol> getSymbolByName(const luthier::hsa::GpuAgent &agent,
                                                                  const std::string &name) const;

    [[nodiscard]] std::vector<LoadedCodeObject> getLoadedCodeObjects() const;

    [[nodiscard]] std::vector<hsa::GpuAgent> getAgents() const;
};

}// namespace luthier::hsa

namespace std {

template<>
struct hash<luthier::hsa::Executable> {
    size_t operator()(const luthier::hsa::Executable &obj) const { return hash<unsigned long>()(obj.hsaHandle()); }
};

template<>
struct less<luthier::hsa::Executable> {
    bool operator()(const luthier::hsa::Executable &lhs, const luthier::hsa::Executable &rhs) const {
        return lhs.hsaHandle() < rhs.hsaHandle();
    }
};

template<>
struct equal_to<luthier::hsa::Executable> {
    bool operator()(const luthier::hsa::Executable &lhs, const luthier::hsa::Executable &rhs) const {
        return lhs.hsaHandle() == rhs.hsaHandle();
    }
};

}// namespace std
#endif