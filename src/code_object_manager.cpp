#include "code_object_manager.hpp"

#include <vector>

#include "context_manager.hpp"
#include "disassembler.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "instrumentation_function.hpp"
#include "log.hpp"

void luthier::CodeObjectManager::registerLuthierHsaExecutables() {

    for (const auto &e: luthier::ContextManager::instance().getHsaExecutables()) {
        for (const auto &a: luthier::ContextManager::instance().getHsaAgents()) {
            if (e.getSymbolByName(a, LUTHIER_RESERVED_MANAGED_VAR).has_value()
                || e.getSymbolByName(a, std::string(LUTHIER_RESERVED_MANAGED_VAR) + ".managed")) {
                toolExecutables_.insert(e);
            }
//            for (const auto &s: e.getSymbols(a)) {
//                if (s.getName().find(LUTHIER_RESERVED_MANAGED_VAR) != std::string::npos
//                    && s.getType() == HSA_SYMBOL_KIND_VARIABLE) {
//
//                }
//            }
        }
    }
    LuthierLogDebug("Number of executables captured: {}", toolExecutables_.size());
}

void luthier::CodeObjectManager::registerHipWrapperKernelsOfInstrumentationFunctions(
    const std::vector<std::tuple<const void *, const char *>> &instrumentationFunctionInfo) {
    registerLuthierHsaExecutables();
    for (const auto &e: toolExecutables_) {
        for (const auto &a: luthier::ContextManager::instance().getHsaAgents()) {
            std::unordered_map<std::string, hsa::ExecutableSymbol> instKernelSymbols;
            std::unordered_map<std::string, hsa::ExecutableSymbol> instFunctionSymbols;
            auto symbols = e.getSymbols(a);
            for (const auto &s: symbols) {
                auto sType = s.getType();
                auto sName = s.getName();
                if (sType == HSA_SYMBOL_KIND_KERNEL) instKernelSymbols.insert({sName.substr(0, sName.find(".kd")), s});
                else if (sType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
                    instFunctionSymbols.insert({LUTHIER_DEVICE_FUNCTION_WRAP + sName, s});
            }
            for (const auto &[instKerShadowPtr, instKerName]: instrumentationFunctionInfo) {
                if (instKernelSymbols.contains(instKerName)) {
                    auto kernelSymbol = instKernelSymbols.at(instKerName);
                    auto functionSymbol = instFunctionSymbols.at(instKerName);
                    if (functions_.contains(instKerShadowPtr)) {
                        functions_.at(instKerShadowPtr).insert({a, {functionSymbol, kernelSymbol}});
                    } else {
                        fmt::println("Function: {:#x}", reinterpret_cast<luthier_address_t>(instKerShadowPtr));
                        functions_.insert({instKerShadowPtr, {{a, {functionSymbol, kernelSymbol}}}});
                    }
                }
            }
        }
    }
    LuthierLogDebug("Number of functions registered: {}", functions_.size());
}

const luthier::hsa::ExecutableSymbol &luthier::CodeObjectManager::getInstrumentationFunction(
    const void *wrapperKernelHostPtr, hsa::GpuAgent agent) const {
    const auto &f = functions_.at(wrapperKernelHostPtr).at(agent).getInstrumentationFunction();
#ifdef LUTHIER_LOG_ENABLE_DEBUG
    auto instrs = luthier::Disassembler::instance().disassemble(f);
//    for (const auto &i: instrs) { LuthierLogDebug("{:#x}: {}", i.getHostAddress(), i.getInstr()); }
#endif
    return f;
}

const luthier::hsa::ExecutableSymbol &luthier::CodeObjectManager::getInstrumentedKernel(
    const hsa::ExecutableSymbol &originalKernel) const {
    if (!instrumentedKernels_.contains(originalKernel))
        throw std::runtime_error(
            fmt::format("The Kernel Descriptor {:#x} has not been instrumented.",
                        reinterpret_cast<luthier_address_t>(originalKernel.getKernelDescriptor())));
    return instrumentedKernels_.at(originalKernel);
}

void luthier::CodeObjectManager::registerInstrumentedKernel(const hsa::ExecutableSymbol &originalCodeKD,
                                                            const hsa::ExecutableSymbol &instrumentedCodeKD) {
    if (!instrumentedKernels_.contains(originalCodeKD))
        instrumentedKernels_.insert({originalCodeKD, instrumentedCodeKD});
}
const luthier::hsa::ExecutableSymbol &luthier::CodeObjectManager::getInstrumentationKernel(
    const void *wrapperKernelHostPtr, hsa::GpuAgent agent) const {
    return functions_.at(wrapperKernelHostPtr).at(agent).getInstrumentationKernel();
}
