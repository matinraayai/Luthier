#include "code_object_manager.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>

#include <vector>

#include "disassembler.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_loaded_code_object.hpp"
#include "instrumentation_function.hpp"
#include "log.hpp"
#include "target_manager.hpp"

void luthier::CodeObjectManager::registerLuthierHsaExecutables() {
    llvm::SmallVector<hsa::Executable> executables;
    hsa::getAllExecutables(executables);

    llvm::SmallVector<hsa::GpuAgent, 8> agents;
    hsa::getGpuAgents(agents);

    for (const auto &e: executables) {
        for (const auto &a: agents) {
            if (e.getSymbolByName(a, LUTHIER_RESERVED_MANAGED_VAR).has_value()
                || e.getSymbolByName(a, std::string(LUTHIER_RESERVED_MANAGED_VAR) + ".managed")) {
                toolExecutables_.insert(e);
            }
        }
    }
    LuthierLogDebug("Number of executables captured: {}", toolExecutables_.size());
}

void luthier::CodeObjectManager::registerHipWrapperKernelsOfInstrumentationFunctions(
    const std::vector<std::tuple<const void *, const char *>> &instrumentationFunctionInfo) {
    registerLuthierHsaExecutables();

    llvm::SmallVector<hsa::GpuAgent, 8> agents;
    hsa::getGpuAgents(agents);

    for (const auto &e: toolExecutables_) {
        for (const auto &a: agents) {
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
    return std::get<hsa::ExecutableSymbol>(instrumentedKernels_.at(originalKernel));
}

void luthier::CodeObjectManager::loadInstrumentedKernel(const luthier::byte_string_t &instrumentedElf,
                                                        const hsa::ExecutableSymbol &originalKernel) {
    if (!instrumentedKernels_.contains(originalKernel)) {
        auto executable = hsa::Executable::create();
        auto agent = originalKernel.getAgent();
        auto reader = hsa::CodeObjectReader::createFromMemory(instrumentedElf);
        executable.loadCodeObject(reader, agent);
        executable.freeze();

        auto originalSymbolName = originalKernel.getName();
        auto instrumentedKernel = executable.getSymbolByName(agent, originalSymbolName);
        LUTHIER_CHECK_WITH_MSG(
            instrumentedKernel.has_value(),
            llvm::formatv("Failed to find symbol {0} in Executable {1}", originalSymbolName, executable.hsaHandle())
                .str());
        LUTHIER_CHECK_WITH_MSG((instrumentedKernel->getType() == HSA_SYMBOL_KIND_KERNEL),
                      llvm::formatv("Symbol {0} was found, but its type is not a kernel", originalSymbolName).str());
        instrumentedKernels_.insert({originalKernel, std::make_tuple(*instrumentedKernel, executable, reader)});
    }
}

const luthier::hsa::ExecutableSymbol &luthier::CodeObjectManager::getInstrumentationKernel(
    const void *wrapperKernelHostPtr, hsa::GpuAgent agent) const {
    return functions_.at(wrapperKernelHostPtr).at(agent).getInstrumentationKernel();
}
luthier::CodeObjectManager::~CodeObjectManager() {
    for (auto &[origSymbol, instInfo]: instrumentedKernels_) {
        auto &[s, e, r] = instInfo;
        e.destroy();
        r.destroy();
    }
    instrumentedKernels_.clear();
    toolExecutables_.clear();
    functions_.clear();
}
