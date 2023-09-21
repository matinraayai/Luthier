#include "code_object_manager.hpp"
#include "code_object_manipulation.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "hsa_intercept.hpp"
#include "log.hpp"
#include <assert.h>
#include <elfio/elfio.hpp>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

namespace {

//luthier::co_manip::code_view_t stripOffKernelLaunch(luthier::co_manip::code_view_t codeObject,
//                                                    const std::string &demangledName) {
//    ELFIO::elfio reader;
//    std::stringstream loadedCodeObjectSS{std::string(reinterpret_cast<const char *>(codeObject.data), codeObject.size)};
//    reader.load(loadedCodeObjectSS, true);
//    auto numSymbols = luthier::co_manip::getSymbolNum(reader);
//    std::cout << "Demangled name: " << demangledName << std::endl;
//    for (unsigned int i = 0; i < numSymbols; i++) {
//        auto info = luthier::co_manip::SymbolInfo(&reader, i);
//        std::string demangledSymName = luthier::co_manip::getDemangledName(info.get_name().c_str());
//        std::cout << "Symbol name: " << info.get_name().c_str() << std::endl;
//        std::cout << "Symbol size: " << info.get_size() << std::endl;
//        std::cout << "Symbol Addr: " << reinterpret_cast<const void *>(info.get_address()) << std::endl;
//    }
//    for (unsigned int i = 0; i < numSymbols; i++) {
//        auto info = luthier::co_manip::SymbolInfo(&reader, i);
//        std::string demangledSymName = luthier::co_manip::getDemangledName(info.get_name().c_str());
//        std::cout << "Symbol name: " << info.get_name().c_str() << std::endl;
//        std::cout << "Symbol size: " << info.get_size() << std::endl;
//        std::cout << "Symbol Addr: " << reinterpret_cast<const void *>(info.get_address()) << std::endl;
//        if (demangledSymName.find("__luthier_wrap__") == std::string::npos and demangledSymName.find(demangledName) != std::string::npos) {
//            std::cout << "Symbol name: " << luthier::co_manip::getDemangledName(info.get_name().c_str()) << std::endl;
//            std::cout << "Symbol size: " << info.get_size() << std::endl;
//            std::cout << "Symbol Addr: " << reinterpret_cast<const void *>(info.get_address()) << std::endl;
//            return {codeObject.data + (luthier_address_t) info.get_address(), info.get_size()};
//        }
//    }
//
//    return {};
//}

}// namespace

void luthier::CodeObjectManager::registerLuthierHsaExecutables() {
    const auto &loaderApi = HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    auto findLuthierExecutable = [](hsa_executable_t exec, void *data) {
        const auto &coreApi = HsaInterceptor::Instance().getSavedHsaTables().core;
        std::vector<hsa_agent_t> agents = ContextManager::Instance().getHsaAgents();
        auto symbolIterator = [](hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data) {
            const auto &coreTable = HsaInterceptor::Instance().getSavedHsaTables().core;
            hsa_symbol_kind_t symbolKind;
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind));

            uint32_t nameSize;
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));

            std::string name;
            name.resize(nameSize);
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()));

            if (name.find("__luthier_reserved") != std::string::npos && symbolKind == HSA_SYMBOL_KIND_VARIABLE) {
                auto executableSet = reinterpret_cast<decltype(executables_) *>(data);
                LuthierLogDebug("Found an executable with handle {} for agent {}", exec.handle, agent.handle);
                executableSet->insert(exec.handle);
                return HSA_STATUS_INFO_BREAK;
            }
            return HSA_STATUS_SUCCESS;
        };
        for (const auto &a: agents) {
            coreApi.hsa_executable_iterate_agent_symbols_fn(exec, a, symbolIterator, data);
        }
        return HSA_STATUS_SUCCESS;
    };
    loaderApi.hsa_ven_amd_loader_iterate_executables(findLuthierExecutable, &executables_);

    LuthierLogDebug("Number of executables captured: {}", executables_.size());
}

void luthier::CodeObjectManager::registerHipWrapperKernelsOfInstrumentationFunctions(const std::vector<std::tuple<const void *, const char *>> &instrumentationFunctionInfo) {
    registerLuthierHsaExecutables();
    // this holds the name of the actual device function, without the __luthier_wrap
    std::vector<std::string> instDeviceFuncNames;
    instDeviceFuncNames.reserve(instrumentationFunctionInfo.size());
    for (const auto &iFuncInfo: instrumentationFunctionInfo) {
        std::string deviceFuncName = std::get<const char *>(iFuncInfo);
        deviceFuncName = deviceFuncName.substr(deviceFuncName.find("__luthier_wrap__") + strlen("__luthier_wrap__"),
                                               deviceFuncName.find('('));
        if (deviceFuncName.empty())
            throw std::runtime_error("The requested instrumentation kernel doesn't have __luthier_wrap__ at its beginning.");
        instDeviceFuncNames.push_back(deviceFuncName);
    }

    auto agents = ContextManager::Instance().getHsaAgents();

    for (const auto &a: agents) {
        for (const auto &e: executables_) {
            auto hostCodeObjects = co_manip::getHostLoadedCodeObjectOfExecutable({e}, a);
            auto deviceCodeObjects = co_manip::getDeviceLoadedCodeObjectOfExecutable({e}, a);
            for (unsigned int i = 0; i < hostCodeObjects.size(); i++) {
                auto hco = hostCodeObjects[i];
                auto dco = deviceCodeObjects[i];
                auto reader = co_manip::ElfView::make_view(hco);
                auto& io = reader->get_elfio();
//                ELFIO::elfio reader;
//                reader.load(hcoSs, true);
                for (unsigned int j = 0; j < co_manip::getSymbolNum(reader); j++) {
                    auto info = co_manip::SymbolView(reader, j);
                    fmt::println("Name of symbol: {}", info.get_name());
                    fmt::println("Size of symbol: {}", info.get_view().size());
                    fmt::println("Type of symbol: {}", (int) info.get_type());
                    fmt::println("Symbol location: {:#x}", reinterpret_cast<luthier_address_t>(info.get_view().data()));
                    fmt::println("Section addr: {:#x}", reinterpret_cast<luthier_address_t>(info.get_section()->get_address()));
                    fmt::println("Section Name: {}", info.get_section()->get_name());
                    for (unsigned int k = 0; k < instrumentationFunctionInfo.size(); k++) {
                        auto deviceFuncName = instDeviceFuncNames[k];
                        if (info.get_name().find(deviceFuncName) != std::string::npos) {
                            luthier_address_t deviceAddress = reinterpret_cast<luthier_address_t>(dco.data()) + reinterpret_cast<luthier_address_t>(info.get_view().data());
                            auto globalFuncPointer = std::get<const void *>(instrumentationFunctionInfo[k]);

                            if (info.get_name().find("__luthier_wrap__") != std::string::npos and info.get_name().find(".kd") != std::string::npos) {
                                //                                assert(info.size == sizeof(kernel_descriptor_t));
                                auto *kd = reinterpret_cast<kernel_descriptor_t *>(deviceAddress);
                                if (!functions_.contains(globalFuncPointer)) {
                                    auto globalFunctionName = std::get<const char *>(instrumentationFunctionInfo[k]);
                                    functions_.insert({globalFuncPointer, {{}, globalFunctionName, deviceFuncName}});
                                }
                                auto &agentToExecMap = functions_[globalFuncPointer].agentToExecMap;
                                if (!agentToExecMap.contains(a.handle)) {
                                    agentToExecMap.insert({a.handle, {}});
                                }
                                agentToExecMap[a.handle].kd = kd;
                            } else {
                                co_manip::code_view_t function{reinterpret_cast<const std::byte*>(info.get_view().data() + reinterpret_cast<luthier_address_t>(dco.data())),
                                                               info.get_view().size()};
                                if (!functions_.contains(globalFuncPointer)) {
                                    auto globalFunctionName = std::get<const char *>(instrumentationFunctionInfo[k]);
                                    functions_.insert({globalFuncPointer, {{}, globalFunctionName, deviceFuncName}});
                                }
                                auto &agentToExecMap = functions_[globalFuncPointer].agentToExecMap;
                                if (!agentToExecMap.contains(a.handle)) {
                                    agentToExecMap.insert({a.handle, {}});
                                }
                                agentToExecMap[a.handle].function = function;
                            }
                        }
                    }
                };
            }
        }
    }
    LuthierLogDebug("Number of functions registered: {}", functions_.size());
    //
    //            if (HSA_STATUS_SUCCESS == coreApi.hsa_executable_get_symbol_by_name_fn({e}, globalFuncName, &agent, &s)) {
    //                fmt::println("Executable {} with agent {} was found for function {}", e, a, globalFuncName);
    //                out.insert({a, {e}});
    //
    //            }
    //            else {
    //                fmt::println("Symbol was not found for function {}!", globalFuncName);
    //            }
    //        }
    //    }
    //    functions_.insert({instrumentationFunctionInfo, {out, std::string(globalFuncName), deviceFuncName, fatBinary}});
}

luthier::co_manip::code_view_t luthier::CodeObjectManager::getInstrumentationFunction(const void *wrapperKernelHostPtr,
                                                                                                           hsa_agent_t agent) const {
    auto f = functions_.at(wrapperKernelHostPtr).agentToExecMap.at(agent.handle).function;
#ifdef LUTHIER_LOG_ENABLE_DEBUG
    auto instrs = luthier::Disassembler::instance().disassemble(agent, f.data());
    for (const auto &i: instrs) {
        LuthierLogDebug("{:#x}: {}", i.getHostAddress(), i.getInstr());
    }
#endif
    return f;
}

const kernel_descriptor_t *luthier::CodeObjectManager::getInstrumentedKernelKD(const kernel_descriptor_t *originalKernelKD) {
    if (!instrumentedKernels_.contains(originalKernelKD))
        throw std::runtime_error(fmt::format("The Kernel Descriptor {:#x} has not been instrumented.", reinterpret_cast<luthier_address_t>(originalKernelKD)));
    return instrumentedKernels_[originalKernelKD];
}

void luthier::CodeObjectManager::registerInstrumentedKernel(const kernel_descriptor_t *originalCodeKD, const kernel_descriptor_t *instrumentedCodeKD) {
    if (!instrumentedKernels_.contains(originalCodeKD))
        instrumentedKernels_.insert({originalCodeKD, instrumentedCodeKD});
}
kernel_descriptor_t *luthier::CodeObjectManager::getKernelDescriptorOfInstrumentationFunction(const void *wrapperKernelHostPtr, hsa_agent_t agent) const {
    return functions_.at(wrapperKernelHostPtr).agentToExecMap.at(agent.handle).kd;
}
