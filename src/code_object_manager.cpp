#include "code_object_manager.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "hsa_intercept.hpp"
#include "log.hpp"
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

    for (const auto& e: luthier::ContextManager::Instance().getHsaExecutables()) {
        for (const auto& a: luthier::ContextManager::Instance().getHsaAgents()) {
            for (const auto& s: e.getSymbols(a)) {
                if (s.getName().find(LUTHIER_RESERVED_MANAGED_VAR) != std::string::npos &&
                    s.getType() == HSA_SYMBOL_KIND_VARIABLE) {
                    toolExecutables_.insert(e);
                }
            }
        }
    }
    LuthierLogDebug("Number of executables captured: {}", toolExecutables_.size());
}

void luthier::CodeObjectManager::registerHipWrapperKernelsOfInstrumentationFunctions(const std::vector<std::tuple<const void *, const char *>> &instrumentationFunctionInfo) {
    registerLuthierHsaExecutables();
    // this holds the name of the actual device function, without the __luthier_wrap
    std::vector<std::string> instDeviceFuncNames;
    instDeviceFuncNames.reserve(instrumentationFunctionInfo.size());
    for (const auto &iFuncInfo: instrumentationFunctionInfo) {
        std::string deviceFuncName = std::get<const char *>(iFuncInfo);
        deviceFuncName = deviceFuncName.substr(deviceFuncName.find(LUTHIER_DEVICE_FUNCTION_WRAP) + strlen(LUTHIER_DEVICE_FUNCTION_WRAP),
                                               deviceFuncName.find('('));
        if (deviceFuncName.empty())
            throw std::runtime_error("The requested instrumentation kernel doesn't have __luthier_wrap__ at its beginning.");
        instDeviceFuncNames.push_back(deviceFuncName);
    }

    for (const auto &e: toolExecutables_) {
        for (const auto &lco: e.getLoadedCodeObjects()) {
            if (lco.getStorageType() == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
                auto storageMemory = lco.getStorageMemory();
                auto loadedMemory = lco.getLoadedMemory();
                auto lcoAgent = lco.getAgent();
                auto reader = co_manip::ElfViewImpl::makeView(storageMemory);
                auto &io = reader->getElfIo();
                for (unsigned int j = 0; j < co_manip::getSymbolNum(reader); j++) {
                    auto info = co_manip::SymbolView(reader, j);
                    //                    fmt::println("Name of symbol: {}", info.getName());
                    //                    fmt::println("Size of symbol: {}", info.getView().size());
                    //                    fmt::println("Type of symbol: {}", (int) info.getType());
                    //                    fmt::println("Symbol location: {:#x}", reinterpret_cast<luthier_address_t>(info.getView().data()));
                    //                    fmt::println("Section addr: {:#x}", reinterpret_cast<luthier_address_t>(info.getSection()->get_address()));
                    //                    fmt::println("Section Name: {}", info.getSection()->get_name());
                    for (unsigned int k = 0; k < instrumentationFunctionInfo.size(); k++) {
                        auto deviceFuncName = instDeviceFuncNames[k];
                        if (info.getName().find(deviceFuncName) != std::string::npos) {
                            luthier_address_t deviceAddress = reinterpret_cast<luthier_address_t>(loadedMemory.data()) + reinterpret_cast<luthier_address_t>(info.getView().data());
                            auto globalFuncPointer = std::get<const void *>(instrumentationFunctionInfo[k]);

                            if (info.getName().find(LUTHIER_DEVICE_FUNCTION_WRAP) != std::string::npos and info.getName().find(".kd") != std::string::npos) {
                                //                                assert(info.size == sizeof(kernel_descriptor_t));
                                auto *kd = reinterpret_cast<kernel_descriptor_t *>(deviceAddress);
                                if (!functions_.contains(globalFuncPointer)) {
                                    auto globalFunctionName = std::get<const char *>(instrumentationFunctionInfo[k]);
                                    functions_.insert({globalFuncPointer, {{}, globalFunctionName, deviceFuncName}});
                                }
                                auto &agentToExecMap = functions_[globalFuncPointer].agentToExecMap;
                                if (!agentToExecMap.contains(lcoAgent)) {
                                    agentToExecMap.insert({lcoAgent, {}});
                                }
                                agentToExecMap[lcoAgent].kd = kd;
                            } else {
                                co_manip::code_view_t function{reinterpret_cast<const std::byte *>(info.getView().data() + reinterpret_cast<luthier_address_t>(loadedMemory.data())),
                                                               info.getView().size()};
                                if (!functions_.contains(globalFuncPointer)) {
                                    auto globalFunctionName = std::get<const char *>(instrumentationFunctionInfo[k]);
                                    functions_.insert({globalFuncPointer, {{}, globalFunctionName, deviceFuncName}});
                                }
                                auto &agentToExecMap = functions_[globalFuncPointer].agentToExecMap;
                                if (!agentToExecMap.contains(lcoAgent)) {
                                    agentToExecMap.insert({lcoAgent, {}});
                                }
                                agentToExecMap[lcoAgent].function = function;
                            }
                        }
                    }
            }
            };
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
                                                                                      hsa::GpuAgent agent) const {
    auto f = functions_.at(wrapperKernelHostPtr).agentToExecMap.at(agent).function;
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
kernel_descriptor_t *luthier::CodeObjectManager::getKernelDescriptorOfInstrumentationFunction(const void *wrapperKernelHostPtr, hsa::GpuAgent agent) const {
    return functions_.at(wrapperKernelHostPtr).agentToExecMap.at(agent).kd;
}
