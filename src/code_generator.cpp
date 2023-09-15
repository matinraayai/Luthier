#include "code_generator.hpp"
#include "amdgpu_elf.hpp"
#include "code_object_manager.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "elfio/elfio.hpp"
#include "hsa_intercept.hpp"
#include "instr.hpp"
#include "log.hpp"
#include <fmt/color.h>
#include <fmt/core.h>
#include <hsa/hsa_ext_amd.h>

std::string getSymbolName(hsa_executable_symbol_t symbol) {
    const auto &coreHsaApiTable = luthier::HsaInterceptor::Instance().getSavedHsaTables().core;
    uint32_t nameSize;
    LUTHIER_HSA_CHECK(coreHsaApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));
    std::string name;
    name.resize(nameSize);
    LUTHIER_HSA_CHECK(coreHsaApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()));
    return name;
}

hsa_status_t registerSymbolWithCodeObjectManager(const hsa_executable_t &executable, const hsa_executable_symbol_t originalSymbol, hsa_agent_t agent) {

    hsa_status_t out = HSA_STATUS_ERROR;
    auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data) {
        auto originalSymbol = reinterpret_cast<hsa_executable_symbol_t *>(data);
        auto originalSymbolName = getSymbolName(*originalSymbol);

        auto &coreTable = luthier::HsaInterceptor::Instance().getSavedHsaTables().core;
        hsa_symbol_kind_t symbolKind;
        LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind));

        fmt::println(stdout, "Symbol kind: {}.", static_cast<int>(symbolKind));

        std::string symbolName = getSymbolName(symbol);

        fmt::println(stdout, "Symbol name: {}.", symbolName);

        if (symbolKind == HSA_SYMBOL_KIND_VARIABLE) {
            luthier_address_t variableAddress;
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &variableAddress));
            std::cout << "Variable location: " << std::hex << variableAddress << std::dec << std::endl;
        }
        if (symbolKind == HSA_SYMBOL_KIND_KERNEL && symbolName == originalSymbolName) {
            luthier_address_t kernelObject;
            luthier_address_t originalKernelObject;
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(*originalSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &originalKernelObject));
            luthier::CodeObjectManager::Instance().registerKD(reinterpret_cast<luthier_address_t>(originalKernelObject), reinterpret_cast<luthier_address_t>(kernelObject));

            std::cout << "original kernel location: " << std::hex << originalKernelObject << std::dec << std::endl;
            std::cout << "Kernel location: " << std::hex << kernelObject << std::dec << std::endl;
            std::vector<luthier::Instr> instList = luthier::Disassembler::Instance().disassemble(kernelObject);
            std::cout << "Disassembly of the KO: " << std::endl;
            for (const auto &i: instList) {
                std::cout << std::hex << i.getDeviceAddress() << std::dec << ": " << i.getInstr() << std::endl;
            }
        }

        //            symbolVec->push_back(symbol);
        return HSA_STATUS_SUCCESS;
    };
    out = hsa_executable_iterate_agent_symbols(executable,
                                               agent,
                                               iterCallback, (void *) &originalSymbol);
    if (out != HSA_STATUS_SUCCESS)
        return HSA_STATUS_ERROR;
    return HSA_STATUS_SUCCESS;
}

hsa_executable_t createExecutable(const char *codeObjectPtr, size_t codeObjectSize, hsa_agent_t agent) {
    auto coreApi = luthier::HsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_code_object_reader_t coReader;
    hsa_executable_t executable;
    LUTHIER_HSA_CHECK(coreApi.hsa_code_object_reader_create_from_memory_fn(codeObjectPtr, codeObjectSize, &coReader));

    LUTHIER_HSA_CHECK(coreApi.hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &executable));

    LUTHIER_HSA_CHECK(coreApi.hsa_executable_load_agent_code_object_fn(executable, agent, coReader, nullptr, nullptr));

    LUTHIER_HSA_CHECK(coreApi.hsa_executable_freeze_fn(executable, nullptr));
    return executable;
}

luthier::elf::mem_backed_code_object_t getLoadedCodeObject(hsa_executable_t executable) {
    auto amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t> *>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, &loadedCodeObjects);

    // Can be removed
    assert(loadedCodeObjects.size() == 1);

    uint64_t lcoBaseAddrDevice;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0], HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &lcoBaseAddrDevice);
    // Query the size of the loaded code object
    uint64_t lcoSizeDevice;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0], HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &lcoSizeDevice);
    return {reinterpret_cast<luthier_address_t>(lcoBaseAddrDevice), static_cast<size_t>(lcoSizeDevice)};
}

luthier::elf::mem_backed_code_object_t getCodeObject(hsa_executable_t executable) {
    auto amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t> *>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, &loadedCodeObjects);

    // Can be removed
    assert(loadedCodeObjects.size() == 1);

    uint64_t lcoBaseAddr;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0], HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE, &lcoBaseAddr);
    // Query the size of the loaded code object
    uint64_t lcoSize;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0], HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE, &lcoSize);
    return {reinterpret_cast<luthier_address_t>(lcoBaseAddr), static_cast<size_t>(lcoSize)};
}

std::string assemble(const std::string &instListStr, hsa_agent_t agent) {

    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataIn));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(dataIn, instListStr.size(), instListStr.data()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(dataIn, "my_source.s"));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_data_set_add(dataSetIn, dataIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetOut));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_action_info(&dataAction));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_isa_name(dataAction, luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName().c_str()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_option_list(dataAction, nullptr, 0));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, dataAction, dataSetIn, dataSetOut));
    amd_comgr_data_t dataOut;
    size_t dataOutSize;
    amd_comgr_action_data_get_data(dataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &dataOut);
    amd_comgr_get_data(dataOut, &dataOutSize, nullptr);
    std::string outElf;
    outElf.resize(dataOutSize);
    amd_comgr_get_data(dataOut, &dataOutSize, outElf.data());

    ELFIO::elfio io;
    std::stringstream elfss{outElf};
    io.load(elfss);

    return {io.sections[".text"]->get_data(), io.sections[".text"]->get_size()};
}

std::string assemble(const std::vector<std::string> &instrVector, hsa_agent_t agent) {

    std::string instString = fmt::format("{}", fmt::join(instrVector, "\n"));
    return assemble(instString, agent);
}

//void* allocateHsaKmtMemory(hsa_agent_t agent, size_t size, luthier::elf::mem_backed_code_object_t codeObject, luthier::elf::mem_backed_code_object_t hostCodeObject) {
//    uint32_t hsaKmtAgentNodeId = luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getAgentDriverNodeIdfromHsa();
//    const auto& amdExtApi = luthier::HsaInterceptor::Instance().getSavedHsaTables().amd_ext;
//    hsa_amd_pointer_info_t loadedCodeObjectPointerInfo;
//    luthier_address_t address = codeObject.data;
//    fmt::println("Address to query: {:#x}", address);
//    LUTHIER_HSA_CHECK(amdExtApi.hsa_amd_pointer_info_fn(reinterpret_cast<void*>(address),
//                                      &loadedCodeObjectPointerInfo, nullptr, nullptr, nullptr));
//    fmt::println("Loaded code object info:");
//    fmt::println("Type: {}", (uint32_t) loadedCodeObjectPointerInfo.type);
//    fmt::println("Agent base address: {:#x}", reinterpret_cast<luthier_address_t>(loadedCodeObjectPointerInfo.agentBaseAddress));
//    fmt::println("Host base address: {:#x}", reinterpret_cast<luthier_address_t>(loadedCodeObjectPointerInfo.hostBaseAddress));
//    fmt::println("size: {}", loadedCodeObjectPointerInfo.sizeInBytes);
//
//    luthier_address_t preferredAddress = codeObject.data;
//    hsa_amd_pointer_info_t preferredAddressInfo;
//    amdExtApi.hsa_amd_pointer_info_fn(reinterpret_cast<void*>(preferredAddress), &preferredAddressInfo, nullptr, nullptr, nullptr);
//    assert(sizeof(hsa_amd_pointer_info_t) == preferredAddressInfo.size);
//    preferredAddress += preferredAddressInfo.sizeInBytes;
//
//    while(preferredAddressInfo.sizeInBytes != 0) {
//        fmt::println("Address to query: {:#x}", preferredAddress);
//        amdExtApi.hsa_amd_pointer_info_fn(reinterpret_cast<void*>(preferredAddress), &preferredAddressInfo, nullptr, nullptr, nullptr);
//        preferredAddress += preferredAddressInfo.sizeInBytes;
//        assert(sizeof(hsa_amd_pointer_info_t) == preferredAddressInfo.size);
//        fmt::println("Code object's memory info:");
//        fmt::println("Type: {}", (uint32_t) preferredAddressInfo.type);
//        fmt::println("Agent base address: {:#x}", reinterpret_cast<luthier_address_t>(preferredAddressInfo.agentBaseAddress));
//        fmt::println("Host base address: {:#x}", reinterpret_cast<luthier_address_t>(preferredAddressInfo.hostBaseAddress));
//        fmt::println("Base address of the loaded code object: {:#x}", codeObject.data);
//        fmt::println("size: {}", preferredAddressInfo.sizeInBytes);
//        fmt::println("size of the code object on device: {}", codeObject.size);
//        fmt::println("size of the code object on host: {}", hostCodeObject.size);
//
//    }
//    fmt::println("Found a potential address!");
//    return reinterpret_cast<void*>(preferredAddress);
//
//
////    LUTHIER_HSAKMT_CHECK(hsaKmtOpenKFD());
////    HsaSystemProperties properties;
////    LUTHIER_HSAKMT_CHECK(hsaKmtAcquireSystemProperties(&properties));
////    HsaPointerInfo hsakmtPtrInfo;
////    LUTHIER_HSAKMT_CHECK(hsaKmtQueryPointerInfo(reinterpret_cast<void*>(codeObject.data), &hsakmtPtrInfo));
////    fmt::println("Agent base address: {:#x}", reinterpret_cast<luthier_address_t>(hsakmtPtrInfo.GPUAddress));
////    fmt::println("Host base address: {:#x}", reinterpret_cast<luthier_address_t>(hsakmtPtrInfo.CPUAddress));
////    fmt::println("size: {}", hsakmtPtrInfo.SizeInBytes);
////    HsaMemFlags flags = hsakmtPtrInfo.MemFlags;
////    flags.ui32.FixedAddress = 1;
//
//
//    fmt::println("Allocating with the HSA extension.");
//
//    struct cbdt {
//        luthier_address_t preferredAddress;
//        size_t size;
//        bool allocated;
//    } callbackData{preferredAddress, size};
////
////    const auto& amdApi = luthier::HsaInterceptor::instance().getSavedHsaTables().amd_ext;
//    const auto& coreApi = luthier::HsaInterceptor::Instance().getSavedHsaTables().core;
////    auto poolIterator = [](hsa_amd_memory_pool_t pool, void* data) {
////        auto callbackData = reinterpret_cast<cbdt *>(data);
////        fmt::println("Address value before anything: {:#x}", callbackData->preferredAddress);
////        const auto& coreApi = luthier::HsaInterceptor::instance().getSavedHsaTables().core;
////        const auto& amdApi = luthier::HsaInterceptor::instance().getSavedHsaTables().amd_ext;
////        hsa_amd_segment_t segment;
////        LUTHIER_HSA_CHECK(amdApi.hsa_amd_memory_pool_get_info_fn(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
////        uint32_t flags;
////        LUTHIER_HSA_CHECK(amdApi.hsa_amd_memory_pool_get_info_fn(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags));
////        bool hostAccessible;
////        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn({pool.handle}, (hsa_region_info_t) HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
////                                                       &hostAccessible));
////
////        size_t regionSize;
////        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn({pool.handle}, (hsa_region_info_t) HSA_REGION_INFO_SIZE, &regionSize));
////#ifdef LUTHIER_LOG_ENABLE_DEBUG
////        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::bisque),
////                   "Memory Flags: {:b}\n", flags);
////        fmt::println(stdout, "Segment: {}", (uint32_t) segment);
////        fmt::println(stdout, "Size: {}", regionSize);
////        fmt::println(stdout, "Host accessible: {}", hostAccessible);
////
////#endif
////        if (segment == HSA_AMD_SEGMENT_GLOBAL && (flags & (uint32_t)HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) && !hostAccessible && !callbackData->allocated) {
////            auto status = amdApi.hsa_amd_memory_pool_allocate_fn(pool, callbackData->size,
////                                                   HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG | HSA_AMD_MEMORY_POOL_FIXED_FLAG,
////                                                   reinterpret_cast<void**>(callbackData->preferredAddress));
////            if (status == HSA_STATUS_SUCCESS) {
////                callbackData->allocated = true;
////                return HSA_STATUS_SUCCESS;
////            }
////            else if (status != HSA_STATUS_ERROR_OUT_OF_RESOURCES)
////                return status;
////            else
////                fmt::println("Failed to allocate!. Current address value: {:#x}", callbackData->preferredAddress);
////        }
////
////        return HSA_STATUS_SUCCESS;
////    };
//////    LUTHIER_HSA_CHECK(amdExtApi.hsa_amd_agent_iterate_memory_pools_fn(agent, poolIterator, &callbackData));
////
////
//    auto regionIterator = [](hsa_region_t region, void* data) {
//        auto cbdata = reinterpret_cast<cbdt*>(data);
//        auto pa = cbdata->preferredAddress;
//        fmt::println("Address value before anything: {:#x}", cbdata->preferredAddress);
//        fmt::println("Requested size: {:#x}", cbdata->size);
//        const auto& coreApi = luthier::HsaInterceptor::Instance().getSavedHsaTables().core;
//        const auto& amdApi = luthier::HsaInterceptor::Instance().getSavedHsaTables().amd_ext;
//        hsa_region_segment_t segment;
//        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment));
//        uint32_t flags;
//        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags));
//        bool hostAccessible;
//        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, (hsa_region_info_t) HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
//                                                       &hostAccessible));
//
//        size_t regionSize;
//        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, (hsa_region_info_t) HSA_REGION_INFO_SIZE, &regionSize));
//#ifdef LUTHIER_LOG_ENABLE_DEBUG
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::bisque),
//                   "Memory Flags: {:b}\n", flags);
//        fmt::println(stdout, "Segment: {}", (uint32_t) segment);
//        fmt::println(stdout, "Size: {:#x}", regionSize);
//        fmt::println(stdout, "Host accessible: {}", hostAccessible);
//
//#endif
//        // NonPaged=1, NoSubstitute = 1, Host Access, Coarse
//        if (segment == HSA_REGION_SEGMENT_GLOBAL && (flags & (uint32_t)HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) && hostAccessible && !cbdata->allocated) {
//            fmt::println("Found a potential region to allocate with!");
//            auto status = amdApi.hsa_amd_memory_pool_allocate_fn({region.handle},
//                                                                      cbdata->size, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG | HSA_AMD_MEMORY_POOL_FIXED_FLAG,
//                                                                      reinterpret_cast<void**>(&pa));
//            if (status == HSA_STATUS_SUCCESS) {
//                cbdata->allocated = true;
//                return HSA_STATUS_SUCCESS;
//            }
//            else if (status != HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
//                return status;
//            }
//            else
//                fmt::println("Failed to allocate!. Current address value: {:#x}", cbdata->preferredAddress);
//
//        }
//        return HSA_STATUS_SUCCESS;
//    };
//
//    LUTHIER_HSA_CHECK(coreApi.hsa_agent_iterate_regions_fn(agent, regionIterator, &callbackData));
//
//    if (callbackData.allocated) {
//        fmt::println("Successfully allocated at {:#x}", preferredAddress);
//        return reinterpret_cast<void*>(callbackData.preferredAddress);
//    }
//    else {
//        throw std::runtime_error("Failed to allocate memory");
//    }
//
////    auto handle = dlopen("libhsa-runtime64.so", RTLD_LAZY);
////    if (handle == nullptr) {
////        fmt::println("dlopen failed: {}\n", dlerror());
////    }
////    else {
////        fmt::println("FOUND IT!");
////        void *function_ptr = ::dlsym(handle, "hsaKmtAllocMemory");
////        if (function_ptr == nullptr) {
////            fmt::println("Function not found :(");
////        } else {
////            fmt::println("Function was found!");
////        }
////    }
//
////    luthier_address_t preferredAddress = reinterpret_cast<luthier_address_t>(hsakmtPtrInfo.GPUAddress) + hsakmtPtrInfo.SizeInBytes;
////    fmt::println("Preferred Address was at: {:#x}", preferredAddress);
////    LUTHIER_HSAKMT_CHECK(hsaKmtAllocMemory(hsaKmtAgentNodeId, size + (4096 - (size % 4096)), flags, reinterpret_cast<void **>(&preferredAddress)));
////    fmt::println("Address was allocated at: {:#x}", preferredAddress);
////    return reinterpret_cast<void*>(preferredAddress);
////     Query where the executable's memory region ends
////     {NonPaged = 1, CachePolicy = 0, ReadOnly = 0, PageSize = 0, HostAccess = 1, NoSubstitute = 1, GDSMemory = 0, Scratch = 0, }
//
////    struct cbdt {
////        hsa_amd_memory_pool_t pool;
////        luthier::elf::mem_backed_code_object_t co;
////    } callbackData{};
////
////    const auto& amdApi = luthier::HsaInterceptor::instance().getSavedHsaTables().amd_ext;
////    auto regionIterator = [](hsa_amd_memory_pool_t pool, void* data) {
////        auto cbdata = reinterpret_cast<cbdt*>(data);
////        const auto& amdApi = luthier::HsaInterceptor::instance().getSavedHsaTables().amd_ext;
////        hsa_amd_memory_pool_info_t
////        hsa_region_segment_t segment;
////        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment));
////        uint32_t flags;
////        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags));
////
////        void* baseAddress;
////        LUTHIER_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, (hsa_region_info_t) HSA_AMD_REGION_INFO_BASE, &baseAddress));
////        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::bisque), "Base address of the memory region: {:#x}\n", reinterpret_cast<luthier_address_t>(baseAddress));
////        auto out = reinterpret_cast<hsa_region_t*>(data);
////        if (segment == HSA_REGION_SEGMENT_GLOBAL && (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED)) {
////            *out = region;
////        }
////
////        return HSA_STATUS_SUCCESS;
////    };
////    LUTHIER_HSA_CHECK(amdApi.hsa_amd_agent_iterate_memory_pools_fn(agent, regionIterator, &callbackData));
////    void* deviceMemory;
////    LUTHIER_HSA_CHECK(coreApi.hsa_memory_allocate_fn(region, size, &deviceMemory));
////    return deviceMemory;
//}
//
void luthier::CodeGenerator::modify(luthier::Instr &instr, void *my_addr) {
    auto kd = instr.getKernelDescriptor();
}

void luthier::CodeGenerator::instrument(luthier::Instr &instr, const std::string &instrumentationFunction, luthier_ipoint_t point) {
    LUTHIER_LOG_FUNCTION_CALL_START

    hsa_agent_t agent = instr.getAgent();
    luthier_address_t instDeviceAddress = instr.getDeviceAddress();
    // Load the instrumentation ELF into the agent, and get its location on the device
    auto instrumentationExecutable = createExecutable(instrumentationFunction.data(), instrumentationFunction.size(), agent);

    //    auto instrmntLoadedCodeObject = luthier::elf::mem_backed_code_object_t(
    //        reinterpret_cast<luthier_address_t>(allocateHsaKmtMemory(agent, instrumentationFunction.size(), getLoadedCodeObject(instr.getExecutable()), getCodeObject(instr.getExecutable()))),
    //        instrumentationFunction.size()
    //        );
    //    auto instrmntTextSectionStart = reinterpret_cast<luthier_address_t>(allocateHsaKmtMemory(agent, instrumentationFunction.size(), getLoadedCodeObject(instr.getExecutable()), getCodeObject(instr.getExecutable())));
    auto instrmntLoadedCodeObject = getLoadedCodeObject(instrumentationExecutable);

    // Get a pointer to the beginning of the .text section of the instrumentation executable
    luthier_address_t instrmntTextSectionStart = instrmntLoadedCodeObject.data + 0x1000;

    // The instrumentation function is inserted first
    std::string dummyInstrmnt = assemble(std::vector<std::string>{"s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)", "s_setpc_b64 s[0:1]"}, agent);

    // Padded with nop
    std::string nopInstr = assemble("s_nop 0", agent);

    std::memcpy(reinterpret_cast<void *>(instrmntTextSectionStart), dummyInstrmnt.data(), dummyInstrmnt.size());

    std::memcpy(reinterpret_cast<void *>(instrmntTextSectionStart + dummyInstrmnt.size()), nopInstr.data(), nopInstr.size());

    // Trampoline starts after the nop
    luthier_address_t trampolineStartAddr = instrmntTextSectionStart + dummyInstrmnt.size() + nopInstr.size();

    // Trampoline is located within the short jump range
    luthier_address_t trampolineInstrOffset = trampolineStartAddr > instDeviceAddress ? trampolineStartAddr - instDeviceAddress : instDeviceAddress - trampolineStartAddr;

    fmt::println(stdout, "trampolineStartAddr: {:#x}", trampolineStartAddr);
    fmt::println(stdout, "instDeviceAddress: {:#x}", instDeviceAddress);

    std::string trampoline;
    if (false) {
        trampoline = assemble("s_getpc_b64 s[2:3]", agent);

        // Get the PC of the instruction after the get PC instruction
        luthier_address_t trampolinePcOffset = trampolineStartAddr + trampoline.size();

        int firstAddOffset = (int) (trampolinePcOffset - instrmntTextSectionStart);

        fmt::println(stdout, "Trampoline PC offset: {:#x}", trampolinePcOffset);
        fmt::println(stdout, "Instrument Code Offset: {:#x}", instrmntTextSectionStart);
        fmt::println(stdout, "The set PC offset: {:#x}", firstAddOffset);

        trampoline += assemble({fmt::format("s_sub_u32 s2, s2, {:#x}", firstAddOffset),
                                "s_subb_u32 s3, s3, 0x0",
                                "s_swappc_b64 s[0:1], s[2:3]",
                                instr.getInstr()},
                               agent);

        trampolinePcOffset = trampolineStartAddr + trampoline.size() + 4;
        //    hostCodeObjectTextSection->append_data(trampoline);
        int lastBranchImmInt;
        short lastBranchImm;
        if (trampolinePcOffset < instr.getDeviceAddress()) {
            lastBranchImmInt = (instr.getDeviceAddress() + 4 - trampolinePcOffset) / 4;
            lastBranchImm = (short) (lastBranchImmInt);
        } else {
            lastBranchImmInt = (trampolinePcOffset - (instr.getDeviceAddress() + 4)) / 4;
            lastBranchImm = -(short) (lastBranchImmInt);
        }

#ifdef LUTHIER_LOG_ENABLE_DEBUG
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Trampoline PC Offset: {:#x}\n", trampolinePcOffset);
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Instr Address: {:#x}\n", instr.getDeviceAddress());
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Last branch imm: {:#x}\n", lastBranchImmInt);
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "After conversion to short: {:#x}\n", lastBranchImm);
#endif

        trampoline += assemble(fmt::format("s_branch {:#x}", lastBranchImm), agent);

        std::memcpy(reinterpret_cast<void *>(trampolineStartAddr), trampoline.data(), trampoline.size());

        const auto &amdExtApi = luthier::HsaInterceptor::Instance().getSavedHsaTables().amd_ext;
        hsa_amd_pointer_info_t instrPtrInfo;
        luthier_address_t address = instr.getDeviceAddress();
        fmt::println("Address to query: {:#x}", address);
        LUTHIER_HSA_CHECK(amdExtApi.hsa_amd_pointer_info_fn(reinterpret_cast<void *>(address), &instrPtrInfo, nullptr, nullptr, nullptr));
        fmt::println("Instruction Info:");
        fmt::println("Type: {}", (uint32_t) instrPtrInfo.type);
        fmt::println("Agent base address: {:#x}", reinterpret_cast<luthier_address_t>(instrPtrInfo.agentBaseAddress));
        fmt::println("Host base address: {:#x}", reinterpret_cast<luthier_address_t>(instrPtrInfo.hostBaseAddress));
        fmt::println("size: {}", instrPtrInfo.sizeInBytes);

        // Overwrite the target instruction
        int firstBranchImmUnconverted;
        short firstBranchImm;
        if (trampolineStartAddr < instr.getDeviceAddress()) {
            firstBranchImmUnconverted = (instr.getDeviceAddress() + 4 - trampolineStartAddr) / 4;
            firstBranchImm = -static_cast<short>(firstBranchImmUnconverted);
        } else {
            firstBranchImmUnconverted = (trampolineStartAddr - (instr.getDeviceAddress() + 4)) / 4;
            firstBranchImm = static_cast<short>(firstBranchImmUnconverted);
        }
#ifdef LUTHIER_LOG_ENABLE_DEBUG
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Trampoline start Address: {:#x}\n", trampolineStartAddr);
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Instr Address: {:#x}\n", instr.getDeviceAddress());
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "First branch imm: {:#x}\n", firstBranchImmUnconverted);
        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "After conversion to short: {:#x}\n", firstBranchImm);
#endif

        //    std::string firstJump = assemble("s_trap 1", agent);
        std::string firstJump = assemble({fmt::format("s_branch {:#x}", firstBranchImm)}, agent);
        if (instr.getSize() == 8)
            firstJump += assemble({std::string("s_nop 0")}, agent);
        std::memcpy(reinterpret_cast<void *>(instr.getDeviceAddress()), firstJump.data(), firstJump.size());
    }

    else {
        trampolineInstrOffset = trampolineStartAddr > instDeviceAddress ? trampolineInstrOffset - 4 : trampolineInstrOffset + 4;
        constexpr uint64_t upperMaskUint64_t = 0xFFFFFFFF00000000;
        constexpr uint64_t lowerMaskUint64_t = 0x00000000FFFFFFFF;
        uint32_t upperTrampolineInstrOffset = (uint32_t) ((trampolineInstrOffset & upperMaskUint64_t) >> 32);
        uint32_t lowerTrampolineInstrOffset = trampolineInstrOffset & lowerMaskUint64_t;

        fmt::println("Upper diff: {:#x}\n", upperTrampolineInstrOffset);
        fmt::println("Lower diff: {:#x}\n", lowerTrampolineInstrOffset);
        fmt::println("Actual diff: {:#x}\n", trampolineInstrOffset);
        std::string targetToTrampolineOffsetInstr = trampolineStartAddr > instDeviceAddress ? fmt::format("s_add_u32 s6, s6, {:#x}", lowerTrampolineInstrOffset) : fmt::format("s_sub_u32 s6, s6, {:#x}", lowerTrampolineInstrOffset);
        std::string longJumpForTarget = assemble(std::vector<std::string>{"s_getpc_b64 s[6:7]", targetToTrampolineOffsetInstr}, agent);

        if (upperTrampolineInstrOffset != 0) {
            longJumpForTarget += trampolineStartAddr > instDeviceAddress ? assemble(fmt::format("s_addc_u32 s7, s7, {:#x}", upperTrampolineInstrOffset), agent) : assemble(fmt::format("s_subb_u32 s7, s7, {:#x}", upperTrampolineInstrOffset), agent);
        }

        longJumpForTarget += assemble("s_swappc_b64 s[2:3], s[6:7]", agent);
        fmt::println("Assembled!!!");
        std::string displacedInstr = std::string(reinterpret_cast<const char *>(instr.getDeviceAddress()), longJumpForTarget.size());

        // Get the PC of the instruction after the get PC instruction
        luthier_address_t trampolinePcOffset = trampolineStartAddr;

        int firstAddOffset = (int) (trampolinePcOffset - instrmntTextSectionStart);

        fmt::println(stdout, "Trampoline PC offset: {:#x}", trampolinePcOffset);
        fmt::println(stdout, "Instrument Code Offset: {:#x}", instrmntTextSectionStart);
        fmt::println(stdout, "The set PC offset: {:#x}", firstAddOffset);

        trampoline = assemble({fmt::format("s_sub_u32 s6, s6, {:#x}", firstAddOffset),
                               "s_subb_u32 s7, s7, 0x0",
                               "s_swappc_b64 s[0:1], s[6:7]"},
                              agent);
        trampoline += displacedInstr;
        trampoline += assemble("s_setpc_b64 s[2:3]", agent);

        std::memcpy(reinterpret_cast<void *>(trampolineStartAddr), trampoline.data(), trampoline.size());
        std::memcpy(reinterpret_cast<void *>(instr.getDeviceAddress()), longJumpForTarget.data(), longJumpForTarget.size());
    }

#ifdef LUTHIER_LOG_ENABLE_DEBUG
    auto finalTargetInstructions =
        luthier::Disassembler::Instance().disassemble(reinterpret_cast<luthier_address_t>(instr.getKernelDescriptor()));
    fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Instrumented Kernel Final View:\n");

    for (const auto &i: finalTargetInstructions) {
        auto printFormat = instr.getDeviceAddress() <= i.getDeviceAddress() && (i.getDeviceAddress() + i.getSize()) <= (instr.getDeviceAddress() + instr.getSize()) ? fmt::emphasis::underline : fmt::emphasis::bold;
        fmt::print(stdout, printFormat, "{:#x}: {:s}\n", i.getDeviceAddress(), i.getInstr());
    }
    auto finalInstrumentationInstructions =
        luthier::Disassembler::Instance().disassemble(agent, instrmntTextSectionStart,
                                                      dummyInstrmnt.size() + nopInstr.size() + trampoline.size());
    fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::orange_red), "Instrumented Kernel Final View:\n");
    for (const auto &i: finalInstrumentationInstructions) {
        fmt::print(stdout, fmt::emphasis::bold, "{:#x}: {:s}\n", i.getDeviceAddress(), i.getInstr());
    }
#endif

#ifdef LUTHIER_LOG_ENABLE_DEBUG
    auto hostInstructions =
        luthier::Disassembler::Instance().disassemble(agent, getCodeObject(instr.getExecutable()).data + 0x1000, 0x54);
    fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Host Executable:\n");

    for (const auto &i: hostInstructions) {
        auto printFormat = instr.getDeviceAddress() <= i.getDeviceAddress() && (i.getDeviceAddress() + i.getSize()) <= (instr.getDeviceAddress() + instr.getSize()) ? fmt::emphasis::underline : fmt::emphasis::bold;
        fmt::print(stdout, printFormat, "{:#x}: {:s}\n", i.getDeviceAddress(), i.getInstr());
    }
#endif

    LUTHIER_LOG_FUNCTION_CALL_END
}
