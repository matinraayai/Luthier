#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "elfio/elfio.hpp"
#include "hsa_intercept.hpp"
#include "instr.hpp"
#include "log.hpp"
#include <fmt/color.h>
#include <fmt/core.h>
#include <hsa/amd_hsa_common.h>
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

luthier::co_manip::code_t luthier::CodeGenerator::compileRelocatableToExecutable(const luthier::co_manip::code_t &code,
                                                                                 hsa_agent_t agent) {
    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    std::string isaName = luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName();

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(dataIn, code.size(), reinterpret_cast<const char *>(code.data())));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(dataIn, "source.o"));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_data_set_add(dataSetIn, dataIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetOut));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_action_info(&dataAction));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_isa_name(dataAction, isaName.c_str()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_option_list(dataAction, nullptr, 0));
    LUTHIER_AMD_COMGR_CHECK(
        amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                            dataAction, dataSetIn, dataSetOut));

    amd_comgr_data_t dataOut;
    size_t dataOutSize;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_data_get_data(dataSetOut, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &dataOut));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(dataOut, &dataOutSize, nullptr));
    luthier::co_manip::code_t executableOut;
    executableOut.resize(dataOutSize);
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(dataOut, &dataOutSize, reinterpret_cast<char *>(executableOut.data())));
    return executableOut;
}

luthier::co_manip::code_t luthier::CodeGenerator::assembleToRelocatable(const std::string &instList, hsa_agent_t agent) {

    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    auto isaName = luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName();

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataIn));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(dataIn, instList.size(), instList.data()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(dataIn, "my_source.s"));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_data_set_add(dataSetIn, dataIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetOut));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_action_info(&dataAction));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_isa_name(dataAction,
                                                               isaName.c_str()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_option_list(dataAction, nullptr, 0));
    LUTHIER_AMD_COMGR_CHECK(
        amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                            dataAction, dataSetIn, dataSetOut));
    amd_comgr_data_t dataOut;
    size_t dataOutSize;
    amd_comgr_action_data_get_data(dataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &dataOut);
    size_t nameSize;
    std::string name;
    amd_comgr_get_data_name(dataOut, &nameSize, nullptr);
    name.resize(nameSize);
    amd_comgr_get_data_name(dataOut, &nameSize, name.data());

    fmt::print(stderr, "Name of the data: {}\n", name);

    amd_comgr_get_data(dataOut, &dataOutSize, nullptr);
    luthier::co_manip::code_t outElf;
    outElf.resize(dataOutSize);
    amd_comgr_get_data(dataOut, &dataOutSize, reinterpret_cast<char *>(outElf.data()));
    auto outView = luthier::co_manip::ElfViewImpl::makeView(outElf);
    return outElf;
}

luthier::co_manip::code_t luthier::CodeGenerator::assembleToRelocatable(const std::vector<std::string> &instList, hsa_agent_t agent) {
    std::string instString = fmt::format("{}", fmt::join(instList, "\n"));
    return assembleToRelocatable(instString, agent);
}

luthier::co_manip::code_t luthier::CodeGenerator::assemble(const std::string &instListStr, hsa_agent_t agent) {
    auto relocatable = assembleToRelocatable(instListStr, agent);
    co_manip::codestream coStream(
        boost_ios::stream<boost_ios::basic_array_source<char>>(
            std::string_view(reinterpret_cast<const char *>(relocatable.data()), relocatable.size()).begin(),
            std::string_view(reinterpret_cast<const char *>(relocatable.data()), relocatable.size()).end()));
    ELFIO::elfio elfio;
    elfio.load(coStream, false);
    //auto &elfio = co_manip::ElfViewImpl::makeView(relocatable)->getElfIo();
    auto textSection = elfio.sections[".text"];

    return {reinterpret_cast<const std::byte *>(textSection->get_data()), textSection->get_size()};
}

luthier::co_manip::code_t luthier::CodeGenerator::assemble(const std::vector<std::string> &instrVector, hsa_agent_t agent) {
    auto relocatable = assembleToRelocatable(instrVector, agent);

    co_manip::codestream coStream(
        boost_ios::stream<boost_ios::basic_array_source<char>>(
            std::string_view(reinterpret_cast<const char *>(relocatable.data()), relocatable.size()).begin(),
            std::string_view(reinterpret_cast<const char *>(relocatable.data()), relocatable.size()).end()));
    ELFIO::elfio elfio;
    elfio.load(coStream, false);
    auto textSection = elfio.sections[".text"];

    return {reinterpret_cast<const std::byte *>(textSection->get_data()), textSection->get_size()};
}

luthier::CodeGenerator::CodeGenerator() {
    const auto &contextManager = luthier::ContextManager::Instance();
    const auto hsaAgents = contextManager.getHsaAgents();
    for (const auto &agent: hsaAgents) {
        auto emptyRelocatable = assembleToRelocatable("s_nop 0", agent);
        emptyRelocatableMap_.insert({agent.handle, emptyRelocatable});
    }
}

hsa_status_t registerSymbolWithCodeObjectManager(const hsa_executable_t &executable,
                                                 const hsa_executable_symbol_t originalSymbol,
                                                 hsa_agent_t agent) {

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
            luthier::CodeObjectManager::instance().registerInstrumentedKernel(reinterpret_cast<kernel_descriptor_t *>(originalKernelObject), reinterpret_cast<kernel_descriptor_t *>(kernelObject));
            std::cout << "original kernel location: " << std::hex << originalKernelObject << std::dec << std::endl;
            std::cout << "Kernel location: " << std::hex << kernelObject << std::dec << std::endl;
            std::vector<luthier::Instr> instList = luthier::Disassembler::instance().disassemble(kernelObject);
            std::cout << "Disassembly of the KO: " << std::endl;
            for (const auto &i: instList) {
                std::cout << std::hex << i.getDeviceAddress() << std::dec << ": " << i.getInstr() << std::endl;
                if (i.getInstr().find("s_add_u32") != std::string::npos) {
                    //                    std::string out = assemble("s_add_u32 s2 s100 0", agent);
                    //                    std::memcpy(reinterpret_cast<void*>(i.getDeviceAddress()), out.data(), out.size());
                }
            }
            // luthier::co_manip::printRSR1(reinterpret_cast<kernel_descriptor_t *>(kernelObject));
            // luthier::co_manip::printRSR2(reinterpret_cast<kernel_descriptor_t *>(kernelObject));
            // luthier::co_manip::printCodeProperties(reinterpret_cast<kernel_descriptor_t *>(kernelObject));
            const kernel_descriptor_t *kernelDescriptor{nullptr};
            const auto &amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
            LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(kernelObject),
                                                                             reinterpret_cast<const void **>(&kernelDescriptor)));
            auto entry_point = reinterpret_cast<luthier_address_t>(kernelObject) + kernelDescriptor->kernel_code_entry_byte_offset;

            //            instList = luthier::Disassembler::Instance().disassemble(agent, entry_point - 0x14c, 0x500);
            // instList = luthier::Disassembler::instance().disassemble(kernelObject);
            // std::cout << "Disassembly of the KO: " << std::endl;
            // for (const auto &i: instList) {
            //     std::cout << std::hex << i.getDeviceAddress() << std::dec << ": " << i.getInstr() << std::endl;
            // }
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

kernel_descriptor_t normalizeTargetAndInstrumentationKDs(kernel_descriptor_t *target, kernel_descriptor_t *instrumentation) {
}

void luthier::CodeGenerator::instrument(Instr &instr, const void *device_func,
                                        luthier_ipoint_t point) {
    LUTHIER_LOG_FUNCTION_CALL_START
    hsa_agent_t agent = instr.getAgent();
    hsa_executable_t instrExecutable = instr.getExecutable();
    auto hco = co_manip::getHostLoadedCodeObjectOfExecutable(instrExecutable, agent);
    co_manip::code_t newCodeObject(hco[0]);

    co_manip::codestream newCOStream(
        boost_ios::stream<boost_ios::basic_array_source<char>>(
            std::string_view(reinterpret_cast<const char *>(newCodeObject.data()), newCodeObject.size()).begin(),
            std::string_view(reinterpret_cast<const char *>(newCodeObject.data()), newCodeObject.size()).end()));
    ELFIO::elfio elfio;
    elfio.load(newCOStream, false);
    std::cout << "Create a elfio object using codestream\n";

    /*luthier_address_t kernelCodeStartAddr, func1StartAddr;

    for (unsigned int i = 0; i < co_manip::getSymbolNum(instrElf); i++) {
        co_manip::SymbolView info(instrElf, i);
        fmt::println("Symbol Info: {}, {}, {:#x}", info.getName(), info.getView().size(),
                     reinterpret_cast<luthier_address_t>(info.getView().data()));
        // Symbol Info: _DYNAMIC, 0, 0xf3f410
        // Symbol Info: _Z10relu_floatiPfS_, 140, 0xf3e410
        // kernel starts at 0xf3e410 ends at 0xf3e49c to add jump instructions modified kernel ends at F3E4E4
        // Symbol Info: _Z10relu_floatiPfS_.kd, 64, 0xf3d910
        if (info.getName().find(".kd") != std::string::npos) {
            fmt::println("sym sec addr: {:#x}", info.getSection()->get_address());
            auto kd = const_cast<kernel_descriptor_t *>(reinterpret_cast<const kernel_descriptor_t *>(info.getView().data()));
            AMD_HSA_BITS_SET(kd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT, 1);
            // AMD_HSA_BITS_SET(kd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT, 8);
            AMD_HSA_BITS_SET(kd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT, 1);
            //            AMD_HSA_BITS_SET(kd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1);
            //            AMD_HSA_BITS_SET(kd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 0);
            // kernelCodeStartAddr = reinterpret_cast<luthier_address_t>(kd) + kd->kernel_code_entry_byte_offset;
            // std::cout<<"kernel code start "<<kernelCodeStartAddr<<std::endl;

        } else if (info.getName() == "_Z10relu_floatiPfS_") {
            kernelCodeStartAddr = reinterpret_cast<luthier_address_t>(info.getView().data());
            auto kernelSize = info.getView().size();
            func1StartAddr = kernelCodeStartAddr + kernelSize + 24;
            fmt::println("kernel code start at address {:#x}", kernelCodeStartAddr);
            fmt::println("func1 start at address {:#x}", func1StartAddr);
        }
    }*/
    //     Require dynamic info about the target kernel
    // 1. Num of threads to calculate memory size needed to store registers THD*4*NV + 4*NS
    // 2. Num of vgpr used NV = 4 available v4, v5, v6, v7 after set GRANULATED_WAVEFRONT_VGPR_COUNT as 1
    // 3. Num of sgpr used NS = 11 available s11, s12, s13, s14, s15
    // Use s12:13 to construct jump PC and s30:31 for return PC after set ENABLE_SGPR_FLAT_SCRATCH_INIT as 1
    // Use v4 to store global thread id
    // Three routines 1) get global thread id; 2) save all vgpr; 3) save all sgpr
    // extra gpr requirement: v s
    //                     1) 0 1    s can be discard and v is permenant global idx
    //                     2) 3 3
    //                     3)
    // put 1) at 0xf3e4e4 and its content is
    // s_load_dword s14, s[4:5], 0x4
    // s_waitcnt lgkmcnt(0)
    // s_and_b32 s14, s14, 0xffff
    // s_mul_i32 s8, s8, s14
    // v_add_u32_e32 v4, s8, v0
    // s_setpc_b64 s[30:31]
    // std::string Func1 = assemble(std::vector<std::string>{
    //     "s_load_dword s14, s[4:5], 0x4",
    // "s_waitcnt lgkmcnt(0)",
    // "s_and_b32 s14, s14, 0xffff",
    // "s_mul_i32 s8, s8, s14",
    // "v_add_u32_e32 v4, s8, v0",
    // "s_setpc_b64 s[30:31]"
    // },agent);
    // std::memcpy(reinterpret_cast<void *>(func1StartAddr), Func1.data(), Func1.size());

    //extra gpr requirement: v4 s14(next one after jump PC sgprs)
    // jump to 1) needs these instructions !!!!!These will be part of trampoline but for now I have to make them part of the kernel which means reconstruct the kernel by inserting three occurance of this part: before kernel start, first call 2), second time call 2)!!!!!!! total 24bytes*3
    // s_getpc_b64 s[12:13] 4bytes
    // s_add_u32 s12, s12,  8bytes
    // s_addc_u32 s13, s13, 0  8bytes
    // s_swappc_b64 s[30:31], s[12:13] 4bytes

    co_manip::code_t myReLU = assemble(std::vector<std::string>{
                                           "s_load_dword s14, s[4:5], 0x4",
                                           "s_waitcnt lgkmcnt(0)",
                                           "s_and_b32 s14, s14, 0xffff",
                                           "s_mul_i32 s8, s8, s14",
                                           "v_add_u32_e32 v0, s8, v0",
                                           "s_endpgm"},
                                       agent);
    // co_manip::code_t myReLU = assemble("s_endpgm", agent);

    elfio.sections[".text"]->set_data(reinterpret_cast<char *>(myReLU.data()), myReLU.size());
    // std::cout << myReLU.size() << std::endl;
    // std::cout << elfio.sections[".text"]->get_size() << std::endl;

    // for (const auto& sec: instrElf->getElfIo().sections) {
    //     fmt::println("Section name {}", sec->get_name());
    //     fmt::println("Section addr {:#x}", sec->get_address());
    // }
    // save the ELF and create an executable
    std::ostringstream ss;
    elfio.save(ss);
    std::string elf = ss.str();
    std::cout << elf << std::endl;

    auto coreTable = HsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_code_object_reader_t reader;
    hsa_executable_t executable;
    LUTHIER_HSA_CHECK(coreTable.hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                                             nullptr, &executable));
    LUTHIER_HSA_CHECK(coreTable.hsa_code_object_reader_create_from_memory_fn(elf.data(),
                                                                             elf.size(), &reader));
    LUTHIER_HSA_CHECK(coreTable.hsa_executable_load_agent_code_object_fn(executable, agent, reader, nullptr, nullptr));
    LUTHIER_HSA_CHECK(coreTable.hsa_executable_freeze_fn(executable, nullptr));
    LUTHIER_HSA_CHECK(registerSymbolWithCodeObjectManager(executable, instr.getSymbol(), agent));
}

//{
//
//    hsa_agent_t agent = instr.getAgent();
//    const auto& coManager = luthier::CodeObjectManager::Instance();
//    luthier::co_manip::code_object_region_t instrumentationFunction = coManager.getCodeObjectOfInstrumentationFunction(device_func, agent);
//    kernel_descriptor_t* kd = luthier::CodeObjectManager::Instance().getKernelDescriptorOfInstrumentationFunction(device_func, agent);
//    fmt::println("Address of kd: {:#x}", reinterpret_cast<luthier_address_t>(kd));
//    co_manip::printRSR1(kd);
//    co_manip::printRSR2(kd);
//    co_manip::printCodeProperties(kd);
//    fmt::println("group_segment_fixed_size: {}", kd->group_segment_fixed_size);
//    fmt::println("private_segment_fixed_size: {}", kd->private_segment_fixed_size);
//    fmt::println("kernarg_size: {}", kd->kernarg_size);
//    for (int i = 0; i < 4; i++)
//        fmt::println("reserved0: {}", kd->reserved0[i]);
//    fmt::println("compute_pgm_rsrc3: {}", kd->compute_pgm_rsrc3);
//    fmt::println("group_segment_fixed_size: {}", kd->group_segment_fixed_size);
//    for (int i = 0; i < 6; i++)
//        fmt::println("reserved2: {}", kd->reserved2[i]);
//
//    auto funcVgprCount = AMD_HSA_BITS_GET(kd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT);
////    auto funcSgprCount = AMD_HSA_BITS_GET(kd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT);
////    auto funcUserSgprCount = AMD_HSA_BITS_GET(kd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT);
////    auto funcDynamicCallStack = AMD_HSA_BITS_GET(kd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK);
////    auto funcEnableSegPtr = AMD_HSA_BITS_GET(kd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
////    auto funcEnableX = AMD_HSA_BITS_GET(kd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X);
//    auto instrKd = instr.getKernelDescriptor();
//    auto entryPointOfInstr = instrKd->kernel_code_entry_byte_offset;
////    *instrKd = *kd;
////    instrKd->kernel_code_entry_byte_offset = entryPointOfInstr;
//    fmt::println("Address of Instr kd: {:#x}", reinterpret_cast<luthier_address_t>(instrKd));
//    fmt::println("Entry: {:#x}", reinterpret_cast<luthier_address_t>(instrKd) + instrKd->kernel_code_entry_byte_offset);
//    co_manip::printRSR1(instrKd);
//    co_manip::printRSR2(instrKd);
//    co_manip::printCodeProperties(instrKd);
//    fmt::println("group_segment_fixed_size: {}", instrKd->group_segment_fixed_size);
//    fmt::println("private_segment_fixed_size: {}", instrKd->private_segment_fixed_size);
//    fmt::println("kernarg_size: {}", instrKd->kernarg_size);
//    for (int i = 0; i < 4; i++)
//        fmt::println("reserved0: {}", instrKd->reserved0[i]);
//    fmt::println("compute_pgm_rsrc3: {}", instrKd->compute_pgm_rsrc3);
//    fmt::println("group_segment_fixed_size: {}", instrKd->group_segment_fixed_size);
//    for (int i = 0; i < 6; i++)
//        fmt::println("reserved2: {}", instrKd->reserved2[i]);
////    auto instrVgprCount = AMD_HSA_BITS_GET(instrKd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT);
////    auto instrSgprCount = AMD_HSA_BITS_GET(instrKd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT);
////    auto instrUserSgprCount = AMD_HSA_BITS_GET(instrKd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT);
////    auto instrDynamicCallStack = AMD_HSA_BITS_GET(instrKd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK);
////    auto instrEnableSegPtr = AMD_HSA_BITS_GET(instrKd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
////    auto instrEnableX = AMD_HSA_BITS_GET(instrKd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X);
////    LuthierLogDebug("Function VGPR count: {}. SGPR count: {}. User SGPR Count: {}, dynamic call stack: {}", funcVgprCount, funcSgprCount, funcUserSgprCount, funcDynamicCallStack);
////    LuthierLogDebug("Function sgpr kernarg seg ptr: {}, Enable X: {}", funcEnableSegPtr, funcEnableX);
////    LuthierLogDebug("Instr VGPR count: {}. SGPR count: {}, User SGPR Count: {}, dynamic call stack: {}", instrVgprCount, instrSgprCount, instrUserSgprCount, instrDynamicCallStack);
////    LuthierLogDebug("Instr sgpr kernarg seg ptr: {}, enable X: {}", instrEnableSegPtr, instrEnableX);
////    AMD_HSA_BITS_SET(instrKd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT, 1);
////    AMD_HSA_BITS_SET(instrKd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT, 5);
////    AMD_HSA_BITS_SET(instrKd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT, 12);
////    AMD_HSA_BITS_SET(instrKd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, 1);
////    AMD_HSA_BITS_SET(instrKd->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT, 1);
////    instrKd->group_segment_fixed_size = 8;
////    instrVgprCount = AMD_HSA_BITS_GET(instrKd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT);
////    instrSgprCount = AMD_HSA_BITS_GET(instrKd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT);
////    instrUserSgprCount = AMD_HSA_BITS_GET(instrKd->compute_pgm_rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT);
////    LuthierLogDebug("Instr VGPR count: {}. SGPR count: {}, User SGPR Count: {}", instrVgprCount, instrSgprCount, instrUserSgprCount);
//
//    luthier_address_t instDeviceAddress = instr.getDeviceAddress();
//    // Load the instrumentation ELF into the agent, and get its location on the device
//    auto instrumentationExecutable = createExecutableMemoryRegion(instrumentationFunction.size, agent);
//
////    auto instrmntLoadedCodeObject = luthier::elf::mem_backed_code_object_t(
////        reinterpret_cast<luthier_address_t>(allocateHsaKmtMemory(agent, instrumentationFunction.size(), getLoadedCodeObject(instr.getExecutable()), getCodeObject(instr.getExecutable()))),
////        instrumentationFunction.size()
////        );
////    auto instrmntTextSectionStart = reinterpret_cast<luthier_address_t>(allocateHsaKmtMemory(agent, instrumentationFunction.size(), getLoadedCodeObject(instr.getExecutable()), getCodeObject(instr.getExecutable())));
////    auto instrmntLoadedCodeObject = luthier::co_manip::getDeviceLoadedCodeObjectOfExecutable(instrumentationExecutable, agent);
//
//
//
//    // Get a pointer to the beginning of the .text section of the instrumentation executable
//    luthier_address_t instrmntTextSectionStart = instrumentationExecutable.data;
//
//    // The instrumentation function is inserted first
//    std::string dummyInstrmnt = assemble(std::vector<std::string>
//        {"s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)", "s_setpc_b64 s[0:1]"}, agent);
//
//    // Padded with nop
//    std::string nopInstr = assemble("s_nop 0", agent);
//
//    std::memcpy(reinterpret_cast<void*>(instrmntTextSectionStart), dummyInstrmnt.data(), dummyInstrmnt.size());
//
//    std::memcpy(reinterpret_cast<void*>(instrmntTextSectionStart + dummyInstrmnt.size()), nopInstr.data(), nopInstr.size());
//
//    // Trampoline starts after the nop
//    luthier_address_t trampolineStartAddr = instrmntTextSectionStart + dummyInstrmnt.size() + nopInstr.size();
//
//    // Trampoline is located within the short jump range
//    luthier_address_t trampolineInstrOffset = trampolineStartAddr > instDeviceAddress ? trampolineStartAddr - instDeviceAddress :
//                                                                                      instDeviceAddress - trampolineStartAddr;
//
//    std::string trampoline;
//    if (trampolineInstrOffset < ((2 << 16) - 1)) {
//        trampoline = assemble("s_getpc_b64 s[2:3]", agent);
//
//        // Get the PC of the instruction after the get PC instruction
//        luthier_address_t trampolinePcOffset = trampolineStartAddr + trampoline.size();
//
//
//        int firstAddOffset = (int) (trampolinePcOffset - instrmntTextSectionStart);
//
//        fmt::println(stdout, "Trampoline PC offset: {:#x}", trampolinePcOffset);
//        fmt::println(stdout, "Instrument Code Offset: {:#x}", instrmntTextSectionStart);
//        fmt::println(stdout, "The set PC offset: {:#x}", firstAddOffset);
//
//
//        trampoline += assemble({fmt::format("s_sub_u32 s2, s2, {:#x}", firstAddOffset),
//                                "s_subb_u32 s3, s3, 0x0",
//                                "s_swappc_b64 s[0:1], s[2:3]",
//                                instr.getInstr()}, agent);
//
//
//        trampolinePcOffset = trampolineStartAddr + trampoline.size() + 4;
//        //    hostCodeObjectTextSection->append_data(trampoline);
//        int lastBranchImmInt;
//        short lastBranchImm;
//        if (trampolinePcOffset < instr.getDeviceAddress()) {
//            lastBranchImmInt = (instr.getDeviceAddress() + 4 - trampolinePcOffset) / 4;
//            lastBranchImm = (short) (lastBranchImmInt);
//        }
//        else {
//            lastBranchImmInt = (trampolinePcOffset - (instr.getDeviceAddress() + 4)) / 4;
//            lastBranchImm = -(short)(lastBranchImmInt);
//        }
//
//#ifdef LUTHIER_LOG_ENABLE_DEBUG
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Trampoline PC Offset: {:#x}\n", trampolinePcOffset);
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Instr Address: {:#x}\n", instr.getDeviceAddress());
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Last branch imm: {:#x}\n", lastBranchImmInt);
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "After conversion to short: {:#x}\n", lastBranchImm);
//#endif
//
//        trampoline += assemble(fmt::format("s_branch {:#x}", lastBranchImm), agent);
//
//        std::memcpy(reinterpret_cast<void*>(trampolineStartAddr), trampoline.data(), trampoline.size());
//
//        const auto& amdExtApi = luthier::HsaInterceptor::Instance().getSavedHsaTables().amd_ext;
//        hsa_amd_pointer_info_t instrPtrInfo;
//        luthier_address_t address = instr.getDeviceAddress();
//        fmt::println("Address to query: {:#x}", address);
//        LUTHIER_HSA_CHECK(amdExtApi.hsa_amd_pointer_info_fn(reinterpret_cast<void*>(address),
//                                                          &instrPtrInfo, nullptr, nullptr, nullptr));
//        fmt::println("Instruction Info:");
//        fmt::println("Type: {}", (uint32_t) instrPtrInfo.type);
//        fmt::println("Agent base address: {:#x}", reinterpret_cast<luthier_address_t>(instrPtrInfo.agentBaseAddress));
//        fmt::println("Host base address: {:#x}", reinterpret_cast<luthier_address_t>(instrPtrInfo.hostBaseAddress));
//        fmt::println("size: {}", instrPtrInfo.sizeInBytes);
//
//
//        // Overwrite the target instruction
//        int firstBranchImmUnconverted;
//        short firstBranchImm;
//        if (trampolineStartAddr < instr.getDeviceAddress()) {
//            firstBranchImmUnconverted = (instr.getDeviceAddress() + 4 - trampolineStartAddr) / 4;
//            firstBranchImm = -static_cast<short>(firstBranchImmUnconverted);
//        }
//        else {
//            firstBranchImmUnconverted = (trampolineStartAddr - (instr.getDeviceAddress() + 4)) / 4;
//            firstBranchImm = static_cast<short>(firstBranchImmUnconverted);
//        }
//#ifdef LUTHIER_LOG_ENABLE_DEBUG
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Trampoline start Address: {:#x}\n", trampolineStartAddr);
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Instr Address: {:#x}\n", instr.getDeviceAddress());
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "First branch imm: {:#x}\n", firstBranchImmUnconverted);
//        fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "After conversion to short: {:#x}\n", firstBranchImm);
//#endif
//
//        //    std::string firstJump = assemble("s_trap 1", agent);
//        std::string firstJump = assemble({fmt::format("s_branch {:#x}", firstBranchImm)}, agent);
//        if (instr.getSize() == 8)
//            firstJump += assemble({std::string("s_nop 0")}, agent);
//        std::memcpy(reinterpret_cast<void *>(instr.getDeviceAddress()), firstJump.data(), firstJump.size());
//    }
//
//    else {
//        trampolineInstrOffset = trampolineStartAddr > instDeviceAddress ? trampolineInstrOffset - 4 : trampolineInstrOffset + 4;
//        constexpr uint64_t upperMaskUint64_t = 0xFFFFFFFF00000000;
//        constexpr uint64_t lowerMaskUint64_t = 0x00000000FFFFFFFF;
//        uint32_t upperTrampolineInstrOffset = (trampolineInstrOffset & upperMaskUint64_t) >> 32;
//        uint32_t lowerTrampolineInstrOffset = trampolineInstrOffset & lowerMaskUint64_t;
//
//        fmt::println("Upper diff: {:#b}\n", upperTrampolineInstrOffset);
//        fmt::println("Lower diff: {:#b}\n", lowerTrampolineInstrOffset);
//        fmt::println("Actual diff: {:#b}\n", trampolineInstrOffset);
//        std::string targetToTrampolineOffsetInstr = trampolineStartAddr > instDeviceAddress ? fmt::format("s_add_u32 s6, s6, {:#x}", lowerTrampolineInstrOffset) :
//                                                                                            fmt::format("s_sub_u32 s6, s6, {:#x}", lowerTrampolineInstrOffset);
//        std::string longJumpForTarget = assemble(std::vector<std::string>{"s_getpc_b64 s[8:9]",
//                                                                          targetToTrampolineOffsetInstr}, agent);
//
//        if (upperTrampolineInstrOffset != 0) {
//            longJumpForTarget += trampolineStartAddr > instDeviceAddress ? assemble(fmt::format("s_addc_u32 s7, s7, {:#x}", upperTrampolineInstrOffset), agent) :
//                                                                         assemble(fmt::format("s_subb_u32 s7, s7, {:#x}", upperTrampolineInstrOffset), agent);
//        }
//
//       longJumpForTarget += assemble("s_swappc_b64 s[2:3], s[8:9]", agent);
//        fmt::println("Assembled!!!");
//        std::string displacedInstr = std::string(reinterpret_cast<const char*>(instr.getDeviceAddress()),
//                                                 longJumpForTarget.size());
//
//        // Get the PC of the instruction after the get PC instruction
//        luthier_address_t trampolinePcOffset = trampolineStartAddr;
//
//
//        int firstAddOffset = (int) (trampolinePcOffset - instrmntTextSectionStart);
//
//        fmt::println(stdout, "Trampoline PC offset: {:#x}", trampolinePcOffset);
//        fmt::println(stdout, "Instrument Code Offset: {:#x}", instrmntTextSectionStart);
//        fmt::println(stdout, "The set PC offset: {:#x}", firstAddOffset);
//
//
//        trampoline = assemble({fmt::format("s_sub_u32 s6, s6, {:#x}", firstAddOffset),
//                                "s_subb_u32 s7, s7, 0x0",
//                                "s_swappc_b64 s[0:1], s[8:9]"}, agent);
//        trampoline += displacedInstr;
//        trampoline += assemble("s_setpc_b64 s[2:3]", agent);
//
//        std::memcpy(reinterpret_cast<void*>(trampolineStartAddr), trampoline.data(), trampoline.size());
//        std::memcpy(reinterpret_cast<void *>(instr.getDeviceAddress()), longJumpForTarget.data(), longJumpForTarget.size());
//    }
//
//#ifdef LUTHIER_LOG_ENABLE_DEBUG
////    fmt::println("content of the header: {}", std::string(reinterpret_cast<const char*>(instr.getDeviceAddress() - 0x1000), 0x750));
////    std::memset(reinterpret_cast<void*>(instr.getDeviceAddress() - 0x1000), 0, 0x750);
//    auto finalTargetInstructions =
//        luthier::Disassembler::Instance().disassemble(reinterpret_cast<luthier_address_t>(instr.getKernelDescriptor()));
//    fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Instrumented Kernel Final View:\n");
//
//    for (const auto& i: finalTargetInstructions) {
//        auto printFormat = instr.getDeviceAddress() <= i.getDeviceAddress() && (i.getDeviceAddress() + i.getSize()) <= (instr.getDeviceAddress() + instr.getSize()) ?
//                                                                                                                                                                    fmt::emphasis::underline :
//                                                                                                                                                                    fmt::emphasis::bold;
//        fmt::print(stdout, printFormat, "{:#x}: {:s}\n", i.getDeviceAddress(), i.getInstr());
//    }
//    auto finalInstrumentationInstructions =
//        luthier::Disassembler::Instance().disassemble(agent, instrmntTextSectionStart,
//                                                    dummyInstrmnt.size() + nopInstr.size() + trampoline.size());
//    fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::orange_red), "Instrumented Kernel Final View:\n");
//    for (const auto& i: finalInstrumentationInstructions) {
//        fmt::print(stdout, fmt::emphasis::bold, "{:#x}: {:s}\n", i.getHostAddress(), i.getInstr());
//    }
//#endif
//
//#ifdef LUTHIER_LOG_ENABLE_DEBUG
//    auto hostInstructions =
//        luthier::Disassembler::Instance().disassemble(agent, luthier::co_manip::getHostLoadedCodeObjectOfExecutable(instr.getExecutable(), agent)[0].data + 0x1000, 0x54);
//    fmt::print(stdout, fmt::emphasis::bold | fg(fmt::color::aquamarine), "Host Executable:\n");
//
//    for (const auto& i: hostInstructions) {
//        auto printFormat = instr.getDeviceAddress() <= i.getHostAddress() && (i.getHostAddress() + i.getSize()) <= (instr.getDeviceAddress() + instr.getSize()) ?
//                                                                                                                                                                    fmt::emphasis::underline :
//                                                                                                                                                                    fmt::emphasis::bold;
//        fmt::print(stdout, printFormat, "{:#x}: {:s}\n", i.getHostAddress(), i.getInstr());
//    }
//#endif
//
//
//
//
//
//    LUTHIER_LOG_FUNCTION_CALL_END
//}
