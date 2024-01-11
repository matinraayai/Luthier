#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "elfio/elfio.hpp"
#include "hsa_intercept.hpp"
#include "instr.hpp"
#include "log.hpp"
#include <fmt/color.h>
#include <fmt/xchar.h>
#include <fmt/core.h>
#include <hsa/amd_hsa_common.h>
#include <hsa/hsa_ext_amd.h>
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include <sstream>
#include "luthier.h"

struct TargetIdentifier {
    llvm::StringRef Arch;
    llvm::StringRef Vendor;
    llvm::StringRef OS;
    llvm::StringRef Environ;
    llvm::StringRef Processor;
    llvm::SmallVector<llvm::StringRef, 2> Features;
};

std::string getSymbolName(hsa_executable_symbol_t symbol) {
    const auto &coreHsaApiTable = luthier::HsaInterceptor::instance().getSavedHsaTables().core;
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

luthier::co_manip::code_t luthier::CodeGenerator::assemble(const std::string &instList, hsa_agent_t agent) {
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
//    auto isaName = luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName();
//
//    std::string Error;
//    const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget(isaName, Error);
//    assert(TheTarget);
//
//    std::unique_ptr<const llvm::MCRegisterInfo> MRI(TheTarget->createMCRegInfo(llvm::StringRef(isaName)));
//    assert(MRI);
//
//
//    llvm::MCTargetOptions MCOptions;
//    std::unique_ptr<const llvm::MCAsmInfo> MAI(
//        TheTarget->createMCAsmInfo(*MRI, isaName, MCOptions));
//
//    assert(MAI);
//
//    std::unique_ptr<const llvm::MCInstrInfo> MII(TheTarget->createMCInstrInfo());
//    assert(MII);
//
//    std::unique_ptr<const llvm::MCSubtargetInfo> STI(
//        TheTarget->createMCSubtargetInfo(isaName, "gfx908", "+sramecc-xnack"));
//    assert(STI);
//
//    // MatchAndEmitInstruction in MCTargetAsmParser.h
//
//
//    // Now that GetTarget() has (potentially) replaced TripleName, it's safe to
//    // construct the Triple object.
//    llvm::Triple TheTriple(isaName);
//
//    std::unique_ptr<llvm::MemoryBuffer> BufferPtr = llvm::MemoryBuffer::getMemBuffer(instListStr);
//
//    llvm::SourceMgr SrcMgr;
//
//    // Tell SrcMgr about this buffer, which is what the parser will pick up.
//    SrcMgr.AddNewSourceBuffer(std::move(BufferPtr), llvm::SMLoc());
//
//    // Package up features to be passed to target/subtarget
//    std::string FeaturesStr;
////    if (MAttrs.size()) {
////        SubtargetFeatures Features;
////        for (unsigned i = 0; i != MAttrs.size(); ++i)
////            Features.AddFeature(MAttrs[i]);
////        FeaturesStr = Features.getString();
////    }
//
////    std::unique_ptr<llvm::MCContext> Ctx(new (std::nothrow)
////                                             llvm::MCContext(llvm::Triple(isaName), MAI.get(), MRI.get(),
////                                                             &SrcMgr,
////                                                             &MCOptions,
////                                                             STI.get()));
////    assert(Ctx);
//
//    // FIXME: This is not pretty. MCContext has a ptr to MCObjectFileInfo and
//    // MCObjectFileInfo needs a MCContext reference in order to initialize itself.
//    llvm::MCContext Ctx(TheTriple, MAI.get(), MRI.get(), STI.get(), &SrcMgr,
//                  &MCOptions);
//    std::unique_ptr<llvm::MCObjectFileInfo> MOFI(
//        TheTarget->createMCObjectFileInfo(Ctx, /*PIC*/ true, /*large code model*/ false));
//    Ctx.setObjectFileInfo(MOFI.get());
//
//    Ctx.setAllowTemporaryLabels(false);
//
//    Ctx.setGenDwarfForAssembly(false);
//
//    llvm::SmallVector<char> out;
//
//    llvm::raw_svector_ostream VOS(out);
//
//    std::unique_ptr<llvm::buffer_ostream> BOS;
//
//    std::unique_ptr<llvm::MCStreamer> Str;
//
//    std::unique_ptr<llvm::MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
//    assert(MCII && "Unable to create instruction info!");
//
//    llvm::MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, Ctx);
//    llvm::MCAsmBackend *MAB = TheTarget->createMCAsmBackend(*STI, *MRI, MCOptions);
//    Str.reset(TheTarget->createMCObjectStreamer(
//        TheTriple, Ctx, std::unique_ptr<llvm::MCAsmBackend>(MAB),
//        MAB->createObjectWriter(VOS),
//        std::unique_ptr<llvm::MCCodeEmitter>(CE), *STI, MCOptions.MCRelaxAll,
//        MCOptions.MCIncrementalLinkerCompatible,
//        /*DWARFMustBeAtTheEnd*/ false));
//
////    Str->initSections(true, *STI);
//
//    // Use Assembler information for parsing.
//    Str->setUseAssemblerInfoForParsing(true);
//
//    std::unique_ptr<llvm::MCAsmParser> Parser(
//        llvm::createMCAsmParser(SrcMgr, Ctx, *Str, *MAI));
//    std::unique_ptr<llvm::MCTargetAsmParser> TAP(
//        TheTarget->createMCAsmParser(*STI, *Parser, *MCII, MCOptions));
//
//    assert(TAP && "this target does not support assembly parsing.\n");
//
////    int SymbolResult = fillCommandLineSymbols(*Parser);
////    if(SymbolResult)
////        return SymbolResult;
//    Parser->setShowParsedOperands(true);
//    Parser->setTargetParser(*TAP);
//    Parser->getLexer().setLexMasmIntegers(true);
//    Parser->getLexer().setLexMasmHexFloats(true);
//    Parser->getLexer().setLexMotorolaIntegers(true);
//
//    int Res = Parser->Run(false);
//
//    return {reinterpret_cast<std::byte*>(out.data()), out.size()};
}

luthier::co_manip::code_t luthier::CodeGenerator::assemble(const std::vector<std::string> &instrVector, hsa_agent_t agent) {
    auto relocatable = assembleToRelocatable(instrVector, agent);
    co_manip::code_char_stream_t coStream(
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

uint64_t luthier::CodeGenerator::allocateGlobalSpace(int numGPRToSave,uint32_t gridSize){
    numRegisters = numGPRToSave;
    allocatedSize = gridSize * numRegisters * 4;
    auto hipMallocFunc = reinterpret_cast<hipError_t (*)(void **, size_t)>(luthier_get_hip_function("hipMalloc"));
    (*hipMallocFunc)(&saved_register, allocatedSize);
    std::cout << "hip allocate " << allocatedSize << " bytes at address " << saved_register << " for me to save registers\n";
}

luthier::CodeGenerator::~CodeGenerator(){
    int* savedRegisterHost = new int[allocatedSize];
//    int savedRegisterHost[allocatedSize];
    reinterpret_cast<hipError_t (*)(void *, void *, size_t, hipMemcpyKind)>(luthier_get_hip_function("hipMemcpy"))(
        savedRegisterHost, saved_register, allocatedSize, hipMemcpyDeviceToHost);
    int gridSize = allocatedSize / (4*numRegisters);
    for (int vindex = 0; vindex <= 0; vindex++) {
        int offset = vindex * gridSize;
        for (int i = 0; i < gridSize; i++) {
            std::cout << "v" << vindex << ":wi " << i << " value " << savedRegisterHost[i + offset] << std::endl;
        }
    }
    auto hipFreeFunc = reinterpret_cast<hipError_t (*)(void **)>(luthier_get_hip_function("hipFree"));
    (*hipFreeFunc)(&saved_register);
    delete[] savedRegisterHost;
}

hsa_status_t registerSymbolWithCodeObjectManager(const hsa_executable_t &executable,
                                                 const hsa_executable_symbol_t originalSymbol,
                                                 hsa_agent_t agent) {

    hsa_status_t out = HSA_STATUS_ERROR;
    auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data) {
        auto originalSymbol = reinterpret_cast<hsa_executable_symbol_t *>(data);
        auto originalSymbolName = getSymbolName(*originalSymbol);

        auto &coreTable = luthier::HsaInterceptor::instance().getSavedHsaTables().core;
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
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(*originalSymbol,
                                                                          HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &originalKernelObject));
            luthier::CodeObjectManager::instance().registerInstrumentedKernel(reinterpret_cast<kernel_descriptor_t *>(originalKernelObject),
                                                                              reinterpret_cast<kernel_descriptor_t *>(kernelObject));
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
            const auto &amdTable = luthier::HsaInterceptor::instance().getHsaVenAmdLoaderTable();
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
    auto &codeObjectManager = luthier::CodeObjectManager::instance();
    luthier::co_manip::code_view_t instrumentationFunc = codeObjectManager.getInstrumentationFunction(device_func, agent);
    kernel_descriptor_t *instrumentationFuncKD = codeObjectManager.getKernelDescriptorOfInstrumentationFunction(device_func, agent);
    hsa_executable_t targetExecutable = instr.getExecutable();

    hsa_executable_symbol_t symbol = instr.getSymbol();
    std::string symbolName = getSymbolName(symbol);

    auto hco = co_manip::getHostLoadedCodeObjectOfExecutable(targetExecutable, agent);
    auto instrumentedElfView = co_manip::ElfViewImpl::makeView(hco[0]);

    auto newElfio = co_manip::createAMDGPUElf(instrumentedElfView->getElfIo(), agent);

    // Find the symbol that requires instrumentation.
    bool foundKDSymbol{false};
    for (unsigned int i = 0; i < co_manip::getSymbolNum(instrumentedElfView) && !foundKDSymbol; i++) {
        co_manip::SymbolView info(instrumentedElfView, i);
        if (info.getName() == symbolName)
            foundKDSymbol = true;
    }
    if (not foundKDSymbol)
        throw std::runtime_error(fmt::format("Failed to find symbol {} in the copied executable", symbolName));
    fmt::println("SGPR granularity: {}", luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getSGPRAllocGranulefromComgrMeta());
    auto meta = GetAttrCodePropMetadata(instrumentedElfView, instrumentedElfView->getKernelMetaDataMap(symbolName));
    fmt::println("Number of SGPRS: {}", meta.usedSGPRs_);
    fmt::println("Number of VGPRS: {}", meta.usedVGPRs_);
    auto myReloc = emptyRelocatableMap_[agent.handle];
    auto out = compileRelocatableToExecutable(myReloc, agent);

    co_manip::code_t newCodeObject(hco[0]);

    co_manip::code_char_stream_t newCOStream(
        boost_ios::stream<boost_ios::basic_array_source<char>>(
            std::string_view(reinterpret_cast<const char *>(newCodeObject.data()), newCodeObject.size()).begin(),
            std::string_view(reinterpret_cast<const char *>(newCodeObject.data()), newCodeObject.size()).end()));
    ELFIO::elfio elfio;
    elfio.load(newCOStream, false);
    std::cout << "Create a elfio object using code_char_stream_t\n";

    ELFIO::symbol_section_accessor symbols(elfio, elfio.sections[".symtab"]);
    for (unsigned int i = 0; i < symbols.get_symbols_num(); i++) {
        std::string name;
        ELFIO::Elf64_Addr value;
        size_t size;
        unsigned char bind;
        unsigned char type;
        ELFIO::Elf_Half sec_index;
        unsigned char other;

        bool ret = symbols.get_symbol(i, name, value, size, bind, type,
                                      sec_index, other);
        if (!ret) {
            throw std::runtime_error(fmt::format("Failed to get symbol info for index {}.", i));
        }
        if (name.find("kd") != std::string::npos) {
            auto kd = reinterpret_cast<kernel_descriptor_t *>(newCodeObject.data() + (size_t) value);
            fmt::println("Symbol Info: {}, {}, {:#x}", name, size,
                         reinterpret_cast<luthier_address_t>(kd));
            auto VgprCount = AMD_HSA_BITS_GET(kd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT);
            fmt::println("AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT: {}", VgprCount);
            auto SgprCount = AMD_HSA_BITS_GET(kd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT);
            fmt::println("AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT: {}", SgprCount);
            AMD_HSA_BITS_SET(kd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT, 3);
            AMD_HSA_BITS_SET(kd->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT, 5);
            co_manip::printRSR1(kd);
            // luthier::co_manip::printCodeProperties(kd);
        }
    }
    co_manip::code_char_stream_t newCOStream_mkd(
        boost_ios::stream<boost_ios::basic_array_source<char>>(
            std::string_view(reinterpret_cast<const char *>(newCodeObject.data()), newCodeObject.size()).begin(),
            std::string_view(reinterpret_cast<const char *>(newCodeObject.data()), newCodeObject.size()).end()));
    ELFIO::elfio elfio_mkd;
    elfio_mkd.load(newCOStream_mkd, false);
    std::cout << "Create a elfio object using code_char_stream_t after modifying kd\n";

    co_manip::code_t myNewProg = assemble(std::vector<std::string>{
                                              "s_add_u32 s0, s0, s11",
                                              "s_addc_u32 s1, s1, 0",
                                              "s_load_dword s40, s[4:5], 0x4",
                                           "s_waitcnt lgkmcnt(0)",
                                           "s_and_b32 s40, s40, 0xffff",
                                           "s_mul_i32 s8, s8, s40",
                                           "v_add_u32_e32 v5, s8, v0",
                                           "v_mov_b32_e32 v4, 0",
                                           "v_ashrrev_i64 v[4:5], 30, v[4:5]",
                                          },
                                          agent);
    uint32_t offsetNeeded = 0;
    fmt::println("s_add_u32 s39, 0, {}", offsetNeeded);
    myNewProg += assemble(std::vector<std::string>{fmt::format("s_add_u32 s39, 0, {}", offsetNeeded)}, agent);

    myNewProg += assemble(std::vector<std::string>{
                              "buffer_store_dwordx4 v[0:3], off, s[0:3], s39",
                              "buffer_store_dword v4, off, s[0:3], s39 offset:16","v_writelane_b32 v5, s0, 0", "v_writelane_b32 v5, s1, 1", "v_writelane_b32 v5, s2, 2", "v_writelane_b32 v5, s3, 3", "v_writelane_b32 v5, s4, 4", "v_writelane_b32 v5, s5, 5", "v_writelane_b32 v5, s6, 6", "v_writelane_b32 v5, s7, 7", "v_writelane_b32 v5, s8, 8", "v_writelane_b32 v5, s9, 9", "v_writelane_b32 v5, s10, 10", "v_writelane_b32 v5, s11, 11", "v_writelane_b32 v5, s12, 12", "v_writelane_b32 v5, s13, 13", "buffer_store_dword v5, off, s[0:3], s39 offset:20","s_waitcnt vmcnt(0)"},
                          agent);
    myNewProg += assemble(std::vector<std::string>{"buffer_load_dwordx4 v[6:9], off, s[0:3], s39", "buffer_load_dword v10, off, s[0:3], s39 offset:16", "buffer_load_dword v11, off, s[0:3], s39 offset:20","s_waitcnt vmcnt(0)"}, agent);

    int vCount = 6;// num
    int start = 6, end = 11;
    int gridSize = 128; // dispatch.grid_size_x
    for (int i = start; i <= end; i++) {
        uint64_t offset = (i-start) * gridSize * 4;
        std::cout << "offset is " << offset << std::endl;

        constexpr uint64_t upperMaskUint64_t = 0xFFFFFFFF00000000;
        constexpr uint64_t lowerMaskUint64_t = 0x00000000FFFFFFFF;
        uint64_t baseAddr = (uint64_t) saved_register + offset;
        uint32_t upperBaseAddr = (uint32_t) ((baseAddr & upperMaskUint64_t) >> 32);
        uint32_t lowerBaseAddr = (uint32_t) (baseAddr & lowerMaskUint64_t);

        fmt::println("s_add_u32 s40, 0, {:#x}", lowerBaseAddr);
        fmt::println("s_add_u32 s41, 0, {:#x}", upperBaseAddr);
        fmt::println("global_store_dword v[12:13], v{}, off", i);

        myNewProg += assemble(std::vector<std::string>{fmt::format("s_add_u32 s40, 0, {:#x}", lowerBaseAddr)}, agent);
        myNewProg += assemble(std::vector<std::string>{fmt::format("s_add_u32 s41, 0, {:#x}", upperBaseAddr)}, agent);
        myNewProg += assemble(std::vector<std::string>{"v_mov_b32_e32 v13, s41", "v_add_co_u32_e32 v12, vcc, s40, v4", "v_addc_co_u32_e32 v13, vcc, v13, v5, vcc"}, agent);
        myNewProg += assemble(std::vector<std::string>{fmt::format("global_store_dword v[12:13], v{}, off", i)}, agent);
    }
    myNewProg += assemble(std::vector<std::string>{"s_endpgm"}, agent);

    elfio_mkd.sections[".text"]->set_data(reinterpret_cast<char *>(myNewProg.data()), myNewProg.size());
    // elfio_mkd.sections[".text"]->set_data(reinterpret_cast<char *>(myNewProg.data()), myNewProg.size());
    std::cout << myNewProg.size() << std::endl;
    std::cout << elfio_mkd.sections[".text"]->get_offset() << std::endl;

    // save the ELF and create an executable
    std::ostringstream ss;
    elfio_mkd.save(ss);
    std::string elf = ss.str();
    out = co_manip::code_t{reinterpret_cast<std::byte*>(elf.data()),elf.size()};


    auto coreTable = HsaInterceptor::instance().getSavedHsaTables().core;
    hsa_code_object_reader_t reader;
    hsa_executable_t executable;
    LUTHIER_HSA_CHECK(coreTable.hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                                             nullptr, &executable));
    LUTHIER_HSA_CHECK(coreTable.hsa_code_object_reader_create_from_memory_fn(out.data(),
                                                                             out.size(), &reader));
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
//        const auto& amdExtApi = luthier::HsaInterceptor::instance()().getSavedHsaTables().amd_ext;
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
