#include "code_generator.hpp"

#include <fmt/color.h>
#include <fmt/core.h>
#include <hsa/amd_hsa_common.h>
#include <hsa/hsa_ext_amd.h>

#include <memory>

#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "code_object_manager.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "elfio/elfio.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_isa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "log.hpp"

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_AVAILABLE_OPCODE_CHECKER
//#define ENABLE_INSTR_PREDICATE_VERIFIER
//#define GET_INSTRINFO_MC_DESC
//#define GET_INSTRINFO_CTOR_DTOR
//#define GET_INSTRINFO_HEADER
#include "AMDGPUGenInstrInfo.inc"
#include "fmt/ranges.h"
#include "llvm/BinaryFormat/ELF.h"

luthier::byte_string_t luthier::CodeGenerator::compileRelocatableToExecutable(const luthier::byte_string_t &code,
                                                                              const hsa::GpuAgent &agent) {
    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    llvm::SmallVector<hsa::Isa, 1> isa;
    agent.getIsa(isa);

    std::string isaName = isa[0].getName();

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
        amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, dataAction, dataSetIn, dataSetOut));

    amd_comgr_data_t dataOut;
    size_t dataOutSize;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_data_get_data(dataSetOut, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &dataOut));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(dataOut, &dataOutSize, nullptr));
    luthier::byte_string_t executableOut;
    executableOut.resize(dataOutSize);
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(dataOut, &dataOutSize, reinterpret_cast<char *>(executableOut.data())));
    return executableOut;
}

luthier::byte_string_t luthier::CodeGenerator::assembleToRelocatable(const std::string &instList,
                                                                     const hsa::GpuAgent &agent) {

    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    llvm::SmallVector<hsa::Isa, 1> isa;
    agent.getIsa(isa);

    std::string isaName = isa[0].getName();

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataIn));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(dataIn, instList.size(), instList.data()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(dataIn, "my_source.s"));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_data_set_add(dataSetIn, dataIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetOut));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_action_info(&dataAction));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_isa_name(dataAction, isaName.c_str()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_option_list(dataAction, nullptr, 0));
    LUTHIER_AMD_COMGR_CHECK(
        amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, dataAction, dataSetIn, dataSetOut));
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
    luthier::byte_string_t outElf;
    outElf.resize(dataOutSize);
    amd_comgr_get_data(dataOut, &dataOutSize, reinterpret_cast<char *>(outElf.data()));
    auto outView = code::ElfView::makeView(outElf);
    return outElf;
}

luthier::byte_string_t luthier::CodeGenerator::assembleToRelocatable(const std::vector<std::string> &instList,
                                                                     const hsa::GpuAgent &agent) {
    std::string instString = fmt::format("{}", fmt::join(instList, "\n"));
    return assembleToRelocatable(instString, agent);
}

luthier::byte_string_t luthier::CodeGenerator::assemble(const std::string &instList, const hsa::GpuAgent &agent) {
    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    llvm::SmallVector<hsa::Isa, 1> isa;
    agent.getIsa(isa);

    std::string isaName = isa[0].getName();

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataIn));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(dataIn, instList.size(), instList.data()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(dataIn, "my_source.s"));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_data_set_add(dataSetIn, dataIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetOut));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_action_info(&dataAction));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_isa_name(dataAction, isaName.c_str()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_option_list(dataAction, nullptr, 0));
    LUTHIER_AMD_COMGR_CHECK(
        amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, dataAction, dataSetIn, dataSetOut));
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
    luthier::byte_string_t outElf;
    outElf.resize(dataOutSize);
    amd_comgr_get_data(dataOut, &dataOutSize, reinterpret_cast<char *>(outElf.data()));
    auto outView = code::ElfView::makeView(outElf);
    return outElf;
}

luthier::byte_string_t luthier::CodeGenerator::assemble(const std::vector<std::string> &instrVector,
                                                        const hsa::GpuAgent &agent) {
    return assembleToRelocatable(instrVector, agent);
}

luthier::CodeGenerator::CodeGenerator() {
    const auto &contextManager = luthier::ContextManager::instance();
    llvm::SmallVector<hsa::GpuAgent, 8> hsaAgents;

    hsa::getGpuAgents(hsaAgents);

    // Find all the unique ISAs supported by the Agents attached to the system
    std::unordered_set<hsa::Isa> uniqueISAs;

    for (const auto &a: hsaAgents) {
        llvm::SmallVector<hsa::Isa, 1> isaList;
        a.getIsa(isaList);
        for (const auto &isa: isaList) { uniqueISAs.insert(isa); }
    }

    // For each unique ISA, identify the supported instructions, and split the target-agnostic and target-specific
    // opcodes to match them later
    for (const auto &isa: uniqueISAs) {
        const auto &targetInfo = contextManager.getLLVMTargetInfo(isa);
        llvm::FeatureBitset subTargetFeatures = targetInfo.getMCSubTargetInfo()->getFeatureBits();
        llvm::StringMap<unsigned int> nonTargetOpcodes;
        llvm::StringMap<unsigned int> targetOpcodes;
        for (unsigned int opCode = 0; opCode < targetInfo.getMCInstrInfo()->getNumOpcodes(); opCode++) {
            llvm::StringRef opName = targetInfo.getMCInstrInfo()->getName(opCode);
            if (llvm::AMDGPU_MC::isOpcodeAvailable(opCode, subTargetFeatures)) {
                llvm::SmallVector<llvm::StringRef> splitName;
                bool isNonTarget = true;
                opName.split(splitName, "_", -1, false);
                for (auto it = splitName.begin(); it != splitName.end();) {
                    if (it->contains("gfx") || it->contains("vi")) {
                        isNonTarget = false;
                        it = splitName.erase(it);
                    } else
                        it++;
                }
                auto opNameWithNoSep = llvm::join(splitName, "_");
                if (isNonTarget) {
                    nonTargetOpcodes.insert({opNameWithNoSep, opCode});
                } else {
                    targetOpcodes.insert({opNameWithNoSep, opCode});
                }
            }
        }

        llvmTargetInstructions_.insert({isa, {}});

        auto &opcodeMap = llvmTargetInstructions_[isa];

        // Target-specific opcodes are the ones actually supported by the code emitter
        for (const auto &[opName, opCode]: targetOpcodes) {
            if (nonTargetOpcodes.contains(opName)) {
                opcodeMap.insert({nonTargetOpcodes[opName], opCode});
            } else {
                // If there are no target-agnostic enums, then push the target-specific enum
                opcodeMap.insert({opCode, opCode});
            }
        }
    }

    for (const auto &agent: hsaAgents) {
        //        auto emptyRelocatable = assembleToRelocatable("s_nop 0", agent);
        //        emptyRelocatableMap_.insert({agent, emptyRelocatable});
    }
}

void luthier::CodeGenerator::instrument(hsa::Instr &instr, const void *deviceFunc, luthier_ipoint_t point) {
    LUTHIER_LOG_FUNCTION_CALL_START
    auto agent = instr.getAgent();
    auto &codeObjectManager = luthier::CodeObjectManager::instance();
    auto &contextManager = luthier::ContextManager::instance();

    const auto &targetInfo = contextManager.getLLVMTargetInfo(agent.getIsa());
//    auto Context = std::make_unique<llvm::LLVMContext>();
//    auto triple = agent.getIsa().getLLVMTargetTriple();
//    auto processor = agent.getIsa().getProcessor();
//    auto featureString = agent.getIsa().getFeatureString();
//    //    auto targetOptions = std::make_unique<llvm::TargetOptions>();
//    auto TheTargetMachine = std::unique_ptr<llvm::LLVMTargetMachine>(
//        reinterpret_cast<llvm::LLVMTargetMachine *>(targetInfo.getTarget()->createTargetMachine(
//            triple, processor, featureString, *targetInfo.getTargetOptions(), llvm::Reloc::Model::PIC_)));
//    //
//    std::unique_ptr<llvm::Module> Module = std::make_unique<llvm::Module>("MyModule", *Context);
//    Module->setDataLayout(TheTargetMachine->createDataLayout());
//    auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TheTargetMachine.get());
    //    llvm::Function f;
    //    llvm::MachineFunction func;
    auto genInst = makeInstruction(
        agent.getIsa(), llvm::AMDGPU::S_ADD_I32, llvm::MCOperand::createReg(llvm::AMDGPU::VGPR3),
        llvm::MCOperand::createReg(llvm::AMDGPU::SGPR6), llvm::MCOperand::createReg(llvm::AMDGPU::VGPR1));

    const auto targetTriple = llvm::Triple(agent.getIsa().getLLVMTargetTriple());
    llvm::MCContext ctx(targetTriple, targetInfo.getMCAsmInfo(), targetInfo.getMCRegisterInfo(),
                        targetInfo.getMCSubTargetInfo());
    auto mcEmitter = targetInfo.getTarget()->createMCCodeEmitter(*targetInfo.getMCInstrInfo(), ctx);

    llvm::SmallVector<char> osBack;
    llvm::SmallVector<llvm::MCFixup> fixUps;
    mcEmitter->encodeInstruction(genInst, osBack, fixUps, *targetInfo.getMCSubTargetInfo());
    fmt::print("Assembled instruction :");
    for (auto s: osBack) { fmt::print("{:#x}", s); }
    fmt::print("\n");

    auto oneMoreTime = Disassembler::instance().disassemble(
        agent.getIsa(), {reinterpret_cast<std::byte *>(osBack.data()), osBack.size()});

    hsa::ExecutableSymbol instrumentationFunc = codeObjectManager.getInstrumentationKernel(deviceFunc, agent);

    //    const std::vector<hsa::Instr> *instFunctionInstructions = Disassembler::instance().disassemble(instrumentationFunc);

    const std::vector<hsa::Instr> *targetFunction = Disassembler::instance().disassemble(instr.getExecutableSymbol());

    for (const auto &i: *targetFunction) {
        std::string instStr;
        llvm::raw_string_ostream instStream(instStr);
        auto inst = i.getInstr();
        fmt::println("{}", targetInfo.getMCInstrInfo()->getName(inst.getOpcode()).str());
        //        fmt::println("Is call? {}", targetInfo.MII_->get(inst.getOpcode()).isCall());
        //        fmt::println("Is control flow? {}",
        //                     targetInfo.MII_->get(inst.getOpcode()).mayAffectControlFlow(inst, *targetInfo.MRI_));
        //        fmt::println("Num max operands? {}", targetInfo.MII_->get(inst.getOpcode()).getNumOperands());
        fmt::println("Num operands? {}", inst.getNumOperands());
        //        fmt::println("May load? {}", targetInfo.MII_->get(inst.getOpcode()).mayLoad());
        fmt::println("Mnemonic: {}", targetInfo.getMCInstPrinter()->getMnemonic(&inst).first);

        for (int j = 0; j < inst.getNumOperands(); j++) {
            //            auto op = inst.getOperand(j);
            //            std::string opStr;
            //            llvm::raw_string_ostream opStream(opStr);
            //            op.print(opStream, targetInfo.MRI_.get());
            //            fmt::println("OP: {}", opStream.str());
            //            if (op.isReg()) {
            //                fmt::println("Reg idx: {}", op.getReg());
            //                fmt::println("Reg name: {}", targetInfo.MRI_->getName(op.getReg()));
            //                auto subRegIterator = targetInfo.MRI_->subregs(op.getReg());
            //                for (auto it = subRegIterator.begin(); it != subRegIterator.end(); it++) {
            //                    fmt::println("\tSub reg Name: {}", targetInfo.MRI_->getName((*it)));
            //                }
            ////                auto superRegIterator = targetInfo.MRI_->superregs(op.getReg());
            ////                for (auto it = superRegIterator.begin(); it != superRegIterator.end(); it++) {
            ////                    fmt::println("\tSuper reg Name: {}", targetInfo.MRI_->getName((*it)));
            ////                }
            //            }
            //            fmt::println("Is op {} reg? {}", j, op.isReg());
            //            fmt::println("Is op {} valid? {}", j, op.isValid());
        }
        targetInfo.getMCInstPrinter()->printInst(&inst, reinterpret_cast<luthier_address_t>(inst.getLoc().getPointer()),
                                                 "", *targetInfo.getMCSubTargetInfo(), instStream);
        fmt::println("{}", instStream.str());
        inst.getOpcode();
    }
    fmt::println("==================================");
    //
    //    for (const auto &i: *targetFunction) {
    //        std::string instStr;
    //        llvm::raw_string_ostream instStream(instStr);
    //        auto inst = i.getInstr();
    //        fmt::println("{}", targetInfo.MII_->getName(inst.getOpcode()).str());
    ////        fmt::println("Is call? {}", targetInfo.MII_->get(inst.getOpcode()).isCall());
    ////        fmt::println("Is control flow? {}",
    ////                     targetInfo.MII_->get(inst.getOpcode()).mayAffectControlFlow(inst, *targetInfo.MRI_));
    //        targetInfo.IP_->printInst(&inst, reinterpret_cast<luthier_address_t>(inst.getLoc().getPointer()), "",
    //                                  *targetInfo.STI_, instStream);
    //        fmt::println("{}", instStream.str());
    //        inst.getOpcode();
    //    }

    auto targetExecutable = instr.getExecutable();

    auto symbol = instr.getExecutableSymbol();
    std::string symbolName = symbol.getName();

    auto storage = targetExecutable.getLoadedCodeObjects()[0].getStorageMemory();
    auto instrumentedElfView = code::ElfView::makeView(storage);

    // Find the symbol that requires instrumentation.
    std::optional<code::SymbolView> storageSymbol = instrumentedElfView->getSymbol(symbolName);
    if (!storageSymbol.has_value())
        throw std::runtime_error(fmt::format("Failed to find symbol {} in the copied executable", symbolName));
    auto meta = storageSymbol->getMetaData();
    fmt::println("Number of SGPRS: {}", meta.usedSGPRs_);
    fmt::println("Number of VGPRS: {}", meta.usedVGPRs_);

    std::unique_ptr<llvm::MCObjectFileInfo> MOFI(
        targetInfo.getTarget()->createMCObjectFileInfo(ctx, /*PIC*/ true, /*large code model*/ false));
    auto textSection = MOFI->getTextSection();
    fmt::println("Does the text section have instructions? {}", textSection->hasInstructions());
    ctx.setObjectFileInfo(MOFI.get());
    //    ctx->setAllowTemporaryLabels(true);
    ctx.setGenDwarfForAssembly(false);

    llvm::SmallVector<char> out;

    llvm::raw_svector_ostream VOS(out);

    llvm::MCCodeEmitter *CE = targetInfo.getTarget()->createMCCodeEmitter(*targetInfo.getMCInstrInfo(), ctx);
    llvm::MCAsmBackend *MAB = targetInfo.getTarget()->createMCAsmBackend(
        *targetInfo.getMCSubTargetInfo(), *targetInfo.getMCRegisterInfo(), targetInfo.getTargetOptions()->MCOptions);

    auto Str = std::unique_ptr<llvm::MCStreamer>(targetInfo.getTarget()->createMCObjectStreamer(
        llvm::Triple(agent.getIsa().getLLVMTargetTriple()), ctx, std::unique_ptr<llvm::MCAsmBackend>(MAB),
        MAB->createObjectWriter(VOS), std::unique_ptr<llvm::MCCodeEmitter>(CE), *targetInfo.getMCSubTargetInfo(), true,
        false,
        /*DWARFMustBeAtTheEnd*/ false));

    //    Str->initSections(false, *targetInfo.STI_);
    Str->switchSection(ctx.getObjectFileInfo()->getTextSection());
    Str->emitCodeAlignment(llvm::Align(ctx.getObjectFileInfo()->getTextSectionAlignment()),
                           targetInfo.getMCSubTargetInfo());

    for (const auto &inst: *targetFunction) { Str->emitInstruction(inst.getInstr(), *targetInfo.getMCSubTargetInfo()); }
    auto kernelName = instr.getExecutableSymbol().getName();
    //
    auto kName = kernelName.substr(0, kernelName.find(".kd"));

    auto KernelDescriptor = instr.getExecutableSymbol().getKernelDescriptor();
    //
    llvm::MCSymbolELF *KernelCodeSymbol = cast<llvm::MCSymbolELF>(ctx.getOrCreateSymbol(llvm::Twine(kName)));

    //    Str->pushSection();
    Str->switchSection(ctx.getObjectFileInfo()->getReadOnlySection());

    Str->emitValueToAlignment(llvm::Align(64), 0, 1, 0);
    ctx.getObjectFileInfo()->getReadOnlySection()->ensureMinAlignment(llvm::Align(64));

    llvm::MCSymbolELF *KernelDescriptorSymbol = cast<llvm::MCSymbolELF>(ctx.getOrCreateSymbol(kernelName));

    // Copy kernel descriptor symbol's binding, other and visibility from the
    // kernel code symbol.
    KernelDescriptorSymbol->setBinding(KernelCodeSymbol->getBinding());
    KernelDescriptorSymbol->setOther(KernelCodeSymbol->getOther());
    KernelDescriptorSymbol->setVisibility(KernelCodeSymbol->getVisibility());
    // Kernel descriptor symbol's type and size are fixed.
    KernelDescriptorSymbol->setType(llvm::ELF::STT_OBJECT);
    KernelDescriptorSymbol->setSize(llvm::MCConstantExpr::create(sizeof(hsa::KernelDescriptor), ctx));

    //    // The visibility of the kernel code symbol must be protected or less to allow
    //    // static relocations from the kernel descriptor to be used.
    if (KernelCodeSymbol->getVisibility() == llvm::ELF::STV_DEFAULT)
        KernelCodeSymbol->setVisibility(llvm::ELF::STV_PROTECTED);
    //
    Str->emitLabel(KernelDescriptorSymbol);
    Str->emitInt32(KernelDescriptor->groupSegmentFixedSize);
    Str->emitInt32(KernelDescriptor->privateSegmentFixedSize);
    Str->emitInt32(KernelDescriptor->kernArgSize);
    //
    for (uint8_t Res: KernelDescriptor->reserved0) Str->emitInt8(Res);

    //    // FIXME: Remove the use of VK_AMDGPU_REL64 in the expression below. The
    //    // expression being created is:
    //    //   (start of kernel code) - (start of kernel descriptor)
    //    // It implies R_AMDGPU_REL64, but ends up being R_AMDGPU_ABS64.
    Str->emitValue(llvm::MCBinaryExpr::createSub(
                       llvm::MCSymbolRefExpr::create(KernelCodeSymbol, llvm::MCSymbolRefExpr::VK_AMDGPU_REL64, ctx),
                       llvm::MCSymbolRefExpr::create(KernelDescriptorSymbol, llvm::MCSymbolRefExpr::VK_None, ctx), ctx),
                   sizeof(KernelDescriptor->kernelCodeEntryByteOffset));
    for (uint8_t Res: KernelDescriptor->reserved1) Str->emitInt8(Res);
    Str->emitInt32(KernelDescriptor->computePgmRsrc3);
    Str->emitInt32(KernelDescriptor->computePgmRsrc1);
    Str->emitInt32(KernelDescriptor->computePgmRsrc2);
    Str->emitInt16(KernelDescriptor->kernelCodeProperties);
    Str->emitInt16(KernelDescriptor->kernArgPreload);
    for (uint8_t Res: KernelDescriptor->reserved2) Str->emitInt8(Res);

    //    Str->popSection();
    //    Str->switchSection(ctx.getObjectFileInfo()->getTextSection());

    fmt::println("Does the text section have instructions? {}", textSection->hasInstructions());

    Str->getTargetStreamer()->finish();
    Str->finish();
    Str->finishImpl();
    auto finalView =
        code::ElfView::makeView(luthier::byte_string_view(reinterpret_cast<std::byte *>(out.data()), out.size()));
    fmt::println("Length of the out vector: {}", out.size());
    fmt::println("Type of the code object created: {}", finalView->getElfIo().get_type());
    fmt::println("Number of symbols: {}", finalView->getNumSymbols());

    finalView->getElfIo().sections[0]->get_data();
    //    fmt::println("Symbol name : {}", finalView->getSymbol(0)->getName());
    //    fmt::println("Symbol name : {}", finalView->getSymbol(1)->getName());
    //    finalView->getSymbol(0)->getData();
    fmt::println("Data works");
    finalView->getElfIo().sections[1]->get_data();

    //    auto executable = compileRelocatableToExecutable({reinterpret_cast<std::byte*>(out.data()), out.size()}, instr.getAgent());
    //    delete Str;
    //    Str->finishImpl();

    //    targetStr->finish();

    CodeObjectManager::instance().loadInstrumentedKernel(luthier::byte_string_t(storage), instr.getExecutableSymbol());
}
