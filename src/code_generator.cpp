#include "code_generator.hpp"

#include <memory>

#include <AMDGPUResourceUsageAnalysis.h>
#include <AMDGPUTargetMachine.h>
#include <amd_comgr/amd_comgr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/LegacyPassManager.h>

#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "error.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_isa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "log.hpp"
#include "target_manager.hpp"
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

// #define GET_REGINFO_ENUM
// #include "AMDGPUGenRegisterInfo.inc"
//
// #define GET_SUBTARGETINFO_ENUM
// #include "AMDGPUGenSubtargetInfo.inc"
//
// #define GET_INSTRINFO_ENUM
// #define GET_AVAILABLE_OPCODE_CHECKER

#include "AMDGPUGenInstrInfo.inc"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Instructions.h"

namespace luthier {

llvm::Error CodeGenerator::compileRelocatableToExecutable(
    const llvm::ArrayRef<uint8_t> &code, const hsa::ISA &isa,
    llvm::SmallVectorImpl<uint8_t> &out) {
  amd_comgr_data_t dataIn;
  amd_comgr_data_set_t dataSetIn, dataSetOut;
  amd_comgr_action_info_t dataAction;

  auto isaName = isa.getName();
  LUTHIER_RETURN_ON_ERROR(isaName.takeError());

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_create_data_set(&dataSetIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_set_data(
      dataIn, code.size(), reinterpret_cast<const char *>(code.data()))));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_set_data_name(dataIn, "source.o"))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_data_set_add(dataSetIn, dataIn))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_data_set(&dataSetOut))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_action_info(&dataAction))));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_isa_name(dataAction, isaName->c_str()))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_option_list(dataAction, nullptr, 0))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                           dataAction, dataSetIn, dataSetOut))));

  amd_comgr_data_t dataOut;
  size_t dataOutSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_action_data_get_data(
          dataSetOut, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &dataOut))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_get_data(dataOut, &dataOutSize, nullptr))));
  out.resize(dataOutSize);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_get_data(
      dataOut, &dataOutSize, reinterpret_cast<char *>(out.data())))));

  return llvm::Error::success();
}

llvm::Error CodeGenerator::compileRelocatableToExecutable(
    const llvm::ArrayRef<uint8_t> &code, const hsa::GpuAgent &agent,
    llvm::SmallVectorImpl<uint8_t> &out) {
  auto Isa = agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());
  return compileRelocatableToExecutable(code, *Isa, out);
}

llvm::Error luthier::CodeGenerator::instrument(hsa::Instr &instr,
                                               const void *deviceFunc,
                                               luthier_ipoint_t point) {
  LUTHIER_LOG_FUNCTION_CALL_START
  auto Agent = instr.getAgent();
  auto &CodeObjectManager = luthier::CodeObjectManager::instance();
  auto &contextManager = luthier::TargetManager::instance();

  auto Isa = Agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());

  auto targetInfo = contextManager.getTargetInfo(*Isa);
  LUTHIER_RETURN_ON_ERROR(targetInfo.takeError());

  llvm::SmallVector<char> reloc;
  llvm::SmallVector<uint8_t> executable;

  auto KernelModuleInfo = luthier::CodeLifter::instance().liftKernelModule(
      instr.getExecutableSymbol());
  LUTHIER_RETURN_ON_ERROR(KernelModuleInfo.takeError());
  auto Module = std::move(std::get<0>(*KernelModuleInfo));
  auto MMIWP = std::move(std::get<1>(*KernelModuleInfo));
  llvm::MCContext &MCContext = MMIWP->getMMI().getContext();

  auto TM = targetInfo->getTargetMachine();

  llvm::legacy::PassManager PM;

  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module->getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  llvm::TargetPassConfig *TPC = TM->createPassConfig(PM);
  PM.add(TPC);
  PM.add(MMIWP.release());
  //        TPC->printAndVerify("MachineFunctionGenerator::assemble");

  //        auto usageAnalysis =
  //        std::make_unique<llvm::AMDGPUResourceUsageAnalysis>();
  PM.add(new llvm::AMDGPUResourceUsageAnalysis());
  // Add target-specific passes.
  //        ET.addTargetSpecificPasses(PM);
  //        TPC->printAndVerify("After
  //        ExegesisTarget::addTargetSpecificPasses");
  // Adding the following passes:
  // - postrapseudos: expands pseudo return instructions used on some targets.
  // - machineverifier: checks that the MachineFunction is well formed.
  // - prologepilog: saves and restore callee saved registers.
  //        for (const char *PassName :
  //             {"postrapseudos", "machineverifier", "prologepilog"})
  //            if (addPass(PM, PassName, *TPC))
  //                return make_error<Failure>("Unable to add a mandatory
  //                pass");
  TPC->setInitialized();

  //        llvm::SmallVector<char> o;
  //        std::string out;
  //        llvm::raw_string_ostream outOs(out);
  llvm::raw_svector_ostream OutOS(reloc);

  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM->addAsmPrinter(PM, OutOS, nullptr, llvm::CodeGenFileType::AssemblyFile,
                        MCContext))
    llvm::outs() << "Failed to add pass manager\n";
  //            return make_error<llvm::Failure>("Cannot add AsmPrinter
  //            passes");

  PM.run(*Module); // Run all the passes

  llvm::outs() << reloc << "\n";

  LUTHIER_RETURN_ON_ERROR(compileRelocatableToExecutable(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(reloc.data()),
                              reloc.size()),
      *Isa, executable));
  llvm::outs() << "Compiled to executable\n";

  LUTHIER_RETURN_ON_ERROR(CodeObjectManager.loadInstrumentedKernel(
      executable, instr.getExecutableSymbol()));

  //  llvm::Expected<const std::vector<hsa::Instr> *> InstFunction =
  //      CodeLifter::instance().disassemble( instr.getExecutableSymbol());
  //    for (const auto& i: *targetFunction) {
  //        const unsigned Opcode = i.getInstr().getOpcode();
  //        const llvm::MCInstrDesc &MCID =
  //        targetInfo.getMCInstrInfo()->get(Opcode); llvm::MachineInstrBuilder
  //        Builder = llvm::BuildMI(MBB, DL, MCID); for (unsigned OpIndex = 0, E
  //        = i.getInstr().getNumOperands(); OpIndex < E;
  //             ++OpIndex) {
  //            const MCOperand &Op = Inst.getOperand(OpIndex);
  //            if (Op.isReg()) {
  //                const bool IsDef = OpIndex < MCID.getNumDefs();
  //                unsigned Flags = 0;
  //                const MCOperandInfo &OpInfo =
  //                MCID.operands().begin()[OpIndex]; if (IsDef &&
  //                !OpInfo.isOptionalDef())
  //                    Flags |= RegState::Define;
  //                Builder.addReg(Op.getReg(), Flags);
  //            } else if (Op.isImm()) {
  //                Builder.addImm(Op.getImm());
  //            } else if (!Op.isValid()) {
  //                llvm_unreachable("Operand is not set");
  //            } else {
  //                llvm_unreachable("Not yet implemented");
  //            }
  //        }
  //    }

  //    new UnreachableInst(Module->getContext(), BB);
  //    return MMI->getOrCreateMachineFunction(*F);

  //        llvm::Function f;
  //    llvm::MachineFunction func;
  //    auto genInst = makeInstruction(
  //        agent.getIsa(), llvm::AMDGPU::S_ADD_I32,
  //        llvm::MCOperand::createReg(llvm::AMDGPU::VGPR3),
  //        llvm::MCOperand::createReg(llvm::AMDGPU::SGPR6),
  //        llvm::MCOperand::createReg(llvm::AMDGPU::VGPR1));

  //    const auto targetTriple =
  //    llvm::Triple(agent.getIsa().getLLVMTargetTriple()); llvm::MCContext
  //    ctx(targetTriple, targetInfo.getMCAsmInfo(),
  //    targetInfo.getMCRegisterInfo(),
  //                        targetInfo.getMCSubTargetInfo());
  //    auto mcEmitter =
  //    targetInfo.getTarget()->createMCCodeEmitter(*targetInfo.getMCInstrInfo(),
  //    ctx);

  //    llvm::SmallVector<char> osBack;
  //    llvm::SmallVector<llvm::MCFixup> fixUps;
  //    mcEmitter->encodeInstruction(genInst, osBack, fixUps,
  //    *targetInfo.getMCSubTargetInfo()); fmt::print("Assembled instruction
  //    :"); for (auto s: osBack) { fmt::print("{:#x}", s); } fmt::print("\n");

  //    auto oneMoreTime = Disassembler::instance().disassemble(
  //        agent.getIsa(), {reinterpret_cast<std::byte *>(osBack.data()),
  //        osBack.size()});

  llvm::Expected<hsa::ExecutableSymbol> instrumentationFunc =
      CodeObjectManager.getInstrumentationFunction(deviceFunc, Agent);
  LUTHIER_RETURN_ON_ERROR(instrumentationFunc.takeError());

  auto instFunctionInstructions =
      CodeLifter::instance().disassemble(*instrumentationFunc);
  LUTHIER_RETURN_ON_ERROR(instFunctionInstructions.takeError());
  //
  for (const auto &i : **instFunctionInstructions) {
    std::string instStr;
    llvm::raw_string_ostream instStream(instStr);
    auto inst = i.getInstr();
    targetInfo->getMCInstPrinter()->printInst(
        &inst, reinterpret_cast<luthier_address_t>(i.getAddress()), "",
        *targetInfo->getMCSubTargetInfo(), llvm::outs());
    llvm::outs() << "\n";
  }
//            targetInfo.getMCInstrInfo()->getName(inst.getOpcode()).str() <<
    //        "\n";
    //        //        fmt::println("Is call? {}",
    //        targetInfo.MII_->get(inst.getOpcode()).isCall());
    //        //        fmt::println("Is control flow? {}",
    //        //
    //        targetInfo.MII_->get(inst.getOpcode()).mayAffectControlFlow(inst,
    //        *targetInfo.MRI_));
    //        //        fmt::println("Num max operands? {}",
    //        targetInfo.MII_->get(inst.getOpcode()).getNumOperands());
    //        llvm::outs() << llvm::formatv("Num operands? {0:X}\n",
    //        inst.getNumOperands());
    //        //        fmt::println("May load? {}",
    //        targetInfo.MII_->get(inst.getOpcode()).mayLoad()); llvm::outs() <<
    //        llvm::formatv("Mnemonic: {0}\n",
    //        targetInfo.getMCInstPrinter()->getMnemonic(&inst).first);
    //
    //        for (int j = 0; j < inst.getNumOperands(); j++) {
    //            //            auto op = inst.getOperand(j);
    //            //            std::string opStr;
    //            //            llvm::raw_string_ostream opStream(opStr);
    //            //            op.print(opStream, targetInfo.MRI_.get());
    //            //            fmt::println("OP: {}", opStream.str());
    //            //            if (op.isReg()) {
    //            //                fmt::println("Reg idx: {}", op.getReg());
    //            //                fmt::println("Reg name: {}",
    //            targetInfo.MRI_->getName(op.getReg()));
    //            //                auto subRegIterator =
    //            targetInfo.MRI_->subregs(op.getReg());
    //            //                for (auto it = subRegIterator.begin(); it !=
    //            subRegIterator.end(); it++) {
    //            //                    fmt::println("\tSub reg Name: {}",
    //            targetInfo.MRI_->getName((*it)));
    //            //                }
    //            ////                auto superRegIterator =
    //            targetInfo.MRI_->superregs(op.getReg());
    //            ////                for (auto it = superRegIterator.begin();
    //            it
    //            != superRegIterator.end(); it++) {
    //            ////                    fmt::println("\tSuper reg Name: {}",
    //            targetInfo.MRI_->getName((*it)));
    //            ////                }
    //            //            }
    //            //            fmt::println("Is op {} reg? {}", j, op.isReg());
    //            //            fmt::println("Is op {} valid? {}", j,
    //            op.isValid());
    //        }

    //        llvm::outs() << instStream.str() << "\n";
    //        inst.getOpcode();
    //    }
    //    llvm::outs() << "==================================\n";*/
    //
    //    for (const auto &i: *targetFunction) {
    //        std::string instStr;
    //        llvm::raw_string_ostream instStream(instStr);
    //        auto inst = i.getInstr();
    //        fmt::println("{}",
    //        targetInfo.MII_->getName(inst.getOpcode()).str());
    ////        fmt::println("Is call? {}",
    /// targetInfo.MII_->get(inst.getOpcode()).isCall()); / fmt::println("Is
    /// control flow? {}", /
    /// targetInfo.MII_->get(inst.getOpcode()).mayAffectControlFlow(inst,
    ///*targetInfo.MRI_));
    //        targetInfo.IP_->printInst(&inst,
    //        reinterpret_cast<luthier_address_t>(inst.getLoc().getPointer()),
    //        "",
    //                                  *targetInfo.STI_, instStream);
    //        fmt::println("{}", instStream.str());
    //        inst.getOpcode();
    //    }

    //  auto targetExecutable = instr.getExecutable();
    //
    //  auto symbol = instr.getExecutableSymbol();
    //  std::string symbolName = symbol.getName();
    //
    //  auto storage =
    //  targetExecutable.getLoadedCodeObjects()[0].getStorageMemory();
    //    auto instrumentedElfView = getELFObjectFileBase(storage);

    //    // Find the symbol that requires instrumentation.
    //    std::optional<code::SymbolView> storageSymbol =
    //    instrumentedElfView->getSymbol(symbolName); if
    //    (!storageSymbol.has_value())
    //        throw std::runtime_error(fmt::format("Failed to find symbol {} in
    //        the copied executable", symbolName));
    //    auto meta = storageSymbol->getMetaData();
    //    fmt::println("Number of SGPRS: {}", meta.usedSGPRs_);
    //    fmt::println("Number of VGPRS: {}", meta.usedVGPRs_);
    //
    //    std::unique_ptr<llvm::MCObjectFileInfo> MOFI(
    //        targetInfo.getTarget()->createMCObjectFileInfo(ctx, /*PIC*/ true,
    //        /*large code model*/ false));
    //    auto textSection = MOFI->getTextSection();
    //    fmt::println("Does the text section have instructions? {}",
    //    textSection->hasInstructions()); ctx.setObjectFileInfo(MOFI.get());
    //    //    ctx->setAllowTemporaryLabels(true);
    //    ctx.setGenDwarfForAssembly(false);
    //
    //    llvm::SmallVector<char> out;
    //
    //    llvm::raw_svector_ostream VOS(out);
    //
    //    llvm::MCCodeEmitter *CE =
    //    targetInfo.getTarget()->createMCCodeEmitter(*targetInfo.getMCInstrInfo(),
    //    ctx); llvm::MCAsmBackend *MAB =
    //    targetInfo.getTarget()->createMCAsmBackend(
    //        *targetInfo.getMCSubTargetInfo(), *targetInfo.getMCRegisterInfo(),
    //        targetInfo.getTargetOptions().MCOptions);
    //
    //    auto Str =
    //    std::unique_ptr<llvm::MCStreamer>(targetInfo.getTarget()->createMCObjectStreamer(
    //        llvm::Triple(agent.getIsa().getLLVMTargetTriple()), ctx,
    //        std::unique_ptr<llvm::MCAsmBackend>(MAB),
    //        MAB->createObjectWriter(VOS),
    //        std::unique_ptr<llvm::MCCodeEmitter>(CE),
    //        *targetInfo.getMCSubTargetInfo(), true, false,
    //        /*DWARFMustBeAtTheEnd*/ false));
    //
    //    //    Str->initSections(false, *targetInfo.STI_);
    //    Str->switchSection(ctx.getObjectFileInfo()->getTextSection());
    //    Str->emitCodeAlignment(llvm::Align(ctx.getObjectFileInfo()->getTextSectionAlignment()),
    //                           targetInfo.getMCSubTargetInfo());
    //
    //    for (const auto &inst: *targetFunction) {
    //    Str->emitInstruction(inst.getInstr(),
    //    *targetInfo.getMCSubTargetInfo()); } auto kernelName =
    //    instr.getExecutableSymbol().getName();
    //    //
    //    auto kName = kernelName.substr(0, kernelName.find(".kd"));
    //
    //    auto KernelDescriptor =
    //    instr.getExecutableSymbol().getKernelDescriptor();
    //    //
    //    llvm::MCSymbolELF *KernelCodeSymbol =
    //    cast<llvm::MCSymbolELF>(ctx.getOrCreateSymbol(llvm::Twine(kName)));
    //
    //    //    Str->pushSection();
    //    Str->switchSection(ctx.getObjectFileInfo()->getReadOnlySection());
    //
    //    Str->emitValueToAlignment(llvm::Align(64), 0, 1, 0);
    //    ctx.getObjectFileInfo()->getReadOnlySection()->ensureMinAlignment(llvm::Align(64));
    //
    //    llvm::MCSymbolELF *KernelDescriptorSymbol =
    //    cast<llvm::MCSymbolELF>(ctx.getOrCreateSymbol(kernelName));
    //
    //    // Copy kernel descriptor symbol's binding, other and visibility from
    //    the
    //    // kernel code symbol.
    //    KernelDescriptorSymbol->setBinding(KernelCodeSymbol->getBinding());
    //    KernelDescriptorSymbol->setOther(KernelCodeSymbol->getOther());
    //    KernelDescriptorSymbol->setVisibility(KernelCodeSymbol->getVisibility());
    //    // Kernel descriptor symbol's type and size are fixed.
    //    KernelDescriptorSymbol->setType(llvm::ELF::STT_OBJECT);
    //    KernelDescriptorSymbol->setSize(llvm::MCConstantExpr::create(sizeof(hsa::KernelDescriptor),
    //    ctx));
    //
    //    //    // The visibility of the kernel code symbol must be protected or
    //    less to allow
    //    //    // static relocations from the kernel descriptor to be used.
    //    if (KernelCodeSymbol->getVisibility() == llvm::ELF::STV_DEFAULT)
    //        KernelCodeSymbol->setVisibility(llvm::ELF::STV_PROTECTED);
    //    //
    //    Str->emitLabel(KernelDescriptorSymbol);
    //    Str->emitInt32(KernelDescriptor->groupSegmentFixedSize);
    //    Str->emitInt32(KernelDescriptor->privateSegmentFixedSize);
    //    Str->emitInt32(KernelDescriptor->kernArgSize);
    //    //
    //    for (uint8_t Res: KernelDescriptor->reserved0) Str->emitInt8(Res);
    //
    //    //    // FIXME: Remove the use of VK_AMDGPU_REL64 in the expression
    //    below. The
    //    //    // expression being created is:
    //    //    //   (start of kernel code) - (start of kernel descriptor)
    //    //    // It implies R_AMDGPU_REL64, but ends up being R_AMDGPU_ABS64.
    //    Str->emitValue(llvm::MCBinaryExpr::createSub(
    //                       llvm::MCSymbolRefExpr::create(KernelCodeSymbol,
    //                       llvm::MCSymbolRefExpr::VK_AMDGPU_REL64, ctx),
    //                       llvm::MCSymbolRefExpr::create(KernelDescriptorSymbol,
    //                       llvm::MCSymbolRefExpr::VK_None, ctx), ctx),
    //                   sizeof(KernelDescriptor->kernelCodeEntryByteOffset));
    //    for (uint8_t Res: KernelDescriptor->reserved1) Str->emitInt8(Res);
    //    Str->emitInt32(KernelDescriptor->computePgmRsrc3);
    //    Str->emitInt32(KernelDescriptor->computePgmRsrc1);
    //    Str->emitInt32(KernelDescriptor->computePgmRsrc2);
    //    Str->emitInt16(KernelDescriptor->kernelCodeProperties);
    //    Str->emitInt16(KernelDescriptor->kernArgPreload);
    //    for (uint8_t Res: KernelDescriptor->reserved2) Str->emitInt8(Res);
    //
    //    //    Str->popSection();
    //    //    Str->switchSection(ctx.getObjectFileInfo()->getTextSection());
    //
    //    fmt::println("Does the text section have instructions? {}",
    //    textSection->hasInstructions());
    //
    //    Str->getTargetStreamer()->finish();
    //    Str->finish();
    //    Str->finishImpl();
    //    auto finalView =
    //        code::ElfView::makeView(luthier::byte_string_view(reinterpret_cast<std::byte
    //        *>(out.data()), out.size()));
    //    fmt::println("Length of the out vector: {}", out.size());
    //    fmt::println("Type of the code object created: {}",
    //    finalView->getElfIo().get_type()); fmt::println("Number of symbols:
    //    {}", finalView->getNumSymbols());
    //
    //    finalView->getElfIo().sections[0]->get_data();
    //    //    fmt::println("Symbol name : {}",
    //    finalView->getSymbol(0)->getName());
    //    //    fmt::println("Symbol name : {}",
    //    finalView->getSymbol(1)->getName());
    //    //    finalView->getSymbol(0)->getData();
    //    fmt::println("Data works");
    //    finalView->getElfIo().sections[1]->get_data();
    //
    //    //    auto executable =
    //    compileRelocatableToExecutable({reinterpret_cast<std::byte*>(out.data()),
    //    out.size()}, instr.getAgent());
    //    //    delete Str;
    //    //    Str->finishImpl();
    //
    //    //    targetStr->finish();
    //
    //    CodeObjectManager::instance().loadInstrumentedKernel(luthier::byte_string_t(storage),
    //    instr.getExecutableSymbol());
    return llvm::Error::success();
  }

} // namespace luthier
