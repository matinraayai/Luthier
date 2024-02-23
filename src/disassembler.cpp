#include "disassembler.hpp"

#include <AMDGPUResourceUsageAnalysis.h>
#include <AMDGPUTargetMachine.h>
#include <GCNSubtarget.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "target_manager.hpp"

namespace luthier {

const Disassembler::DisassemblyInfo &luthier::Disassembler::getDisassemblyInfo(const luthier::hsa::Isa &isa) {
    if (!disassemblyInfoMap_.contains(isa)) {
        const auto &targetInfo = TargetManager::instance().getTargetInfo(isa);
        const auto targetTriple = llvm::Triple(isa.getLLVMTargetTriple());
        std::unique_ptr<llvm::MCContext> ctx(new (std::nothrow) llvm::MCContext(
            targetTriple, targetInfo.getMCAsmInfo(), targetInfo.getMCRegisterInfo(), targetInfo.getMCSubTargetInfo()));
        std::unique_ptr<const llvm::MCDisassembler> disAsm(
            targetInfo.getTarget()->createMCDisassembler(*(targetInfo.getMCSubTargetInfo()), *ctx));
        disassemblyInfoMap_.insert({isa, DisassemblyInfo{std::move(ctx), std::move(disAsm)}});
    }
    return disassemblyInfoMap_.at(isa);
}

std::vector<llvm::MCInst> Disassembler::disassemble(const hsa::Isa &isa, llvm::ArrayRef<uint8_t> code) {
    const auto &disassemblyInfo = getDisassemblyInfo(isa);
    const auto &targetInfo = TargetManager::instance().getTargetInfo(isa);
    const auto &disAsm = disassemblyInfo.disAsm_;

    size_t maxReadSize = targetInfo.getMCAsmInfo()->getMaxInstLength();
    size_t idx = 0;
    auto currentAddress = reinterpret_cast<luthier_address_t>(code.data());
    std::vector<llvm::MCInst> instructions;

    //TODO: Check if currentAddress needs to be bundled with MCINst
    while (idx < code.size()) {
        size_t readSize = (idx + maxReadSize) < code.size() ? maxReadSize : code.size() - idx;
        size_t instSize{};
        llvm::MCInst inst;
        std::string annotations;
        llvm::raw_string_ostream annotationsStream(annotations);
        auto readBytes = arrayRefFromStringRef(toStringRef(code).substr(idx, readSize));
        if (disAsm->getInstruction(inst, instSize, readBytes, currentAddress, annotationsStream)
            != llvm::MCDisassembler::Success) {
            llvm::report_fatal_error("Failed to disassemble instructions");
        }
        inst.setLoc(llvm::SMLoc::getFromPointer(reinterpret_cast<const char *>(currentAddress)));

        idx += instSize;
        currentAddress += instSize;
        instructions.push_back(inst);
    }

    return instructions;
}

const std::vector<hsa::Instr> *luthier::Disassembler::disassemble(const hsa::ExecutableSymbol &symbol,
                                                                  std::optional<hsa::Isa> isa,
                                                                  std::optional<size_t> size) {
    if (!disassembledSymbols_.contains(symbol)) {
        const auto symbolType = symbol.getType();
        LUTHIER_CHECK((symbolType != HSA_SYMBOL_KIND_VARIABLE));
        if (!isa.has_value()) isa.emplace(symbol.getAgent().getIsa());

        llvm::StringRef code = toStringRef(
            symbol.getType() == HSA_SYMBOL_KIND_KERNEL ? symbol.getKernelCode() : symbol.getIndirectFunctionCode());

        if (size.has_value()) code = code.substr(0, *size > code.size() ? code.size() : *size);

        std::vector<llvm::MCInst> instructions = disassemble(*isa, arrayRefFromStringRef(code));
        disassembledSymbols_.insert({symbol, std::make_unique<std::vector<hsa::Instr>>()});
        auto &out = disassembledSymbols_.at(symbol);
        out->reserve(instructions.size());
        for (const auto &inst: instructions) { out->push_back(hsa::Instr(inst, symbol)); }
    }
    return disassembledSymbols_.at(symbol).get();
}

std::vector<llvm::MCInst> luthier::Disassembler::disassemble(const llvm::object::ELFSymbolRef& symbol,
                                                             std::optional<size_t> size) {
    auto symbolType = symbol.getELFType();
    LUTHIER_CHECK((symbolType == llvm::ELF::STT_FUNC));

    auto symbolNameOrError = symbol.getName();
    LUTHIER_CHECK(llvm::errorToBool(symbolNameOrError.takeError()));
    auto symbolName = *symbolNameOrError;
    // If the kd symbol was passed, get the function symbol associated with it in the parent ELF.


    auto triple = symbol.getObject()->makeTriple();
    auto isa = hsa::Isa::fromName(triple.normalize().c_str());

    auto addressOrError = symbol.getAddress();
    LUTHIER_CHECK(llvm::errorToBool(addressOrError.takeError()));
    auto address = *addressOrError;
    llvm::StringRef code(reinterpret_cast<const char*>(address), symbol.getSize());

    if (size.has_value()) code = code.substr(0, *size > code.size() ? code.size() : *size);

    return disassemble(isa, llvm::arrayRefFromStringRef(code));
}

luthier::Disassembler::~Disassembler() {
    disassemblyInfoMap_.clear();
    moduleInfoMap_.clear();
    disassembledSymbols_.clear();
}
void luthier::Disassembler::liftKernelModule(const hsa::ExecutableSymbol &symbol, llvm::SmallVectorImpl<char> &out) {
    if (!moduleInfoMap_.contains(symbol)) {
        auto agent = symbol.getAgent();
        LUTHIER_CHECK((symbol.getType() == HSA_SYMBOL_KIND_KERNEL));
        auto isa = agent.getIsa();
        auto targetInfo = luthier::TargetManager::instance().getTargetInfo(isa);

        auto theTargetMachine = targetInfo.getTargetMachine();

        auto mCInstInfo = theTargetMachine->getMCInstrInfo();

        auto context = std::make_unique<llvm::LLVMContext>();
        LUTHIER_CHECK(context);

        auto symbolName = symbol.getName();

        auto module = std::make_unique<llvm::Module>(symbol.getName(), *context);
        LUTHIER_CHECK(module);
        module->setDataLayout(theTargetMachine->createDataLayout());

        auto mmiwp = std::make_unique<llvm::MachineModuleInfoWrapperPass>(theTargetMachine);

        LUTHIER_CHECK(mmiwp);

        llvm::Type *const returnType = llvm::Type::getVoidTy(module->getContext());
        LUTHIER_CHECK(returnType);
        llvm::Type *const memParamType =
            llvm::PointerType::get(llvm::Type::getInt32Ty(module->getContext()), llvm::AMDGPUAS::GLOBAL_ADDRESS);
        LUTHIER_CHECK(memParamType);
        llvm::FunctionType *FunctionType = llvm::FunctionType::get(returnType, {memParamType}, false);
        LUTHIER_CHECK(FunctionType);
        llvm::Function *F = llvm::Function::Create(FunctionType, llvm::GlobalValue::ExternalLinkage,
                                                   symbolName.substr(0, symbolName.size() - 3), *module);
        LUTHIER_CHECK(F);
        F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

        llvm::outs() << "Number of arguments: " << F->arg_size() << "\n";

        llvm::BasicBlock *BB = llvm::BasicBlock::Create(module->getContext(), "", F);
        LUTHIER_CHECK(BB);
        new llvm::UnreachableInst(module->getContext(), BB);

        // Construct the attributes of the Function, which will result in the MF attributes getting populated
        F->addFnAttr("amdgpu-no-dispatch-ptr");
        F->addFnAttr("amdgpu-no-queue-ptr");
        F->addFnAttr("amdgpu-no-dispatch-id");
        F->addFnAttr("amdgpu-no-workgroup-id-y");
        F->addFnAttr("amdgpu-no-workitem-id-y");
        F->addFnAttr("amdgpu-no-workgroup-id-z");
        F->addFnAttr("amdgpu-no-workitem-id-z");
        F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");
        F->addFnAttr("uniform-work-group-size", "true");

        llvm::outs() << "Preloaded Args: " << symbol.getKernelDescriptor()->kernArgPreload << "\n";

        auto &MF = mmiwp->getMMI().getOrCreateMachineFunction(*F);

        MF.setAlignment(llvm::Align(4096));

        const std::vector<hsa::Instr> *targetFunction = Disassembler::instance().disassemble(symbol);

        llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
        MF.push_back(MBB);

        for (const auto &i: *targetFunction) {
            auto inst = i.getInstr();

            const unsigned Opcode = inst.getOpcode();

            auto subTargetInfo = theTargetMachine->getSubtargetImpl(*F);

            const llvm::MCInstrDesc &MCID = mCInstInfo->get(Opcode);
            llvm::MachineInstrBuilder Builder = llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
            for (unsigned OpIndex = 0, E = i.getInstr().getNumOperands(); OpIndex < E; ++OpIndex) {
                const llvm::MCOperand &Op = inst.getOperand(OpIndex);
                if (Op.isReg()) {
                    const bool IsDef = OpIndex < MCID.getNumDefs();
                    unsigned Flags = 0;
                    const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
                    if (IsDef && !OpInfo.isOptionalDef()) Flags |= llvm::RegState::Define;
                    llvm::outs() << "Reg name: ";
                    Op.print(llvm::outs(), targetInfo.getMCRegisterInfo());
                    llvm::outs() << " " << llvm::hexdigit(Flags) << "\n";
                    Builder.addReg(Op.getReg(), Flags);
                } else if (Op.isImm()) {
                    Builder.addImm(Op.getImm());
                } else if (!Op.isValid()) {
                    llvm_unreachable("Operand is not set");
                } else {
                    llvm_unreachable("Not yet implemented");
                }
            }
        }
        //        MBB->dump();
        //        llvm::outs() << "Number of blocks : " << MF.getNumBlockIDs() << "\n";
        //        MF.dump();
        auto TII = reinterpret_cast<const llvm::SIInstrInfo *>(theTargetMachine->getSubtargetImpl(*F)->getInstrInfo());
        auto TRI =
            reinterpret_cast<const llvm::SIRegisterInfo *>(theTargetMachine->getSubtargetImpl(*F)->getRegisterInfo());
        auto MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
        MFI->addPrivateSegmentBuffer(*TRI);
        MFI->addKernargSegmentPtr(*TRI);

        for (auto &BB: MF) {
            for (auto &Inst: BB) {
                llvm::outs() << "MIR: ";
                Inst.print(llvm::outs(), true, false, false, true, TII);
                std::string error;
                llvm::StringRef errorRef(error);
                bool isInstCorrect = TII->verifyInstruction(Inst, errorRef);
                llvm::outs() << "Is instruction correct: " << isInstCorrect << "\n";
                if (!isInstCorrect) {

                    llvm::outs() << "May read Exec : " << TII->mayReadEXEC(MF.getRegInfo(), Inst) << "\n";
                    Inst.addOperand(llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
                    TII->fixImplicitOperands(Inst);
                    Inst.addImplicitDefUseOperands(MF);
                    llvm::outs() << "After correction: ";
                    Inst.print(llvm::outs(), true, false, false, true, TII);

                    llvm::outs() << "Is correct now: " << TII->verifyInstruction(Inst, errorRef) << "\n";
                }
                llvm::outs() << "Error: " << errorRef << "\n";
                for (auto &op: Inst.operands()) {
                    if (op.isReg()) {
                        llvm::outs() << "Reg: ";
                        op.print(llvm::outs(), TRI);
                        llvm::outs() << "\n";
                        llvm::outs() << "is implicit: " << op.isImplicit() << "\n";
                        //                        if (op.isImplicit() && op.readsReg() && op.isUse() && op.getReg().id() == llvm::AMDGPU::EXEC) {
                        //                            op.setImplicit(true);
                        //
                        //                        }
                    }
                }
                llvm::outs() << "==============================================================\n";
            }
        }
        llvm::MachineFunctionProperties &properties = MF.getProperties();
        properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
        properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
        properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
        properties.reset(llvm::MachineFunctionProperties::Property::TracksLiveness);

        MF.getRegInfo().freezeReservedRegs(MF);

        //        auto elfOrError = getELFObjectFileBase(symbol.getExecutable().getLoadedCodeObjects()[0].getStorageMemory());
        //        if (llvm::errorToBool(elfOrError.takeError())) {
        //            llvm::report_fatal_error("Failed to parse the elf.");
        //        }
        //        auto noteOrError = getElfNoteMetadataRoot(elfOrError.get().get());
        //        if (llvm::errorToBool(noteOrError.takeError())) {
        //            llvm::report_fatal_error("Failed to parse the note section");
        //        }
        //        noteOrError.get().toYAML(llvm::outs());
        //        llvm::outs() << "\n";
        ////        MF.dump();
        //        properties.print(llvm::outs());
        //        llvm::outs() << "\n";

        // We create the pass manager, run the passes to populate AsmBuffer.
        llvm::MCContext &MCContext = mmiwp->getMMI().getContext();
        llvm::legacy::PassManager PM;

        llvm::TargetLibraryInfoImpl tlii(llvm::Triple(module->getTargetTriple()));
        PM.add(new llvm::TargetLibraryInfoWrapperPass(tlii));

        llvm::TargetPassConfig *TPC = theTargetMachine->createPassConfig(PM);
        PM.add(TPC);
        PM.add(mmiwp.release());
        //        TPC->printAndVerify("MachineFunctionGenerator::assemble");

        //        auto usageAnalysis = std::make_unique<llvm::AMDGPUResourceUsageAnalysis>();
        PM.add(new llvm::AMDGPUResourceUsageAnalysis());
        // Add target-specific passes.
        //        ET.addTargetSpecificPasses(PM);
        //        TPC->printAndVerify("After ExegesisTarget::addTargetSpecificPasses");
        // Adding the following passes:
        // - postrapseudos: expands pseudo return instructions used on some targets.
        // - machineverifier: checks that the MachineFunction is well formed.
        // - prologepilog: saves and restore callee saved registers.
        //        for (const char *PassName :
        //             {"postrapseudos", "machineverifier", "prologepilog"})
        //            if (addPass(PM, PassName, *TPC))
        //                return make_error<Failure>("Unable to add a mandatory pass");
        TPC->setInitialized();

        //        llvm::SmallVector<char> o;
        //        std::string out;
        //        llvm::raw_string_ostream outOs(out);
        llvm::raw_svector_ostream outOs(out);
        // AsmPrinter is responsible for generating the assembly into AsmBuffer.
        if (theTargetMachine->addAsmPrinter(PM, outOs, nullptr, llvm::CodeGenFileType::ObjectFile, MCContext))
            llvm::outs() << "Failed to add pass manager\n";
        //            return make_error<llvm::Failure>("Cannot add AsmPrinter passes");

        PM.run(*module);// Run all the passes

        llvm::outs() << out << "\n";
    }
}

}// namespace luthier