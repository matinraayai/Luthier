#include "luthier/ToolCodeGen/DebugInfoPass.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/MachineInstrTraceAddressAnalysis.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <unordered_set>

#define DEBUG_TYPE "luthier-populate-debug-info"

namespace luthier {

llvm::PreservedAnalyses DebugInfoPass::run(Prototype &IP,
                                           PrototypeAnalysisManager &IPAM) {
  llvm::Module &M = IP.getTargetModule();
  llvm::LLVMContext &Ctx = M.getContext();

  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<TargetModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  const MemoryAllocationAccessor &SegAccessor =
      MAM.getResult<MemoryAllocationAnalysis>(IP.getTargetModule())
          .getAccessor();

  llvm::FunctionAnalysisManager &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();

  llvm::DenseMap<const luthier::object::AMDGCNObjectFile *,
                 llvm::DenseMap<llvm::StringRef, llvm::DISubprogram *>>
      CodeObjectLinkageNameToDISP;

  llvm::DenseMap<const luthier::object::AMDGCNObjectFile *,
                 llvm::DenseMap<uint64_t, llvm::DISubprogram *>>
      CodeObjectDIEOffsetToDISP;

  llvm::DenseMap<const luthier::object::AMDGCNObjectFile *,
                 std::unique_ptr<llvm::DWARFContext>>
      CodeObjectToDWARFCtx;

  llvm::DenseSet<llvm::DISubprogram *> DISPSet;

  for (llvm::Function &F : M) {
    auto EntryPoint = getFunctionEntryPoint(F);
    if (!EntryPoint) {
      LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] No entry point for function "
                              << F.getName() << "\n");
      continue;
    }
    uint64_t EntryPointAddr = EntryPoint->getEntryPointAddress();

    // Get allocation descriptor
    auto AllocationOrErr = SegAccessor.getAllocationDescriptor(EntryPointAddr);
    if (!AllocationOrErr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] No allocation descriptor for function "
                 << F.getName() << "\n");
      llvm::consumeError(AllocationOrErr.takeError());
      continue;
    }

    // Retrieve the code object
    const auto *CodeObject = AllocationOrErr->getAllocationCodeObject();
    if (!CodeObject) {
      LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] No CodeObject for function "
                              << F.getName() << "\n");
      continue;
    }

    uint64_t AllocBaseAddr = reinterpret_cast<uint64_t>(
        AllocationOrErr->getDeviceAllocation().data());

    uint64_t EntryPointOffset = EntryPointAddr - AllocBaseAddr;

    llvm::DISubprogram *FuncDISP = nullptr;

    // Eagerly populate all subprogram DIEs in a new CodeObject
    if (CodeObjectToDWARFCtx.find(CodeObject) == CodeObjectToDWARFCtx.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] Creating DWARFContext for CodeObject\n");
      CodeObjectToDWARFCtx[CodeObject] =
          llvm::DWARFContext::create(*CodeObject);

      if (CodeObjectToDWARFCtx.at(CodeObject)->getNumCompileUnits() == 0) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "[DebugInfoPass] Skipping CodeObject with no CompileUnits\n");
        continue;
      }

      for (const std::unique_ptr<llvm::DWARFUnit> &CU :
           CodeObjectToDWARFCtx[CodeObject]->compile_units()) {

        llvm::DIBuilder DIB(M);

        llvm::StringRef CUName =
            CU->getUnitDIE().getName(llvm::DINameKind::ShortName);
        llvm::StringRef CUDir = CU->getCompilationDir();

        LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] Creating DICompileUnit '"
                                << CUName << "' in dir '" << CUDir << "'\n");

        llvm::DIFile *CUFile = DIB.createFile(CUName, CUDir);

        auto SrcLang = llvm::DISourceLanguageName(
            static_cast<uint16_t>(llvm::dwarf::toUnsigned(
                CU->getUnitDIE().find(llvm::dwarf::DW_AT_language),
                llvm::dwarf::DW_LANG_C_plus_plus)));

        DIB.createCompileUnit(SrcLang, CUFile, "Luthier Debug Info Pass", false,
                              "", 0);

        // Create DISubprograms for all subprogram DIEs in this CU
        for (const auto &DieInfo : CU->dies()) {
          llvm::DWARFDie Die(CU.get(), &DieInfo);

          switch (Die.getTag()) {
          case (llvm::dwarf::DW_TAG_subprogram): {
            llvm::StringRef LinkageNameRef(Die.getLinkageName());
            llvm::StringRef ShortNameRef(Die.getShortName());
            // ShortName fallback
            llvm::StringRef FunctionName =
                LinkageNameRef.empty() ? ShortNameRef : LinkageNameRef;

            if (FunctionName.empty()) {
              break;
            }

            if (Die.find(llvm::dwarf::DW_AT_declaration)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[DebugInfoPass] Skipping Function Declaration "
                         << FunctionName << "\n");
              continue;
            }

            uint32_t LineNumber = Die.getDeclLine();

            llvm::StringRef FileName;
            llvm::StringRef DirName;

            std::string DeclFileResult = Die.getDeclFile(
                llvm::DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath);

            if (!DeclFileResult.empty()) {
              llvm::StringRef PathRef(DeclFileResult);
              FileName = llvm::sys::path::filename(PathRef);
              DirName = llvm::sys::path::parent_path(PathRef);
            }

            llvm::DIFile *UnitFile = DIB.createFile(FileName, DirName);

            llvm::DISubroutineType *SubType =
                DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));

            // Create a DISubprogram
            llvm::DISubprogram *DISP = DIB.createFunction(
                UnitFile, ShortNameRef, LinkageNameRef, UnitFile, LineNumber,
                SubType, LineNumber, llvm::DINode::FlagZero,
                llvm::DISubprogram::SPFlagDefinition);

            if (CodeObjectLinkageNameToDISP[CodeObject].lookup(FunctionName)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[DebugInfoPass] Function " << FunctionName
                         << " will overwrite a cached DISubprogram entry\n");
            }

            // Add DISubprogram to ObjectFile -> {LinkageName, DISubprogram}
            // cache
            CodeObjectLinkageNameToDISP[CodeObject][FunctionName] = DISP;
            CodeObjectDIEOffsetToDISP[CodeObject][Die.getOffset()] = DISP;

            LLVM_DEBUG(llvm::dbgs()
                       << "[DebugInfoPass] Subprogram DIE: short='"
                       << ShortNameRef << "' linkage='" << LinkageNameRef
                       << "' line=" << LineNumber << " source_file='"
                       << UnitFile->getFilename() << "'\n");
            break;
          }

            // case (llvm::dwarf::DW_TAG_inlined_subroutine): {
            //   break;
            // }

            // case (llvm::dwarf::DW_TAG_lexical_block): {
            // break;
            // }

          default:
            break;
          }
        }

        DIB.finalize();
      }
    }

    llvm::DWARFDie FunctionDie;

    llvm::DWARFContext::DIEsForAddress EntryFunctionDie =
        CodeObjectToDWARFCtx.at(CodeObject)
            ->getDIEsForAddress(EntryPointOffset);

    FunctionDie = EntryFunctionDie.FunctionDIE;

    if (!FunctionDie) {
      LLVM_DEBUG(
          llvm::dbgs() << llvm::formatv(
              "[DebugInfoPass] No DWARFDie exists for function at offset {0:x}",
              EntryPointOffset));
      continue;
    }

    while (FunctionDie && !FunctionDie.isSubprogramDIE()) {
      FunctionDie = FunctionDie.getParent();
    }

    uint64_t FunctionDieOffset = FunctionDie.getOffset();
    FuncDISP = CodeObjectDIEOffsetToDISP[CodeObject].lookup(FunctionDieOffset);

    if (!FuncDISP) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] No matching DISubprogram for "
                 << FunctionDie.getName(llvm::DINameKind::LinkageName) << "\n");

      continue;
    }

    if (!DISPSet.insert(FuncDISP).second) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] Failed to insert DISubprogram:"
                 << FuncDISP->getName() << " for Function "
                 << FunctionDie.getName(llvm::DINameKind::LinkageName)
                 << " into the DISubpgrogram set\n");
      continue;
    }

    F.setSubprogram(FuncDISP);

    // PC Section to IR instrcution map for matching MIR to IR instructions
    llvm::DenseMap<llvm::MDNode *, llvm::SmallVector<llvm::Instruction *>>
        PCSectionIRMap;

    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB) {
        if (auto *PCS = I.getMetadata(llvm::LLVMContext::MD_pcsections)) {
          PCSectionIRMap[PCS].push_back(&I);
        }
      }
    }

    auto *MFResult = FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);

    if (!MFResult) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "[DebugInfoPass] No MachineFunction found for function {0}\n",
          F.getName()))));
      continue;
    }

    auto &MF = MFResult->getMF();

    llvm::MachineFunctionAnalysisManager &MFAM =
        FAM.getResult<llvm::MachineFunctionAnalysisManagerFunctionProxy>(F)
            .getManager();

    const auto &MITraceAddrMap =
        MFAM.getResult<MachineInstrTraceAddressAnalysis>(MF).getMIToTraceMap();

    auto &LinkageNameToDISP = CodeObjectLinkageNameToDISP[CodeObject];

    llvm::DILineInfoSpecifier DILineInfoSpecifier(
        llvm::symbolize::FileLineInfoKind::AbsoluteFilePath,
        llvm::symbolize::FunctionNameKind::LinkageName);

    // Iterate over all Traced Instructions
    for (llvm::MachineBasicBlock &MBB : MF) {
      for (llvm::MachineInstr &MI : MBB) {

        uint64_t TraceAddr = MITraceAddrMap.lookup(&MI);

        if (!TraceAddr) {
          continue;
        }

        // Calculate offset from code object load base
        uint64_t Offset = TraceAddr - AllocBaseAddr;

        LLVM_DEBUG(
            llvm::dbgs() << llvm::formatv(
                "[DebugInfoPass] MI at Trace Addr {0:x} From Entry Addr {1:x}, "
                "Alloc Base {2:x}, DWARF offset {3:x}\n",
                TraceAddr, EntryPointAddr, AllocBaseAddr, Offset));

        llvm::object::SectionedAddress SectionnedAddress;
        SectionnedAddress.Address = Offset;

        llvm::DIInliningInfo InliningInfo =
            CodeObjectToDWARFCtx[CodeObject]->getInliningInfoForAddress(
                SectionnedAddress, DILineInfoSpecifier);

        llvm::DIScope *Scope = F.getSubprogram();

        uint32_t NumFrames = InliningInfo.getNumberOfFrames();

        if (NumFrames == 0) {
          continue;
        }

        llvm::DILocation *DILoc = nullptr;
        for (uint32_t Idx = NumFrames; Idx > 0; --Idx) {
          uint32_t FrameIdx = Idx - 1;

          const auto &Frame = InliningInfo.getFrame(FrameIdx);

          if (FrameIdx == NumFrames - 1) {
            DILoc = llvm::DILocation::get(Ctx, Frame.Line, Frame.Column, Scope);
          } else {
            Scope = LinkageNameToDISP.lookup(Frame.FunctionName);
            if (!Scope) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[DebugInfoPass] Function: " << Frame.FunctionName
                         << " has no DISubprogram. The InlinedAt chain "
                            "terminates early.\n");
              break;
            }
            llvm::DILocation *CurrentLoc = llvm::DILocation::get(
                Ctx, Frame.Line, Frame.Column, Scope, DILoc);
            DILoc = CurrentLoc;
          }
        }

        MI.setDebugLoc(DILoc);

        auto It = PCSectionIRMap.find(MI.getPCSections());
        if (It != PCSectionIRMap.end()) {
          for (llvm::Instruction *I : It->second) {
            I->setDebugLoc(llvm::DebugLoc(DILoc));
          }
        }

        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "[DebugInfoPass] MI at Trace Addr {3:x} attached to "
                       "{0}:{1}:{2}\n",
                       llvm::sys::path::filename(DILoc->getFilename()),
                       DILoc->getLine(), DILoc->getColumn(), TraceAddr););
      }
    }
  }

  return llvm::PreservedAnalyses::all();
}

llvm::PreservedAnalyses
DebugInfoPrinterPass::run(Prototype &IP, PrototypeAnalysisManager &IPAM) {
  llvm::Module &M = IP.getTargetModule();

  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<TargetModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  const MemoryAllocationAccessor &SegAccessor =
      MAM.getResult<MemoryAllocationAnalysis>(M).getAccessor();

  std::unordered_set<const llvm::object::ObjectFile *> DumpedObjects;

  OS << "=== DWARF Debug Info for Module: " << M.getName() << " ===\n";

  for (llvm::Function &F : M) {

    auto EntryPointOpt = getFunctionEntryPoint(F);
    if (!EntryPointOpt)
      continue;

    uint64_t EntryPointAddr = EntryPointOpt->getEntryPointAddress();
    auto AllocationOrErr = SegAccessor.getAllocationDescriptor(EntryPointAddr);
    if (!AllocationOrErr) {
      llvm::consumeError(AllocationOrErr.takeError());
      continue;
    }

    const auto *CodeObject = AllocationOrErr->getAllocationCodeObject();
    if (!CodeObject)
      continue;

    if (DumpedObjects.insert(CodeObject).second) {
      OS << "\n--- Code Object DWARF Context ---\n";

      std::unique_ptr<llvm::DWARFContext> DWARFCtx =
          llvm::DWARFContext::create(*CodeObject);

      llvm::DIDumpOptions DumpOpts;
      DumpOpts.ShowChildren = true;
      DumpOpts.ShowParents = true;
      DumpOpts.ShowForm = true;
      DumpOpts.SummarizeTypes = true;
      DumpOpts.Verbose = true;

      DWARFCtx->dump(OS, DumpOpts);
    }
  }

  OS << "\n=== End DWARF Debug Info ===\n";

  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
