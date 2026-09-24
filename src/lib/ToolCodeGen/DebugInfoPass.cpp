#include "luthier/ToolCodeGen/DebugInfoPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
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
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "luthier-debug-info"

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

  llvm::DIBuilder DIB(M);

  std::unordered_map<const llvm::object::ObjectFile *,
                     std::unique_ptr<llvm::DWARFContext>>
      ContextCache;

  // Avoid CU offset collision when multiple ObjectFiles are present
  std::unordered_map<const llvm::object::ObjectFile *,
                     std::unordered_map<uint64_t, llvm::DICompileUnit *>>
      ObjFileCUOffsetToDICU;

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

    if (ContextCache.find(CodeObject) == ContextCache.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] Creating DWARFContext for CodeObject\n");
      ContextCache[CodeObject] = llvm::DWARFContext::create(*CodeObject);

      // TODO: eagerly populate all DIEs in the code object
    }

    // Fetch DWARFContext for CodeObject
    llvm::DWARFContext *DWARFCtx = ContextCache[CodeObject].get();

    // Skip if no debug info is found for this CodeObject
    if (DWARFCtx->getNumCompileUnits() == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] No compile units in DWARF for "
                 << CodeObject->getFileName() << "\n");
      continue;
    }

    // DISubprogram required as scope for DILocations
    llvm::DISubprogram *FuncDISP = nullptr;

    // get all CUs in DWARFCtx
    for (const std::unique_ptr<llvm::DWARFUnit> &CU :
         DWARFCtx->compile_units()) {
      uint64_t CUOffset = CU->getOffset();
      llvm::DICompileUnit *DICU = nullptr;

      // Get or Build the DICompileUnit
      if (ObjFileCUOffsetToDICU[CodeObject].count(CUOffset)) {
        DICU = ObjFileCUOffsetToDICU[CodeObject][CUOffset];
      } else {
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

        DICU = DIB.createCompileUnit(SrcLang, CUFile, "Luthier Debug Info Pass",
                                     false, "", 0);
        ObjFileCUOffsetToDICU[CodeObject][CUOffset] = DICU;
      }

      // Create DISubprograms for all subprogram DIEs in this CU
      for (const auto &DieInfo : CU->dies()) {
        llvm::DWARFDie Die(CU.get(), &DieInfo);

        switch (Die.getTag()) {
        case (llvm::dwarf::DW_TAG_subprogram): {
          const char *LinkageName = Die.getLinkageName();
          const char *ShortName = Die.getShortName();
          llvm::StringRef LinkageNameRef = LinkageName ? LinkageName : "";
          llvm::StringRef ShortNameRef = ShortName ? ShortName : "";
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

          // TODO: Populate subroutine return and arg types
          llvm::DISubroutineType *SubType =
              DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));

          // Create a DISubprogram
          llvm::DISubprogram *DISP = DIB.createFunction(
              DICU, ShortNameRef, LinkageNameRef, UnitFile, LineNumber, SubType,
              LineNumber, llvm::DINode::FlagZero,
              llvm::DISubprogram::SPFlagDefinition);

          LLVM_DEBUG(llvm::dbgs()
                     << "[DebugInfoPass] Subprogram DIE: short='"
                     << ShortNameRef << "' linkage='" << LinkageNameRef
                     << "' line=" << LineNumber << " source_file= '"
                     << UnitFile->getFilename() << "'\n");

          // Link DISubprogram to Function if linkage name matches
          if (LinkageNameRef == F.getName() && !FuncDISP) {
            FuncDISP = DISP;
            F.setSubprogram(FuncDISP);
          }
          break;
        }

        // TODO: populate `DW_TAG_inlined_subroutine` DIEs
        case (llvm::dwarf::DW_TAG_inlined_subroutine): {
          break;
        }

        default:
          break;
        }
      }
    }

    if (!FuncDISP) {
      LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] No matching DISubprogram for "
                              << F.getName() << "\n");
      continue;
    }

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
          "No MachineFunction found for function {0}", F.getName()))));
      continue;
    }

    auto &MF = MFResult->getMF();

    // Iterate over all Traced Instructions
    for (llvm::MachineBasicBlock &MBB : MF) {
      for (llvm::MachineInstr &MI : MBB) {
        auto *MDNode = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
        if (!MDNode)
          continue;

        std::optional<uint64_t> TraceAddrOpt = MDNode->getTraceInstrAddress();
        if (!TraceAddrOpt)
          continue;

        uint64_t TraceAddr = *TraceAddrOpt;

        // Calculate offset from code object load base
        uint64_t Offset = TraceAddr - AllocBaseAddr;

        // Record the MI to trace + offset mapping
        MIToTrace[&MI] = {TraceAddr, Offset};

        LLVM_DEBUG(
            llvm::dbgs() << llvm::formatv(
                "[DebugInfoPass] MI from entry addr {0:x}, trace addr {1:x}, "
                "alloc base {2:x}, DWARF offset {3:x}\n",
                EntryPointAddr, TraceAddr, AllocBaseAddr, Offset));

        // TODO: hoist getLineTableForUnit out of MBB,MI loops

        // DWARF line table lookup using offset
        for (const std::unique_ptr<llvm::DWARFUnit> &CU :
             DWARFCtx->compile_units()) {
          const llvm::DWARFDebugLine::LineTable *LT =
              DWARFCtx->getLineTableForUnit(CU.get());
          if (!LT)
            continue;

          uint32_t RowIdx = LT->lookupAddress(
              {Offset, llvm::object::SectionedAddress::UndefSection});

          if (RowIdx != LT->UnknownRowIndex) {
            const llvm::DWARFDebugLine::Row &Row = LT->Rows[RowIdx];

            // Assign DILocation to MI
            llvm::DILocation *Loc =
                llvm::DILocation::get(Ctx, Row.Line, Row.Column, FuncDISP);
            MI.setDebugLoc(llvm::DebugLoc(Loc));

            // Propagate DILocation to IR
            auto It = PCSectionIRMap.find(MI.getPCSections());
            if (It != PCSectionIRMap.end()) {
              for (llvm::Instruction *I : It->second) {
                I->setDebugLoc(llvm::DebugLoc(Loc));
              }
            }

            LLVM_DEBUG({
              std::string FileName;
              if (LT->getFileNameByIndex(Row.File, CU->getCompilationDir(),
                                         llvm::DILineInfoSpecifier::
                                             FileLineInfoKind::AbsoluteFilePath,
                                         FileName)) {
                llvm::dbgs() << llvm::formatv(
                    "[DebugInfoPass] Attached {0}:{1}:{2} to MI\n",
                    llvm::sys::path::filename(FileName), Row.Line, Row.Column);
              }
            });
            // First CU whose line table covers this address wins.
            break;
          }
        }
      }
    }
  }

  DIB.finalize();

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
