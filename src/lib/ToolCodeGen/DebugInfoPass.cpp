#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/DebugInfoPass.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "luthier-debug-info"

namespace luthier {

llvm::PreservedAnalyses DebugInfoPass::run(llvm::Module &M,
                                           llvm::ModuleAnalysisManager &AM) {
  llvm::LLVMContext &Ctx = M.getContext();

  const MemoryAllocationAccessor &SegAccessor =
      AM.getResult<MemoryAllocationAnalysis>(M).getAccessor();

  auto &MFAMProxy =
      AM.getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(M);
  auto &MFAM = MFAMProxy.getManager();

  auto &FAMProxy = AM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M);
  auto &FAM = FAMProxy.getManager();

  llvm::DIBuilder DIB(M);

  // Code object to DWARFContext cache
  std::unordered_map<const llvm::object::ObjectFile *,
                     std::unique_ptr<llvm::DWARFContext>>
      ContextCache;

  // DWARF CompileUnit offset to DICompileUnit cache
  std::unordered_map<uint64_t, llvm::DICompileUnit *> OffsetToCUMap;

  // early exit flag
  bool AnyDebugInfoFound = false;

  for (llvm::Function &F : M) {
    if (!F.hasFnAttribute(ElfSymbolAttr))
      continue;

    // Get entry point address
    auto EntryPointOpt = getFunctionEntryPoint(F);
    if (!EntryPointOpt) {
      LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] No entry point for function "
                              << F.getName() << "\n");
      continue;
    }
    uint64_t EntryPointAddr = EntryPointOpt->getEntryPointAddress();

    // Get allocation descriptor and code object
    auto AllocationOrErr = SegAccessor.getAllocationDescriptor(EntryPointAddr);
    if (!AllocationOrErr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] No allocation descriptor for function "
                 << F.getName() << "\n");
      llvm::consumeError(AllocationOrErr.takeError());
      continue;
    }

    const auto *CodeObject = AllocationOrErr->getAllocationCodeObject();
    if (!CodeObject) {
      LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] No CodeObject for function "
                              << F.getName() << "\n");
      continue;
    }

    // Code object load base address
    // DWARF addresses are relative to this
    uint64_t AllocBaseAddr = reinterpret_cast<uint64_t>(
        AllocationOrErr->getDeviceAllocation().data());

    // Create/cache DWARFContext
    if (ContextCache.find(CodeObject) == ContextCache.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] Creating DWARFContext for CodeObject\n");
      ContextCache[CodeObject] = llvm::DWARFContext::create(*CodeObject);
    }

    llvm::DWARFContext *DICtx = ContextCache[CodeObject].get();

    // Skip if no debug info is found for this CodeObject
    if (DICtx->getNumCompileUnits() == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[DebugInfoPass] No compile units in DWARF for "
                 << F.getName() << "\n");
      continue;
    }
    AnyDebugInfoFound = true;

    // -------------------------------------------
    // Poplulate IR level debug metadata from DICtx
    // DISubprogram required as scope before populating DILocations
    // -------------------------------------------

    // init function scope
    llvm::DISubprogram *FuncSP = nullptr;

    // get all CUs in DICtx
    for (const std::unique_ptr<llvm::DWARFUnit> &CU : DICtx->compile_units()) {
      uint64_t CUOffset = CU->getOffset();
      llvm::DICompileUnit *DICU = nullptr;

      // Populate the offset to CU Map
      if (OffsetToCUMap.count(CUOffset)) {
        DICU = OffsetToCUMap[CUOffset];
      } else {
        llvm::StringRef CUName =
            CU->getUnitDIE().getName(llvm::DINameKind::ShortName);
        llvm::StringRef CUDir = CU->getCompilationDir();

        LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] Creating DICompileUnit '"
                                << CUName << "' in dir '" << CUDir << "'\n");

        llvm::DIFile *CUFile = DIB.createFile(CUName, CUDir);

        // ? replace hardcoded C++ src language

        unsigned SrcLang = llvm::dwarf::DW_LANG_C_plus_plus;
        DICU = DIB.createCompileUnit(SrcLang, CUFile, "Luthier Debug Info Pass",
                                     false, "", 0);
        OffsetToCUMap[CUOffset] = DICU;
      }

      // Create DISubprograms for all subprogram DIEs in this CU
      for (const auto &DieInfo : CU->dies()) {
        llvm::DWARFDie Die(CU.get(), &DieInfo);
        if (Die.getTag() != llvm::dwarf::DW_TAG_subprogram)
          continue;

        const char *LinkageName = Die.getLinkageName();
        const char *ShortName = Die.getShortName();
        llvm::StringRef LinkageNameRef = LinkageName ? LinkageName : "";
        llvm::StringRef ShortNameRef = ShortName ? ShortName : "";
        uint32_t LineNumber = Die.getDeclLine();

        LLVM_DEBUG(llvm::dbgs()
                   << "[DebugInfoPass] Subprogram DIE: short='" << ShortNameRef
                   << "' linkage='" << LinkageNameRef << "' line=" << LineNumber
                   << "\n");

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

        // init empty subroutine type
        llvm::DISubroutineType *SubType =
            DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));

        llvm::DISubprogram *SP = DIB.createFunction(
            DICU, ShortNameRef.empty() ? LinkageNameRef : ShortNameRef,
            LinkageNameRef, UnitFile, LineNumber, SubType, LineNumber,
            llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition);

        // Link Subprogram DI with the function if linkage name matches
        if (LinkageNameRef == F.getName() && !FuncSP) {
          FuncSP = SP;
          F.setSubprogram(FuncSP);
        }
      }
    }

    // Fallback if DISubprogram was not mached by linkage name
    if (!FuncSP) {
      LLVM_DEBUG(llvm::dbgs() << "[DebugInfoPass] No matching DISubprogram for "
                              << F.getName() << ", creating fallback\n");

      // Use the first compile unit as scope
      llvm::DICompileUnit *FallbackCU = nullptr;
      if (!OffsetToCUMap.empty()) {
        FallbackCU = OffsetToCUMap.begin()->second;
      }
      if (FallbackCU) {
        llvm::DIFile *FallbackFile =
            DIB.createFile(F.getName(), FallbackCU->getDirectory());
        llvm::DISubroutineType *SubType =
            DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
        FuncSP = DIB.createFunction(
            FallbackCU, F.getName(), F.getName(), FallbackFile, 0, SubType, 0,
            llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition);
        F.setSubprogram(FuncSP);
      }
    }

    // -------------------------------------
    // Populating MIR Debug Info
    // -------------------------------------

    auto &MFAnalysis = FAM.getResult<llvm::MachineFunctionAnalysis>(F);
    llvm::MachineFunction *MF = &MFAnalysis.getMF();

    // Throwing error if MF not found
    if (!MF) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "No MachineFunction found for function {0}", F.getName()))));
      continue;
    }

    if (!FuncSP)
      continue;

    // PC Section to IR instrcution map used to match MIR to IR instructions
    llvm::DenseMap<llvm::MDNode*, llvm::SmallVector<llvm::Instruction*>> PCSectionIRMap;

    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB) {
        if (auto *PCS = I.getMetadata(llvm::LLVMContext::MD_pcsections)) {
          PCSectionIRMap[PCS].push_back(&I);
        }
      }
    }

    // Iterate over all Traced Instructions
    for (llvm::MachineBasicBlock &MBB : *MF) {
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
        Mapping.Map[&MI] = {TraceAddr, Offset};

        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "[DebugInfoPass] MI trace addr {0:x}, entry {1:x}, "
                       "alloc base {2:x}, DWARF offset {3:x}\n",
                       TraceAddr, EntryPointAddr, AllocBaseAddr, Offset));

        // DWARF line table lookup using offset
        for (const std::unique_ptr<llvm::DWARFUnit> &CU :
             DICtx->compile_units()) {
          const llvm::DWARFDebugLine::LineTable *LT =
              DICtx->getLineTableForUnit(CU.get());
          if (!LT)
            continue;

          uint32_t RowIdx = LT->lookupAddress(
              {Offset, llvm::object::SectionedAddress::UndefSection});

          if (RowIdx != LT->UnknownRowIndex) {
            const llvm::DWARFDebugLine::Row &Row = LT->Rows[RowIdx];

            // Assign DILocation to MI
            llvm::DILocation *Loc =
                llvm::DILocation::get(Ctx, Row.Line, Row.Column, FuncSP);
            MI.setDebugLoc(llvm::DebugLoc(Loc));
            
            // Match MIR to PC sections to propagate DILocation
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
            break;
          }
        }
      }
    }
  }

  if (!AnyDebugInfoFound)
    return llvm::PreservedAnalyses::all();

  DIB.finalize();

  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<llvm::MachineFunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
  return PA;
}

llvm::PreservedAnalyses
DebugInfoPrinterPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  const MemoryAllocationAccessor &SegAccessor =
      AM.getResult<MemoryAllocationAnalysis>(M).getAccessor();

  std::unordered_set<const llvm::object::ObjectFile *> DumpedObjects;

  OS << "=== DWARF Debug Info for Module: " << M.getName() << " ===\n";

  for (llvm::Function &F : M) {
    if (!F.hasFnAttribute(ElfSymbolAttr))
      continue;

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

      std::unique_ptr<llvm::DWARFContext> DICtx =
          llvm::DWARFContext::create(*CodeObject);

      llvm::DIDumpOptions DumpOpts;
      DumpOpts.ShowChildren = true;
      DumpOpts.ShowParents = true;
      DumpOpts.ShowForm = true;
      DumpOpts.SummarizeTypes = false;
      DumpOpts.Verbose = true;

      DICtx->dump(OS, DumpOpts);
    }
  }

  OS << "\n=== End DWARF Debug Info ===\n";

  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
