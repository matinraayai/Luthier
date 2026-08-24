//===-- InstrFuzzerMain.cpp - AMDGPU Instruction Semantic Fuzzer ----------===//
//
// Reference-path bring-up. For each representative opcode:
//   analyze -> classify + build binding steps -> build the MachineInstr ->
//   machine-verify -> emit a loadable code object -> dispatch on the GPU.
//
// Tier-2 additionally raises the reference kernel back to LLVM IR through
// luthier::TraceFunctionTranslator, recompiles it, and compares the translated
// kernel's outputs against the reference's -- a mismatch is a translator bug.
//
// Nothing here is pinned to one GPU generation: representatives are named by
// canonical AMDGPU pseudo enum and resolved against the subtarget under test
// (see resolveOpcode), and reps with no encoding on the running target are
// skipped rather than failed.
//
//===----------------------------------------------------------------------===//
#include "FuzzerDriver.h"
#include "HSADispatcher.h"
#include "InstrDescriptor.h"
#include "MachineKernelBuilder.h"
#include "RefKernelSupport.h" // waveSize()

#include "luthier/ToolCodeGen/TraceFunctionTranslator.h"

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/InitializePasses.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

using namespace luthier::test;

namespace {

std::unique_ptr<HSADispatcher> GDispatcher;
std::unique_ptr<llvm::TargetMachine> GTM;
std::unique_ptr<InstrDescriptor> GDesc;
std::unique_ptr<FuzzerDriver> GDriver;
/// Wavefront size of the target the tests are running on (32 or 64). The
/// full-wave / cross-lane tests size a wave from this rather than assuming 64,
/// so they are correct on both RDNA wave32 and CDNA / wave64.
unsigned GWaveSize = 64;
/// Holds the throwaway module/function the subtarget is queried through (see
/// GTII); kept alive for the lifetime of the environment.
std::unique_ptr<llvm::LLVMContext> GProbeCtx;
std::unique_ptr<llvm::Module> GProbeMod;
/// SIInstrInfo for the target under test, used to ask LLVM whether a given
/// pseudo has an encoding here. Obtaining a GCNSubtarget needs a Function, so
/// FuzzerEnv creates one throwaway function to hang it off.
const llvm::SIInstrInfo *GTII = nullptr;

std::optional<unsigned> findOpcode(llvm::StringRef Name) {
  for (unsigned I = 0, E = GDesc->getNumOpcodes(); I < E; ++I)
    if (GDesc->getName(I) == Name)
      return I;
  return std::nullopt;
}

/// True if \p Opcode is a pseudo that lowers to a real encoding on the target
/// under test. Pseudos are generation-scoped: LLVM's `_mc` multiclasses emit
/// one pseudo per encoding rule (e.g. DS_READ_B32 reads M0, DS_READ_B32_gfx9
/// does not), and only the matching one has an encoding on a given subtarget.
bool isEncodable(unsigned Opcode) {
  return GTII && GTII->pseudoToMCOpcode(static_cast<int>(Opcode)) >= 0;
}

/// Resolve a canonical pseudo opcode to the variant that is actually encodable
/// on the target under test.
///
/// Tests name instructions by their canonical AMDGPU enum (e.g.
/// \c llvm::AMDGPU::DS_READ_B32). On GFX9+ the encodable pseudo for the DS
/// family is the `_gfx9` sibling instead -- and that sibling is also what
/// Luthier's real-to-pseudo mapper hands the translator for GFX10 code, so it
/// is the variant the fuzzer must exercise. Only sibling *pseudos* are
/// considered; the `_gfx10` / `_vi` / `_gfx6_gfx7` names in the opcode enum are
/// real encodings, never build targets.
///
/// \returns the encodable opcode, or std::nullopt if this subtarget has none.
/// That is a legitimate answer, not an error: an instruction a generation
/// removed outright (V_MAD_MIX_F32 on GFX10) simply has no representative here,
/// and callers skip it rather than failing.
std::optional<unsigned> resolveOpcode(unsigned CanonicalOpcode) {
  if (isEncodable(CanonicalOpcode))
    return CanonicalOpcode;
  /// The only pseudo-level generation split LLVM models this way.
  static constexpr llvm::StringLiteral SiblingSuffix = "_gfx9";
  const std::string Sibling =
      (GDesc->getName(CanonicalOpcode) + SiblingSuffix).str();
  auto Op = findOpcode(Sibling);
  /// The `_gfx9` suffix is overloaded in the AMDGPU opcode enum, and the
  /// difference matters: for the DS family it names a sibling *pseudo* (the
  /// variant that does not read M0), but for others -- e.g.
  /// V_ADD_CO_U32_e32_gfx9 -- it names the GFX9 *real encoding* of the plain
  /// pseudo. Only a pseudo may be built. Emitting a real encoding puts a GFX9
  /// instruction word into a GFX10 kernel, which the hardware happily runs as
  /// something else: the reference kernel then returns zeros and the fuzzer
  /// blames the translator for a divergence it did not cause.
  if (Op && isEncodable(*Op) && GTII->get(*Op).isPseudo())
    return Op;
  return std::nullopt;
}

class FuzzerEnv : public ::testing::Environment {
public:
  void SetUp() override {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUAsmParser();
    llvm::initializeCodeGen(*llvm::PassRegistry::getPassRegistry());

    GDispatcher = std::make_unique<HSADispatcher>();
    auto Err = GDispatcher->init();
    ASSERT_FALSE(static_cast<bool>(Err))
        << "HSA init failed: " << llvm::toString(std::move(Err));
    ASSERT_GT(GDispatcher->getNumGpuAgents(), 0u) << "no GPU agents found";

    const llvm::Triple TT("amdgcn-amd-amdhsa");
    std::string Error;
    const llvm::Target *T = llvm::TargetRegistry::lookupTarget(TT, Error);
    ASSERT_NE(T, nullptr) << "AMDGPU target not found: " << Error;
    llvm::TargetOptions Opts;
    // Wave-mode selection: defaults to wave64 so results are stable across
    // subtargets (LLVM's own AMDGPU default is wave32 on RDNA, wave64-only on
    // CDNA). Override with LUTHIER_WAVE=32 or 64, e.g. to compare against
    // RDNA's wave32 mode explicitly.
    std::string Features = "+wavefrontsize64,-wavefrontsize32";
    if (const char *W = std::getenv("LUTHIER_WAVE")) {
      llvm::StringRef WS(W);
      if (WS == "32")
        Features = "+wavefrontsize32,-wavefrontsize64";
      else if (WS == "64")
        Features = "+wavefrontsize64,-wavefrontsize32";
    }
    GTM.reset(T->createTargetMachine(TT, GDispatcher->getGpuTarget(), Features,
                                     Opts, std::nullopt, std::nullopt));
    ASSERT_NE(GTM, nullptr);
    GWaveSize = waveSize(*GTM);
    llvm::errs() << "Target " << GDispatcher->getGpuTarget() << ", wave"
                 << GWaveSize << "\n";

    // A GCNSubtarget (and so SIInstrInfo) is only reachable through a Function,
    // so hang one throwaway function off a throwaway module to get at the
    // subtarget's encoding tables. Used by isEncodable / resolveOpcode.
    GProbeCtx = std::make_unique<llvm::LLVMContext>();
    GProbeMod = std::make_unique<llvm::Module>("opcode_probe", *GProbeCtx);
    GProbeMod->setTargetTriple(TT);
    GProbeMod->setDataLayout(GTM->createDataLayout());
    auto *ProbeFTy =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*GProbeCtx), false);
    auto *ProbeF =
        llvm::Function::Create(ProbeFTy, llvm::GlobalValue::ExternalLinkage,
                               "opcode_probe", GProbeMod.get());
    GTII = static_cast<const llvm::GCNSubtarget *>(GTM->getSubtargetImpl(*ProbeF))
               ->getInstrInfo();
    ASSERT_NE(GTII, nullptr);

    GDesc = std::make_unique<InstrDescriptor>(*GTM);
    GDriver = std::make_unique<FuzzerDriver>(*GDispatcher, *GTM, *GDesc);
  }

  void TearDown() override {
    GDriver.reset();
    GDesc.reset();
    GTII = nullptr;
    GProbeMod.reset();
    GProbeCtx.reset();
    GTM.reset();
    if (GDispatcher)
      GDispatcher->shutdown();
    GDispatcher.reset();
  }
};

/// Tier-1: build the reference kernel, print the binding log, run the machine
/// verifier, and emit a code object. \returns the built ELF on success.
llvm::Expected<llvm::SmallVector<char, 0>> tier1(unsigned CanonicalOpcode) {
  auto Op = resolveOpcode(CanonicalOpcode);
  if (!Op)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        GDesc->getName(CanonicalOpcode).str() +
            " has no encoding on this subtarget");
  const llvm::StringRef OpcodeName = GDesc->getName(*Op);
  InstrProfile P = GDesc->analyze(*Op);
  MachineKernelBuilder Builder(*GTM);
  KernargLayout Layout;
  auto KCtxOrErr = Builder.build(P, Layout);
  if (!KCtxOrErr)
    return KCtxOrErr.takeError();

  llvm::errs() << "--- " << OpcodeName << " ---\n"
               << KCtxOrErr->BindingLog;
  KCtxOrErr->MF->print(llvm::errs());
  llvm::errs() << "\n";

  if (!KCtxOrErr->MF->verify(nullptr, "reference kernel", &llvm::errs(),
                             /*AbortOnError=*/false))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "machine verifier failed");

  return Builder.emitToELF(*KCtxOrErr);
}

/// One representative instruction for a behavioural subgroup.
///
/// \c Opcode is the canonical AMDGPU pseudo enum; \c resolveOpcode maps it to
/// whichever generation-specific sibling this subtarget can actually encode, so
/// the tables stay subtarget-agnostic. A rep with no representative here is
/// skipped by the runners, not failed.
struct Reps {
  unsigned Opcode;
  const char *Subgroup;
};

/// The opcode a rep resolves to on this subtarget.
std::optional<unsigned> resolveRep(const Reps &R) {
  return resolveOpcode(R.Opcode);
}

/// Display name for a rep: the instruction actually built where one resolves,
/// otherwise the canonical name so diagnostics still identify it.
std::string repName(const Reps &R) {
  if (auto Op = resolveRep(R))
    return GDesc->getName(*Op).str();
  return GDesc->getName(R.Opcode).str();
}

/// Tier-1 for a rep, resolved for this subtarget.
llvm::Expected<llvm::SmallVector<char, 0>> tier1(const Reps &R) {
  auto Op = resolveRep(R);
  if (!Op)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   GDesc->getName(R.Opcode).str() +
                                       " has no encoding on this subtarget");
  return tier1(*Op);
}
// Emittable representatives (one+ per subgroup) — must pass tier 1 & 2.
// V_MADAK (literal at end) and V_MADMK (literal in the middle) both exercise
// the Sg2-literal path; both have a gfx9 encoding.
const Reps kReps[] = {
    {llvm::AMDGPU::V_ADD_F32_e32, "Sg1"},        {llvm::AMDGPU::V_AND_B32_e32, "Sg1"},
    {llvm::AMDGPU::V_MADAK_F32, "Sg2-literal-end"}, {llvm::AMDGPU::V_MADMK_F32, "Sg2-literal-mid"},
    {llvm::AMDGPU::V_CNDMASK_B32_e32, "Sg2-cndmask"},
    {llvm::AMDGPU::V_ADDC_U32_e32, "Sg3-carry"}, {llvm::AMDGPU::V_SUBB_U32_e32, "Sg3-carry"},
    {llvm::AMDGPU::V_ADD_CO_U32_e32, "carry-out-only"},
    {llvm::AMDGPU::V_MAC_F32_e32, "tied"},       {llvm::AMDGPU::V_FMAC_F32_e32, "tied"},
};

// The FMA-literal VOP2 pseudos. Whether these are emittable is a property of
// the subtarget, not a fixed fact: they have no gfx908 encoding but a perfectly
// good GFX10 one. The test below asserts the builder's behaviour *matches* what
// the subtarget reports, rather than assuming either answer.
const unsigned kFMALiteral[] = {llvm::AMDGPU::V_FMAAK_F32,
                                llvm::AMDGPU::V_FMAMK_F32};

// VOP1 (single-source VALU: vdst, src0) representatives. Same prolog/epilog
// logic as VOP2, degenerate to one source; no modifier operands. 64-bit VOP1
// ops (v_cvt_f64_*, v_rcp_f64, ...) are rejected cleanly by the 32-bit guard.
const Reps kVOP1Reps[] = {
    {llvm::AMDGPU::V_MOV_B32_e32, "VOP1-copy"},        {llvm::AMDGPU::V_NOT_B32_e32, "VOP1-bitwise"},
    {llvm::AMDGPU::V_CVT_F32_I32_e32, "VOP1-cvt"},     {llvm::AMDGPU::V_CVT_I32_F32_e32, "VOP1-cvt"},
    {llvm::AMDGPU::V_RCP_F32_e32, "VOP1-transcend"},   {llvm::AMDGPU::V_FLOOR_F32_e32, "VOP1-round"},
    {llvm::AMDGPU::V_BFREV_B32_e32, "VOP1-bitwise"},   {llvm::AMDGPU::V_FFBH_U32_e32, "VOP1-bitcount"},
};

// VOPC (compare: src0, src1 -> VCC mask) representatives. Same VGPR inputs as
// VOP2; no explicit VGPR output — the result is the implicit VCC def (captured
// via E3). e32 form only (no modifiers).
const Reps kVOPCReps[] = {
    {llvm::AMDGPU::V_CMP_EQ_F32_e32, "VOPC-f32"}, {llvm::AMDGPU::V_CMP_LT_F32_e32, "VOPC-f32"},
    {llvm::AMDGPU::V_CMP_EQ_U32_e32, "VOPC-u32"}, {llvm::AMDGPU::V_CMP_NE_U32_e32, "VOPC-u32"},
    {llvm::AMDGPU::V_CMP_LT_I32_e32, "VOPC-i32"}, {llvm::AMDGPU::V_CMP_GT_U32_e32, "VOPC-u32"},
};

// VOP3 (3-input / e64) representatives. Integer forms (add3/med3) have no
// modifier operands; float forms (fma/med3_f32) interleave src_modifiers +
// clamp/omod, all disabled (encoded 0). src0..src2 are scalar-or-vector.
const Reps kVOP3Reps[] = {
    {llvm::AMDGPU::V_ADD3_U32_e64, "VOP3-int3"},   {llvm::AMDGPU::V_MED3_I32_e64, "VOP3-int3"},
    {llvm::AMDGPU::V_MED3_U32_e64, "VOP3-int3"},   {llvm::AMDGPU::V_MAD_U32_U24_e64, "VOP3-madint"},
    {llvm::AMDGPU::V_FMA_F32_e64, "VOP3-fma+mods"}, {llvm::AMDGPU::V_MED3_F32_e64, "VOP3-f32+mods"},
};

// VOP3P (packed math) representatives. op_sel/op_sel_hi/neg_lo/neg_hi disabled.
const Reps kVOP3PReps[] = {
    {llvm::AMDGPU::V_PK_ADD_F16, "VOP3P-pk"},  {llvm::AMDGPU::V_PK_MUL_F16, "VOP3P-pk"},
    // GFX10 replaced V_MAD_MIX_F32 with V_FMA_MIX_F32, which is already a rep
    // in its own right -- so the mix subgroup stays covered there and
    // V_MAD_MIX_F32 simply skips on subtargets that lack it.
    {llvm::AMDGPU::V_FMA_MIX_F32, "VOP3P-mix"},
    {llvm::AMDGPU::V_MAD_MIX_F32, "VOP3P-mix"},
};

// SOP (scalar ALU) representatives. SOP1/SOP2 (SGPR in/out, SCC on most),
// SOPK (16-bit literal, tied dst). Scalar path + SCC seed/capture.
const Reps kSOPReps[] = {
    {llvm::AMDGPU::S_MOV_B32, "SOP1"},        {llvm::AMDGPU::S_BREV_B32, "SOP1"},
    {llvm::AMDGPU::S_NOT_B32, "SOP1-scc"},    {llvm::AMDGPU::S_ADD_U32, "SOP2-scc"},
    {llvm::AMDGPU::S_AND_B32, "SOP2-scc"},    {llvm::AMDGPU::S_LSHL_B32, "SOP2-scc"},
    {llvm::AMDGPU::S_ADDC_U32, "SOP2-scc-io"}, {llvm::AMDGPU::S_CSELECT_B32, "SOP2-scc-in"},
    {llvm::AMDGPU::S_MOVK_I32, "SOPK"},       {llvm::AMDGPU::S_ADDK_I32, "SOPK-tied-scc"},
    {llvm::AMDGPU::S_MULK_I32, "SOPK-tied"},
};

// GLOBAL (FLAT-family) representatives — the _SADDR 32-bit forms.
const Reps kGlobalReps[] = {
    {llvm::AMDGPU::GLOBAL_LOAD_DWORD_SADDR, "GLOBAL-Load"},
    {llvm::AMDGPU::GLOBAL_STORE_DWORD_SADDR, "GLOBAL-Store"},
    {llvm::AMDGPU::GLOBAL_ATOMIC_ADD_SADDR, "GLOBAL-AtomicNoRet"},
    {llvm::AMDGPU::GLOBAL_ATOMIC_ADD_SADDR_RTN, "GLOBAL-AtomicRet"},
};

// Plain FLAT representatives — 64-bit flat address (global aperture) + FLAT_SCR.
// Sub-dword GLOBAL representatives — byte / short load (zero + sign) and store.
const Reps kGlobalSubDwordReps[] = {
    {llvm::AMDGPU::GLOBAL_LOAD_UBYTE_SADDR, "GLOBAL-LoadU8"},
    {llvm::AMDGPU::GLOBAL_LOAD_SBYTE_SADDR, "GLOBAL-LoadS8"},
    {llvm::AMDGPU::GLOBAL_LOAD_USHORT_SADDR, "GLOBAL-LoadU16"},
    {llvm::AMDGPU::GLOBAL_LOAD_SSHORT_SADDR, "GLOBAL-LoadS16"},
    {llvm::AMDGPU::GLOBAL_STORE_BYTE_SADDR, "GLOBAL-Store8"},
    {llvm::AMDGPU::GLOBAL_STORE_SHORT_SADDR, "GLOBAL-Store16"},
};

// CMPSWAP representatives — the data operand packs {swap, cmp}; 32- and 64-bit.
const Reps kGlobalCmpSwapReps[] = {
    {llvm::AMDGPU::GLOBAL_ATOMIC_CMPSWAP_SADDR, "GLOBAL-CmpSwap"},
    {llvm::AMDGPU::GLOBAL_ATOMIC_CMPSWAP_SADDR_RTN, "GLOBAL-CmpSwapRet"},
    {llvm::AMDGPU::GLOBAL_ATOMIC_CMPSWAP_X2_SADDR, "GLOBAL-CmpSwap64"},
    {llvm::AMDGPU::GLOBAL_ATOMIC_CMPSWAP_X2_SADDR_RTN, "GLOBAL-CmpSwap64Ret"},
};

// MUBUF (untyped buffer) OFFSET-form representatives — V# + scalar offset.
const Reps kMUBUFReps[] = {
    {llvm::AMDGPU::BUFFER_LOAD_DWORD_OFFSET, "MUBUF-Load"},
    {llvm::AMDGPU::BUFFER_STORE_DWORD_OFFSET, "MUBUF-Store"},
    {llvm::AMDGPU::BUFFER_LOAD_DWORDX2_OFFSET, "MUBUF-Load-x2"},
    {llvm::AMDGPU::BUFFER_LOAD_DWORDX4_OFFSET, "MUBUF-Load-x4"},
    {llvm::AMDGPU::BUFFER_STORE_DWORDX2_OFFSET, "MUBUF-Store-x2"},
    {llvm::AMDGPU::BUFFER_LOAD_UBYTE_OFFSET, "MUBUF-LoadU8"},
    {llvm::AMDGPU::BUFFER_LOAD_USHORT_OFFSET, "MUBUF-LoadU16"},
    {llvm::AMDGPU::BUFFER_STORE_BYTE_OFFSET, "MUBUF-Store8"},
    {llvm::AMDGPU::BUFFER_ATOMIC_ADD_OFFSET, "MUBUF-AtomicNoRet"},
    {llvm::AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_RTN, "MUBUF-AtomicRet"},
};

// MUBUF vaddr-form representatives — OFFEN / IDXEN / BOTHEN (vaddr held at 0).
const Reps kMUBUFVAddrReps[] = {
    {llvm::AMDGPU::BUFFER_LOAD_DWORD_OFFEN, "MUBUF-Load-offen"},
    {llvm::AMDGPU::BUFFER_STORE_DWORD_OFFEN, "MUBUF-Store-offen"},
    {llvm::AMDGPU::BUFFER_LOAD_DWORD_IDXEN, "MUBUF-Load-idxen"},
    {llvm::AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_RTN, "MUBUF-AtomicRet-offen"},
};

// MTBUF (typed buffer) representatives — 32-bit UINT format passthrough.
const Reps kMTBUFReps[] = {
    {llvm::AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFSET, "MTBUF-Load"},
    {llvm::AMDGPU::TBUFFER_STORE_FORMAT_X_OFFSET, "MTBUF-Store"},
    {llvm::AMDGPU::TBUFFER_LOAD_FORMAT_XY_OFFSET, "MTBUF-Load-xy"},
    {llvm::AMDGPU::TBUFFER_LOAD_FORMAT_XYZW_OFFSET, "MTBUF-Load-xyzw"},
    {llvm::AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFEN, "MTBUF-Load-offen"},
};

// Non-SADDR GLOBAL representatives — a full 64-bit vaddr, no saddr base pointer.
const Reps kGlobalVAddrReps[] = {
    {llvm::AMDGPU::GLOBAL_LOAD_DWORD, "GLOBALv-Load"},
    {llvm::AMDGPU::GLOBAL_STORE_DWORD, "GLOBALv-Store"},
    {llvm::AMDGPU::GLOBAL_ATOMIC_ADD, "GLOBALv-AtomicNoRet"},
    {llvm::AMDGPU::GLOBAL_ATOMIC_ADD_RTN, "GLOBALv-AtomicRet"},
};

const Reps kFlatReps[] = {
    {llvm::AMDGPU::FLAT_LOAD_DWORD, "FLAT-Load"},
    {llvm::AMDGPU::FLAT_STORE_DWORD, "FLAT-Store"},
    {llvm::AMDGPU::FLAT_ATOMIC_ADD, "FLAT-AtomicNoRet"},
    {llvm::AMDGPU::FLAT_ATOMIC_ADD_RTN, "FLAT-AtomicRet"},
};


// SCRATCH (flat-scratch) representatives — SADDR (SGPR byte offset) and plain
// (VGPR byte offset), load + store.
const Reps kScratchReps[] = {
    {llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR, "SCRATCH-SADDR-Load"},
    {llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR, "SCRATCH-SADDR-Store"},
    {llvm::AMDGPU::SCRATCH_LOAD_DWORD, "SCRATCH-Load"},
    {llvm::AMDGPU::SCRATCH_STORE_DWORD, "SCRATCH-Store"},
};

// Sub-dword SCRATCH reps — byte / short load and store (SADDR + plain).
const Reps kScratchSubDwordReps[] = {
    {llvm::AMDGPU::SCRATCH_LOAD_UBYTE_SADDR, "SCRATCH-LoadU8"},
    {llvm::AMDGPU::SCRATCH_LOAD_USHORT_SADDR, "SCRATCH-LoadU16"},
    {llvm::AMDGPU::SCRATCH_STORE_BYTE_SADDR, "SCRATCH-Store8"},
    {llvm::AMDGPU::SCRATCH_STORE_SHORT_SADDR, "SCRATCH-Store16"},
    {llvm::AMDGPU::SCRATCH_LOAD_UBYTE, "SCRATCH-LoadU8-v"},
    {llvm::AMDGPU::SCRATCH_STORE_BYTE, "SCRATCH-Store8-v"},
};

// Wide (multi-dword) memory reps: DWORDX2 / X4 tuples across GLOBAL, FLAT and
// SCRATCH, load + store.
const Reps kWideMemReps[] = {
    {llvm::AMDGPU::GLOBAL_LOAD_DWORDX2_SADDR, "GLOBAL-Load-x2"},
    {llvm::AMDGPU::GLOBAL_STORE_DWORDX2_SADDR, "GLOBAL-Store-x2"},
    {llvm::AMDGPU::GLOBAL_LOAD_DWORDX4_SADDR, "GLOBAL-Load-x4"},
    {llvm::AMDGPU::GLOBAL_STORE_DWORDX4_SADDR, "GLOBAL-Store-x4"},
    {llvm::AMDGPU::FLAT_LOAD_DWORDX2, "FLAT-Load-x2"},
    {llvm::AMDGPU::FLAT_STORE_DWORDX2, "FLAT-Store-x2"},
    {llvm::AMDGPU::SCRATCH_LOAD_DWORDX2_SADDR, "SCRATCH-Load-x2"},
    {llvm::AMDGPU::SCRATCH_STORE_DWORDX2_SADDR, "SCRATCH-Store-x2"},
    {llvm::AMDGPU::SCRATCH_LOAD_DWORDX4_SADDR, "SCRATCH-Load-x4"},
    {llvm::AMDGPU::SCRATCH_STORE_DWORDX4_SADDR, "SCRATCH-Store-x4"},
};

// SMEM (scalar memory) representatives — scalar loads of every width and every
// offset form (immediate, SGPR, SGPR+immediate).
const Reps kSMEMReps[] = {
    {llvm::AMDGPU::S_LOAD_DWORD_IMM, "SMEM-Load"},
    {llvm::AMDGPU::S_LOAD_DWORDX2_IMM, "SMEM-Load-x2"},
    {llvm::AMDGPU::S_LOAD_DWORDX4_IMM, "SMEM-Load-x4"},
    {llvm::AMDGPU::S_LOAD_DWORDX8_IMM, "SMEM-Load-x8"},
    {llvm::AMDGPU::S_LOAD_DWORDX16_IMM, "SMEM-Load-x16"},
    {llvm::AMDGPU::S_LOAD_DWORD_SGPR, "SMEM-Load-soff"},
    {llvm::AMDGPU::S_LOAD_DWORDX4_SGPR, "SMEM-Load-x4-soff"},
};

// S_BUFFER_LOAD representatives — sbase is a 128-bit V# resource descriptor
// (built in-kernel from the data-buffer pointer), across widths and offset forms.
const Reps kBufferSMEMReps[] = {
    {llvm::AMDGPU::S_BUFFER_LOAD_DWORD_IMM, "SBUF-Load"},
    {llvm::AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM, "SBUF-Load-x2"},
    {llvm::AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM, "SBUF-Load-x4"},
    {llvm::AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM, "SBUF-Load-x8"},
    {llvm::AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM, "SBUF-Load-x16"},
    {llvm::AMDGPU::S_BUFFER_LOAD_DWORD_SGPR, "SBUF-Load-soff"},
    {llvm::AMDGPU::S_BUFFER_LOAD_DWORDX4_SGPR, "SBUF-Load-x4-soff"},
};

// DS (LDS) representatives — one per shape handled this session.
const Reps kDSReps[] = {
    {llvm::AMDGPU::DS_READ_B32, "DS-Load"},
    {llvm::AMDGPU::DS_WRITE_B32, "DS-Store"},
    {llvm::AMDGPU::DS_ADD_U32, "DS-AtomicNoRet"},
    {llvm::AMDGPU::DS_ADD_F32, "DS-AtomicNoRet"},
    {llvm::AMDGPU::DS_ADD_RTN_U32, "DS-AtomicRet"},
};

// DS cross-lane permute representatives — full-wave (64-lane) gather/scatter.
const Reps kDSPermuteReps[] = {
    {llvm::AMDGPU::DS_BPERMUTE_B32, "DS-BPermute"},
    {llvm::AMDGPU::DS_PERMUTE_B32, "DS-Permute"},
};

// Wide-operand VOP representatives: 64-bit register tuples, including the
// mixed-width V_LSHLREV_B64 (32-bit shift amount + 64-bit source/dest).
const Reps kWideVOPReps[] = {
    {llvm::AMDGPU::V_ADD_F64_e64, "VOP3-64"},   {llvm::AMDGPU::V_MAX_F64_e64, "VOP3-64"},
    {llvm::AMDGPU::V_LSHLREV_B64_e64, "VOP3-64-mixed"},
};

// Wide-operand SOP representatives: 64-bit scalar tuples, including the
// mixed-width S_LSHL_B64 (64-bit source, 32-bit shift amount).
const Reps kWideSOPReps[] = {
    {llvm::AMDGPU::S_AND_B64, "SOP2-64"}, {llvm::AMDGPU::S_OR_B64, "SOP2-64"},
    {llvm::AMDGPU::S_NOT_B64, "SOP1-64"}, {llvm::AMDGPU::S_LSHL_B64, "SOP2-64-mixed"},
};

} // namespace

//===----------------------------------------------------------------------===//
// Tier 1: every representative must machine-verify and emit an ELF.
//===----------------------------------------------------------------------===//
TEST(RefPathVOP2, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_GE(ELF->size(), 4u) << repName(R);
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "Tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// Literal-in-middle VOP2 pseudos must be rejected cleanly (no crash).
//===----------------------------------------------------------------------===//
TEST(RefPathVOP2, UnsupportedRejectedCleanly) {
  for (unsigned Op : kFMALiteral) {
    const std::string Name = GDesc->getName(Op).str();
    auto ELF = tier1(Op);
    // Either outcome is correct; what must hold is that the builder agrees with
    // the subtarget and never crashes. On a target with no encoding it must
    // reject cleanly; where one exists it must actually build.
    if (!resolveOpcode(Op)) {
      EXPECT_FALSE(static_cast<bool>(ELF))
          << Name << " unexpectedly built; expected a clean rejection";
      if (!ELF)
        llvm::errs() << Name << ": rejected -> "
                     << llvm::toString(ELF.takeError()) << "\n";
    } else {
      EXPECT_TRUE(static_cast<bool>(ELF))
          << Name << " is encodable here but failed to build: "
          << llvm::toString(ELF.takeError());
      if (ELF)
        llvm::errs() << Name << ": encodable here, built " << ELF->size()
                     << " bytes\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// Sanity: V_ADD_F32 still computes a + b on the GPU.
//===----------------------------------------------------------------------===//
TEST(RefPathVOP2, AddF32ComputesSum) {
  auto Op = resolveOpcode(llvm::AMDGPU::V_ADD_F32_e32);
  ASSERT_TRUE(Op.has_value());
  const float A = 1.5f, B = 2.25f;
  uint32_t ABits, BBits;
  std::memcpy(&ABits, &A, 4);
  std::memcpy(&BBits, &B, 4);
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {ABits, BBits});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_GE(R.Outputs.size(), 1u);
  float Got;
  std::memcpy(&Got, &R.Outputs[0].Reference, 4);
  EXPECT_FLOAT_EQ(Got, A + B);
}

//===----------------------------------------------------------------------===//
// Tier 2: every representative dispatches and produces output.
//===----------------------------------------------------------------------===//
TEST(RefPathVOP2, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xC0FFEE);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    // Reference dispatched twice with identical inputs -> outputs must agree.
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): "
                 << Res.Outputs.size() << " output(s), "
                 << (Res.Passed ? "stable" : "UNSTABLE") << "\n";
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "Tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// VOP1 tier 1: build + verify + emit (reuses the VOP2 builder).
//===----------------------------------------------------------------------===//
TEST(RefPathVOP1, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kVOP1Reps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_GE(ELF->size(), 4u) << repName(R);
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "VOP1 tier-1 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// VOP1 functional: mov copies, not inverts.
//===----------------------------------------------------------------------===//
TEST(RefPathVOP1, MovAndNotCompute) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  const uint32_t In = 0x12345678;

  auto Mov = resolveOpcode(llvm::AMDGPU::V_MOV_B32_e32);
  ASSERT_TRUE(Mov.has_value());
  TestResult M = GDriver->testInstruction(Agent, *Mov, /*Seed=*/0, {In});
  llvm::errs() << formatResult(M);
  ASSERT_TRUE(M.ErrorMsg.empty()) << M.ErrorMsg;
  ASSERT_EQ(M.Outputs.size(), 1u);
  EXPECT_EQ(M.Outputs[0].Reference, In) << "v_mov should copy the input";

  auto Not = resolveOpcode(llvm::AMDGPU::V_NOT_B32_e32);
  ASSERT_TRUE(Not.has_value());
  TestResult N = GDriver->testInstruction(Agent, *Not, /*Seed=*/0, {In});
  llvm::errs() << formatResult(N);
  ASSERT_TRUE(N.ErrorMsg.empty()) << N.ErrorMsg;
  ASSERT_EQ(N.Outputs.size(), 1u);
  EXPECT_EQ(N.Outputs[0].Reference, ~In) << "v_not should invert the input";
}

//===----------------------------------------------------------------------===//
// VOP1 tier 2: dispatch and produce stable output.
//===----------------------------------------------------------------------===//
TEST(RefPathVOP1, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kVOP1Reps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x5A17ED);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "VOP1 tier-2 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// VOPC tier 1: build + verify + emit (reuses the VOP builder).
//===----------------------------------------------------------------------===//
TEST(RefPathVOPC, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kVOPCReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_GE(ELF->size(), 4u) << repName(R);
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "VOPC tier-1 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// VOPC functional: the VCC mask (lane-0 bit) reflects the comparison.
//===----------------------------------------------------------------------===//
TEST(RefPathVOPC, EqU32SetsMask) {
  auto Op = resolveOpcode(llvm::AMDGPU::V_CMP_EQ_U32_e32);
  ASSERT_TRUE(Op.has_value());
  const auto &Agent = GDispatcher->getGpuAgent(0);

  // Equal operands -> lane-0 VCC bit set.
  TestResult Eq = GDriver->testInstruction(Agent, *Op, /*Seed=*/0, {7u, 7u});
  llvm::errs() << formatResult(Eq);
  ASSERT_TRUE(Eq.ErrorMsg.empty()) << Eq.ErrorMsg;
  ASSERT_EQ(Eq.Outputs.size(), 1u);
  EXPECT_EQ(Eq.Outputs[0].Reference & 1u, 1u) << "7==7 should set VCC bit 0";

  // Unequal operands -> lane-0 VCC bit clear.
  TestResult Ne = GDriver->testInstruction(Agent, *Op, /*Seed=*/0, {7u, 9u});
  llvm::errs() << formatResult(Ne);
  ASSERT_TRUE(Ne.ErrorMsg.empty()) << Ne.ErrorMsg;
  ASSERT_EQ(Ne.Outputs.size(), 1u);
  EXPECT_EQ(Ne.Outputs[0].Reference & 1u, 0u) << "7==9 should clear VCC bit 0";
}

//===----------------------------------------------------------------------===//
// VOPC tier 2: dispatch and produce stable output.
//===----------------------------------------------------------------------===//
TEST(RefPathVOPC, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kVOPCReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xC0AFFE);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "VOPC tier-2 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// VOP3 / VOP3P tier 1: build + verify + emit (reuses the VOP builder). The
// modifier immediates must be encoded as 0, not the literal constant.
//===----------------------------------------------------------------------===//
static void runVOP3Tier1(llvm::ArrayRef<Reps> RepList, const char *Label) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : RepList) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_GE(ELF->size(), 4u) << repName(R);
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << Label << " tier-1 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

static void runVOP3Tier2(llvm::ArrayRef<Reps> RepList, const char *Label) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : RepList) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x3F00D);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << Label << " tier-2 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathVOP3, Tier1BuildVerifyEmit) { runVOP3Tier1(kVOP3Reps, "VOP3"); }
TEST(RefPathVOP3, Tier2Dispatch) { runVOP3Tier2(kVOP3Reps, "VOP3"); }
TEST(RefPathVOP3P, Tier1BuildVerifyEmit) { runVOP3Tier1(kVOP3PReps, "VOP3P"); }
TEST(RefPathVOP3P, Tier2Dispatch) { runVOP3Tier2(kVOP3PReps, "VOP3P"); }

//===----------------------------------------------------------------------===//
// VOP3 functional: V_ADD3_U32 = src0 + src1 + src2 (no modifiers).
//===----------------------------------------------------------------------===//
TEST(RefPathVOP3, Add3Computes) {
  auto Op = resolveOpcode(llvm::AMDGPU::V_ADD3_U32_e64);
  ASSERT_TRUE(Op.has_value());
  const uint32_t A = 100, B = 20, C = 3;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {A, B, C});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, A + B + C) << "v_add3_u32 = s0+s1+s2";
}

//===----------------------------------------------------------------------===//
// SOP (scalar ALU) tier 1: build + verify + emit.
//===----------------------------------------------------------------------===//
TEST(RefPathSOP, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kSOPReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_GE(ELF->size(), 4u) << repName(R);
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "SOP tier-1 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// SOP functional: S_ADD_U32 (sum + SCC carry) and S_CSELECT_B32 (seeded SCC).
//===----------------------------------------------------------------------===//
TEST(RefPathSOP, AddU32SumAndCarry) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_ADD_U32);
  ASSERT_TRUE(Op.has_value());
  const auto &Agent = GDispatcher->getGpuAgent(0);

  // No carry: 100 + 20 = 120, SCC = 0.
  TestResult A = GDriver->testInstruction(Agent, *Op, /*Seed=*/0, {100u, 20u});
  llvm::errs() << formatResult(A);
  ASSERT_TRUE(A.ErrorMsg.empty()) << A.ErrorMsg;
  ASSERT_EQ(A.Outputs.size(), 2u); // sdst, scc_out
  EXPECT_EQ(A.Outputs[0].Reference, 120u);
  EXPECT_EQ(A.Outputs[1].Reference & 1u, 0u) << "no unsigned carry";

  // Carry: 0xFFFFFFFF + 2 wraps, SCC = 1.
  TestResult C =
      GDriver->testInstruction(Agent, *Op, /*Seed=*/0, {0xFFFFFFFFu, 2u});
  llvm::errs() << formatResult(C);
  ASSERT_TRUE(C.ErrorMsg.empty()) << C.ErrorMsg;
  ASSERT_EQ(C.Outputs.size(), 2u);
  EXPECT_EQ(C.Outputs[0].Reference, 1u);
  EXPECT_EQ(C.Outputs[1].Reference & 1u, 1u) << "unsigned carry out";
}

TEST(RefPathSOP, CselectPicksBySCC) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_CSELECT_B32);
  ASSERT_TRUE(Op.has_value());
  const auto &Agent = GDispatcher->getGpuAgent(0);
  // Inputs in layout order: [src0, src1, scc_in]. S_CSELECT: d = SCC ? s0 : s1.
  const uint32_t S0 = 0xAAAA, S1 = 0xBBBB;

  TestResult T = GDriver->testInstruction(Agent, *Op, /*Seed=*/0, {S0, S1, 1u});
  llvm::errs() << formatResult(T);
  ASSERT_TRUE(T.ErrorMsg.empty()) << T.ErrorMsg;
  ASSERT_EQ(T.Outputs.size(), 1u);
  EXPECT_EQ(T.Outputs[0].Reference, S0) << "SCC=1 -> src0";

  TestResult F = GDriver->testInstruction(Agent, *Op, /*Seed=*/0, {S0, S1, 0u});
  llvm::errs() << formatResult(F);
  ASSERT_TRUE(F.ErrorMsg.empty()) << F.ErrorMsg;
  ASSERT_EQ(F.Outputs.size(), 1u);
  EXPECT_EQ(F.Outputs[0].Reference, S1) << "SCC=0 -> src1";
}

//===----------------------------------------------------------------------===//
// SOP tier 2: dispatch and produce stable output.
//===----------------------------------------------------------------------===//
TEST(RefPathSOP, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kSOPReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x5C0FFEE);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "SOP tier-2 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// Wide (64-bit) operands: VOP and SOP register tuples split into per-dword
// kernarg/output fields.
//===----------------------------------------------------------------------===//
static uint64_t joinU64(uint32_t Lo, uint32_t Hi) {
  return uint64_t(Lo) | (uint64_t(Hi) << 32);
}

// The wide-operand builder is width-general (per-dword kernarg/output fields,
// VGPR/SGPR tuple allocation, sub-register seed/capture). SOP exercises it
// end-to-end below. For 64-bit VALU (VOP3 f64/b64) ops, the opcode the fuzzer
// resolves can present anomalous operand register classes (a 96-bit VReg_96 def
// and an AGPR-only 96-bit source that accept neither a 64-bit VGPR nor a 64-bit
// AGPR), so the builder rejects them cleanly rather than mis-emitting. This
// documents that graceful rejection; a target that does expose clean 64-bit
// VALU classes builds instead, which the test also accepts.
TEST(RefPathWide, VOPWideRejectedCleanly) {
  for (const Reps &R : kWideVOPReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    auto ELF = tier1(R);
    if (ELF) {
      // If a future toolchain exposes clean 64-bit VALU classes, the build will
      // succeed — that is also acceptable, just note it.
      llvm::errs() << repName(R) << " (" << R.Subgroup << "): built " << ELF->size()
                   << " bytes (clean 64-bit VALU classes available)\n";
      continue;
    }
    std::string Msg = llvm::toString(ELF.takeError());
    llvm::errs() << repName(R) << " (" << R.Subgroup
                 << "): rejected cleanly: " << Msg << "\n";
    EXPECT_NE(Msg.find("no physreg"), std::string::npos)
        << repName(R) << ": expected an operand-class rejection, got: " << Msg;
  }
}

TEST(RefPathWide, SOPTier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kWideSOPReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  EXPECT_EQ(Ok, Total);
}

// S_AND_B64 ANDs two 64-bit scalar tuples (and sets SCC = result != 0).
TEST(RefPathWide, AndB64Computes) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_AND_B64);
  ASSERT_TRUE(Op.has_value());
  const uint64_t A = 0xF0F0F0F00A0A0A0AULL, B = 0x0FF00FF0FF00FF00ULL;
  TestResult R = GDriver->testInstruction(
      GDispatcher->getGpuAgent(0), *Op, /*Seed=*/0,
      {uint32_t(A), uint32_t(A >> 32), uint32_t(B), uint32_t(B >> 32)});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_GE(R.Outputs.size(), 2u);
  EXPECT_EQ(joinU64(R.Outputs[0].Reference, R.Outputs[1].Reference), A & B);
}

// S_LSHL_B64: mixed-width op (64-bit source, 32-bit shift amount). 1 << 32.
TEST(RefPathWide, LshlB64MixedWidth) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_LSHL_B64);
  ASSERT_TRUE(Op.has_value());
  // Layout order: src0.lo, src0.hi, src1 (shift, 32-bit).
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {1u, 0u, 32u});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_GE(R.Outputs.size(), 2u);
  EXPECT_EQ(joinU64(R.Outputs[0].Reference, R.Outputs[1].Reference),
            uint64_t(1) << 32);
}

// Every wide SOP representative dispatches twice and produces stable output.
TEST(RefPathWide, SOPTier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kWideSOPReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x0DE);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "Wide SOP tier-2 passed " << Ok << "/"
               << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// GLOBAL (FLAT-family) tier 1: build + verify + emit.
//===----------------------------------------------------------------------===//
TEST(RefPathGlobal, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_GE(ELF->size(), 4u) << repName(R);
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "GLOBAL tier-1 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// Not-yet-supported FLAT-family forms must be rejected cleanly (no crash): the
// non-SADDR global (needs a 64-bit vaddr) and CMPSWAP.
//===----------------------------------------------------------------------===//
TEST(RefPathGlobal, UnsupportedRejectedCleanly) {
  // D16 (partial-register) loads are still rejected cleanly.
  const unsigned Names[] = {llvm::AMDGPU::GLOBAL_LOAD_UBYTE_D16_SADDR,
                            llvm::AMDGPU::GLOBAL_LOAD_SHORT_D16_SADDR};
  for (unsigned Op : Names) {
    auto Resolved = resolveOpcode(Op);
    if (!Resolved)
      continue; // not on this subtarget
    const std::string Name = GDesc->getName(*Resolved).str();
    auto ELF = tier1(*Resolved);
    EXPECT_FALSE(static_cast<bool>(ELF))
        << Name << " unexpectedly built; expected a clean rejection";
    if (!ELF)
      llvm::errs() << Name << ": rejected -> "
                   << llvm::toString(ELF.takeError()) << "\n";
  }
}

//===----------------------------------------------------------------------===//
// GLOBAL functional: load returns the stored value; atomic-RTN adds.
//===----------------------------------------------------------------------===//
TEST(RefPathGlobal, LoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_LOAD_DWORD_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0xCAFEF00D; // inputs: [mem_init]
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "load should return mem[0]";
}

TEST(RefPathGlobal, AtomicRtnAdds) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_ATOMIC_ADD_SADDR_RTN);
  ASSERT_TRUE(Op.has_value());
  // inputs: [mem_init, vdata]; outputs: [vdst(pre), mem_after].
  const uint32_t Base = 1000, Add = 337;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Base, Add});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Base) << "RTN returns the pre-op value";
  EXPECT_EQ(R.Outputs[1].Reference, Base + Add) << "mem after = base + addend";
}

//===----------------------------------------------------------------------===//
// GLOBAL tier 2: dispatch and produce stable output.
//===----------------------------------------------------------------------===//
TEST(RefPathGlobal, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x6106A1);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "GLOBAL tier-2 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// Non-SADDR GLOBAL (full 64-bit vaddr, no saddr): tier 1 / functional / tier 2.
//===----------------------------------------------------------------------===//
TEST(RefPathGlobalVAddr, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalVAddrReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "GLOBAL-vaddr tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathGlobalVAddr, LoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_LOAD_DWORD);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0x0DDBA115;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "vaddr load should return mem[0]";
}

TEST(RefPathGlobalVAddr, AtomicRtnAdds) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_ATOMIC_ADD_RTN);
  ASSERT_TRUE(Op.has_value());
  const uint32_t Base = 700, Add = 55;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Base, Add});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Base) << "RTN returns the pre-op value";
  EXPECT_EQ(R.Outputs[1].Reference, Base + Add) << "mem after = base + addend";
}

TEST(RefPathGlobalVAddr, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalVAddrReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue;
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x6106A2);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "GLOBAL-vaddr tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// MUBUF (untyped buffer, OFFSET form): a 128-bit V# built in-kernel + a scalar
// offset. Load / store / atomic, DWORD / wide / sub-dword.
//===----------------------------------------------------------------------===//
TEST(RefPathMUBUF, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kMUBUFReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "MUBUF tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathMUBUF, LoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::BUFFER_LOAD_DWORD_OFFSET);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0xB07F0042; // arbitrary
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "buffer load should return buffer[0]";
}

TEST(RefPathMUBUF, StoreLands) {
  auto Op = resolveOpcode(llvm::AMDGPU::BUFFER_STORE_DWORD_OFFSET);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0x5A5AA5A5; // inputs: [vdata]; outputs: [mem_after]
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "buffer store should land in buffer[0]";
}

TEST(RefPathMUBUF, AtomicRtnAdds) {
  auto Op = resolveOpcode(llvm::AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_RTN);
  if (!Op)
    GTEST_SKIP() << "no BUFFER_ATOMIC_ADD_OFFSET_RTN on this subtarget";
  // inputs: [mem_init, vdata=addend]; outputs: [vdst(pre), mem_after].
  const uint32_t Base = 4000, Add = 91;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Base, Add});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Base) << "RTN returns the pre-op value";
  EXPECT_EQ(R.Outputs[1].Reference, Base + Add) << "mem after = base + addend";
}

TEST(RefPathMUBUF, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kMUBUFReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue;
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xB0FFE7);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "MUBUF tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// MUBUF vaddr forms (OFFEN / IDXEN / BOTHEN) and MTBUF (typed). vaddr is held
// at 0 so the access stays at element 0; MTBUF uses a 32-bit UINT passthrough.
//===----------------------------------------------------------------------===//
static void bufTier(llvm::ArrayRef<Reps> RepList, const char *Label) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : RepList) {
    auto Op = resolveRep(R);
    if (!Op)
      continue;
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xB0FFE8);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << Label << " passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathMUBUFVAddr, BuildAndDispatch) {
  bufTier(kMUBUFVAddrReps, "MUBUF-vaddr");
}

// OFFEN is per-lane: each of 64 lanes reads its own buffer element buffer[tid].
TEST(RefPathMUBUFVAddr, OffenLoadPerLane) {
  auto Op = resolveOpcode(llvm::AMDGPU::BUFFER_LOAD_DWORD_OFFEN);
  ASSERT_TRUE(Op.has_value());
  const unsigned N = GWaveSize;
  std::vector<uint32_t> In(N);
  for (unsigned L = 0; L < N; ++L)
    In[L] = 0xD00D0000u + L; // each lane's own element value
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, In);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), N);
  for (unsigned L = 0; L < N; ++L)
    EXPECT_EQ(R.Outputs[L].Reference, 0xD00D0000u + L)
        << "lane " << L << " should load buffer[" << L << "]";
}

// Per-lane atomic: each lane adds its own addend to its own element (no
// contention), so lane L returns its pre-value and mem[L] = base[L] + add[L].
TEST(RefPathMUBUFVAddr, OffenAtomicPerLane) {
  auto Op = resolveOpcode(llvm::AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_RTN);
  if (!Op)
    GTEST_SKIP() << "no BUFFER_ATOMIC_ADD_OFFEN_RTN on this subtarget";
  const unsigned N = GWaveSize;
  // inputs per lane: [mem_init, vdata=addend]; outputs per lane: [vdst, mem_after].
  std::vector<uint32_t> In(2 * N);
  for (unsigned L = 0; L < N; ++L) {
    In[L * 2 + 0] = 1000 + L;   // base[L]
    In[L * 2 + 1] = L + 1;      // add[L]
  }
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, In);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2 * N);
  for (unsigned L = 0; L < N; ++L) {
    EXPECT_EQ(R.Outputs[L * 2 + 0].Reference, 1000 + L) << "lane " << L << " pre";
    EXPECT_EQ(R.Outputs[L * 2 + 1].Reference, 1000 + L + (L + 1))
        << "lane " << L << " mem after";
  }
}

TEST(RefPathMTBUF, BuildAndDispatch) { bufTier(kMTBUFReps, "MTBUF"); }

TEST(RefPathMTBUF, FormatXLoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::TBUFFER_LOAD_FORMAT_X_OFFSET);
  if (!Op)
    GTEST_SKIP() << "no TBUFFER_LOAD_FORMAT_X_OFFSET on this subtarget";
  const uint32_t V = 0x40490FDB; // pi as float bits; 32-UINT passes it through
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "32-UINT tbuffer load passes through";
}

//===----------------------------------------------------------------------===//
// Sub-dword GLOBAL (byte / short): the memory footprint is 1 or 2 bytes; the
// value lives in a 32-bit register (loads zero/sign-extend).
//===----------------------------------------------------------------------===//
TEST(RefPathGlobalSubDword, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalSubDwordReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    ++Ok;
  }
  llvm::errs() << "GLOBAL-subdword tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathGlobalSubDword, UByteLoadReturnsLowByte) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_LOAD_UBYTE_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0xABCDEF12; // init stores low byte; ubyte load zero-extends
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V & 0xFFu) << "ubyte load = low byte";
}

TEST(RefPathGlobalSubDword, UShortLoadReturnsLowHalf) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_LOAD_USHORT_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0xABCDEF12;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V & 0xFFFFu) << "ushort load = low half";
}

TEST(RefPathGlobalSubDword, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalSubDwordReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue;
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x5B17E);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "GLOBAL-subdword tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// CMPSWAP: the data operand packs {swap, cmp}; element/result stay 1 (or 2)
// dwords. inputs: [mem_init, vdata.0, vdata.1]; outputs: [vdst(pre)?, mem_after].
//===----------------------------------------------------------------------===//
TEST(RefPathGlobalCmpSwap, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalCmpSwapReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    ++Ok;
  }
  llvm::errs() << "GLOBAL-cmpswap tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

// When mem == cmp the swap happens and RTN returns the pre-op value.
TEST(RefPathGlobalCmpSwap, MatchSwaps) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_ATOMIC_CMPSWAP_SADDR_RTN);
  ASSERT_TRUE(Op.has_value());
  // inputs: [mem_init=100, vdata.0=swap=555, vdata.1=cmp=100].
  const uint32_t Init = 100, Swap = 555, Cmp = 100;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Init, Swap, Cmp});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Init) << "RTN returns the pre-op value";
  EXPECT_EQ(R.Outputs[1].Reference, Swap) << "mem == cmp -> swapped";
}

// When mem != cmp the swap does not happen.
TEST(RefPathGlobalCmpSwap, MismatchKeeps) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_ATOMIC_CMPSWAP_SADDR_RTN);
  ASSERT_TRUE(Op.has_value());
  const uint32_t Init = 100, Swap = 555, Cmp = 999; // cmp != mem
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Init, Swap, Cmp});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Init) << "RTN returns the pre-op value";
  EXPECT_EQ(R.Outputs[1].Reference, Init) << "mem != cmp -> unchanged";
}

TEST(RefPathGlobalCmpSwap, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kGlobalCmpSwapReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue;
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xC5A9);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "GLOBAL-cmpswap tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// Plain FLAT tier 1 / functional / tier 2.
//===----------------------------------------------------------------------===//
TEST(RefPathFlat, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kFlatReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "FLAT tier-1 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathFlat, LoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::FLAT_LOAD_DWORD);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0x1337BEEF;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "flat load should return mem[0]";
}

TEST(RefPathFlat, AtomicRtnAdds) {
  auto Op = resolveOpcode(llvm::AMDGPU::FLAT_ATOMIC_ADD_RTN);
  ASSERT_TRUE(Op.has_value());
  const uint32_t Base = 500, Add = 44;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Base, Add});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Base) << "RTN returns the pre-op value";
  EXPECT_EQ(R.Outputs[1].Reference, Base + Add) << "mem after = base + addend";
}

TEST(RefPathFlat, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kFlatReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xF1A7);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "FLAT tier-2 passed " << Ok << "/" << Total
               << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// SMEM (scalar memory): scalar loads of every width / offset form. The kernel
// initializes a host-visible buffer, invalidates the scalar cache, runs the
// S_LOAD under test, and captures the sdst SGPR tuple.
//===----------------------------------------------------------------------===//
TEST(RefPathSMEM, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kSMEMReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "SMEM tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// SMEM functional: the scalar load returns the value the kernel wrote.
//===----------------------------------------------------------------------===//
TEST(RefPathSMEM, LoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_LOAD_DWORD_IMM);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0xABCD1234; // inputs: [mem_init]
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "s_load should return buffer[0]";
}

TEST(RefPathSMEM, LoadX2ReturnsBothDwords) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_LOAD_DWORDX2_IMM);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V0 = 0x11112222, V1 = 0x33334444;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V0, V1});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, V0);
  EXPECT_EQ(R.Outputs[1].Reference, V1);
}

TEST(RefPathSMEM, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kSMEMReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue; // not on this subtarget
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x5EED);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "SMEM tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// S_BUFFER_LOAD: same load path but sbase is a 128-bit V# resource descriptor
// built in-kernel from the data-buffer pointer.
//===----------------------------------------------------------------------===//
TEST(RefPathBufferSMEM, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kBufferSMEMReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "S_BUFFER_LOAD tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathBufferSMEM, LoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_BUFFER_LOAD_DWORD_IMM);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0xB0FFEE42; // inputs: [mem_init]
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V) << "s_buffer_load should return buffer[0]";
}

TEST(RefPathBufferSMEM, LoadX4ReturnsAllDwords) {
  auto Op = resolveOpcode(llvm::AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V[4] = {0xAAAA0000, 0xBBBB1111, 0xCCCC2222, 0xDDDD3333};
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V[0], V[1], V[2], V[3]});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 4u);
  for (unsigned D = 0; D < 4; ++D)
    EXPECT_EQ(R.Outputs[D].Reference, V[D]) << "dword " << D;
}

TEST(RefPathBufferSMEM, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kBufferSMEMReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue; // not on this subtarget
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xB0F);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "S_BUFFER_LOAD tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// SCRATCH (flat-scratch): the kernel enables the flat-scratch ABI, so the op
// under test runs against a real private segment supplied by HSA at dispatch.
//===----------------------------------------------------------------------===//
TEST(RefPathScratch, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kScratchReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "SCRATCH tier-1 passed " << Ok << "/"
               << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

// A scratch load returns the value the kernel stored to the same slot.
TEST(RefPathScratch, LoadReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t Val = 0xCAFEF00D;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Val});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, Val) << "scratch load should return slot[0]";
}

// A scratch store makes the value observable on a subsequent scratch load.
TEST(RefPathScratch, StoreThenReadback) {
  auto Op = resolveOpcode(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t Val = 0x0B0E0000 | 0xBEEF;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Val});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, Val) << "readback should see the stored val";
}

TEST(RefPathScratch, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kScratchReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x5C7A);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "SCRATCH tier-2 passed " << Ok << "/"
               << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// Sub-dword SCRATCH (byte / short).
//===----------------------------------------------------------------------===//
TEST(RefPathScratchSubDword, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kScratchSubDwordReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    ++Ok;
  }
  llvm::errs() << "SCRATCH-subdword tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

TEST(RefPathScratchSubDword, UByteRoundTrips) {
  auto Op = resolveOpcode(llvm::AMDGPU::SCRATCH_LOAD_UBYTE_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V = 0x99AABBCD;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, V & 0xFFu) << "ubyte load = low byte";
}

TEST(RefPathScratchSubDword, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kScratchSubDwordReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue;
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x5C7B);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "SCRATCH-subdword tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// Wide (multi-dword) FLAT / GLOBAL / SCRATCH: DWORDX2 / X4 tuples split into
// per-dword kernarg/output fields.
//===----------------------------------------------------------------------===//
// Dump the raised IR of the V_ADD_F32 reference kernel (translation path).
TEST(TranslationPath, RaiseVAddF32Dump) {
  auto Op = resolveOpcode(llvm::AMDGPU::V_ADD_F32_e32);
  ASSERT_TRUE(Op.has_value());
  InstrProfile P = GDesc->analyze(*Op);
  MachineKernelBuilder Builder(*GTM);
  KernargLayout Layout;
  auto KCtx = Builder.build(P, Layout);
  ASSERT_TRUE(static_cast<bool>(KCtx)) << llvm::toString(KCtx.takeError());

  llvm::Error Err = llvm::Error::success();
  luthier::TraceFunctionTranslator T(*KCtx->MF, Err);
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  T.translate();

  llvm::errs() << "=== raised IR for " << P.Name << " ===\n";
  KCtx->MF->getFunction().print(llvm::errs());
  llvm::errs() << "\n";
}

// End-to-end: the translation path (raised IR, recompiled) computes a + b on
// the GPU, matching the reference path.
TEST(TranslationPath, VAddF32MatchesReference) {
  auto Op = resolveOpcode(llvm::AMDGPU::V_ADD_F32_e32);
  ASSERT_TRUE(Op.has_value());
  const float A = 3.5f, B = 1.25f;
  uint32_t ABits, BBits;
  std::memcpy(&ABits, &A, 4);
  std::memcpy(&BBits, &B, 4);
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {ABits, BBits},
                                          /*CompareTranslation=*/true);
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  float Ref, Tr;
  std::memcpy(&Ref, &R.Outputs[0].Reference, 4);
  std::memcpy(&Tr, &R.Outputs[0].Translated, 4);
  EXPECT_FLOAT_EQ(Ref, A + B) << "reference";
  EXPECT_FLOAT_EQ(Tr, A + B) << "translation";
  EXPECT_TRUE(R.Outputs[0].Matches) << "translation must match reference";
}

// Survey the translation path across representative opcodes: raise each
// reference kernel to IR, recompile, and compare against the reference on the
// GPU. Informational (semantics coverage is still growing), but a mismatch or
// translation error is logged per opcode. The V_ADD_F32 sentinel must match.
TEST(TranslationPath, SurveyRepresentatives) {
  const unsigned Names[] = {
      llvm::AMDGPU::V_ADD_F32_e32, llvm::AMDGPU::V_MUL_F32_e32,
      llvm::AMDGPU::V_SUB_F32_e32, llvm::AMDGPU::V_MAX_F32_e32,
      llvm::AMDGPU::V_MIN_F32_e32, llvm::AMDGPU::V_AND_B32_e32,
      llvm::AMDGPU::V_OR_B32_e32,  llvm::AMDGPU::V_XOR_B32_e32,
      llvm::AMDGPU::V_MOV_B32_e32, llvm::AMDGPU::S_ADD_U32,
      llvm::AMDGPU::S_AND_B32,     llvm::AMDGPU::S_OR_B32,
  };
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Matched = 0, Mismatched = 0, Errored = 0, Sentinel = 0;
  for (unsigned Canonical : Names) {
    auto Op = resolveOpcode(Canonical);
    const std::string Name = GDesc->getName(Op.value_or(Canonical)).str();
    if (!Op) {
      llvm::errs() << Name << ": not on this subtarget\n";
      continue;
    }
    TestResult R = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x7A1E,
                                            /*FixedInputs=*/{},
                                            /*CompareTranslation=*/true);
    if (!R.ErrorMsg.empty() && R.Outputs.empty()) {
      llvm::errs() << Name << ": ERROR " << R.ErrorMsg << "\n";
      ++Errored;
      continue;
    }
    if (R.Passed) {
      llvm::errs() << Name << ": match\n";
      ++Matched;
      if (llvm::StringRef(Name) == "V_ADD_F32_e32")
        Sentinel = 1;
    } else {
      llvm::errs() << Name << ": MISMATCH (" << R.ErrorMsg << ")\n";
      ++Mismatched;
    }
  }
  llvm::errs() << "translation survey: " << Matched << " matched, "
               << Mismatched << " mismatched, " << Errored << " errored\n";
  EXPECT_EQ(Sentinel, 1u) << "V_ADD_F32 translation must match the reference";
}

TEST(RefPathWideMem, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kWideMemReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "wide-mem tier-1 passed " << Ok << "/"
               << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

// A DWORDX2 global load returns both stored dwords.
TEST(RefPathWideMem, GlobalLoadX2ReturnsStored) {
  auto Op = resolveOpcode(llvm::AMDGPU::GLOBAL_LOAD_DWORDX2_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t Lo = 0x11112222, Hi = 0x33334444;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Lo, Hi});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Lo);
  EXPECT_EQ(R.Outputs[1].Reference, Hi);
}

// A DWORDX4 scratch store is observable on a subsequent DWORDX4 scratch load.
TEST(RefPathWideMem, ScratchStoreX4Readback) {
  auto Op = resolveOpcode(llvm::AMDGPU::SCRATCH_STORE_DWORDX4_SADDR);
  ASSERT_TRUE(Op.has_value());
  const uint32_t V0 = 0xA0A0A0A0, V1 = 0xB1B1B1B1, V2 = 0xC2C2C2C2,
                 V3 = 0xD3D3D3D3;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {V0, V1, V2, V3});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 4u);
  EXPECT_EQ(R.Outputs[0].Reference, V0);
  EXPECT_EQ(R.Outputs[1].Reference, V1);
  EXPECT_EQ(R.Outputs[2].Reference, V2);
  EXPECT_EQ(R.Outputs[3].Reference, V3);
}

TEST(RefPathWideMem, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kWideMemReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0x1DE4);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "wide-mem tier-2 passed " << Ok << "/"
               << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// DS tier 1: build + verify + emit.
//===----------------------------------------------------------------------===//
TEST(RefPathDS, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kDSReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_GE(ELF->size(), 4u) << repName(R);
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "DS tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// DS functional: LDS init + readback actually computes the right thing.
//===----------------------------------------------------------------------===//
TEST(RefPathDS, AtomicRetComputesAdd) {
  auto Op = resolveOpcode(llvm::AMDGPU::DS_ADD_RTN_U32);
  ASSERT_TRUE(Op.has_value());
  // Inputs in layout order: [lds_init, data0]. vdst = pre-add value = init;
  // lds_after = init + data0.
  const uint32_t Init = 100, Addend = 5;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Init, Addend});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 2u);
  EXPECT_EQ(R.Outputs[0].Reference, Init) << "vdst should be the pre-add value";
  EXPECT_EQ(R.Outputs[1].Reference, Init + Addend)
      << "LDS after should be init + data0";
}

TEST(RefPathDS, LoadReturnsWrittenValue) {
  auto Op = resolveOpcode(llvm::AMDGPU::DS_READ_B32);
  ASSERT_TRUE(Op.has_value());
  const uint32_t Val = 0xABCD1234;
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, {Val});
  llvm::errs() << formatResult(R);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), 1u);
  EXPECT_EQ(R.Outputs[0].Reference, Val) << "read should return the LDS value";
}

//===----------------------------------------------------------------------===//
// DS tier 2: dispatch and produce output.
//===----------------------------------------------------------------------===//
TEST(RefPathDS, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kDSReps) {
    auto Op = resolveRep(R);
    if (!Op) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xD5123);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): " << Res.Outputs.size()
                 << " output(s), " << (Res.Passed ? "stable" : "UNSTABLE")
                 << "\n";
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "DS tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

//===----------------------------------------------------------------------===//
// DS cross-lane permute (full-wave, 64 lanes). Each lane gets its own data0 /
// addr and its own vdst output slot.
//===----------------------------------------------------------------------===//
TEST(RefPathDSPermute, Tier1BuildVerifyEmit) {
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kDSPermuteReps) {
    if (!resolveRep(R)) {
      llvm::errs() << repName(R) << " (" << R.Subgroup
                   << "): no encoding on this subtarget, skipped\n";
      continue;
    }
    ++Total;
    auto ELF = tier1(R);
    if (!ELF) {
      ADD_FAILURE() << repName(R) << " (" << R.Subgroup
                    << "): " << llvm::toString(ELF.takeError());
      continue;
    }
    EXPECT_EQ(std::memcmp(ELF->data(), "\177ELF", 4), 0) << repName(R);
    llvm::errs() << repName(R) << " (" << R.Subgroup << "): ELF " << ELF->size()
                 << " bytes\n";
    ++Ok;
  }
  llvm::errs() << "DS-permute tier-1 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

// ds_bpermute_b32 is a gather: vdst[lane] = data0[ addr[lane] >> 2 ]. Seed each
// lane's addr to point one lane ahead, and check every lane gathered correctly.
TEST(RefPathDSPermute, BPermuteGathers) {
  auto Op = resolveOpcode(llvm::AMDGPU::DS_BPERMUTE_B32);
  ASSERT_TRUE(Op.has_value());
  const unsigned N = GWaveSize;
  std::vector<uint32_t> In(2 * N);
  for (unsigned L = 0; L < N; ++L) {
    In[L * 2 + 0] = 0xA000 + L;             // data0[L]
    In[L * 2 + 1] = ((L + 1) & (N - 1)) * 4; // addr[L] -> lane (L+1)
  }
  TestResult R = GDriver->testInstruction(GDispatcher->getGpuAgent(0), *Op,
                                          /*Seed=*/0, In);
  ASSERT_TRUE(R.ErrorMsg.empty()) << R.ErrorMsg;
  ASSERT_EQ(R.Outputs.size(), N);
  for (unsigned L = 0; L < N; ++L)
    EXPECT_EQ(R.Outputs[L].Reference, 0xA000 + ((L + 1) & (N - 1)))
        << "lane " << L << " should have gathered from lane "
        << ((L + 1) & (N - 1));
}

TEST(RefPathDSPermute, Tier2Dispatch) {
  const auto &Agent = GDispatcher->getGpuAgent(0);
  unsigned Ok = 0, Total = 0;
  for (const Reps &R : kDSPermuteReps) {
    auto Op = resolveRep(R);
    if (!Op)
      continue;
    ++Total;
    TestResult Res = GDriver->testInstruction(Agent, *Op, /*Seed=*/0xD5B0F);
    if (!Res.ErrorMsg.empty()) {
      ADD_FAILURE() << repName(R) << ": " << Res.ErrorMsg;
      continue;
    }
    EXPECT_TRUE(Res.Passed) << repName(R) << ": " << Res.ErrorMsg;
    if (Res.Passed)
      ++Ok;
  }
  llvm::errs() << "DS-permute tier-2 passed " << Ok << "/" << Total << "\n";
  EXPECT_EQ(Ok, Total);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new FuzzerEnv);
  return RUN_ALL_TESTS();
}
