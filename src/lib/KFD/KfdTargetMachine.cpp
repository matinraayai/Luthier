//===-- KfdTargetMachine.cpp -----------------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements \c luthier/KFD/KfdTargetMachine.h: turns the driver's answer about
/// a device into an \c llvm::TargetMachine.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/KfdTargetMachine.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"

#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TargetParser/AMDGPUTargetParser.h>

#define DEBUG_TYPE "luthier-kfd-target-machine"

namespace luthier {

llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
buildTargetMachineForKfdDispatch(
    uint32_t GpuId, const llvm::amdhsa::kernel_descriptor_t &KD) {
  kfd::IsaInfo Info;
  LUTHIER_RETURN_ON_ERROR(kfd::queryIsaInfo(GpuId).moveInto(Info));

  std::string CPU;
  LUTHIER_RETURN_ON_ERROR(
      kfd::archNameForIsaVersion(Info.Major, Info.Minor, Info.Stepping)
          .moveInto(CPU));

  // Only name a feature the architecture actually has. Handing LLVM "-sramecc"
  // for a chip without sramecc is not fatal, but it is a diagnostic the user
  // cannot act on, and asking LLVM which features exist costs nothing.
  const unsigned ArchAttrs =
      llvm::AMDGPU::getArchAttrAMDGCN(llvm::AMDGPU::parseArchAMDGCN(CPU));

  llvm::SubtargetFeatures Features;
  if ((ArchAttrs & llvm::AMDGPU::FEATURE_SRAMECC) != 0)
    Features.AddFeature("sramecc", Info.SrameccEnabled);
  if ((ArchAttrs & llvm::AMDGPU::FEATURE_XNACK) != 0)
    Features.AddFeature("xnack", Info.XnackEnabled);

  // Fold the per-kernel wavefront size and CU/WGP execution mode out of the
  // kernel descriptor, exactly as the HSA path does (HSATool.h:170-190). These
  // are properties of the kernel rather than of the ISA, and the lifted MIR
  // depends on the subtarget reflecting them: EXEC-mask predication width
  // follows the wavefront size, and the re-lowered KD's WGP_MODE / TG_SPLIT bits
  // are derived from the cumode feature. Both features only exist on gfx10+;
  // earlier hardware is always wave64 and CU mode and has no such features to
  // set.
  if (Info.Major >= 10) {
    const bool IsWave32 = AMDHSA_BITS_GET(
        KD.kernel_code_properties,
        llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
    Features.AddFeature(IsWave32 ? "wavefrontsize32" : "wavefrontsize64");

    // WGP_MODE set => the kernel runs in WGP mode (cumode disabled); clear =>
    // CU mode (cumode enabled).
    const bool IsWGPMode = AMDHSA_BITS_GET(
        KD.compute_pgm_rsrc1, llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE);
    Features.AddFeature("cumode", !IsWGPMode);
    // TODO: add missing tgsplit feature, as on the HSA path
  }

  const llvm::Triple TheTriple("amdgcn-amd-amdhsa");
  std::string ErrMsg;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TheTriple, ErrMsg);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      TheTarget != nullptr,
      llvm::formatv("TargetRegistry::lookupTarget failed for triple {0}: {1}. "
                    "The AMDGPU target must be registered before building a "
                    "target machine.",
                    TheTriple.str(), ErrMsg)));

  llvm::TargetOptions TMOpts;
  std::unique_ptr<llvm::TargetMachine> TM(TheTarget->createTargetMachine(
      TheTriple, CPU, Features.getString(), TMOpts, /*RM=*/std::nullopt));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      TM != nullptr, "createTargetMachine returned nullptr."));
  TM->setOptLevel(llvm::CodeGenOptLevel::Default);

  LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                 "[KfdTargetMachine] built {0} {1} for gpu_id {2}.\n", CPU,
                 Features.getString(), GpuId));
  return TM;
}

} // namespace luthier
