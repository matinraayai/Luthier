//===-- KfdTargetMachineTest.cpp -------------------------------------------===//
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
/// Tests naming a GPU's instruction set from the KFD driver alone.
///
/// \par The test that carries the weight
/// \c AgreesWithWhatHsaReports. Deriving an ISA from sysfs and an ioctl is a
/// re-implementation of something ROCR already does, and the failure mode is not
/// a crash -- it is a subtly wrong subtarget, which lifts and compiles and then
/// produces wrong code. The only real defence is to run both derivations on the
/// same device and require them to agree, so that is what that test does. It is
/// skipped where there is no GPU, and it is the reason this file links HSA at
/// all: HSA appears here as the *oracle*, never as a dependency of the code
/// under test.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/IsaInfo.h"
#include "luthier/KFD/KfdAgent.h"
#include "luthier/KFD/KfdTargetMachine.h"
#include "luthier/KFD/Topology.h"

#include "common/GpuAvailability.h"
#include "common/HsaApiTable.h"

#include <gtest/gtest.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/AMDGPUTargetParser.h>

#include <cstdio>
#include <string>
#include <vector>

using luthier::kfd::archNameForIsaVersion;
using luthier::kfd::queryIsaInfo;

namespace {

/// Consume an Expected's error and return its message.
std::string errorMessage(llvm::Error E) { return llvm::toString(std::move(E)); }

/// Every GPU gpu_id the topology reports, in node order.
std::vector<uint32_t> allGpuIds() {
  std::vector<uint32_t> Ids;
  for (unsigned Node = 0; Node < 64; Node++) {
    std::string Path = "/sys/class/kfd/kfd/topology/nodes/" +
                       std::to_string(Node) + "/gpu_id";
    FILE *F = fopen(Path.c_str(), "r");
    if (F == nullptr)
      continue;
    unsigned Id = 0;
    if (fscanf(F, "%u", &Id) == 1 && Id != 0)
      Ids.push_back(Id);
    fclose(F);
  }
  return Ids;
}

//===----------------------------------------------------------------------===//
// Version -> architecture name
//===----------------------------------------------------------------------===//

/// Pins the decoding of gfx_target_version through to a name, on the shapes that
/// differ from one another: a plain stepping, a stepping that is rendered as a
/// hex letter, and a two-digit major.
TEST(KfdArchName, KnownVersionsResolve) {
  struct Case {
    unsigned Major, Minor, Stepping;
    const char *Expected;
  };
  // gfx90a is the one worth having: its stepping is 10, so any decode that
  // assumes a decimal digit produces "gfx9010" and matches nothing.
  const Case Cases[] = {
      {9, 0, 8, "gfx908"},
      {9, 0, 10, "gfx90a"},
      {9, 4, 2, "gfx942"},
      {10, 3, 0, "gfx1030"},
  };
  for (const Case &C : Cases) {
    auto Name = archNameForIsaVersion(C.Major, C.Minor, C.Stepping);
    ASSERT_TRUE(static_cast<bool>(Name))
        << C.Expected << ": " << errorMessage(Name.takeError());
    EXPECT_EQ(C.Expected, *Name);
  }
}

/// An unknown device must be an error rather than a plausible-looking name.
/// Lifting code for a chip LLVM cannot model is not something to attempt on a
/// best-effort basis.
TEST(KfdArchName, UnknownVersionIsAnError) {
  auto Name = archNameForIsaVersion(99, 9, 9);
  ASSERT_FALSE(static_cast<bool>(Name));
  const std::string Msg = errorMessage(Name.takeError());
  EXPECT_NE(std::string::npos, Msg.find("99.9.9"));
  EXPECT_NE(std::string::npos, Msg.find("LLVM"));
}

/// Round-trip against LLVM's own list. Guards the assumption the lookup rests
/// on: that a version triple names at most one architecture, so matching on it
/// is unambiguous.
TEST(KfdArchName, EveryLlvmArchRoundTrips) {
  llvm::SmallVector<llvm::StringRef, 64> Arches;
  llvm::AMDGPU::fillValidArchListAMDGCN(Arches);
  ASSERT_FALSE(Arches.empty());

  unsigned Checked = 0;
  for (const llvm::StringRef Arch : Arches) {
    const llvm::AMDGPU::IsaVersion V = llvm::AMDGPU::getIsaVersion(Arch);
    if (V.Major == 0)
      continue; // generic targets carry no concrete version
    auto Name = archNameForIsaVersion(V.Major, V.Minor, V.Stepping);
    ASSERT_TRUE(static_cast<bool>(Name)) << Arch.str();
    const llvm::AMDGPU::IsaVersion Back =
        llvm::AMDGPU::getIsaVersion(llvm::StringRef(*Name));
    EXPECT_EQ(V.Major, Back.Major) << Arch.str() << " -> " << *Name;
    EXPECT_EQ(V.Minor, Back.Minor) << Arch.str() << " -> " << *Name;
    EXPECT_EQ(V.Stepping, Back.Stepping) << Arch.str() << " -> " << *Name;
    Checked++;
  }
  EXPECT_GT(Checked, 0u);
}

//===----------------------------------------------------------------------===//
// Querying the driver
//===----------------------------------------------------------------------===//

TEST(KfdIsaInfo, UnknownGpuIdIsAnError) {
  auto Info = queryIsaInfo(0xFFFFFFFFU);
  ASSERT_FALSE(static_cast<bool>(Info));
  EXPECT_NE(std::string::npos,
            errorMessage(Info.takeError()).find("topology node"));
}

/// gpu_id 0 is a CPU node. Rejecting it here rather than letting it resolve is
/// what stops a caller that lost track of an identifier from lifting code
/// against whatever the first node happens to be.
TEST(KfdIsaInfo, GpuIdZeroIsAnError) {
  auto Info = queryIsaInfo(0U);
  ASSERT_FALSE(static_cast<bool>(Info));
  llvm::consumeError(Info.takeError());
}

TEST(KfdIsaInfo, RealGpuResolvesToAKnownArchitecture) {
  const std::vector<uint32_t> Ids = allGpuIds();
  if (Ids.empty())
    GTEST_SKIP() << "no KFD GPU node on this machine";

  for (const uint32_t Id : Ids) {
    auto Info = queryIsaInfo(Id);
    ASSERT_TRUE(static_cast<bool>(Info))
        << "gpu_id " << Id << ": " << errorMessage(Info.takeError());
    auto Name =
        archNameForIsaVersion(Info->Major, Info->Minor, Info->Stepping);
    ASSERT_TRUE(static_cast<bool>(Name))
        << "gpu_id " << Id << ": " << errorMessage(Name.takeError());
    EXPECT_EQ(0u, Name->find("gfx")) << *Name;
  }
}

//===----------------------------------------------------------------------===//
// The oracle: agree with HSA on real hardware
//===----------------------------------------------------------------------===//

/// Split "amdgcn-amd-amdhsa--gfx908:sramecc+:xnack-" into its parts.
struct ParsedTargetId {
  std::string Arch;
  bool HasSramecc{false}, SrameccOn{false};
  bool HasXnack{false}, XnackOn{false};
};

ParsedTargetId parseTargetId(const std::string &FullName) {
  ParsedTargetId P;
  const size_t Sep = FullName.rfind("--");
  std::string Rest =
      Sep == std::string::npos ? FullName : FullName.substr(Sep + 2);

  size_t Colon = Rest.find(':');
  P.Arch = Rest.substr(0, Colon);
  while (Colon != std::string::npos) {
    const size_t Next = Rest.find(':', Colon + 1);
    const std::string Feature =
        Rest.substr(Colon + 1, Next == std::string::npos ? std::string::npos
                                                         : Next - Colon - 1);
    const bool On = !Feature.empty() && Feature.back() == '+';
    if (Feature.rfind("sramecc", 0) == 0) {
      P.HasSramecc = true;
      P.SrameccOn = On;
    } else if (Feature.rfind("xnack", 0) == 0) {
      P.HasXnack = true;
      P.XnackOn = On;
    }
    Colon = Next;
  }
  return P;
}

/// The two derivations must agree, device by device.
///
/// HSA is the oracle here and nothing more -- the code under test never calls
/// it, and cannot, in the applications this exists for. Matching the *bridge* as
/// well as the ISA is deliberate: the agent-to-gpu_id mapping goes through the
/// driver node id, and neither HSA_AGENT_INFO_NODE nor
/// HSA_AMD_AGENT_INFO_DRIVER_NODE_ID is a gpu_id -- both are node indices. A
/// test that compared only the ISA would pass while resolving the wrong device
/// on a multi-GPU machine, which is exactly this machine.
TEST(KfdIsaInfo, AgreesWithWhatHsaReports) {
  if (!luthier::test::hsaGpuAvailable())
    GTEST_SKIP() << "no HSA GPU on this machine";

  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());

  struct Visitor {
    unsigned Compared{0};
  } V;

  const hsa_status_t Walk = hsa_iterate_agents(
      [](hsa_agent_t Agent, void *Data) -> hsa_status_t {
        auto &V = *static_cast<Visitor *>(Data);

        hsa_device_type_t Type{};
        if (hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type) !=
                HSA_STATUS_SUCCESS ||
            Type != HSA_DEVICE_TYPE_GPU)
          return HSA_STATUS_SUCCESS;

        // Agent -> KFD node index -> gpu_id. The second hop is the sysfs walk,
        // and it is the hop that has no HSA equivalent.
        uint32_t NodeId = 0;
        EXPECT_EQ(HSA_STATUS_SUCCESS,
                  hsa_agent_get_info(
                      Agent,
                      static_cast<hsa_agent_info_t>(
                          HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                      &NodeId));
        // gpu_id lives in its own file, not in properties.
        char Path[128];
        snprintf(Path, sizeof(Path),
                 "/sys/class/kfd/kfd/topology/nodes/%u/gpu_id", NodeId);
        FILE *F = fopen(Path, "r");
        if (F == nullptr)
          return HSA_STATUS_SUCCESS;
        unsigned Id = 0;
        const int Scanned = fscanf(F, "%u", &Id);
        fclose(F);
        if (Scanned != 1 || Id == 0)
          return HSA_STATUS_SUCCESS;

        // Round-trip the bridge the code under test actually uses.
        EXPECT_EQ(NodeId, luthier::kfd::topologyNodeForGpuId(Id).value_or(~0U))
            << "gpu_id " << Id << " does not map back to node " << NodeId;

        char IsaName[128] = {};
        hsa_isa_t Isa{};
        EXPECT_EQ(HSA_STATUS_SUCCESS,
                  hsa_agent_get_info(
                      Agent, static_cast<hsa_agent_info_t>(HSA_AGENT_INFO_ISA),
                      &Isa));
        EXPECT_EQ(HSA_STATUS_SUCCESS,
                  hsa_isa_get_info_alt(Isa, HSA_ISA_INFO_NAME, IsaName));

        const ParsedTargetId Hsa = parseTargetId(IsaName);

        auto Info = luthier::kfd::queryIsaInfo(Id);
        EXPECT_TRUE(static_cast<bool>(Info));
        if (!Info) {
          llvm::consumeError(Info.takeError());
          return HSA_STATUS_SUCCESS;
        }
        auto Name = luthier::kfd::archNameForIsaVersion(
            Info->Major, Info->Minor, Info->Stepping);
        EXPECT_TRUE(static_cast<bool>(Name));
        if (!Name) {
          llvm::consumeError(Name.takeError());
          return HSA_STATUS_SUCCESS;
        }

        EXPECT_EQ(Hsa.Arch, *Name)
            << "HSA and KFD disagree on the architecture of gpu_id " << Id;
        if (Hsa.HasSramecc)
          EXPECT_EQ(Hsa.SrameccOn, Info->SrameccEnabled)
              << "sramecc disagrees for " << *Name << " (HSA said " << IsaName
              << ")";
        if (Hsa.HasXnack)
          EXPECT_EQ(Hsa.XnackOn, Info->XnackEnabled)
              << "xnack disagrees for " << *Name << " (HSA said " << IsaName
              << ")";

        V.Compared++;
        return HSA_STATUS_SUCCESS;
      },
      &V);
  EXPECT_EQ(HSA_STATUS_SUCCESS, Walk);
  (void)hsa_shut_down();

  EXPECT_GT(V.Compared, 0u)
      << "HSA reported a GPU but none could be compared, so this test asserted "
         "nothing";
}

//===----------------------------------------------------------------------===//
// End to end: the TargetMachine itself
//===----------------------------------------------------------------------===//

/// The deliverable. Everything above tests an input to this; this checks that
/// the inputs actually assemble into a subtarget LLVM accepts, which is a
/// separate way to be wrong -- a CPU string LLVM does not recognise is silently
/// ignored by createTargetMachine rather than refused.
TEST(KfdTargetMachine, BuildsForARealDevice) {
  const std::vector<uint32_t> Ids = allGpuIds();
  if (Ids.empty())
    GTEST_SKIP() << "no KFD GPU node on this machine";

  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();

  // A zeroed descriptor is enough: the two fields read out of it only apply on
  // gfx10+, and what is under test is that the driver-derived parts arrive.
  llvm::amdhsa::kernel_descriptor_t KD{};

  for (const uint32_t Id : Ids) {
    auto TM = luthier::buildTargetMachineForKfdDispatch(Id, KD);
    ASSERT_TRUE(static_cast<bool>(TM))
        << "gpu_id " << Id << ": " << errorMessage(TM.takeError());
    ASSERT_NE(nullptr, TM->get());

    EXPECT_EQ("amdgcn-amd-amdhsa", (*TM)->getTargetTriple().str());

    auto Info = queryIsaInfo(Id);
    ASSERT_TRUE(static_cast<bool>(Info));
    auto Name = archNameForIsaVersion(Info->Major, Info->Minor, Info->Stepping);
    ASSERT_TRUE(static_cast<bool>(Name));

    // The CPU must be the architecture we derived, not empty. An unrecognised
    // CPU leaves createTargetMachine with a generic subtarget, which lifts
    // wrong code rather than failing.
    EXPECT_EQ(*Name, (*TM)->getTargetCPU().str())
        << "gpu_id " << Id << " built a target machine for the wrong CPU";
  }
}

/// A device that does not exist must not produce a target machine at all.
TEST(KfdTargetMachine, UnknownDeviceIsAnError) {
  llvm::amdhsa::kernel_descriptor_t KD{};
  auto TM = luthier::buildTargetMachineForKfdDispatch(0xFFFFFFFFU, KD);
  ASSERT_FALSE(static_cast<bool>(TM));
  llvm::consumeError(TM.takeError());
}

//===----------------------------------------------------------------------===//
// gpu_id -> HSA agent
//===----------------------------------------------------------------------===//

/// Round-trip the bridge the loader depends on.
///
/// Worth pinning rather than trusting: no HSA attribute returns a \c gpu_id, so
/// the mapping goes agent -> topology node index -> \c gpu_id through sysfs, and
/// the two identifiers are different enough in kind that swapping them reads
/// perfectly well and resolves the wrong device. On a single-GPU machine a
/// swapped mapping would even pass; this asserts per agent, so a multi-GPU box
/// catches it.
TEST(KfdAgent, EveryGpuAgentRoundTripsThroughItsGpuId) {
  if (!luthier::test::hsaGpuAvailable())
    GTEST_SKIP() << "no HSA GPU on this machine";

  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());
  ::CoreApiTable Table = luthier::test::buildCoreApiTable();
  const luthier::hsa::ApiTableContainer<::CoreApiTable> Core(Table);

  unsigned Checked = 0;
  for (const uint32_t Id : allGpuIds()) {
    auto AgentOrErr = luthier::kfd::agentForGpuId(Core, Id);
    ASSERT_TRUE(static_cast<bool>(AgentOrErr))
        << "gpu_id " << Id << ": " << errorMessage(AgentOrErr.takeError());

    // The agent we got back must report the node whose gpu_id we asked for.
    uint32_t Node = 0;
    ASSERT_EQ(HSA_STATUS_SUCCESS,
              hsa_agent_get_info(*AgentOrErr,
                                 static_cast<hsa_agent_info_t>(
                                     HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                                 &Node));
    EXPECT_EQ(Id, luthier::kfd::gpuIdForTopologyNode(Node).value_or(0))
        << "the agent returned for gpu_id " << Id << " reports node " << Node
        << ", which is a different device";
    Checked++;
  }
  (void)hsa_shut_down();

  EXPECT_GT(Checked, 0u) << "a GPU was reported but none was checked";
}

/// A device the driver does not have must be an error, not some other agent.
TEST(KfdAgent, UnknownGpuIdIsAnError) {
  if (!luthier::test::hsaGpuAvailable())
    GTEST_SKIP() << "no HSA GPU on this machine";
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());
  ::CoreApiTable Table = luthier::test::buildCoreApiTable();
  const luthier::hsa::ApiTableContainer<::CoreApiTable> Core(Table);

  auto AgentOrErr = luthier::kfd::agentForGpuId(Core, 0xFFFFFFFFU);
  ASSERT_FALSE(static_cast<bool>(AgentOrErr));
  const std::string Msg = errorMessage(AgentOrErr.takeError());
  EXPECT_NE(std::string::npos, Msg.find("4294967295"));
  (void)hsa_shut_down();
}

} // namespace
