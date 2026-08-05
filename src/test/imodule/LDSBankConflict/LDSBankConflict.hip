//===-- LDSBankConflict.hip - LDS Bank Conflict IModule ----------------===//
//
// Device-only instrumentation module for LDS bank conflict detection.
// Stripped from examples/LDSBankConflict/LDSBankConflict.hip.
//
//===----------------------------------------------------------------------===//
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

__attribute__((device)) float NumBankConflicts;

__attribute__((device)) uint64_t NumSingleAddressLDSInstructions;

__attribute__((device)) uint64_t NumDoubleAddressLDSInstructions;

__attribute__((device, used)) void
countUniqueBanksPerWarpSingleAddress(unsigned int M0, unsigned int VAddr,
                                     short Offset, short CacheSizeLog2) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumSingleAddressLDSInstructions, 1);
  }

  unsigned int Address = M0 + VAddr + Offset;
  unsigned int BankIdx = Address >> CacheSizeLog2;
  auto NumWorkItemsWithSameBankID =
      static_cast<float>(__popcll(__match_any(BankIdx)));
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID);
}

__attribute__((device, used)) void
countUniqueBanksPerWarpDoubleAddress(unsigned int M0, unsigned int VAddr,
                                     uint8_t Offset0, uint8_t Offset1,
                                     short CacheSizeLog2) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumDoubleAddressLDSInstructions, 1);
  }

  unsigned int Address0 = VAddr + (Offset0 << 2);
  unsigned int Address1 = VAddr + (Offset1 << 2);
  unsigned int BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned int BankIdx1 = Address1 >> CacheSizeLog2;
  auto NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  auto NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned int BankIdx0IthThread = __shfl_sync(ExecMask, BankIdx0, I);
      if (BankIdx0IthThread == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned int BankIdx1IthThread = __shfl_sync(ExecMask, BankIdx1, I);
      if (BankIdx1IthThread == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }

  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID1);
}

__attribute__((device, used)) void
countUniqueBanksPerWarpDoubleAddressTwoSource32(unsigned int M0,
                                                unsigned int VAddr,
                                                uint16_t Offset0,
                                                uint16_t Offset1,
                                                uint8_t CacheSizeLog2) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumDoubleAddressLDSInstructions, 1);
  }

  unsigned int Address0 = M0 + VAddr + (Offset0 << 8);
  unsigned int Address1 = M0 + VAddr + (Offset1 << 8);
  unsigned int BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned int BankIdx1 = Address1 >> CacheSizeLog2;
  auto NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  auto NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned int BankIdx0IthThread = __shfl_sync(ExecMask, BankIdx0, I);
      if (BankIdx0IthThread == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned int BankIdx1IthThread = __shfl_sync(ExecMask, BankIdx1, I);
      if (BankIdx1IthThread == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }

  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID1);
}

__attribute__((device, used)) void
countUniqueBanksPerWarpDoubleAddressTwoSource64(unsigned int M0,
                                                unsigned int VAddr,
                                                uint16_t Offset0,
                                                uint16_t Offset1,
                                                uint8_t CacheSizeLog2) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumDoubleAddressLDSInstructions, 1);
  }

  unsigned int Address0 = M0 + VAddr + (Offset0 << 9);
  unsigned int Address1 = M0 + VAddr + (Offset1 << 9);
  unsigned int BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned int BankIdx1 = Address1 >> CacheSizeLog2;
  auto NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  auto NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned int BankIdx0IthThread = __shfl_sync(ExecMask, BankIdx0, I);
      if (BankIdx0IthThread == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned int BankIdx1IthThread = __shfl_sync(ExecMask, BankIdx1, I);
      if (BankIdx1IthThread == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }

  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID1);
}
