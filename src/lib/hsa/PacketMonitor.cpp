//===-- HsaPacketMonitor.cpp ----------------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Implements the \c PacketMonitor interface.
//===----------------------------------------------------------------------===//
#include <luthier/hsa/KernelDescriptor.h>
#include <luthier/hsa/PacketMonitor.h>

namespace luthier::hsa {

llvm::Expected<std::tuple<hsa_executable_t, hsa_loaded_code_object_t,
                          hsa_executable_symbol_t>>
PacketMonitor::getKernelObjectDefinition(const uint64_t KernelObject) const {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      KernelObject != 0, "Kernel object cannot be zero"));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      HsaApiTable.wasRegistrationCallbackInvoked(),
      "The API table snapshot is not yet initialized"));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      LoaderTable != nullptr, "The Loader table is not yet initialized"));
  return KernelDescriptor::fromKernelObject(KernelObject)
      ->getExecutableDefinition(
          *LoaderTable->hsa_ven_amd_loader_query_executable,
          *LoaderTable
               ->hsa_ven_amd_loader_executable_iterate_loaded_code_objects,
          *LoaderTable->hsa_ven_amd_loader_loaded_code_object_get_info,
          *HsaApiTable.getFunction<
              &::CoreApiTable::hsa_executable_iterate_agent_symbols_fn>(),
          *HsaApiTable.getFunction<
              &::CoreApiTable::hsa_executable_symbol_get_info_fn>());
}

} // namespace luthier::hsa