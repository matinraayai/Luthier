//
// Created by User on 3/19/2025.
//

#include "TestingUtils.h"

#include <amd_comgr/amd_comgr.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/llvm/streams.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Support/FileSystem.h>
#include <luthier/comgr/ComgrError.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <string>
#include <tooling_common/CodeLifter.hpp>

llvm::Error compile_and_link(std::string filepath)
{
  // read in .s file
  llvm::ErrorOr <std::unique_ptr<llvm::MemoryBuffer>> BuffOrErr =
      llvm::MemoryBuffer::getFile(filepath);

  // https://github.com/ROCm/llvm-project/blob/7addc3557e2d6e0a1aa133d625c62e5ee04bc5bf/llvm/tools/llvm-dwarfdump/llvm-dwarfdump.cpp#L322
  if (BuffOrErr.getError()) {
    llvm::errs() << "Error opening file: " << filepath << "\n";
    // TODO - handle this error
  }

  std::unique_ptr<llvm::MemoryBuffer> Buff = std::move(BuffOrErr.get());

  // TODO - move this logic into utils file for unit tests
  amd_comgr_data_t DataIn, DataReloc;
  amd_comgr_data_set_t DataSetIn, DataSetOut, DataSetReloc;
  amd_comgr_action_info_t DataAction;

  // create data in set
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_ERROR_CHECK(
    amd_comgr_create_data_set(&DataSetIn)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataIn)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_set_data(DataIn, Buff->getBuffer().size(),
      Buff->getBuffer().data())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_set_data_name(DataIn, filepath.c_str())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_data_set_add(DataSetIn, DataIn)));

  // create action info and set ISA name??
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_create_action_info(&DataAction)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx908")));

  // compile source to executable
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_create_data_set(&DataSetOut)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_action_info_set_option_list(DataAction, NULL, 0)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE, DataAction, DataSetIn, DataSetOut)));

  // link to executable on drive
  amd_comgr_data_set_t DataSetLinked;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_create_data_set(&DataSetLinked)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
      DataAction, DataSetOut, DataSetLinked)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
    amd_comgr_create_data_set(&DataSetReloc)));

  /*
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_ERROR_CHECK(
    amd_comgr_action_data_get_data(DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE,
        0, &DataReloc)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_ERROR_CHECK(
    amd_comgr_get_data()));
    */
  return llvm::Error::success();
}