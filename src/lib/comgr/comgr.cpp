//===-- comgr.cpp - AMD CoMGR High-level Wrapper --------------------------===//
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
/// This files implements wrappers around AMD CoMGR functionality frequently
/// used by Luthier.
//===----------------------------------------------------------------------===//
#include "comgr/comgr.hpp"
#include <amd_comgr/amd_comgr.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/comgr/ComgrError.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/object/ObjectFileUtils.h>

namespace luthier::comgr {

llvm::Error linkRelocatableToExecutable(llvm::ArrayRef<char> Code,
                                        llvm::SmallVectorImpl<char> &Out) {
  llvm::TimeTraceScope Scope("Comgr Executable Linking");
  amd_comgr_data_t DataIn;
  amd_comgr_data_set_t DataSetIn, DataSetOut;
  amd_comgr_action_info_t DataAction;

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_create_data_set(&DataSetIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      amd_comgr_set_data(DataIn, Code.size(), Code.data())));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_set_data_name(DataIn, "source.o"))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_data_set_add(DataSetIn, DataIn))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_data_set(&DataSetOut))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_action_info(&DataAction))));

  /// Get the ISA of the object file and set the action's ISA name
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> CodeAsObjFileOrErr =
      luthier::object::createObjectFile(llvm::toStringRef(Code));
  LUTHIER_RETURN_ON_ERROR(CodeAsObjFileOrErr.takeError());

  llvm::Expected<std::string> ObjFileIsaOrErr =
      luthier::object::getObjectFileTarget(**CodeAsObjFileOrErr);
  LUTHIER_RETURN_ON_ERROR(ObjFileIsaOrErr.takeError());

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_action_info_set_isa_name(
          DataAction, ObjFileIsaOrErr->c_str()))));
  const char *LinkOptions[]{"-Wl,--unresolved-symbols=ignore-all"};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_option_list(DataAction, LinkOptions, 1))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                           DataAction, DataSetIn, DataSetOut))));

  amd_comgr_data_t DataOut;
  size_t DataOutSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_action_data_get_data(
          DataSetOut, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataOut))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_get_data(DataOut, &DataOutSize, nullptr))));
  Out.resize(DataOutSize);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_get_data(DataOut, &DataOutSize, Out.data()))));
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_destroy_data_set(DataSetIn)));
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_destroy_data_set(DataSetOut)));
  return llvm::Error::success();
}

} // namespace luthier::comgr