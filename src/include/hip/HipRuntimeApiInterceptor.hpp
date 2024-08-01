//===-- HipRuntimeApiInterceptor.hpp - HIP Runtime API Interceptor --------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains Luthier's HIP Runtime API Interceptor Singleton,
/// implemented using the rocprofiler-sdk API for capturing the HIP runtime API
/// tables and installing wrappers around them.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_HIP_HIP_RUNTIME_API_INTERCEPTOR_HPP
#define LUTHIER_HIP_HIP_RUNTIME_API_INTERCEPTOR_HPP

#include <functional>
#include <hip/amd_detail/hip_api_trace.hpp>
#include <llvm/ADT/DenseSet.h>

#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/error.hpp"
#include "common/singleton.hpp"
#include <luthier/types.h>
#include <rocprofiler-sdk/hip/api_args.h>
#include <rocprofiler-sdk/hip/api_id.h>

// TODO: 1. Overhaul Python generation script to use the new profiler API
//  enums + Generate Wrappers at CMake time by running the intercept generation
//  Python script + Enable HIP API runtime callbacks again
//  2. Move setting the callbacks to constructor arguments
//  3. Make a table of (APIID -> (Orig Func, Wrapper Func) in the auto generated
//  Python file to allow for complete uninstallation of wrapper functions when
//  No callback is requested from both the user and the tool
//  4. Update the Header of hip_intercept.cpp once the Python script is
//  overhauled

namespace luthier::hip {

class HipRuntimeApiInterceptor
    : public ROCmLibraryApiInterceptor<rocprofiler_hip_runtime_api_id_t,
                                       rocprofiler_hip_api_args_t,
                                       HipDispatchTable, HipDispatchTable>,
      public Singleton<HipRuntimeApiInterceptor> {

public:
  HipRuntimeApiInterceptor() = default;
  ~HipRuntimeApiInterceptor() {
    SavedRuntimeApiTable = {};
    Singleton<HipRuntimeApiInterceptor>::~Singleton();
  }

  void enableUserCallback(rocprofiler_hip_runtime_api_id_t op);

  void disableUserCallback(rocprofiler_hip_runtime_api_id_t op);

  void enableInternalCallback(rocprofiler_hip_runtime_api_id_t Op);

  void disableInternalCallback(rocprofiler_hip_runtime_api_id_t Op);

  void enableAllUserCallbacks() {
    for (std::underlying_type<rocprofiler_hip_runtime_api_id_t>::type I =
             rocprofiler_hip_runtime_api_id_t::
                 ROCPROFILER_HIP_RUNTIME_API_ID_hipApiName;
         I <
         rocprofiler_hip_runtime_api_id_t::ROCPROFILER_HIP_RUNTIME_API_ID_LAST;
         I++) {
      enableUserCallback(rocprofiler_hip_runtime_api_id_t(I));
    }
  }

  void disableAllUserCallbacks() {
    for (std::underlying_type<rocprofiler_hip_runtime_api_id_t>::type I =
             rocprofiler_hip_runtime_api_id_t::
                 ROCPROFILER_HIP_RUNTIME_API_ID_hipApiName;
         I <
         rocprofiler_hip_runtime_api_id_t::ROCPROFILER_HIP_RUNTIME_API_ID_LAST;
         I++) {
      disableUserCallback(rocprofiler_hip_runtime_api_id_t(I));
    }
  }

  void enableAllInternalCallbacks() {
    for (std::underlying_type<rocprofiler_hip_runtime_api_id_t>::type I =
             rocprofiler_hip_runtime_api_id_t::
                 ROCPROFILER_HIP_RUNTIME_API_ID_hipApiName;
         I <
         rocprofiler_hip_runtime_api_id_t::ROCPROFILER_HIP_RUNTIME_API_ID_LAST;
         I++) {
      enableInternalCallback(rocprofiler_hip_runtime_api_id_t(I));
    }
  }

  void disableAllInternalCallbacks() {
    for (std::underlying_type<rocprofiler_hip_runtime_api_id_t>::type I =
             rocprofiler_hip_runtime_api_id_t::
                 ROCPROFILER_HIP_RUNTIME_API_ID_hipApiName;
         I <
         rocprofiler_hip_runtime_api_id_t::ROCPROFILER_HIP_RUNTIME_API_ID_LAST;
         I++) {
      disableInternalCallback(rocprofiler_hip_runtime_api_id_t(I));
    }
  }

  llvm::Error captureApiTable(HipDispatchTable *Table) {
    RuntimeApiTable = Table;
    SavedRuntimeApiTable = *Table;
    return llvm::Error::success();
  }
};

} // namespace luthier::hip

#endif
