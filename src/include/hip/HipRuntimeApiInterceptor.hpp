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
#include <llvm/ADT/DenseSet.h>

#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/error.hpp"
#include "common/singleton.hpp"
#include <luthier/hip_trace_api.h>
#include <luthier/types.h>

namespace luthier::hip {

class HipRuntimeApiInterceptor
    : public ROCmLibraryApiInterceptor<luthier::hip::RuntimeApiEvtID,
                                       luthier::hip::ApiEvtArgs,
                                       HipDispatchTable, HipDispatchTable>,
      public Singleton<HipRuntimeApiInterceptor> {

public:
  HipRuntimeApiInterceptor() = default;
  ~HipRuntimeApiInterceptor() {
    SavedRuntimeApiTable = {};
    Singleton<HipRuntimeApiInterceptor>::~Singleton();
  }

  bool enableUserCallback(luthier::hip::RuntimeApiEvtID op);

  void disableUserCallback(luthier::hip::RuntimeApiEvtID op);

  bool enableInternalCallback(luthier::hip::RuntimeApiEvtID Op);

  void disableInternalCallback(luthier::hip::RuntimeApiEvtID Op);

  llvm::Error captureApiTable(HipDispatchTable *Table) {
    RuntimeApiTable = Table;
    SavedRuntimeApiTable = *Table;
    return llvm::Error::success();
  }
};

} // namespace luthier::hip

#endif
