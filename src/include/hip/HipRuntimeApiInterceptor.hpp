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
    : public ROCmLibraryApiInterceptor<luthier::hip::ApiEvtID,
                                       luthier::hip::ApiEvtArgs,
                                       HipDispatchTable, HipDispatchTable>,
      public Singleton<HipRuntimeApiInterceptor> {

public:
  HipRuntimeApiInterceptor() = default;
  ~HipRuntimeApiInterceptor() {
    *RuntimeApiTable = SavedRuntimeApiTable;
    SavedRuntimeApiTable = {};
    Singleton<HipRuntimeApiInterceptor>::~Singleton();
  }

  llvm::Error enableUserCallback(luthier::hip::ApiEvtID Op);

  llvm::Error disableUserCallback(luthier::hip::ApiEvtID Op);

  llvm::Error enableInternalCallback(luthier::hip::ApiEvtID Op);

  llvm::Error disableInternalCallback(luthier::hip::ApiEvtID Op);

  llvm::Error captureApiTable(HipDispatchTable *Table) {
    RuntimeApiTable = Table;
    SavedRuntimeApiTable = *Table;
    Status = API_TABLE_CAPTURED;
    return ROCmLibraryApiInterceptor<luthier::hip::ApiEvtID,
                                     luthier::hip::ApiEvtArgs, HipDispatchTable,
                                     HipDispatchTable>::captureApiTable(Table);

  }
};

} // namespace luthier::hip

#endif
