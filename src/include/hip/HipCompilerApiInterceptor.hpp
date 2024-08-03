//===-- HipCompilerApiInterceptor.hpp - Luthier's HIP API Interceptor -----===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains Luthier's HIP API Interceptor Singleton, implemented
/// using the rocprofiler-sdk API for capturing HIP compiler and runtime API
/// tables.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_HIP_HIP_COMPILER_API_INTERCEPT_HPP
#define LUTHIER_HIP_HIP_COMPILER_API_INTERCEPT_HPP

#include <functional>
#include <llvm/ADT/DenseSet.h>

#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/error.hpp"
#include "common/singleton.hpp"
#include <luthier/hip_trace_api.h>
#include <luthier/types.h>

namespace luthier::hip {

class HipCompilerApiInterceptor
    : public ROCmLibraryApiInterceptor<
          luthier::hip::CompilerApiEvtID, luthier::hip::ApiEvtArgs,
          HipCompilerDispatchTable, HipCompilerDispatchTable>,
      public Singleton<HipCompilerApiInterceptor> {
private:
public:
  HipCompilerApiInterceptor() = default;
  ~HipCompilerApiInterceptor() {
    SavedRuntimeApiTable = {};
    Singleton<HipCompilerApiInterceptor>::~Singleton();
  }

  bool enableUserCallback(luthier::hip::CompilerApiEvtID Op);

  void disableUserCallback(luthier::hip::CompilerApiEvtID Op);

  bool enableInternalCallback(luthier::hip::CompilerApiEvtID Op);

  void disableInternalCallback(luthier::hip::CompilerApiEvtID Op);

  llvm::Error captureApiTable(HipCompilerDispatchTable *Table) {
    RuntimeApiTable = Table;
    SavedRuntimeApiTable = *Table;
    return llvm::Error::success();
  }
};

} // namespace luthier::hip

#endif
