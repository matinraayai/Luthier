//===-- Context.cpp - Luthier tool's Context Logic implementation ---------===//
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
/// This file implements the \c Context singleton class logic.
//===----------------------------------------------------------------------===//
#include "luthier/tooling/Context.h"
#include "common/Log.hpp"
#include "intrinsic/ImplicitArgPtr.hpp"
#include "intrinsic/ReadReg.hpp"
#include "intrinsic/SAtomicAdd.hpp"
#include "intrinsic/WriteExec.hpp"
#include "intrinsic/WriteReg.hpp"
#include "luthier/hsa/Executable.h"
#include "luthier/hsa/LoadedCodeObjectCache.h"
#include "luthier/llvm/EagerManagedStatic.h"
#include "luthier/llvm/streams.h"
#include "luthier/luthier.h"
#include "luthier/rocprofiler-sdk/RocprofilerError.h"
#include "luthier/types.h"
#include "tooling_common/CodeGenerator.hpp"
#include "tooling_common/CodeLifter.hpp"
#include "tooling_common/TargetManager.hpp"
#include "tooling_common/ToolExecutableLoader.hpp"
#include <llvm/Support/Error.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/hsa/PacketMointor.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-controller"

namespace luthier {

static EagerManagedStatic<llvm::cl::OptionCategory>
    TimeTracingOptionCategory("Luthier Tooling Time Tracing Options");

static EagerManagedStatic<llvm::cl::opt<bool>>
    TimeTrace("time-trace", llvm::cl::desc("Record luthier tool's time trace"),
              llvm::cl::cat(*TimeTracingOptionCategory));

static EagerManagedStatic<llvm::cl::opt<unsigned>> TimeTraceGranularity(
    "time-trace-granularity",
    llvm::cl::desc(
        "Minimum time granularity (in microseconds) traced by time profiler"),
    llvm::cl::init(500), llvm::cl::Hidden,
    llvm::cl::cat(*TimeTracingOptionCategory));

static EagerManagedStatic<llvm::cl::opt<std::string>>
    TimeTraceFile("time-trace-file",
                  llvm::cl::desc("Specify time trace file destination"),
                  llvm::cl::value_desc("filename"),
                  llvm::cl::cat(*TimeTracingOptionCategory));

template <> Context *Singleton<Context>::Instance{nullptr};

Context::Context(hsa::PacketMonitor::CallbackType PacketCallback,
                 llvm::Error &Err) {
  llvm::ErrorAsOutParameter EAO(Err);
  // Initialize all Luthier singletons
  MDParser = new amdgpu::hsamd::MetadataParser();

  HsaCoreApiTableSnapshot =
      new rocprofiler::HsaApiTableSnapshot<::CoreApiTable>(Err);
  if (Err)
    return;
  HsaAmdExtTableSnapshot =
      new rocprofiler::HsaApiTableSnapshot<AmdExtTable>(Err);
  if (Err)
    return;
  HipCompilerTableSnapshot =
      new rocprofiler::HipApiTableSnapshot<ROCPROFILER_HIP_COMPILER_TABLE>(Err);
  if (Err)
    return;
  VenLoaderSnapshot =
      new rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>(Err);
  if (Err)
    return;
  TM = new TargetManager(*HsaCoreApiTableSnapshot);
  CodeObjectCache = new hsa::LoadedCodeObjectCache(
      *HsaCoreApiTableSnapshot, *VenLoaderSnapshot, *MDParser, Err);
  if (Err)
    return;
  TEL = new ToolExecutableLoader(*HsaCoreApiTableSnapshot, *VenLoaderSnapshot,
                                 *CodeObjectCache, *MDParser, Err);
  if (Err)
    return;
  CL = new CodeLifter(*HsaCoreApiTableSnapshot, *VenLoaderSnapshot);
  CG = new CodeGenerator(*HsaCoreApiTableSnapshot, *VenLoaderSnapshot);

  // Register Luthier intrinsics with the Code Generator
  CG->registerIntrinsic("luthier::readReg",
                        {readRegIRProcessor, readRegMIRProcessor});
  CG->registerIntrinsic("luthier::writeReg",
                        {writeRegIRProcessor, writeRegMIRProcessor});
  CG->registerIntrinsic("luthier::writeExec",
                        {writeExecIRProcessor, writeExecMIRProcessor});
  CG->registerIntrinsic(
      "luthier::implicitArgPtr",
      {implicitArgPtrIRProcessor, implicitArgPtrMIRProcessor});
  CG->registerIntrinsic("luthier::sAtomicAdd",
                        {sAtomicAddIRProcessor, sAtomicAddMIRProcessor});

  PacketMonitor =
      new hsa::PacketMonitor(*HsaCoreApiTableSnapshot, *HsaAmdExtTableSnapshot,
                             *VenLoaderSnapshot, PacketCallback, Err);
  if (Err)
    return;
  // Enable the LLVM time tracer if enabled
  if (*TimeTrace)
    llvm::timeTraceProfilerInitialize(*TimeTraceGranularity,
                                      "libLuthierTooling.so");
}

Context::~Context() {
  delete CG;
  delete CL;
  delete TEL;
  delete CodeObjectCache;
  delete VenLoaderSnapshot;
  delete HipCompilerTableSnapshot;
  delete HsaAmdExtTableSnapshot;
  delete HsaCoreApiTableSnapshot;
  delete TM;
  delete MDParser;
  if (*TimeTrace) {
    LUTHIER_REPORT_FATAL_ON_ERROR(
        llvm::timeTraceProfilerWrite(*TimeTraceFile, "luthier-profile"));
    llvm::timeTraceProfilerCleanup();
  }
}

} // namespace luthier
