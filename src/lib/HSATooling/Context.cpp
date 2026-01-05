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
#include "luthier/HSATooling/Context.h"
#include "luthier/HSA/Executable.h"
#include "luthier/HSATooling/CodeGenerator.h"
#include "luthier/HSATooling/CodeLifter.h"
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/HSATooling/PacketMointor.h"
#include "luthier/HSATooling/TargetManager.h"
#include "luthier/HSATooling/ToolExecutableLoader.h"
#include "luthier/Intrinsic/ImplicitArgPtr.h"
#include "luthier/Intrinsic/ReadReg.h"
#include "luthier/Intrinsic/SAtomicAdd.h"
#include "luthier/Intrinsic/WriteExec.h"
#include "luthier/Intrinsic/WriteReg.h"
#include "luthier/LLVM/EagerManagedStatic.h"
#include "luthier/LLVM/streams.h"
#include "luthier/Rocprofiler/RocprofilerError.h"
#include "luthier/Tooling/InstrumentationPMDriver.h"
#include "luthier/luthier.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TimeProfiler.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-controller"

namespace luthier {

static EagerManagedStatic<InstrumentationPMDriverOptions>
    InstrumentationPMOptions;

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

  IPR = new IntrinsicProcessorRegistry();

  CG = new CodeGenerator(*HsaCoreApiTableSnapshot, *VenLoaderSnapshot,
                         *InstrumentationPMOptions);

  PacketMonitor = new hsa::PacketMonitor(
      *HsaCoreApiTableSnapshot, *HsaAmdExtTableSnapshot, *VenLoaderSnapshot,
      std::move(PacketCallback), Err);
  if (Err)
    return;
  // Enable the LLVM time tracer if enabled
  if (*TimeTrace)
    llvm::timeTraceProfilerInitialize(*TimeTraceGranularity,
                                      "libLuthierTooling.so");
}

Context::~Context() {
  delete CG;
  delete IPR;
  delete CL;
  delete TEL;
  delete CodeObjectCache;
  delete VenLoaderSnapshot;
  delete HipCompilerTableSnapshot;
  delete HsaAmdExtTableSnapshot;
  delete HsaCoreApiTableSnapshot;
  delete MDParser;
  if (*TimeTrace) {
    LUTHIER_REPORT_FATAL_ON_ERROR(
        llvm::timeTraceProfilerWrite(*TimeTraceFile, "luthier-profile"));
    llvm::timeTraceProfilerCleanup();
  }
  delete TM;
}

} // namespace luthier
