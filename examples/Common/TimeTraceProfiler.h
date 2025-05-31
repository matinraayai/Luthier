//===-- TimeTraceProfiler.h -------------------------------------*- C++ -*-===//
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
/// Describes the \c TimeTraceProfiler object, a convenience
/// wrapper for using LLVM's time trace functionality in the example Luthier
/// tools.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_COMMON_TIME_TRACE_PROFILER_H
#define LUTHIER_EXAMPLES_COMMON_TIME_TRACE_PROFILER_H
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/common/Singleton.h>
#include <luthier/llvm/LLVMError.h>
#include <mutex>
#include <rocprofiler-sdk/defines.h>

namespace luthier {

/// \brief defines a set of CLI options used by the \c TimeTraceProfiler
class TimeTracerProfilerOptions {
private:
  llvm::cl::OptionCategory TimeTracingOptionCategory{"Time Tracing Options"};

  llvm::cl::opt<bool> TimeTraceEnabled{
      "time-trace",
      llvm::cl::desc("Enables recording of LLVM time trace from "
                     "when the Time Trace Profiler object is initialized"),
      llvm::cl::init(false), llvm::cl::cat(TimeTracingOptionCategory)};

  llvm::cl::opt<unsigned> TimeTraceGranularity{
      "time-trace-granularity",
      llvm::cl::desc("Minimum time granularity (in microseconds) traced by "
                     "time profiler"),
      llvm::cl::init(500), llvm::cl::Hidden,
      llvm::cl::cat(TimeTracingOptionCategory)};

  llvm::cl::opt<std::string> TimeTraceFile{
      "time-trace-file",
      llvm::cl::desc("If time-trace is enabled, specifies the "
                     "time trace file destination"),
      llvm::cl::value_desc("filename"),
      llvm::cl::cat(TimeTracingOptionCategory)};

  llvm::cl::opt<std::string> TimeTraceProcess{
      "time-trace-procname",
      llvm::cl::desc("If --time-trace is enabled, "
                     "specifies time trace process name"),
      llvm::cl::value_desc("process name"),
      llvm::cl::cat(TimeTracingOptionCategory)};

public:
  [[nodiscard]] bool isEnabled() const { return TimeTraceEnabled; }

  [[nodiscard]] unsigned getTimeTraceGranularity() const {
    return TimeTraceGranularity;
  }

  [[nodiscard]] const std::string &getTimeTraceFileName() const {
    return TimeTraceFile;
  }

  [[nodiscard]] const std::string &getTimeTraceProcessName() const {
    return TimeTraceProcess;
  }

  TimeTracerProfilerOptions(const TimeTracerProfilerOptions &other) = delete;

  TimeTracerProfilerOptions &
  operator=(const TimeTracerProfilerOptions &other) = delete;

  TimeTracerProfilerOptions() = default;
};

/// \brief A component used to manage the LLVM time trace
/// functionality based on user options passed on the CLI
class TimeTraceProfiler {
private:
  TimeTracerProfilerOptions *Options;

  explicit TimeTraceProfiler(std::unique_ptr<TimeTracerProfilerOptions> Options)
      : Options(Options.release()) {

    if (Options && Options->isEnabled()) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_ERROR_CHECK(!llvm::timeTraceProfilerEnabled(),
                              "LLVM Time tracer has already been initialized"));
      llvm::timeTraceProfilerInitialize(Options->getTimeTraceGranularity(),
                                        Options->getTimeTraceProcessName());
    }
  }

  ~TimeTraceProfiler() {
    if (Options && Options->isEnabled()) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_ERROR_CHECK(llvm::timeTraceProfilerEnabled(),
                              "LLVM Time tracer has already been destroyed"));
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_LLVM_ERROR_CHECK(llvm::timeTraceProfilerWrite(
              Options->getTimeTraceFileName(), "luthier-profile")));
      llvm::timeTraceProfilerCleanup();
    }
  }
};

} // namespace luthier

#endif