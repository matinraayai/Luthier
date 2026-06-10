//===-- HostcallConsumer.h - Luthier host hostcall listener -----*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file HostcallConsumer.h
/// A self-contained host-side hostcall listener for instrumented kernels. The
/// ROCclr runtime only provisions a hostcall buffer + consumer when the
/// *application* itself uses hostcall (printf etc.); Luthier-instrumented
/// kernels that hostcall (e.g. the indirect-branch resolver's slow path) need
/// their own. This replicates the essential half of ROCclr's
/// \c amd::enableHostcalls (\c device/devhostcall.cpp): allocate a system
/// fine-grained, atomics-capable buffer accessible to a GPU agent, lay it out
/// per the device-libs ABI (\c HostcallABI.h), arm a doorbell signal, and run a
/// listener thread that drains the ready stack and dispatches
/// \c SERVICE_FUNCTION_CALL packets by invoking the host function pointer the
/// device put in the payload.
///
/// The device half is \c __ockl_hostcall_internal, which reads the buffer
/// pointer out of the kernel's COV5 hostcall implicit arg
/// (\c cov5::HostcallPtr) — Luthier publishes \c getBufferPointer() there via
/// the custom-kernarg buffer.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_HOSTCALL_CONSUMER_H
#define LUTHIER_HSA_TOOLING_HOSTCALL_CONSUMER_H

#include "luthier/HSA/ApiTable.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include <atomic>
#include <cstdint>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <thread>

namespace luthier {

/// Owns one device-libs-compatible hostcall buffer, its doorbell signal, and a
/// listener thread, for one GPU agent. Non-copyable; tears everything down (set
/// doorbell to \c SIGNAL_DONE, join the thread, destroy the signal, free the
/// buffer) on destruction.
class HostcallConsumer {
public:
  /// Allocate + initialize a hostcall buffer for \p GpuAgent, arm the doorbell,
  /// and launch the listener thread. \p NumPackets must exceed the agent's max
  /// concurrent waves so no wave ever blocks waiting for a free packet; the
  /// default is a generous upper bound.
  static llvm::Expected<std::unique_ptr<HostcallConsumer>>
  create(const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
         const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
         hsa_agent_t GpuAgent, uint32_t NumPackets = 2048);

  ~HostcallConsumer();

  HostcallConsumer(const HostcallConsumer &) = delete;
  HostcallConsumer &operator=(const HostcallConsumer &) = delete;

  /// Device-visible base of the hostcall buffer. Publish this into the kernel's
  /// COV5 hostcall implicit arg so \c __ockl_hostcall_internal finds it.
  [[nodiscard]] void *getBufferPointer() const { return Buffer; }

private:
  HostcallConsumer(hsa::ApiTableContainer<::CoreApiTable> Core,
                   hsa::ApiTableContainer<::AmdExtTable> AmdExt,
                   uint32_t NumPackets)
      : Core(Core), AmdExt(AmdExt), NumPackets(NumPackets) {}

  /// Listener loop: block on the doorbell, drain the ready stack.
  void consumeLoop();
  /// Drain the ready stack once and dispatch each active workitem's packet.
  void processPackets();

  hsa::ApiTableContainer<::CoreApiTable> Core;
  hsa::ApiTableContainer<::AmdExtTable> AmdExt;
  uint32_t NumPackets{0};
  /// Device-visible buffer base (system fine-grained, GPU-accessible).
  void *Buffer{nullptr};
  /// Doorbell the device pulses to announce work; the listener waits on it.
  hsa_signal_t Doorbell{};
  std::atomic<bool> Stop{false};
  std::thread Worker;
};

} // namespace luthier

#endif // LUTHIER_HSA_TOOLING_HOSTCALL_CONSUMER_H
