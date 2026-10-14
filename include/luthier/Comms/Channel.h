//===-- Channel.h - Device-to-host message channel for a tool ---*- C++ -*-===//
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
/// \file
/// A generic GPU-to-CPU channel for Luthier tools. It knows nothing about
/// what is sent: a hook calls \c Channel::send with a few dwords per active
/// lane and a \c MessageInfo, and the host hands every message to the
/// registered \c MessageHandler s, which pick theirs by \c WaveHeader::Tag.
///
/// Usage, in the tool:
/// \code
///   struct MyTool : luthier::Tool<MyTool, llvm::MachineFunction>,
///                   luthier::comms::Channel<MyTool> {
///     __attribute__((device, used)) static void myHook(...) {
///       const uint32_t Lane[1] = {...};
///       send<1>(SubBuffer, {MyTag, Site, Info, BlockX, BlockY, BlockZ,
///                           luthier::comms::device::readHwLocation()},
///               Lane);
///     }
///   };
///   // once:          setChannelConfig(...); addHandler(MyHandler);
///   // per dispatch:  openChannel(KD) before submitting the packet,
///   //                closeChannel() after it completed.
/// \endcode
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_CHANNEL_H
#define LUTHIER_COMMS_CHANNEL_H

#include "luthier/Comms/Device.h"
#include "luthier/Comms/MessageProcessor.h"
#include "luthier/Comms/SharedBuffers.h"

#include <hip/hip_runtime.h>

#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSA/hsa.h>

#include <memory>
#include <vector>

namespace luthier::comms {

template <typename ToolT> class Channel {
public:
  //===--------------------------------------------------------------------===//
  // Device code
  //===--------------------------------------------------------------------===//

  /// Where the hooks write; set by \c openChannel before each dispatch.
  __attribute__((device)) static ChannelDescriptor Desc;

  /// One lock per sub-buffer, serialising the waves that share it.
  __attribute__((device)) static uint32_t Locks[MaxSubBuffers];

  /// Sends \p Lane (this lane's \p N dwords) from every active lane as one
  /// message through sub-buffer \p SubBuffer, which must be wave-uniform
  /// (see \c subBufferFor).
  template <unsigned N>
  __attribute__((device, always_inline)) static void
  send(uint32_t SubBuffer, const device::MessageInfo &M,
       const uint32_t (&Lane)[N]) {
#if defined(__HIP_DEVICE_COMPILE__)
    device::submit<N>(Desc, Locks, SubBuffer, M, Lane);
#endif
  }

  /// A wave-uniform sub-buffer choice that spreads waves by workgroup.
  __attribute__((device, always_inline)) static uint32_t
  subBufferFor(uint32_t BlockX, uint32_t BlockY, uint32_t BlockZ) {
#if defined(__HIP_DEVICE_COMPILE__)
    return (BlockX + 7919u * BlockY + 104729u * BlockZ) * 2654435761u >> 8;
#else
    return 0;
#endif
  }

  //===--------------------------------------------------------------------===//
  // Host code
  //===--------------------------------------------------------------------===//

  struct Config {
    uint32_t NumSubBuffers = 256; ///< Power of two, <= MaxSubBuffers.
    uint64_t SubBufferBytes = 64 * 1024;
  };

  void setChannelConfig(const Config &C) { Cfg = C; }

  /// Messages go to every handler; each one checks the tag.
  void addHandler(MessageHandler &H) { Handlers.push_back(&H); }

  const MessageProcessor *processor() const { return Processor.get(); }
  bool droppedMessages() const { return Dropped; }

  /// Points the instrumented kernel at empty buffers and starts draining.
  /// Call after the dispatch was overridden with its instrumented variant
  /// and before it is submitted; \p KD is the original kernel descriptor.
  llvm::Error openChannel(const llvm::amdhsa::kernel_descriptor_t &KD) {
    const auto Core = self().getCoreApiTableSnapshot().getTable();
    const auto AmdExt = self().getAmdExtTableSnapshot().getTable();
    if (!Processor) {
      auto BuffersOrErr = SharedBuffers::create(
          Core, AmdExt, {Cfg.NumSubBuffers, Cfg.SubBufferBytes});
      LUTHIER_RETURN_ON_ERROR(BuffersOrErr.takeError());
      Buffers = std::move(*BuffersOrErr);
      Processor = std::make_unique<MessageProcessor>(*Buffers, Handlers);
    }
    Buffers->reset();
    const ChannelDescriptor D = Buffers->describe();
    LUTHIER_RETURN_ON_ERROR(
        writeDeviceGlobal(&Channel::Desc, &D, sizeof(D), KD));
    static const uint32_t ZeroLocks[MaxSubBuffers] = {};
    LUTHIER_RETURN_ON_ERROR(
        writeDeviceGlobal(&Channel::Locks, ZeroLocks, sizeof(ZeroLocks), KD));
    Processor->start();
    return llvm::Error::success();
  }

  /// Drains what is left once the dispatch has completed.
  void closeChannel() {
    Processor->stop();
    Dropped |= Buffers->errorBits() & 1u;
  }

  /// Copies \p Bytes from \p Src into the device global \p Var of the
  /// instrumented code of \p KD. For tools' own per-dispatch device state.
  template <typename T>
  llvm::Error writeDeviceGlobal(T *Var, const void *Src, size_t Bytes,
                                const llvm::amdhsa::kernel_descriptor_t &KD) {
    const auto Core = self().getCoreApiTableSnapshot().getTable();
    auto SymOrErr = self().lookupGlobalVariable(Var, &KD);
    LUTHIER_RETURN_ON_ERROR(SymOrErr.takeError());
    auto AddrOrErr = hsa::executableSymbolGetAddress(Core, *SymOrErr);
    LUTHIER_RETURN_ON_ERROR(AddrOrErr.takeError());
    return LUTHIER_HSA_CALL_ERROR_CHECK(
        Core.template callFunction<hsa_memory_copy>(
            reinterpret_cast<void *>(*AddrOrErr), Src, Bytes),
        "comms: failed to write a device global");
  }

private:
  ToolT &self() { return static_cast<ToolT &>(*this); }

  Config Cfg;
  std::vector<MessageHandler *> Handlers;
  std::unique_ptr<SharedBuffers> Buffers;
  std::unique_ptr<MessageProcessor> Processor;
  bool Dropped = false;
};

template <typename ToolT>
__attribute__((device)) ChannelDescriptor Channel<ToolT>::Desc;
template <typename ToolT>
__attribute__((device)) uint32_t Channel<ToolT>::Locks[MaxSubBuffers];

} // namespace luthier::comms

#endif
