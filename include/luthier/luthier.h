//===-- luthier.h - Luthier Interface  --------------------------*- C++ -*-===//
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
/// This file defines the public-facing interface of Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_H
#define LUTHIER_H
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include "luthier/common/ErrorCheck.h"
#include "luthier/tooling/InstrumentationTask.h"
#include "luthier/tooling/LiftedRepresentation.h"
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/Support/Error.h>
#include <luthier/Consts.h>
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/hip/TraceApi.h>
#include <luthier/hsa/Instr.h>
#include <luthier/hsa/KernelDescriptor.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>
#include <luthier/hsa/TraceApi.h>
#include <luthier/types.h>

namespace luthier {

//===----------------------------------------------------------------------===//
// Luthier callback APIs (required to be implemented by all tools)
//===----------------------------------------------------------------------===//

/// A callback function required to be defined in a Luthier tool. The tooling
/// library invokes this function before and after Luthier's internal components
/// are initialized \n
/// This function is typically used for setting tool callback functions that are
/// considered optional for a luthier tool
/// (e.g. <tt>hsa::setAtApiTableCaptureEvtCallback</tt>),
/// printing a banner for the tool, or doing non-HIP/HSA related initializations
/// \warning It is not safe to use HSA or HIP functions in this callback,
/// as at this point of execution neither HIP or HSA are initialized.
/// \warning This callback should not enable callbacks for HIP/HSA APIs, as
/// at this point of execution, HIP/HSA API tables have not been captured by
/// Luthier; Use \c hsa::setAtApiTableCaptureEvtCallback and
/// \c hip::setAtApiTableCaptureEvtCallback instead
/// \warning It is unsafe to access statically-initialized LLVM command-line
/// arguments of the tool. Wait until any of the HIP/HSA API tables to be
/// captured to access them
/// \note This function is required to be implemented by all Luthier tools
/// \param Phase \c API_EVT_PHASE_BEFORE when called before tool initialization
/// or \c API_EVT_PHASE_AFTER when called after tool initialization
void atToolInit(ApiEvtPhase Phase);

/// A callback invoked before and after Luthier's sub-systems are
/// destroyed and finalized\n
/// Use this callback to print out/process results that are already on the host,
/// and finalize the tool\n
/// \warning No HSA or HIP function should be used inside this callback,
/// as at this point both HSA and HIP runtimes are finalized\n
/// \warning If any interactions are to happen inside this callback and
/// Luthier's internal components, it must be only be done when \p Phase
/// is \c API_EVT_PHASE_BEFORE
/// \note This function is required to be implemented by all Luthier tools
/// \param Phase \c API_EVT_PHASE_BEFORE when called before tool finalization
/// or \c API_EVT_PHASE_AFTER when called after tool finalization
void atToolFini(ApiEvtPhase Phase);

/// A function defined by all tools which returns the tool's name
/// The returned name will be passed to rocprofiler-sdk as the Luthier tool's
/// identifier
/// \return the Luthier tool's name
llvm::StringRef getToolName();

//===----------------------------------------------------------------------===//
// HSA/HIP callback functions
//===----------------------------------------------------------------------===//

namespace hsa {

/// If called, invokes the \p Callback before and after the HSA API table has
/// been captured by Luthier\n
/// Use this function to request callbacks for each \c luthier::hsa::ApiEvtID
/// (only after tables have been captured)
/// If the \p Callback is called during <tt>API_EVT_PHASE_AFTER</tt>, it
/// is allowed to use HSA functions via <tt>luthier::hsa::getHsaApiTable</tt>
/// \param Callback the function to be called before/after the HSA API tables
/// has been captured by Luthier
void setAtApiTableCaptureEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback);

/// If called, performs the \p Callback before and after each HSA API or event
/// being actively captured by Luthier\n
/// \param Callback the function to be called before/after each HSA API or event
/// being actively captured by Luthier
/// \sa hsa_trace_api.h, enableHsaApiEvtIDCallback, disableHsaApiEvtIDCallback,
/// enableAllHsaApiEvtCallbacks, disableAllHsaCallbacks
void setAtHsaApiEvtCallback(
    const std::function<void(ApiEvtArgs *, ApiEvtPhase, ApiEvtID)> &Callback);

/// Enables capturing of the given \p Op and performs a HSA API/EVT
/// callback everytime it is reached in the application
/// \note This function must be called after the HSA API tables have been
/// captured by Luthier. The tool will be notified of this event by the
/// \c luthier::setAtApiTableCaptureEvtCallback function, when \c ApiEvtPhase
/// is <tt>API_EVT_PHASE_AFTER</tt>.
/// \param ApiID the API/EVT ID to be captured
/// \returns an \c llvm::Error if any issue was encountered during the process
/// \sa setAtHsaApiEvtCallback, disableHsaApiEvtIDCallback,
/// enableAllHsaApiEvtCallbacks, disableAllHsaCallbacks
llvm::Error enableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID);

/// Disables capturing of the given HSA \p Op and its callback
/// \note This function must be called after the HSA API tables have been
/// captured by Luthier. The tool will be notified of this event by the
/// \c luthier::setAtApiTableCaptureEvtCallback function, when \c ApiEvtPhase
/// is <tt>API_EVT_PHASE_AFTER</tt>.
/// \param ApiID the API/EVT ID to stop capturing
/// \returns an \c llvm::Error if any issue was encountered during the process
/// \sa setAtHsaApiEvtCallback, enableHsaApiEvtIDCallback,
/// enableAllHsaApiEvtCallbacks, disableAllHsaCallbacks
llvm::Error disableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID);

} // namespace hsa

namespace hsa {

//===----------------------------------------------------------------------===//
//  HSA/HIP functionality on the tool side
//===----------------------------------------------------------------------===//

/// Use this function to call HSA functions without it being intercepted
/// \return a reference to the original HSA API table i.e. the table containing
/// the original, un-intercepted version of the HSA API functions
const HsaApiTable &getHsaApiTable();

/// Use the AMD vendor loader API only to query segment descriptors
/// \return a reference to the original AMD
const hsa_ven_amd_loader_1_03_pfn_s &getHsaVenAmdLoaderTable();

} // namespace hsa

namespace hip {

/// \return a const reference to the original HIP Compiler API table
const HipCompilerDispatchTable &getSavedCompilerTable();

/// \return a const reference to the original HIP Dispatch API table
const HipDispatchTable &getSavedDispatchTable();

} // namespace hip

//===----------------------------------------------------------------------===//
//  Inspection API
//===----------------------------------------------------------------------===//

/// Disassembles the \p Kernel into a list of <tt>hsa::Instr</tt>\n
/// Disassembly only occurs on the first time this function is invoked on
/// \p Kernel. Subsequent calls will use a result cached internally\n
/// \note This function only provides a raw LLVM MC view of the instructions;
/// For instrumentation, use <tt>lift</tt>
/// \param Kernel the kernel symbol to be disassembled
/// \return an \c llvm::ArrayRef to an internally cached vector of
/// <tt>hsa::Instr</tt>s, or an \c llvm::Error if an issue was encountered
/// during the process
llvm::Expected<llvm::ArrayRef<hsa::Instr>>
disassemble(const hsa::LoadedCodeObjectKernel &Kernel);

/// Disassembles the \p Func into a list of <tt>hsa::Instr</tt>.\n
/// Disassembly only occurs on the first time this function is invoked on
/// \p Func. Subsequent calls will use a result cached internally.\n
/// \note This function only provides a raw LLVM MC view of the instructions;
/// For instrumentation, use <tt>lift</tt>
/// \param Func the device function to be disassembled
/// \return an \c llvm::ArrayRef to an internally cached vector of
/// <tt>hsa::Instr</tt>s, or an \c llvm::Error if an issue was encountered
/// during the process
llvm::Expected<llvm::ArrayRef<hsa::Instr>>
disassemble(const hsa::LoadedCodeObjectDeviceFunction &Func);

/// Lifts the given \p Kernel and return a reference to its
/// <tt>LiftedRepresentation</tt>.\n
/// The lifted result gets cached internally on the first invocation.
/// \param [in] Kernel an \c hsa::LoadedCodeObjectKernel to be lifted
/// \return a reference to the internally-cached <tt>LiftedRepresentation</tt>
/// if successful, or an \c llvm::Error describing the issue encountered.
/// \sa LiftedRepresentation, \sa lift
llvm::Expected<const luthier::LiftedRepresentation &>
lift(const hsa::LoadedCodeObjectKernel &Kernel);

/// Lifts the given \p Executable and returns a reference to its
/// <tt>LiftedRepresentation</tt>.\n
/// The lifted result gets cached internally on the first invocation.
/// \param [in] Executable an \c hsa_executable_t to be lifted
/// \return a reference to the internally-cached <tt>LiftedRepresentation</tt>
/// if successful, or an \c llvm::Error describing the issue encountered.
/// \sa LiftedRepresentation, \sa lift
llvm::Expected<const luthier::LiftedRepresentation &>
lift(hsa_executable_t Executable);

//===----------------------------------------------------------------------===//
//  Instrumentation API
//===----------------------------------------------------------------------===//

/// Instruments the \p LR by applying the \p Mutator and
/// \param LR the \c LiftedRepresentation about to be instrumented
/// \param Mutator the lambda that instruments and modifies the \p LR
/// \return returns a new \c LiftedRepresentation containing the instrumented
/// code, or an \c llvm::Error if an issue was encountered during
/// instrumentation
llvm::Expected<std::unique_ptr<LiftedRepresentation>>
instrument(const LiftedRepresentation &LR,
           llvm::function_ref<llvm::Error(InstrumentationTask &,
                                          LiftedRepresentation &)>
               Mutator);

/// Applies the assembly printer pass on the \p LR to generate object files or
/// assembly files for each of its <tt>llvm::Module</tt>s and
/// <tt>llvm::MachineModuleInfo</tt>s
/// \note After printing, and all of the <tt>LR</tt>'s
/// <tt>llvm::MachineModuleInfo</tt>s will be deleted; This is due to an LLVM
/// design shortcoming which is being worked on
/// \param [in] LR the \c LiftedRepresentation to be printed into an assembly
/// file; Its \c llvm::LLVMTargetMachine can be used to control the
/// \c llvm::TargetOptions of the compilation process
/// \param [out] CompiledObjectFiles the printed assembly file
/// \param [in] FileType Type of the assembly file printed; Can be either
/// \c llvm::CodeGenFileType::AssemblyFile or
/// \c llvm::CodeGenFileType::ObjectFile
/// \return an \c llvm::Error in case of any issues encountered during the
/// process
llvm::Error printLiftedRepresentation(
    LiftedRepresentation &LR,
    llvm::SmallVectorImpl<
        std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>>
        &CompiledObjectFiles,
    llvm::CodeGenFileType FileType = llvm::CodeGenFileType::ObjectFile);

// TODO: Implement link to executable, and load methods individually +
//  update the instrumentAndLoad docs

/// Instruments the <tt>Kernel</tt>'s lifted representation \p LR by
/// applying the instrumentation task <tt>ITask</tt> to it.\n After
/// instrumentation, loads the instrumented code onto the same device as the
/// \p Kernel
/// \param Kernel the kernel that's about to be instrumented
/// \param LR the lifted representation of the \p Kernel
/// \param ITask the instrumentation task, describing the instrumentation to
/// be performed on the <tt>kernel</tt>'s <tt>LR</t>
/// \return an \c llvm::Error describing if the operation succeeded or
/// failed
llvm::Error
instrumentAndLoad(const hsa::LoadedCodeObjectKernel &Kernel,
                  const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset);

/// Instruments the <tt>Exec</tt>'s lifted representation \p LR by applying
/// the instrumentation task <tt>ITask</tt> to it.\n
/// After instrumentation, loads the instrumented code onto the same device
/// as the \p Exec
/// \param Exec
/// \param LR
/// \param ITask
/// \return
llvm::Error
instrumentAndLoad(hsa_executable_t Exec, const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset);

/// Checks if the \p Kernel is instrumented under the given \p Preset or not
/// \param [in] Kernel the \c hsa::LoadedCodeObjectKernel of the app
/// \param [in] Preset the preset name the kernel was instrumented under
/// \return on success, returns \c true if the \p Kernel is instrumented, \c
// false otherwise. Returns an \c llvm::Error if the \p Kernel HSA symbol handle
// is invalid
llvm::Expected<bool>
isKernelInstrumented(const hsa::LoadedCodeObjectKernel &Kernel,
                     llvm::StringRef Preset);

/// Overrides the kernel object field of the Packet with its instrumented
/// version under the given \p Preset, forcing HSA to launch the
/// instrumented version instead\n Modifies the rest of the launch
/// configuration (e.g. private segment size) if needed\n Note that this
/// function should be called every time an instrumented kernel needs to be
/// launched, since the content of the dispatch packet will always be set by
/// the target application to the original, un-instrumented version\n To
/// launch the original version of the kernel, simply refrain from calling
/// this function
/// \param Packet Packet the HSA dispatch packet intercepted from an HSA
/// queue,
// containing the kernel launch parameters/configuration
/// \param Preset the preset the kernel was instrumented under
/// \return an \c llvm::Error reporting
llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                                     llvm::StringRef Preset);

/// \brief If a tool contains an instrumentation hook it \b must
/// use this macro once. Luthier hooks are annotated via the the
/// \p LUTHIER_HOOK_CREATE macro. \n
///
/// \p MARK_LUTHIER_DEVICE_MODULE macro defines a managed variable of
/// type \p char named \p __luthier_reserved in the tool device code.
/// This managed variable ensures that: \n
/// 1. <b>The HIP runtime is forced to load the tool code object before the
/// first HIP kernel is launched on the device, without requiring eager binary
/// loading to be enabled</b>: The Clang compiler embeds the device code of a
/// Luthier tool and its bitcode into a static HIP FAT binary bundled within the
/// tool's shared object. During runtime, the tool's FAT binary gets
/// registered with the HIP runtime; However, by default, the HIP runtime loads
/// FAT binaries in a lazy fashion, only loading it onto a device if:
/// a. a kernel is launched from it on the said device, or
/// b. it contains a managed variable. \n
/// Including a managed variable is the only way to ensure the tool's FAT binary
/// is loaded in time without interfering with the loading mechanism of HIP
/// runtime.
/// \n
/// 2. <b>Luthier can easily identify a tool's code object by a constant time
/// symbol hash lookup</b>.
/// \n
/// If the target application is not using the HIP runtime, then no kernel is
/// launched by the HIP runtime, meaning that the tool FAT binary does not ever
/// get loaded. In that scenario, as the HIP runtime is present solely for
/// Luthier's function, the `HIP_ENABLE_DEFERRED_LOADING` environment
/// variable must be set to zero to ensure Luthier tool code objects get loaded
/// right away on all devices.
///
/// * \sa LUTHIER_HOOK_ANNOTATE
#define MARK_LUTHIER_DEVICE_MODULE                                             \
  __attribute__((managed, used)) char LUTHIER_RESERVED_MANAGED_VAR = 0;

#define LUTHIER_HOOK_ANNOTATE                                                  \
  __attribute__((                                                              \
      device, used,                                                            \
      annotate(LUTHIER_STRINGIFY(LUTHIER_HOOK_ATTRIBUTE)))) extern "C" void

#define LUTHIER_EXPORT_HOOK_HANDLE(HookName)                                   \
  __attribute__((global, used)) extern "C" void LUTHIER_CAT(                         \
      LUTHIER_HOOK_HANDLE_PREFIX, HookName)(){};

#define LUTHIER_GET_HOOK_HANDLE(HookName)                                      \
  reinterpret_cast<const void *>(                                              \
      LUTHIER_CAT(LUTHIER_HOOK_HANDLE_PREFIX,HookName))
} // namespace luthier

#endif
