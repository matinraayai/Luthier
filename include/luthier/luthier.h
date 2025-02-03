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
/// \c HIP_ENABLE_WARP_SYNC_BUILTINS is defined before HIP runtime headers
/// are included
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/Support/Error.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/consts.h>
#include <luthier/hip/TraceApi.h>
/// Undef the ill-defined \c ICMP_NE in HIP headers
#undef ICMP_NE
#include <luthier/hsa/Instr.h>
#include <luthier/hsa/KernelDescriptor.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>
#include <luthier/hsa/TraceApi.h>
#include <luthier/intrinsic/Intrinsics.h>
#include <luthier/tooling/InstrumentationTask.h>
#include <luthier/tooling/LiftedRepresentation.h>
#include <luthier/types.h>

namespace luthier {

//===----------------------------------------------------------------------===//
// Luthier callback APIs (required to be implemented by all tools)
//===----------------------------------------------------------------------===//

/// A callback function required to be defined in a Luthier tool. It is invoked
/// by the tooling library before and after Luthier's internal components
/// are initialized. \n
/// This function is typically used for setting other tool callback functions
/// that are not necessarily required for a luthier tool
/// (e.g. <tt>hsa::setAtApiTableCaptureEvtCallback</tt>),
/// printing a banner for the tool, allocating global variables, or
/// doing non-HIP/HSA related initializations
/// \note This function is required to be implemented by all Luthier tools
/// \note This function is invoked by the tooling library inside the
/// \c rocprofiler_configure function, which registers the Luthier tool with the
/// rocprofiler-sdk library. Therefore, all restrictions imposed by rocprofiler
/// inside \c rocprofiler_configure applies here as well
/// \warning It is not safe to perform HSA or HIP functions in this callback,
/// as at this point of execution neither HIP or HSA are initialized. Use
/// <tt>luthier::hip::setAtHipCompilerTableCaptureEvtCallback</tt>,
/// <tt>setAtHipCompilerTableCaptureEvtCallback</tt>, and
/// <tt>hsa::setAtApiTableCaptureEvtCallback</tt> instead to be notified when
/// it is safe to make HIP/HSA call
/// \warning This callback should not enable callbacks for HSA APIs, as
/// at this point of execution, HSA API tables have not been captured by
/// Luthier; Use <tt>hsa::setAtApiTableCaptureEvtCallback</tt> instead to
/// set a callback for HSA calls inside the application
/// \param Phase \c API_EVT_PHASE_BEFORE when called before tool initialization
/// or \c API_EVT_PHASE_AFTER when called after tool initialization
void atToolInit(ApiEvtPhase Phase);

/// A callback invoked before and after Luthier's sub-systems are
/// destroyed and finalized\n
/// Use this callback to print out/process results that are already on the host,
/// de-allocate any global variables, or finalize the tool\n
/// \note This function is required to be implemented by all Luthier tools
/// \note This function is invoked by the tooling library inside the
/// rocprofiler-sdk's tool finalizer function. Therefore, any restrictions
/// imposed by rocprofiler-sdk imposed on tool finalizer functions also apply
/// here
/// \warning The tool must not make any calls to the tooling library which
/// relies on a persistent internal state (e.g. <tt>luthier::outs</tt>,
/// requesting API tables for HIP/HSA) in the \c API_EVT_PHASE_AFTER phase, as
/// at this point, all internal global variables of the tooling library have
/// been destroyed via \c llvm_shutdown
/// \param Phase \c API_EVT_PHASE_BEFORE when called before tool finalization
/// or \c API_EVT_PHASE_AFTER when called after tool finalization
void atToolFini(ApiEvtPhase Phase);

/// A function required to be defined by all tools, which the tool's name
/// \note The returned \c llvm::StringRef will be passed to rocprofiler-sdk as
/// the Luthier tool's identifier, and must remain valid throughout the
/// lifetime of the Luthier tool
/// \return the Luthier tool's name
llvm::StringRef getToolName();

//===----------------------------------------------------------------------===//
// HSA/HIP/Rocprofiler callback functions
//===----------------------------------------------------------------------===//

namespace rocprofiler_sdk {

/// Sets a \p Callback that can be used to create a rocprofiler-sdk context and
/// request rocprofiler-sdk services
/// \note This function gets invoked as part of <tt>rocprofiler_configure</tt>'s
/// initializer function, therefore any restrictions imposed by rocprofiler-sdk
/// also applies here
void setServiceInitCallback(const std::function<void()> &Callback);

} // namespace rocprofiler_sdk

namespace hsa {

/// If set, invokes the \p Callback before and after the HSA API table has
/// been provided to Luthier by rocprofiler-sdk\n
/// Use this function to request callbacks for each \c luthier::hsa::ApiEvtID
/// after the tables have been captured
/// \note During <tt>API_EVT_PHASE_AFTER</tt>, it is safe to use HSA
/// functions via <tt>luthier::hsa::getHsaApiTable</tt>
/// \param Callback the function to be called before/after the HSA API tables
/// has been captured by Luthier
void setAtApiTableCaptureEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback);

/// Enables callbacks to be invoked when the given \p ApiID is reached inside
/// the target application
/// \note This function must be called after the HSA API tables have been
/// captured by the tooling library i.e. inside the callback function provided
/// to <tt>hsa::setAtApiTableCaptureEvtCallback</tt>.
/// \warning The Luthier tool must declare all possible <tt>hsa::ApiEvtID</tt>s
/// it intends to monitor inside the callback function provided
/// to <tt>hsa::setAtApiTableCaptureEvtCallback</tt>. After the HSA API table
/// capture event has passed, the tool cannot request for callbacks for any
/// additional events, as no additional modifications are possible to the
/// ROCr runtime's API table. It is safe, however, to enable and disable
/// callbacks for <tt>hsa::ApiEvtID</tt> already requested by the tool.
/// \note Use \c hsa::setAtHsaApiEvtCallback to set the callback invoked at
/// each HSA event.
/// \param ApiID the API/EVT ID to be captured via a callback
/// \returns an \c llvm::Error if any issue was encountered during the process
/// \sa setAtHsaApiEvtCallback, disableHsaApiEvtIDCallback
llvm::Error enableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID);

/// If called, invokes \p Callback before and after each HSA API or event
/// inside the target application\n
/// \param Callback the function to be called before/after each HSA API or event
/// being actively captured by Luthier
/// \sa hsa_trace_api.h, enableHsaApiEvtIDCallback, disableHsaApiEvtIDCallback
void setAtHsaApiEvtCallback(
    const std::function<void(ApiEvtArgs *, ApiEvtPhase, ApiEvtID)> &Callback);


/// Disables capturing of the given HSA \p Op and its callback
/// \note This function must be called after the HSA API tables have been
/// captured by the tooling library i.e. inside the callback function provided
/// to <tt>hsa::setAtApiTableCaptureEvtCallback</tt>
/// \param ApiID the API/EVT ID to stop capturing
/// \returns an \c llvm::Error if any issue was encountered during the process
/// \sa setAtHsaApiEvtCallback, enableHsaApiEvtIDCallback
llvm::Error disableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID);

} // namespace hsa

namespace hip {

/// If called, invokes the \p Callback before and after the HIP Compiler API
/// table has been provided to Luthier by rocprofiler-sdk\n
/// Use this function to be notified exactly when the HIP compiler tabl has been
/// initialized i.e. it is safe to perform HIP compiler calls
/// \param Callback the function to be called before/after the HIP Compiler API
/// tables has been captured by Luthier
void setAtHipCompilerTableCaptureEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback);

/// If set, invokes the \p Callback before and after the HIP Dispatch API
/// table has been provided to Luthier by rocprofiler-sdk\n
/// Use this function to be notified exactly when the HIP runtime has been
/// initialized i.e. it is safe to perform HIP runtime calls
/// \param Callback the function to be called before/after the HIP Dispatch API
/// tables has been captured by Luthier
void setAtHipDispatchTableCaptureEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback);

} // namespace hip

namespace hsa {

//===----------------------------------------------------------------------===//
//  HSA/HIP functionality on the tool side
//===----------------------------------------------------------------------===//

/// \return a reference to the original HSA API table i.e. the table containing
/// the original, un-intercepted version of the HSA API functions
/// \warning Never invoke HSA APIs directly inside a Luthier tool;
/// Use this function to call HSA functions without it being re-intercepted by
/// Luthier instead
const HsaApiTable &getHsaApiTable();

/// \return a reference to the original AMD Loader table
/// \sa hsa_ven_amd_loader_1_03_pfn_s
const hsa_ven_amd_loader_1_03_pfn_s &getHsaVenAmdLoaderTable();

} // namespace hsa

namespace hip {

/// \return a const reference to the original HIP Compiler API table i.e.
/// the table containing the original, un-intercepted version of the
/// HIP Compiler API functions
/// \warning Never invoke HIP Compiler APIs directly inside a Luthier tool;
/// Use this function to call HIP Compiler API functions without it being
/// re-intercepted by Luthier instead
const HipCompilerDispatchTable &getSavedCompilerTable();

/// \return a const reference to the original HIP Dispatch API table i.e.
/// the table containing the original, un-intercepted version of the
/// HIP Dispatch API functions
/// \warning Never invoke HIP Dispatch APIs directly inside a Luthier tool;
/// Use this function to call HIP Dispatch API functions without it being
/// re-intercepted by Luthier instead
const HipDispatchTable &getSavedDispatchTable();

} // namespace hip

//===----------------------------------------------------------------------===//
//  Inspection APIs
//===----------------------------------------------------------------------===//

/// Disassembles the \p Kernel into a list of <tt>hsa::Instr</tt>\n
/// Disassembly only occurs on the first time this function is invoked on
/// the \p Kernel. Subsequent calls will use a result cached internally\n
/// \note This function only provides a raw LLVM MC view of the instructions;
/// For instrumentation, use <tt>lift</tt> instead
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
/// For instrumentation, use <tt>lift</tt> instead
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

/// Instruments the \p LR by applying the \p Mutator to it
/// \param LR the \c LiftedRepresentation about to be instrumented
/// \param Mutator a function that instruments and modifies the \p LR
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
/// file; Its \c llvm::TargetMachine can be used to control the
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
/// \param Exec the executable being instrumented
/// \param LR the lifted representation of the \p Exec
/// \param Mutator a function that applies an instrumentation task and mutates
/// the lifted representation
/// \return \c llvm::Error describing any issues that might have been
/// encountered during the process
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
/// false otherwise. Returns an \c llvm::Error if the \p Kernel HSA symbol handle
/// is invalid
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
/// \sa LUTHIER_HOOK_ANNOTATE
#define MARK_LUTHIER_DEVICE_MODULE                                             \
  __attribute__((managed, used)) char LUTHIER_RESERVED_MANAGED_VAR = 0;

#define LUTHIER_HOOK_ANNOTATE                                                  \
  __attribute__((                                                              \
      device, used,                                                            \
      annotate(LUTHIER_STRINGIFY(LUTHIER_HOOK_ATTRIBUTE)))) extern "C" void

#define LUTHIER_EXPORT_HOOK_HANDLE(HookName)                                   \
  __attribute__((global, used)) extern "C" void LUTHIER_CAT(                   \
      LUTHIER_HOOK_HANDLE_PREFIX, HookName)(){};

#define LUTHIER_GET_HOOK_HANDLE(HookName)                                      \
  reinterpret_cast<const void *>(                                              \
      LUTHIER_CAT(LUTHIER_HOOK_HANDLE_PREFIX, HookName))
} // namespace luthier

#endif
