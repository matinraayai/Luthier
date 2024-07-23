//===-- luthier.h - Luthier Interface  --------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the public-facing interface of Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_H
#define LUTHIER_H
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/Support/Error.h>

#include <hip/amd_detail/hip_api_trace.hpp>
#include <luthier/hip_trace_api.h>
#include <luthier/hsa_trace_api.h>
#include <luthier/instr.h>
#include <luthier/instrumentation_task.h>
#include <luthier/kernel_descriptor.h>
#include <luthier/lifted_representation.h>
#include <luthier/types.h>

namespace luthier {

/// A function that will be called before and after Luthier's sub-systems are
/// initialized, with timing indicated by \p Phase\n
/// Use this function to print a banner for the tool, parse arguments,
/// or (only after init) setup different types of Luthier callbacks\n
/// No HSA or HIP function should be used inside this function, as at this
/// point non of them would likely to be initialized\n
/// This function is required to be implemented by tools
/// \param Phase \c API_EVT_PHASE_BEFORE when called before tool initialization
/// or \c API_EVT_PHASE_AFTER when called after tool initialization
void atToolInit(ApiEvtPhase Phase);

/// A function that will be called before and after Luthier's sub-systems are
/// destroyed and finalized\n
/// Use this function to print out/process results that are already on the host,
/// and clean up after the tool\n
/// No HSA or HIP function should be used inside this function, as at this
/// point non of them are likely to be initialized\n
/// This function is required to be implemented by tools
/// \param Phase \c API_EVT_PHASE_BEFORE when called before tool finalization
/// or \c API_EVT_PHASE_AFTER when called after tool finalization
void atFinalization(ApiEvtPhase Phase);

namespace hsa {

//===----------------------------------------------------------------------===//
// HSA/HIP callback functions
//===----------------------------------------------------------------------===//

/// If called, performs the \p Callback before and after the HSA API table has
/// been captured by Luthier\n
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
/// \param ApiID the API/EVT ID to be captured
/// \sa setAtHsaApiEvtCallback, disableHsaApiEvtIDCallback,
/// enableAllHsaApiEvtCallbacks, disableAllHsaCallbacks
void enableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID);

/// Disables capturing of the given HSA \p Op and its callback
/// \param ApiID the API/EVT to stop capturing
/// \sa setAtHsaApiEvtCallback, enableHsaApiEvtIDCallback,
/// enableAllHsaApiEvtCallbacks, disableAllHsaCallbacks
void disableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID);

/// Convenience method for enabling all HSA API/EVT IDs capture and callbacks
void enableAllHsaApiEvtCallbacks();

/// Convenience method for disabling all HSA API/EVT
void disableAllHsaCallbacks();

} // namespace hsa

/// If called, calls the \p Callback function before/after Luthier finalizes
/// and releases the captured HIP/HSA API tables\n
/// During \c API_EVT_PHASE_BEFORE of the \p Callback is the tool's last chance
/// to perform calls to HIP/HSA functions
/// \param Callback the function to be called before/after the HSA/HIP API
/// tables are being released by Luthier
void setAtApiTableReleaseEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback);

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

/// Returns the \c SymbolKind of the given \p Symbol\n
/// Use this instead of querying the \c HSA_EXECUTABLE_SYMBOL_INFO_TYPE info
/// using HSA; If the \p Symbol is of type \c KERNEL or \c VARIABLE, then it
/// is safe to use with the HSA API; For symbols of type \c DEVICE_FUNCTION
/// strictly use Luthier APIs
/// \param Symbol the \c hsa_executable_symbol_t to be queried
/// \return Type of symbol on success, an \c llvm::Error in case the \c Symbol
/// is invalid
llvm::Expected<SymbolKind> getSymbolKind(hsa_executable_symbol_t Symbol);

/// Returns the name of the \p Symbol\n
/// Use this instead of querying the \c HSA_EXECUTABLE_SYMBOL_INFO_NAME info
/// using HSA; If the \p Symbol is of type \c KERNEL or \c VARIABLE, then it
/// is safe to use with the HSA API; For symbols of type \c DEVICE_FUNCTION
/// strictly use Luthier APIs
/// \param Symbol the \c hsa_executable_symbol_t to be queried
/// \return Name of the symbol on success, or an \c llvm::Error in case the
/// \c Symbol is invalid
llvm::Expected<llvm::StringRef> getSymbolName(hsa_executable_symbol_t Symbol);

/// Returns the executable this \p Symbol belongs to\n
/// Use this instead of the Loader API since Luthier internally caches the
/// mapping between a symbol and its executable
/// \param Symbol the \c hsa_executable_symbol_t being queried
/// \return on success, returns the \c hsa_executable_t of <tt>Symbol</tt>; an
/// \c llvm::Error if any issue is encountered during the process
llvm::Expected<hsa_executable_t>
getExecutableOfSymbol(hsa_executable_symbol_t Symbol);

/// Returns the loaded code object which defines the <tt>Symbol</tt>. If the
/// \p Symbol is external (not defined by any l
/// \param Symbol
/// \return the \c hsa_loaded_code_object_t that defines <tt>Symbol</tt> if
/// \p Symbol is not external; If the \p Symbol is external, returns
/// <tt>std::nullopt</tt>; If an error is encountered during the process,
/// returns an \c llvm::Error
llvm::Expected<std::optional<hsa_loaded_code_object_t>>
getDefiningLoadedCodeObject(hsa_executable_symbol_t Symbol);

} // namespace hsa

namespace hip {

/// \return a const reference to the original HIP Compiler API table
/// TODO: introduce thread-local temp callback disabling
const HipCompilerDispatchTable &getSavedCompilerTable();

} // namespace hip

//===----------------------------------------------------------------------===//
//  Inspection API
//===----------------------------------------------------------------------===//

/// Disassembles the Func into a list of <tt>hsa::Instr</tt>.\n
/// Disassembly only occurs when this function is called on the function for the
/// first time. Subsequent calls will use a result cached internally.\n
/// This function is provided for convenience; For instrumentation,
/// use <tt>lift</tt>.
/// \param Func the \c hsa_executable_symbol_t of type \c KERNEL or
/// \c DEVICE_FUNCTION to be disassembled
/// \return a <tt>const</tt> reference to an internally cached vector of
/// <tt>hsa::Instr</tt>s, or an \c llvm::Error if an issue was encountered
/// during the process.
llvm::Expected<const std::vector<hsa::Instr> &>
disassemble(hsa_executable_symbol_t Func, bool includeDebugInfo);

/// Lifts the given \p Kernel and return a reference to its
/// <tt>LiftedRepresentation</tt>.\n
/// The lifted result gets cached internally on the first invocation.
/// \param [in] Kernel an \c hsa_executable_symbol_t of type \c KERNEL
/// \return a reference to the internally-cached <tt>LiftedRepresentation</tt>
/// if successful, or an \c llvm::Error describing the issue encountered.
/// \sa LiftedRepresentation, \sa lift
llvm::Expected<const luthier::LiftedRepresentation &>
lift(hsa_executable_symbol_t Kernel, bool includeDebugInfo);

/// Lifts the given \p Executable and returns a reference to its
/// <tt>LiftedRepresentation</tt>.\n
/// The lifted result gets cached internally on the first invocation.
/// \param [in] Executable an \c hsa_executable_t to be lifted
/// \return a reference to the internally-cached <tt>LiftedRepresentation</tt>
/// if successful, or an \c llvm::Error describing the issue encountered.
/// \sa LiftedRepresentation, \sa lift
llvm::Expected<const luthier::LiftedRepresentation &>
lift(hsa_executable_t Executable, bool includeDebugInfo);

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

/// Instruments the <tt>Kernel</tt>'s lifted representation \p LR by applying
/// the instrumentation task <tt>ITask</tt> to it.\n
/// After instrumentation, loads the instrumented code onto the same
/// device as the \p Kernel
/// \param Kernel the kernel that's about to be instrumented
/// \param LR the lifted representation of the \p Kernel
/// \param ITask the instrumentation task, describing the instrumentation to
/// be performed on the <tt>kernel</tt>'s <tt>LR</t>
/// \return an \c llvm::Error describing if the operation succeeded or failed
llvm::Error
instrumentAndLoad(hsa_executable_symbol_t Kernel,
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
/// \param [in] Kernel an \c hsa_executable_symbol_t of \c KERNEL type
/// \param [in] Preset the preset name the kernel was instrumented under
/// \return on success, returns \c true if the \p Kernel is instrumented, \c
// false otherwise. Returns an \c llvm::Error if the \p Kernel HSA symbol handle
// is invalid
llvm::Expected<bool> isKernelInstrumented(hsa_executable_symbol_t Kernel,
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
  __attribute__((managed, used)) char __luthier_reserved = 0;

#define LUTHIER_HOOK_ANNOTATE                                                  \
  __attribute__((device, used, annotate("luthier_hook"))) extern "C" void

#define LUTHIER_EXPORT_HOOK_HANDLE(HookName)                                   \
  __attribute__((global,                                                       \
                 used)) extern "C" void __luthier_hook_handle_##HookName(){};

#define LUTHIER_GET_HOOK_HANDLE(HookName)                                      \
  reinterpret_cast<const void *>(__luthier_hook_handle_##HookName)
} // namespace luthier

////
/////*********************************************************************
//// *
//// *          NVBit inspection APIs  (provided by NVBit)
//// *
//// **********************************************************************/

////
/////* Get line information for a particular instruction offset if available,
//// * binary must be compiled with --generate-line-info   (-lineinfo) */
////bool nvbit_get_line_info(CUcontext cuctx, CUfunction cufunc, uint32_t
/// offset, /                         char** file_name, char** dir_name,
/// uint32_t* line);
////
/////* Get the SM family */
////uint32_t nvbit_get_sm_family(CUcontext cuctx);
////
/////* Allows to get PC address of the function */
////uint64_t nvbit_get_func_addr(CUfunction func);
////
////
/////* Allows to get shmem base address from CUcontext
//// * shmem range is [shmem_base_addr, shmem_base_addr+16MB) and
//// * the base address is 16MB aligned.  */
////uint64_t nvbit_get_shmem_base_addr(CUcontext cuctx);
////
/////* Allows to get local memory base address from CUcontext
//// * local mem range is [shmem_base_addr, shmem_base_addr+16MB) and
//// * the base address is 16MB aligned.  */
////uint64_t nvbit_get_local_mem_base_addr(CUcontext cuctx);
////
/////*********************************************************************
//// *
//// *          NVBit injection APIs  (provided by NVBit)
//// *
//// **********************************************************************/
////
////
/////* This function inserts a device function call named "dev_func_name",
//// * before or after Instr (ipoint_t { IPOINT_BEFORE, IPOINT_AFTER}).
//// * It is important to remember that calls to device functions are
//// * identified by name (as opposed to function pointers) and that
//// * we declare the function as:
//// *
//// *        extern "C" __device__ __noinline__
//// *
//// * to prevent the compiler from optimizing out this device function
//// * during compilation.
//// *
//// * Multiple device functions can be inserted before or after and the
//// * order in which they get executed is defined by the order in which
//// * they have been inserted. */
////

/////* Add int32_t argument to last injected call, value of the predicate for
/// this / * instruction */ /
/// void nvbit_add_call_arg_pred_val(const Instr*
/// instr, /                                 bool is_variadic_arg = false);
////
/////* Add int32_t argument to last injected call, value of the entire predicate
//// * register for this thread */
////void nvbit_add_call_arg_pred_reg(const Instr* instr,
////                                 bool is_variadic_arg = false);
////
/* Add uint32_t argument to last injected call, constant 32-bit value */
// void luthier_add_call_arg_const_val32(luthier_instruction_t instr,
//                                       uint32_t val);
////
/////* Add uint64_t argument to last injected call, constant 64-bit value */
// void luthier_add_call_arg_const_val64(luthier_instruction_t instr,
//                                       uint64_t val);
////
/////* Add uint32_t argument to last injected call, content of the register
/// reg_num / */ /void nvbit_add_call_arg_reg_val(const Instr* instr, int
/// reg_num, /                                bool is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, content of the
//// * uniform register reg_num */
////void nvbit_add_call_arg_ureg_val(const Instr* instr, int reg_num,
////                                 bool is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, 32-bit at launch value at
/// offset / * "offset", set at launch time with nvbit_set_at_launch */ /void
/// nvbit_add_call_arg_launch_val32(const Instr* instr, int offset, / bool
/// is_variadic_arg = false);
////
/////* Add uint64_t argument to last injected call, 64-bit at launch value at
/// offset / * "offset", set at launch time with nvbit_set_at_launch */ /void
/// nvbit_add_call_arg_launch_val64(const Instr* instr, int offset, / bool
/// is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, constant bank value at
//// * c[bankid][bankoffset] */
////void nvbit_add_call_arg_cbank_val(const Instr* instr, int bankid,
////                                  int bankoffset, bool is_variadic_arg =
/// false);
////
/////* The 64-bit memory reference address accessed by this instruction
////  Typically memory instructions have only 1 MREF so in general id = 0 */
////void nvbit_add_call_arg_mref_addr64(const Instr* instr, int id = 0,
////                                    bool is_variadic_arg = false);
////

////
/////*********************************************************************
//// *
//// *          NVBit device level APIs  (provided by NVBit)
//// *
//// **********************************************************************/
////
////#ifdef __CUDACC__
/////* device function used to read/write register values
//// * writes are permanent into application state */
// Save callee registers
// Call
// Sync
// Read content of register to SGPR[30:31]
// return SGPR[30:31]
// SGPR3 (where it actually is in the app) -> SGRP[30:31]
//
//
// __device__ __noinline__ int32_t nvbit_read_reg(uint64_t reg_num);
////__device__ __noinline__ void nvbit_write_reg(uint64_t reg_num, int32_t
/// reg_val);
// -> instrmnt -> nvbit_write_reg(R2, 2000);
// R5 = R5 + 2
// -> instrmnt -> nvbit_write_reg(R2, 2000); R5 = 2000;
////__device__ __noinline__ int32_t nvbit_read_ureg(uint64_t reg_num);
////__device__ __noinline__ void nvbit_write_ureg(uint64_t reg_num, int32_t
/// reg_val);
////__device__ __noinline__ int32_t nvbit_read_pred_reg(void);
////__device__ __noinline__ void nvbit_write_pred_reg(int32_t reg_val);
////__device__ __noinline__ int32_t nvbit_read_upred_reg(void);
////__device__ __noinline__ void nvbit_write_upred_reg(int32_t reg_val);
////#endif
////
/////*********************************************************************
//// *
//// *          NVBit control APIs  (provided by NVBit)
//// *
//// **********************************************************************/
////

////
/////* Set arguments at launch time, that will be loaded on input argument of
//// * the instrumentation function */
////void nvbit_set_at_launch(CUcontext ctx, CUfunction func, void* buf,
////                         uint32_t nbytes);
////
/////* Notify nvbit of a pthread used by the tool, this pthread will not
//// * trigger any call backs even if executing CUDA events of kernel launches.
//// * Multiple pthreads can be registered one after the other. */
////void nvbit_set_tool_pthread(pthread_t tool_pthread);
////void nvbit_unset_tool_pthread(pthread_t tool_pthread);
////
// NOLINTEND

#endif
