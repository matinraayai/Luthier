#ifndef SIBIR_H
#define SIBIR_H
#include "error.h"
#include "instr.hpp"
#include "sibir_types.hpp"
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>

//extern "C" {

//
///*********************************************************************
// *
// *                  NVBit tool callbacks
// *     (implement these functions to get a callback from NVBit)
// *
// **********************************************************************/
///* This function is called as soon as the program starts, no GPU calls
// * should be made at this moment */
void sibir_at_init();
//
///* This function is called just before the program terminates, no GPU calls
// * should be made at this moment */
void sibir_at_term();
//
////
/////* This is the function called every beginning (is_exit = 0) and
//// * end (is_exit = 1) of a CUDA driver call.
//// * cbid identifies the CUDA driver call as specified by the enum
//// * nvbit_api_cuda_t, see tools_cuda_api_meta.h for the list of cbid.
//// * Name is its the driver call name.
//// * params is pointer to* one of the structures defined in the
//// * generated_cuda_meta.h.
//// * Params must be casted to the correct struct based on the cbid.
//// * For instance if cbid = cuMemcpyDtoH_v2 then params must be casted to
//// * (cuMemcpyDtoH_v2_params *)
//// * */
void sibir_at_hip_event(void* args, sibir_api_phase_t phase, int hip_api_id);


void sibir_at_hsa_event(hsa_api_args_t* cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id);

/**
 * Returns the original HSA API table to avoid re-instrumentation of HSA functions.
 * @return saved HSA API Table
 */
const HsaApiTable* sibir_get_hsa_table();

const hsa_ven_amd_loader_1_03_pfn_s* sibir_get_hsa_ven_amd_loader();


void* sibir_get_hip_function(const char* funcName);



std::vector<sibir::Instr> sibir_disassemble_kernel_object(uint64_t kernel_object);


#define SIBIR_EXPORT_FUNC(f)               \
    __global__ void __sibir_wrap__##f() {  \
        void (*pfun)() = (void (*)())f;    \
        if (pfun == (void (*)())1) pfun(); \
    }

//static inline const char* sibir_hip_api_name(uint32_t hip_api_id) {
//    if (hip_api_id < 1000)
//        return hip_api_name(hip_api_id);
//    else
//        return
//}

//void print_instructions(const std::vector<Inst>& isa);
////
/////*********************************************************************
//// *
//// *          NVBit inspection APIs  (provided by NVBit)
//// *
//// **********************************************************************/
/////* Get vector of related functions */
////std::vector<CUfunction> nvbit_get_related_functions(CUcontext ctx,
////                                                    CUfunction func);
////
/////* Get vector of instructions composing the CUfunction */
////const std::vector<Instr*>& nvbit_get_instrs(CUcontext ctx, CUfunction func);
////
/////* Get control flow graph (CFG) */
////const CFG_t& nvbit_get_CFG(CUcontext ctx, CUfunction func);
////
/////* Allows to get a function name from its CUfunction */
////const char* nvbit_get_func_name(CUcontext ctx, CUfunction f,
////                                bool mangled = false);
////
/////* Get line information for a particular instruction offset if available,
//// * binary must be compiled with --generate-line-info   (-lineinfo) */
////bool nvbit_get_line_info(CUcontext cuctx, CUfunction cufunc, uint32_t offset,
////                         char** file_name, char** dir_name, uint32_t* line);
////
/////* Get the SM family */
////uint32_t nvbit_get_sm_family(CUcontext cuctx);
////
/////* Allows to get PC address of the function */
////uint64_t nvbit_get_func_addr(CUfunction func);
////
/////* Returns true if function is a kernel (i.e. __global__ ) */
////bool nvbit_is_func_kernel(CUcontext ctx, CUfunction func);
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
/////* Enumeration used by nvbit_insert_call to specify where we want to insert
//// * the device function for a given Instr, if before or after */
////typedef enum { IPOINT_BEFORE, IPOINT_AFTER } ipoint_t;
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

/**
 *
 * @param instr
 * @param dev_func_name
 * @param point
 */
void sibir_insert_call(sibir::Instr* instr, const char* dev_func_name, sibir_ipoint_t point);

/////* Add int32_t argument to last injected call, value of the predicate for this
//// * instruction */
////void nvbit_add_call_arg_pred_val(const Instr* instr,
////                                 bool is_variadic_arg = false);
////
/////* Add int32_t argument to last injected call, value of the entire predicate
//// * register for this thread */
////void nvbit_add_call_arg_pred_reg(const Instr* instr,
////                                 bool is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, constant 32-bit value */
////void nvbit_add_call_arg_const_val32(const Instr* instr, uint32_t val,
////                                    bool is_variadic_arg = false);
////
/////* Add uint64_t argument to last injected call, constant 64-bit value */
////void nvbit_add_call_arg_const_val64(const Instr* instr, uint64_t val,
////                                    bool is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, content of the register reg_num
//// */
////void nvbit_add_call_arg_reg_val(const Instr* instr, int reg_num,
////                                bool is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, content of the
//// * uniform register reg_num */
////void nvbit_add_call_arg_ureg_val(const Instr* instr, int reg_num,
////                                 bool is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, 32-bit at launch value at offset
//// * "offset", set at launch time with nvbit_set_at_launch */
////void nvbit_add_call_arg_launch_val32(const Instr* instr, int offset,
////                                     bool is_variadic_arg = false);
////
/////* Add uint64_t argument to last injected call, 64-bit at launch value at offset
//// * "offset", set at launch time with nvbit_set_at_launch */
////void nvbit_add_call_arg_launch_val64(const Instr* instr, int offset,
////                                     bool is_variadic_arg = false);
////
/////* Add uint32_t argument to last injected call, constant bank value at
//// * c[bankid][bankoffset] */
////void nvbit_add_call_arg_cbank_val(const Instr* instr, int bankid,
////                                  int bankoffset, bool is_variadic_arg = false);
////
/////* The 64-bit memory reference address accessed by this instruction
////  Typically memory instructions have only 1 MREF so in general id = 0 */
////void nvbit_add_call_arg_mref_addr64(const Instr* instr, int id = 0,
////                                    bool is_variadic_arg = false);
////
/////* Remove the original instruction */
////void nvbit_remove_orig(const Instr* instr);
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
////__device__ __noinline__ int32_t nvbit_read_reg(uint64_t reg_num);
////__device__ __noinline__ void nvbit_write_reg(uint64_t reg_num, int32_t reg_val);
////__device__ __noinline__ int32_t nvbit_read_ureg(uint64_t reg_num);
////__device__ __noinline__ void nvbit_write_ureg(uint64_t reg_num, int32_t reg_val);
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
/////* Run instrumented on original function (and its related functions)
//// * based on flag value */
void sibir_enable_instrumented(hsa_kernel_dispatch_packet_t* dispatch_packet, sibir_address_t func, bool flag);
//                               bool apply_to_related = true);
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

#endif
////
/////*********************************************************************
//// *
//// *              Macros to read environment variables
//// *
//// **********************************************************************/
////
/**

////#define PRINT_VAR(env_var, help, var)                                      \
////    std::cout << std::setw(20) << env_var << " = " << var << " - " << help \
////              << std::endl;
////
////#define GET_VAR_INT(var, env_var, def, help) \
////    if (getenv(env_var)) {                   \
////        var = atoi(getenv(env_var));         \
////    } else {                                 \
////        var = def;                           \
////    }                                        \
////    PRINT_VAR(env_var, help, var)
////
////#define GET_VAR_LONG(var, env_var, def, help) \
////    if (getenv(env_var)) {                    \
////        var = atol(getenv(env_var));          \
////    } else {                                  \
////        var = def;                            \
////    }                                         \
////    PRINT_VAR(env_var, help, var)
////
////#define GET_VAR_STR(var, env_var, help) \
////    if (getenv(env_var)) {              \
////        std::string s(getenv(env_var)); \
////        var = s;                        \
////    }                                   \
////    PRINT_VAR(env_var, help, var)
*/