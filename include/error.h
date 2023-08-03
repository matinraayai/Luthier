#ifndef ERROR_H
#define ERROR_H

#include <amd_comgr/amd_comgr.h>
#include <hip/hip_runtime_api.h>
#include <hsakmt/hsakmt.h>
#include <hsa/hsa.h>
#include <fmt/core.h>
#include <stdexcept>

// Workaround for GCC or other compilers that don't have this macro built-in
// Source: https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
#if !defined(__FILE_NAME__)

#define __FILE_NAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#endif

#define SIBIR_ROCM_LIB_ERROR_MSG "{:s}, line {:d}: Sibir {:s} call to function {:s} failed with "\
                                 "Error Code {:d}.\n\nError code reason according to the library: {:s}."

#define SIBIR_ROCM_LIB_API_CHECK(LIB_NAME, LIB_ERROR_TYPE, LIB_ERROR_SUCCESS_TYPE) \
        inline void check_##LIB_NAME##_error(LIB_ERROR_TYPE err, const char* callName, const char* fileName, const int line) { \
            if (err != LIB_ERROR_SUCCESS_TYPE) { \
                const char* errMsg = "Unknown Error"; \
                LIB_NAME##_status_string(err, &errMsg);                            \
                std::string what = fmt::format(fmt::runtime(SIBIR_ROCM_LIB_ERROR_MSG),\
                                           fileName, line, #LIB_NAME, callName, static_cast<int>(err), errMsg);                                                             \
                throw std::runtime_error(what);\
            }\
        }

SIBIR_ROCM_LIB_API_CHECK(hsa, hsa_status_t, HSA_STATUS_SUCCESS)

SIBIR_ROCM_LIB_API_CHECK(amd_comgr, amd_comgr_status_t, AMD_COMGR_STATUS_SUCCESS)

#undef SIBIR_ROCM_LIB_API_CHECK

inline void check_hip_error(hipError_t err, const char* callName, const char* fileName, const int line) {
    if (err != hipSuccess) {
        const char* errMsg = hipGetErrorString(err);
        std::string what = fmt::format(fmt::runtime(SIBIR_ROCM_LIB_ERROR_MSG),
                                       fileName, line, "hip", callName, static_cast<int>(err), errMsg);
        throw std::runtime_error(what);
    }
}


inline void check_hsakmt_error(HSAKMT_STATUS err, const char* callName, const char* fileName, const int line) {
    std::string errMsg;
    switch(err) {
        case HSAKMT_STATUS_SUCCESS: break;
        case HSAKMT_STATUS_ERROR:
            errMsg = "General error return otherwise not specified";
            break;
        case HSAKMT_STATUS_DRIVER_MISMATCH:
            errMsg = "User mode component is not compatible with kernel HSA driver";
            break;
        case HSAKMT_STATUS_INVALID_PARAMETER:
            errMsg = "KFD identifies input parameters invalid";
            break;
        case HSAKMT_STATUS_INVALID_HANDLE:
            errMsg = "KFD identifies handle parameter invalid";
            break;
        case HSAKMT_STATUS_INVALID_NODE_UNIT:
            errMsg = "KFD identifies node or unit parameter invalid";
            break;
        case HSAKMT_STATUS_NO_MEMORY:
            errMsg = "No memory available (when allocating queues or memory)";
            break;
        case HSAKMT_STATUS_BUFFER_TOO_SMALL:
            errMsg = "A buffer needed to handle a request is too small";
            break;
        case HSAKMT_STATUS_NOT_IMPLEMENTED:
            errMsg = "KFD function is not implemented for this set of parameters";
            break;
        case HSAKMT_STATUS_NOT_SUPPORTED:
            errMsg = "KFD function is not supported on this node";
            break;
        case HSAKMT_STATUS_UNAVAILABLE:
            errMsg = "KFD function is not available currently on this node (but may be at a later time)";
            break;
        case HSAKMT_STATUS_OUT_OF_RESOURCES:
            errMsg = "KFD function request exceeds the resources currently available";
            break;
        case HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED:
            errMsg = "KFD driver path not opened";
            break;
        case HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR:
            errMsg = "user-kernel mode communication failure";
            break;
        case HSAKMT_STATUS_KERNEL_ALREADY_OPENED:
            errMsg = "KFD driver path already opened";
            break;
        case HSAKMT_STATUS_HSAMMU_UNAVAILABLE:
            errMsg = "ATS/PRI 1.1 (Address Translation Services) not available (IOMMU driver not installed or not-available)";
            break;
        case HSAKMT_STATUS_WAIT_FAILURE:
            errMsg = "The wait operation failed";
            break;
        case HSAKMT_STATUS_WAIT_TIMEOUT:
            errMsg = "The wait operation timed out";
            break;
        case HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED:
            errMsg = "Memory buffer already registered";
            break;
        case HSAKMT_STATUS_MEMORY_NOT_REGISTERED:
            errMsg = "Memory buffer not registered";
            break;
        case HSAKMT_STATUS_MEMORY_ALIGNMENT:
            errMsg = "Memory parameter not aligned";
            break;
    }
    if (err != HSAKMT_STATUS_SUCCESS) {
        std::string what = fmt::format(fmt::runtime(SIBIR_ROCM_LIB_ERROR_MSG),
                                       fileName, line, callName, static_cast<int>(err), errMsg);
        throw std::runtime_error(what);
    }
}

#undef SIBIR_ROCM_LIB_ERROR_MSG

#define SIBIR_AMD_COMGR_CHECK(call) check_amd_comgr_error(call, #call, __FILE_NAME__, __LINE__)

#define SIBIR_HSA_CHECK(call) check_hsa_error(call, #call, __FILE_NAME__, __LINE__)

#define SIBIR_HIP_CHECK(call) check_hip_error(call, #call, __FILE_NAME__, __LINE__)

#define SIBIR_HSAKMT_CHECK(call) check_hsakmt_error(call, #call, __FILE_NAME__, __LINE__)

#endif
