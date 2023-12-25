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

#define LUTHIER_ROCM_LIB_ERROR_MSG "{:s}, line {:d}: Luthier {:s} call to function {:s} failed with "\
                                 "Error Code {:d}.\n\nError code reason according to the library: {:s}\n."

#define LUTHIER_ROCM_LIB_API_CHECK(LIB_NAME, LIB_ERROR_TYPE, LIB_ERROR_SUCCESS_TYPE) \
        inline void check_##LIB_NAME##_error(LIB_ERROR_TYPE err, const char* callName, const char* fileName, const int line) { \
            if (err != LIB_ERROR_SUCCESS_TYPE) { \
                const char* errMsg = "Unknown Error"; \
                LIB_NAME##_status_string(err, &errMsg);                            \
                std::string what = fmt::format(fmt::runtime(LUTHIER_ROCM_LIB_ERROR_MSG),\
                                           fileName, line, #LIB_NAME, callName, static_cast<int>(err), errMsg);                                                             \
                throw std::runtime_error(what);\
            }\
        }

LUTHIER_ROCM_LIB_API_CHECK(hsa, hsa_status_t, HSA_STATUS_SUCCESS)

LUTHIER_ROCM_LIB_API_CHECK(amd_comgr, amd_comgr_status_t, AMD_COMGR_STATUS_SUCCESS)

#undef LUTHIER_ROCM_LIB_API_CHECK

inline void check_hip_error(hipError_t err, const char* callName, const char* fileName, const int line) {
    if (err != hipSuccess) {
        const char* errMsg = hipGetErrorString(err);
        std::string what = fmt::format(fmt::runtime(LUTHIER_ROCM_LIB_ERROR_MSG),
                                       fileName, line, "hip", callName, static_cast<int>(err), errMsg);
        throw std::runtime_error(what);
    }
}

#undef LUTHIER_ROCM_LIB_ERROR_MSG

#define LUTHIER_AMD_COMGR_CHECK(call) check_amd_comgr_error(call, #call, __FILE_NAME__, __LINE__)

#define LUTHIER_HSA_CHECK(call) check_hsa_error(call, #call, __FILE_NAME__, __LINE__)

#define LUTHIER_HIP_CHECK(call) check_hip_error(call, #call, __FILE_NAME__, __LINE__)

#endif
