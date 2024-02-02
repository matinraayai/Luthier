#ifndef ERROR_HPP
#define ERROR_HPP

#include <amd_comgr/amd_comgr.h>
#include <hsa/hsa.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/FormatVariadicDetails.h>

// Workaround for GCC or other compilers that don't have this macro built-in
// Source: https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
#if !defined(__FILE_NAME__)

#define __FILE_NAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#endif

#define LUTHIER_ROCM_LIB_ERROR_MSG                                         \
    "{0:s}, line {1:d}: Luthier {2:s} call to function {3:s} failed with " \
    "Error Code {4:d}.\n\nError code reason according to the library: {5:s}\n."

#define LUTHIER_ROCM_LIB_API_CHECK(LIB_NAME, LIB_ERROR_TYPE)                                                  \
    inline void check_##LIB_NAME##_error(LIB_ERROR_TYPE err, const char* callName, const char* fileName,      \
                                         const int line, const LIB_ERROR_TYPE expected) {                     \
        if (err != expected) {                                                                                \
            const char* errMsg = "Unknown Error";                                                             \
            LIB_NAME##_status_string(err, &errMsg);                                                           \
            std::string what = llvm::formatv(LUTHIER_ROCM_LIB_ERROR_MSG, fileName, line, #LIB_NAME, callName, \
                                             static_cast<int>(err), errMsg)                                   \
                                   .str();                                                                    \
            llvm::report_fatal_error(what.c_str());                                                           \
        }                                                                                                     \
    }

LUTHIER_ROCM_LIB_API_CHECK(hsa, hsa_status_t)

LUTHIER_ROCM_LIB_API_CHECK(amd_comgr, amd_comgr_status_t)

#undef LUTHIER_ROCM_LIB_API_CHECK

//inline void checkHipError(hipError_t err, const char* callName, const char* fileName, const int line) {
//    if (err != hipSuccess) {
//        const char* errMsg =
//            luthier::HipInterceptor::instance().getHipFunction<const char* (*) (hipError_t)>("hipGetErrorString")(err);
//        std::string what =
//            llvm::formatv(LUTHIER_ROCM_LIB_ERROR_MSG, fileName, line, "hip", callName, static_cast<int>(err), errMsg);
//        llvm::report_fatal_error(what.c_str());
//    }
//}

#undef LUTHIER_ROCM_LIB_ERROR_MSG

#define LUTHIER_AMD_COMGR_CHECK(call) \
    check_amd_comgr_error(call, #call, __FILE_NAME__, __LINE__, AMD_COMGR_STATUS_SUCCESS)

#define LUTHIER_HSA_CHECK(call) check_hsa_error(call, #call, __FILE_NAME__, __LINE__, HSA_STATUS_SUCCESS)

#define LUTHIER_CHECK_WITH_MSG(pred, msg)                                                                   \
    if (!pred) {                                                                                            \
        llvm::report_fatal_error(                                                                           \
            llvm::formatv("Luthier check on file {0}, line {1} failed: {2}.", __FILE_NAME__, __LINE__, msg) \
                .str()                                                                                      \
                .c_str());                                                                                  \
    }

#define LUTHIER_CHECK(pred)                                                                                           \
    if (!pred) {                                                                                                      \
        llvm::report_fatal_error(llvm::formatv("Luthier check for expression {0} on file {1}, line {2} failed.", \
                                               #pred, __FILE_NAME__, __LINE__)                                   \
                                     .str()                                                                           \
                                     .c_str());                                                                       \
    }

//#define LUTHIER_HIP_CHECK(call) check_hip_error(call, #call, __FILE_NAME__, __LINE__)

#endif
