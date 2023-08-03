#ifndef ERROR_CHECK_HPP
#define ERROR_CHECK_HPP

#include <amd_comgr/amd_comgr.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <fmt/core.h>
#include <stdexcept>


#define SIBIR_ROCM_LIB_API_CHECK(LIB_NAME, LIB_ERROR_TYPE, LIB_ERROR_SUCCESS_TYPE) \
        inline void check_##LIB_NAME##_error(LIB_ERROR_TYPE err, const char* callName, const char* fileName, const int line) { \
            if (err != LIB_ERROR_SUCCESS_TYPE) { \
                const char* errMsg = "Unknown Error"; \
                LIB_NAME##_status_string(err, &errMsg);                            \
                std::string what = fmt::format(fmt::runtime("{:s}, line {:d}: Sibir {:s} call to function {:s} failed; Error Code: {:d}.\nError Details: {:s}."),\
                                           fileName, line, #LIB_NAME, callName, static_cast<int>(err), errMsg);                                                             \
                throw std::runtime_error(what);\
            }\
        }

SIBIR_ROCM_LIB_API_CHECK(hsa, hsa_status_t, HSA_STATUS_SUCCESS)

SIBIR_ROCM_LIB_API_CHECK(amd_comgr, amd_comgr_status_t, AMD_COMGR_STATUS_SUCCESS)

#undef SIBIR_ROCM_LIB_API_CHECK

#define SIBIR_AMD_COMGR_CHECK(call) check_amd_comgr_error(call, #call, __FILE__, __LINE__)

#define SIBIR_HSA_CHECK(call) check_hsa_error(call, #call, __FILE__, __LINE__)


#endif
