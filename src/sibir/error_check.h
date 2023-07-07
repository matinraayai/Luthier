#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include <amd_comgr.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>

#define SIBIR_ROCM_LIB_API_CHECK(LIB_NAME, LIB_ERROR_TYPE, LIB_ERROR_SUCCESS_TYPE) \
        inline void check_##LIB_NAME##_error(LIB_ERROR_TYPE err, const char* callName) { \
           if (err != LIB_ERROR_SUCCESS_TYPE) { \
                const char* errMsg = "Unknown Error"; \
                LIB_NAME##_status_string(err, &errMsg); \
                fprintf(stderr, "%s, line %d: Sibir %s call to function %s failed!\n", __FILE_NAME__, __LINE__, #LIB_NAME, callName); \
                fprintf(stderr, "Error Code: %d.\n", err); \
                fprintf(stderr, "More info: %s\n", errMsg); \
           } \
        }

SIBIR_ROCM_LIB_API_CHECK(hsa, hsa_status_t, HSA_STATUS_SUCCESS)

SIBIR_ROCM_LIB_API_CHECK(amd_comgr, amd_comgr_status_t, AMD_COMGR_STATUS_SUCCESS)

#undef SIBIR_ROCM_LIB_API_CHECK

#define SIBIR_AMD_COMGR_CHECK(call) check_amd_comgr_error(call, #call)

#define SIBIR_HSA_CHECK(call) check_hsa_error(call, #call)


#endif
