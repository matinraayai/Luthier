#include <sibir.h>
#include <iostream>

void sibir_at_init() {
    std::cout << "Hi from Sibir!" << std::endl;
}


void sibir_at_term() {
    std::cout << "Bye from Sibir!" << std::endl;
}

void sibir_at_hip_event(uint32_t cid, const hip_api_data_t* callback_data) {
    fprintf(stdout, "<%s id(%u)\tcorrelation_id(%lu) %s> ",
            roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0),
            cid,
            callback_data->correlation_id,
            (callback_data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");
    if (callback_data->phase == ACTIVITY_API_PHASE_ENTER) {
        switch (cid) {
            case HIP_API_ID_hipMemcpy:
                fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
                        callback_data->args.hipMemcpy.dst,
                        callback_data->args.hipMemcpy.src,
                        (uint32_t)(callback_data->args.hipMemcpy.sizeBytes),
                        (uint32_t)(callback_data->args.hipMemcpy.kind));
                break;
            case HIP_API_ID_hipMalloc:
                fprintf(stdout, "ptr(%p) size(0x%x)",
                        callback_data->args.hipMalloc.ptr,
                        (uint32_t)(callback_data->args.hipMalloc.size));
                break;
            case HIP_API_ID_hipFree:
                fprintf(stdout, "ptr(%p)", callback_data->args.hipFree.ptr);
                break;
            case HIP_API_ID_hipModuleLaunchKernel:
                fprintf(stdout, "kernel(\"%s\") stream(%p)",
                        hipKernelNameRef(callback_data->args.hipModuleLaunchKernel.f),
                        callback_data->args.hipModuleLaunchKernel.stream);
                break;
            default:
                break;
        }
    } else {
        switch (cid) {
            case HIP_API_ID_hipMalloc:
                fprintf(stdout, "*ptr(0x%p)",
                        *(callback_data->args.hipMalloc.ptr));
                break;
            default:
                break;
        }
    }
    fprintf(stdout, "\n"); fflush(stdout);
}