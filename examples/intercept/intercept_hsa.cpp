#include <luthier.h>
#include <fstream>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>
#include <string>

void luthier_at_init() {
    std::cout << "Calling luthier_enable_hsa_op_callback" << std::endl;
    luthier_enable_hsa_op_callback(39);
    luthier_enable_hsa_op_callback(139);
}


void luthier_at_term() {
    std::cout << "HSA Intercept Tool is terminating!" << std::endl;
}

void luthier_at_hip_event(void* args, luthier_api_phase_t phase, int hip_api_id) {
    //hi
}


void luthier_at_hsa_event(hsa_api_args_t* cb_data, luthier_api_phase_t phase, hsa_api_id_t api_id) {
    if (api_id == 39) luthier_disable_hsa_op_callback(39);
    fprintf(stdout, "<call to (%d)\t on %s> ",
            api_id,
            phase == LUTHIER_API_PHASE_ENTER ? "entry" : "exit"
            );
    fprintf(stdout, "\n"); fflush(stdout);
}
