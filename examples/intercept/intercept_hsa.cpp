#include <luthier.h>
#include <fstream>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <string>

void luthier_at_init() {
    std::cout << "Calling luthier_enable_hsa_op_callback" << std::endl;
    //luthier_enable_hsa_op_callback(HSA_API_ID_hsa_signal_load_relaxed);
    //luthier_enable_hsa_op_callback(HSA_API_ID_hsa_signal_create);
    //luthier_enable_hsa_op_callback(HSA_API_ID_hsa_amd_memory_pool_free);
    //luthier_disable_hsa_op_callback(HSA_EVT_ID_hsa_queue_packet_submit);
    luthier_enable_all_hsa_callbacks();
}


void luthier_at_term() {
    std::cout << "HSA Intercept Tool is terminating!" << std::endl;
}

void luthier_at_hip_event(void* args, luthier_api_evt_phase_t phase, int hip_api_id) {
    //pass
}


void luthier_at_hsa_event(hsa_api_evt_args_t* cb_data, luthier_api_evt_phase_t phase, hsa_api_evt_id_t api_id) {
    //if (api_id == HSA_API_ID_hsa_signal_load_relaxed) luthier_disable_hsa_op_callback(HSA_API_ID_hsa_signal_load_relaxed);
    //if (api_id == HSA_API_ID_hsa_signal_create) luthier_disable_hsa_all_callback();
    fprintf(stdout, "<call to (%d)\t on %s> ",
            api_id,
            phase == LUTHIER_API_EVT_PHASE_ENTER ? "entry" : "exit"
            );
    fprintf(stdout, "\n"); fflush(stdout);
}
