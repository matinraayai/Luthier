#include <luthier.h>
#include <fstream>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <string>

void luthier_at_hsa_event(hsa_api_evt_args_t* cb_data, luthier_api_evt_phase_t phase, hsa_api_evt_id_t api_id) {}

void luthier_at_init() {
    std::cout << "Calling luthier_enable_hip_op_callback" << std::endl;
    //luthier_enable_hip_op_callback(108);
    luthier_enable_all_hip_callbacks();
}


#include <iostream>
#include <luthier/luthier.h>
#include <string>

namespace luthier {

void hsa::atHsaApiTableLoad() {
  std::cout << "Calling enableHsaOpCallback" << std::endl;
  // luthier_enable_hsa_op_callback(HSA_API_ID_hsa_signal_load_relaxed);
  // luthier_enable_hsa_op_callback(HSA_API_ID_hsa_signal_create);
  // luthier_enable_hsa_op_callback(HSA_API_ID_hsa_amd_memory_pool_free);
  // luthier_disable_hsa_op_callback(HSA_EVT_ID_hsa_queue_packet_submit);
  hsa::enableAllHsaCallbacks();
}

void hsa::atHsaApiTableUnload() {
  std::cout << "HSA Intercept Tool is terminating!" << std::endl;
}

// Below should not print if callbacks are disabled
void hsa::atHsaEvt(luthier::hsa::ApiEvtArgs *CBData, luthier::ApiEvtPhase Phase,
                   luthier::hsa::ApiEvtID ApiID){

  // Test to see if disabling callbacks works
  int *pointer;
  {
    DisableInterceptScope mtx;
    // want to call hipMalloc and check all the hip functions for their error codes and
    // make sure they equate to success
    // look at <hip/hip_runtime_api.h>
    hipMallocManaged(&pointer, 20); // TODO: outputs status code make sure it is equal to hip_success
    hipMemset(pointer, 5, 4);

    // TODO: Also print out thread number
    std::cout << "Content of pointer: " << *pointer << std::endl;
    // TODO:  If this works properly, the above will print first and below second
    // TODO:  test by appending to preload

    hipFree(pointer);
  }
  // if (api_id == HSA_API_ID_hsa_signal_load_relaxed)
  // luthier_disable_hsa_op_callback(HSA_API_ID_hsa_signal_load_relaxed); if
  // (api_id == HSA_API_ID_hsa_signal_create)
  // luthier_disable_hsa_all_callback();
  fprintf(stdout, "<call to (%d)\t on %s> ", ApiID,
          Phase == luthier::API_EVT_PHASE_ENTER ? "entry" : "exit");
  fprintf(stdout, "\n");
  fflush(stdout);
}

} // namespace luthier