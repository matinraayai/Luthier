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
                   luthier::hsa::ApiEvtID ApiID) {
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