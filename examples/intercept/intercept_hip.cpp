#include <fstream>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <luthier/luthier.h>
#include <string>

namespace luthier {

void luthier_at_hsa_event(hsa_api_evt_args_t *cb_data,
                          luthier_api_evt_phase_t phase,
                          hsa_api_evt_id_t api_id) {}

void luthier::atHsaApiTableLoad() {
  std::cout << "Calling luthier_enable_hip_op_callback" << std::endl;
  // luthier_enable_hip_op_callback(108);
  luthier_enable_all_hip_callbacks();
}

void luthier::atHsaApiTableUnload() {
  std::cout << "HIP Intercept Tool is terminating!" << std::endl;
}

void luthier_at_hip_event(void *args, luthier_api_evt_phase_t phase,
                          int hip_api_id) {
  // if (hip_api_id == 108) luthier_enable_hip_all_callback();
  fprintf(stdout, "<call to %d (%s)\t on %s> ", hip_api_id,
          hip_api_name(hip_api_id),
          phase == LUTHIER_API_EVT_PHASE_ENTER ? "entry" : "exit");
  fprintf(stdout, "\n");
  fflush(stdout);
}
} // namespace luthier
