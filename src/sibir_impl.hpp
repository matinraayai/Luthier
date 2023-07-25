#ifndef SIBIR_IMPL_HPP
#define SIBIR_IMPL_HPP
#include "sibir_types.hpp"

namespace sibir::impl {
void init();
void finalize();
void hipStartupCallback(void *cb_data, sibir_api_phase_t phase, int api_id);
void hipApiCallback(void *cb_data, sibir_api_phase_t phase, int api_id);
void hsaApiCallback(hsa_api_args_t *cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id);
};// namespace sibir::impl

#endif
