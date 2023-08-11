#ifndef LUTHIER_IMPL_HPP
#define LUTHIER_IMPL_HPP
#include "luthier_types.hpp"

namespace luthier::impl {
void init();
void finalize();
void hipStartupCallback(void *cb_data, luthier_api_phase_t phase, int api_id);
void hipApiCallback(void *cb_data, luthier_api_phase_t phase, int api_id);
void hsaApiCallback(hsa_api_args_t *cb_data, luthier_api_phase_t phase, hsa_api_id_t api_id);
};// namespace luthier::impl

#endif
