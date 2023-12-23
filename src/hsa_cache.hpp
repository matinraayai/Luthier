#ifndef HSA_CACHE_HPP
#define HSA_CACHE_HPP
#include <hsa/hsa.h>
#include <string>
#include "hsa_primitive.hpp"

namespace luthier::hsa {

class Cache : public HandleType<hsa_cache_t> {
 private:
    explicit Cache(hsa_cache_t cache) : HandleType<hsa_cache_t>(cache) {};

 public:
    std::string getName();

    uint8_t getLevel();

    unsigned int getSize();
};

}

#endif