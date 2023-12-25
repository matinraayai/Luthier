#ifndef HSA_ISA_HPP
#define HSA_ISA_HPP
#include "hsa_handle_type.hpp"
#include "hsa_intercept.hpp"

namespace luthier::hsa {

class Isa : public HandleType<hsa_isa_t> {
 public:

    explicit Isa(hsa_isa_t isa): HandleType<hsa_isa_t>(isa) {};

    static Isa fromName(const char* isaName);

    std::string getName() const;

};

}


#endif