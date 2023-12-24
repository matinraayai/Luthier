#ifndef HSA_PRIMITIVE_HPP
#define HSA_PRIMITIVE_HPP
#include "hsa_intercept.hpp"
#include "hsa_type.hpp"

namespace luthier::hsa {

template<typename HT>
class HandleType : public Type<HT> {
 protected:
    explicit HandleType(HT primitive) : Type<HT>(primitive){};

 public:
    [[nodiscard]] uint64_t hsaHandle() const { return this->asHsaType().handle; }

};

}// namespace luthier::hsa

template<typename HT>
inline bool operator==(const luthier::hsa::HandleType<HT> &lhs, const luthier::hsa::HandleType<HT> &rhs) {
    return lhs.hsaHandle() == rhs.hsaHandle();
}

template<typename HT>
inline bool operator==(const luthier::hsa::HandleType<HT> &lhs, const HT &rhs) {
    return lhs.hsaHandle() == rhs.handle;
}

template<typename HT>
inline bool operator==(const HT &lhs, const luthier::hsa::HandleType<HT> &rhs) {
    return lhs.handle == rhs.hsaHandle();
}

#endif