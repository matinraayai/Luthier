#ifndef HSA_HANDLE_TYPE_HPP
#define HSA_HANDLE_TYPE_HPP
#include "hsa_type.hpp"

namespace luthier::hsa {

template <typename HT> class HandleType : public Type<HT> {
protected:
  explicit HandleType(HT Primitive) : Type<HT>(Primitive){};

public:
  [[nodiscard]] virtual uint64_t hsaHandle() const {
    return this->asHsaType().handle;
  }

  HandleType(const HandleType &Type) : HandleType(Type.asHsaType()){};

  HandleType &operator=(const HandleType &Other) {
    Type<HT>::operator=(Other);
    return *this;
  }

  HandleType &operator=(HandleType &&Other) noexcept {
    Type<HT>::operator=(Other);
    return *this;
  }
};

} // namespace luthier::hsa

template <typename HT>
inline bool operator==(const luthier::hsa::HandleType<HT> &Lhs,
                       const luthier::hsa::HandleType<HT> &Rhs) {
  return Lhs.hsaHandle() == Rhs.hsaHandle();
}

template <typename HT>
inline bool operator==(const luthier::hsa::HandleType<HT> &Lhs, const HT &Rhs) {
  return Lhs.hsaHandle() == Rhs.handle;
}

template <typename HT>
inline bool operator==(const HT &Lhs, const luthier::hsa::HandleType<HT> &Rhs) {
  return Lhs.handle == Rhs.hsaHandle();
}

#endif