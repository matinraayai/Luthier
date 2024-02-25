#ifndef HSA_ISA_HPP
#define HSA_ISA_HPP
#include "hsa_handle_type.hpp"
#include <llvm/Support/Error.h>

namespace luthier::hsa {

class ISA : public HandleType<hsa_isa_t> {
public:
  explicit ISA(hsa_isa_t Isa) : HandleType<hsa_isa_t>(Isa){};

  static llvm::Expected<ISA> fromName(const char *IsaName);

  [[nodiscard]] llvm::Expected<std::string> getName() const;

  [[nodiscard]] llvm::Expected<std::string> getArchitecture() const;

  [[nodiscard]] llvm::Expected<std::string> getVendor() const;

  [[nodiscard]] llvm::Expected<std::string> getOS() const;

  [[nodiscard]] llvm::Expected<std::string> getEnvironment() const;

  [[nodiscard]] llvm::Expected<std::string> getProcessor() const;

  [[nodiscard]] llvm::Expected<bool> isXNACSupported() const;

  [[nodiscard]] llvm::Expected<bool> isSRAMECCSupported() const;

  [[nodiscard]] llvm::Expected<std::string> getLLVMTarget() const;

  [[nodiscard]] llvm::Expected<std::string> getLLVMTargetTriple() const;

  [[nodiscard]] llvm::Expected<std::string> getFeatureString() const;
};

} // namespace luthier::hsa

namespace std {

template <> struct hash<luthier::hsa::ISA> {
  size_t operator()(const luthier::hsa::ISA &obj) const {
    return hash<unsigned long>()(obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &lhs,
                  const luthier::hsa::ISA &rhs) const {
    return lhs.hsaHandle() < rhs.hsaHandle();
  }
};

template <> struct less_equal<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &lhs,
                  const luthier::hsa::ISA &rhs) const {
    return lhs.hsaHandle() <= rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

template <> struct not_equal_to<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &lhs,
                  const luthier::hsa::ISA &rhs) const {
    return lhs.hsaHandle() != rhs.hsaHandle();
  }
};

template <> struct greater<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &lhs,
                  const luthier::hsa::ISA &rhs) const {
    return lhs.hsaHandle() > rhs.hsaHandle();
  }
};

template <> struct greater_equal<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &lhs,
                  const luthier::hsa::ISA &rhs) const {
    return lhs.hsaHandle() >= rhs.hsaHandle();
  }
};

} // namespace std

#endif