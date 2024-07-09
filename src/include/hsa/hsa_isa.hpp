#ifndef HSA_ISA_HPP
#define HSA_ISA_HPP
#include "hsa_handle_type.hpp"
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

namespace luthier::hsa {

class ISA : public HandleType<hsa_isa_t> {
public:
  explicit ISA(hsa_isa_t Isa) : HandleType<hsa_isa_t>(Isa){};

  static llvm::Expected<ISA> fromName(const char *IsaName);

  static llvm::Expected<ISA> fromLLVM(const llvm::Triple &TT,
                                      llvm::StringRef CPU,
                                      const llvm::SubtargetFeatures &Features);

  [[nodiscard]] llvm::Expected<std::string> getName() const;

  [[nodiscard]] llvm::Expected<std::string> getArchitecture() const;

  [[nodiscard]] llvm::Expected<std::string> getVendor() const;

  [[nodiscard]] llvm::Expected<std::string> getOS() const;

  [[nodiscard]] llvm::Expected<std::string> getEnvironment() const;

  [[nodiscard]] llvm::Expected<std::string> getProcessor() const;

  [[nodiscard]] llvm::Expected<bool> isXNACSupported() const;

  [[nodiscard]] llvm::Expected<bool> isSRAMECCSupported() const;

  [[nodiscard]] llvm::Expected<llvm::Triple> getTargetTriple() const;

  [[nodiscard]] llvm::Expected<llvm::SubtargetFeatures>
  getSubTargetFeatures() const;
};

} // namespace luthier::hsa

namespace std {

template <> struct hash<luthier::hsa::ISA> {
  size_t operator()(const luthier::hsa::ISA &Obj) const {
    return hash<unsigned long>()(Obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() < Rhs.hsaHandle();
  }
};

template <> struct less_equal<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() <= Rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

template <> struct not_equal_to<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() != Rhs.hsaHandle();
  }
};

template <> struct greater<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() > Rhs.hsaHandle();
  }
};

template <> struct greater_equal<luthier::hsa::ISA> {
  bool operator()(const luthier::hsa::ISA &Lhs,
                  const luthier::hsa::ISA &Rhs) const {
    return Lhs.hsaHandle() >= Rhs.hsaHandle();
  }
};

} // namespace std

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::ISA> {
  static inline luthier::hsa::ISA getEmptyKey() {
    return luthier::hsa::ISA(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::ISA getTombstoneKey() {
    return luthier::hsa::ISA(
        {DenseMapInfo<decltype(hsa_isa_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::ISA &ISA) {
    return DenseMapInfo<decltype(hsa_isa_t::handle)>::getHashValue(
        ISA.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::ISA &Lhs,
                      const luthier::hsa::ISA &Rhs) {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace llvm

#endif
