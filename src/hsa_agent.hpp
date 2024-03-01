#ifndef HSA_AGENT_HPP
#define HSA_AGENT_HPP
#include "hsa_handle_type.hpp"
#include "hsa_intercept.hpp"
#include "hsa_isa.hpp"
#include <hsa/hsa.h>
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

class GpuAgent : public HandleType<hsa_agent_t> {
public:
  explicit GpuAgent(hsa_agent_t Agent) : HandleType<hsa_agent_t>(Agent){};

  llvm::Error getIsa(llvm::SmallVectorImpl<ISA> &IsaList) const;

  [[nodiscard]] llvm::Expected<hsa::ISA> getIsa() const;
};

} // namespace luthier::hsa

namespace std {

template <> struct hash<luthier::hsa::GpuAgent> {
  size_t operator()(const luthier::hsa::GpuAgent &obj) const {
    return hash<unsigned long>()(obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() < rhs.hsaHandle();
  }
};

template <> struct less_equal<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() <= rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

template <> struct not_equal_to<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() != rhs.hsaHandle();
  }
};

template <> struct greater<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() > rhs.hsaHandle();
  }
};

template <> struct greater_equal<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() >= rhs.hsaHandle();
  }
};

} // namespace std

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::GpuAgent> {
  static inline luthier::hsa::GpuAgent getEmptyKey() {
    return luthier::hsa::GpuAgent(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::GpuAgent getTombstoneKey() {
    return luthier::hsa::GpuAgent(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::GpuAgent &Agent) {
    return DenseMapInfo<decltype(hsa_agent_t::handle)>::getHashValue(
        Agent.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::GpuAgent &lhs,
                      const luthier::hsa::GpuAgent &rhs) {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace llvm


#endif