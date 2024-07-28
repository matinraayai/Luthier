#ifndef HIP_PLATFORM_HPP
#define HIP_PLATFORM_HPP
#include <llvm/ADT/DenseMap.h>

namespace luthier::hip {

/**
 * \brief in charge of caching useful information about the HIP runtime state
 */
class Platform {

public:
  Platform(const Platform &) = delete;
  Platform &operator=(const Platform &) = delete;

  static inline Platform &instance() {
    static Platform Instance;
    return Instance;
  }

private:
  Platform() = default;
  ~Platform() = default;


};

} // namespace luthier::hip

#endif