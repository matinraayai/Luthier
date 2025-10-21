//===-- Bits.h --------------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Copyright 2025 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KLEE_BITS_H
#define LUTHIER_KLEE_BITS_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <llvm/Support/DataTypes.h>
#include <type_traits>

namespace luthier::klee {
namespace bits32 {
/// \pre(0 <= N <= 32)
/// \post(retval = max([truncateToNBits(i,N) for i in naturals()]))
inline unsigned maxValueOfNBits(unsigned N) {
  assert(N <= 32);
  if (N == 0)
    return 0;
  return (UINT32_C(-1)) >> (32 - N);
}

/// \pre(0 < N <= 32)
inline unsigned truncateToNBits(unsigned X, unsigned N) {
  assert(N > 0 && N <= 32);
  return X & ((UINT32_C(-1)) >> (32 - N));
}

inline unsigned withoutRightmostBit(unsigned X) { return X & (X - 1); }

inline unsigned isolateRightmostBit(unsigned X) { return X & -X; }

inline unsigned isPowerOfTwo(unsigned X) {
  if (X == 0)
    return 0;
  return !(X & (X - 1));
}

/// \pre(withoutRightmostBit(x) == 0)
/// \post((1 << retval) == x)
inline unsigned indexOfSingleBit(unsigned X) {
  assert(withoutRightmostBit(X) == 0);
  unsigned Res = 0;
  if (X & 0xFFFF0000)
    Res += 16;
  if (X & 0xFF00FF00)
    Res += 8;
  if (X & 0xF0F0F0F0)
    Res += 4;
  if (X & 0xCCCCCCCC)
    Res += 2;
  if (X & 0xAAAAAAAA)
    Res += 1;
  assert(Res < 32);
  assert((UINT32_C(1) << Res) == X);
  return Res;
}
} // namespace bits32

namespace bits64 {
/// \pre(0 <= N <= 64)
/// \post(retval = max([truncateToNBits(i,N) for i in naturals()]))
inline uint64_t maxValueOfNBits(unsigned N) {
  assert(N <= 64);
  if (N == 0)
    return 0;
  return ((UINT64_C(-1)) >> (64 - N));
}

/// \pre(0 < N <= 64)
inline uint64_t truncateToNBits(uint64_t X, unsigned N) {
  assert(N > 0 && N <= 64);
  return X & ((UINT64_C(-1)) >> (64 - N));
}

inline uint64_t withoutRightmostBit(uint64_t X) { return X & (X - 1); }

inline uint64_t isolateRightmostBit(uint64_t X) { return X & -X; }

inline uint64_t isPowerOfTwo(uint64_t X) {
  if (X == 0)
    return 0;
  return !(X & (X - 1));
}

/// \pre((x&(x-1)) == 0)
/// \post((1 << retval) == x)
inline unsigned indexOfSingleBit(uint64_t X) {
  assert((X & (X - 1)) == 0);
  unsigned Res = bits32::indexOfSingleBit(static_cast<unsigned>(X | (X >> 32)));
  if (X & (UINT64_C(0xFFFFFFFF) << 32))
    Res += 32;
  assert(Res < 64);
  assert((UINT64_C(1) << Res) == X);
  return Res;
}
} // namespace bits64

template <typename T>
[[nodiscard]] static constexpr auto countLeadingZeroes(T &&X) noexcept
    -> std::enable_if_t<!std::numeric_limits<std::decay_t<T>>::is_signed &&
                            std::numeric_limits<std::decay_t<T>>::digits ==
                                std::numeric_limits<unsigned>::digits,
                        int> {
  assert(X > 0);
  return __builtin_clz(static_cast<unsigned>(X));
}

template <typename T>
[[nodiscard]] static constexpr auto countLeadingZeroes(T &&X) noexcept
    -> std::enable_if_t<!std::numeric_limits<std::decay_t<T>>::is_signed &&
                            std::numeric_limits<std::decay_t<T>>::digits ==
                                std::numeric_limits<unsigned long>::digits &&
                            std::numeric_limits<unsigned>::digits !=
                                std::numeric_limits<unsigned long>::digits,
                        int> {
  assert(X > 0);
  return __builtin_clzl(static_cast<unsigned long>(X));
}

template <typename T>
[[nodiscard]] static constexpr auto countLeadingZeroes(T &&X) noexcept
    -> std::enable_if_t<
        !std::numeric_limits<std::decay_t<T>>::is_signed &&
            std::numeric_limits<std::decay_t<T>>::digits ==
                std::numeric_limits<unsigned long long>::digits &&
            std::numeric_limits<unsigned>::digits !=
                std::numeric_limits<unsigned long long>::digits &&
            std::numeric_limits<unsigned long>::digits !=
                std::numeric_limits<unsigned long long>::digits,
        int> {
  assert(X > 0);
  return __builtin_clzll(static_cast<unsigned long long>(X));
}

template <typename T>
[[nodiscard]] static constexpr auto countTrailingZeroes(T &&X) noexcept
    -> std::enable_if_t<!std::numeric_limits<std::decay_t<T>>::is_signed &&
                            std::numeric_limits<std::decay_t<T>>::digits ==
                                std::numeric_limits<unsigned>::digits,
                        int> {
  assert(X > 0);
  return __builtin_ctz(static_cast<unsigned>(X));
}

template <typename T>
[[nodiscard]] static constexpr auto countTrailingZeroes(T &&X) noexcept
    -> std::enable_if_t<!std::numeric_limits<std::decay_t<T>>::is_signed &&
                            std::numeric_limits<std::decay_t<T>>::digits ==
                                std::numeric_limits<unsigned long>::digits &&
                            std::numeric_limits<unsigned>::digits !=
                                std::numeric_limits<unsigned long>::digits,
                        int> {
  assert(X > 0);
  return __builtin_ctzl(static_cast<unsigned long>(X));
}

template <typename T>
[[nodiscard]] static constexpr auto countTrailingZeroes(T &&X) noexcept
    -> std::enable_if_t<
        !std::numeric_limits<std::decay_t<T>>::is_signed &&
            std::numeric_limits<std::decay_t<T>>::digits ==
                std::numeric_limits<unsigned long long>::digits &&
            std::numeric_limits<unsigned>::digits !=
                std::numeric_limits<unsigned long long>::digits &&
            std::numeric_limits<unsigned long>::digits !=
                std::numeric_limits<unsigned long long>::digits,
        int> {
  assert(X > 0);
  return __builtin_ctzll(static_cast<unsigned long long>(X));
}

[[nodiscard]] static constexpr std::size_t
roundUpToMultipleOf4096(std::size_t const X) {
  return ((X - 1) | static_cast<std::size_t>(4096 - 1)) + 1;
}
} // namespace luthier::klee

#endif
