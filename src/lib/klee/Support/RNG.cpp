//===-- RNG.cpp -------------------------------------------------*- C++ -*-===//
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
#include "klee/ADT/RNG.h"
#include "klee/Support/OptionCategories.h"

namespace {
llvm::cl::opt<RNG::result_type> RNGInitialSeed(
    "rng-initial-seed", llvm::cl::init(5489U),
    llvm::cl::desc("seed value for random number generator (default=5489)"),
    llvm::cl::cat(klee::MiscCat));
}

namespace luthier::klee {
RNG::RNG() : std::mt19937(RNGInitialSeed.getValue()) {}

RNG::RNG(RNG::result_type Seed) : std::mt19937(Seed) {}

unsigned int RNG::getInt32() {
  static_assert((min)() == 0);
  static_assert((max)() == -1u);
  return (*this)();
}

double RNG::getDoubleL() {
  return getInt32() * (1.0 / 4294967296.0);
  /* divided by 2^32 */
}

double RNG::getDouble() {
  return (static_cast<double>(getInt32()) + 0.5) * (1.0 / 4294967296.0);
  /* divided by 2^32 */
}

bool RNG::getBool() {
  unsigned Bits = getInt32();
  Bits ^= Bits >> 16U;
  Bits ^= Bits >> 8U;
  Bits ^= Bits >> 4U;
  Bits ^= Bits >> 2U;
  Bits ^= Bits >> 1U;
  return Bits & 1U;
}

} // namespace luthier::klee
