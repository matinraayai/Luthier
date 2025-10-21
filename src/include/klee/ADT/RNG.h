//===-- RNG.h ---------------------------------------------------*- C++ -*-===//
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
#ifndef LUTHIER_KLEE_RNG_H
#define LUTHIER_KLEE_RNG_H
#include <random>

namespace luthier::klee {

struct RNG : std::mt19937 {
  RNG();
  explicit RNG(RNG::result_type Seed);

  /// Generates a random number on [0,0xffffffff]-interval
  unsigned int getInt32();

  /// generates a random number on [0,1)-real-interval
  double getDoubleL();

  /// generates a random number on (0,1)-real-interval
  double getDouble();

  /// generators a random flop
  bool getBool();
};

} // namespace luthier::klee

#endif /* KLEE_RNG_H */
