//===-- types.h - Luthier Types  --------------------------------*- C++ -*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// \file
/// This file describes simple types and enums used by Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TYPES_H
#define LUTHIER_TYPES_H

namespace luthier {

/// Luthier address type
typedef unsigned long address_t;

/// Phase of the API/Event callback
enum ApiEvtPhase : unsigned short {
  /// Before API/Event occurs
  API_EVT_PHASE_BEFORE = 0,
  /// After API/Event has occurred
  API_EVT_PHASE_AFTER = 1
};

} // namespace luthier

#endif
