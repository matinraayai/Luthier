//===-- LegacyPassSupport.h -------------------------------------*- C++ -*-===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// Includes utilities for support legacy passes in Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_LEGACY_PASS_SUPPORT_H
#define LUTHIER_TOOLING_LEGACY_PASS_SUPPORT_H

#define LUTHIER_INITIALIZE_LEGACY_PASS_PROTOTYPE(PASS_CLASS)                   \
  void initialize##PASS_CLASS(::llvm::PassRegistry &Registry)

#define LUTHIER_INITIALIZE_LEGACY_PASS_BODY(PASS_CLASS, ARG, NAME, CFG,        \
                                            ANALYSIS)                          \
  LUTHIER_INITIALIZE_LEGACY_PASS_PROTOTYPE(PASS_CLASS) {                       \
    static ::llvm::once_flag Initialize##PASS_CLASS##PassFlag;                 \
    auto initialize##PASS_CLASS##PassOnce = [](::llvm::PassRegistry &R) {      \
      auto *PI =                                                               \
          new llvm::PassInfo(NAME, ARG, &PASS_CLASS::ID,                       \
                             static_cast<::llvm::PassInfo::NormalCtor_t>(      \
                                 ::llvm::callDefaultCtor<PASS_CLASS>),         \
                             CFG, ANALYSIS);                                   \
      R.registerPass(*PI, true);                                               \
    };                                                                         \
    llvm::call_once(Initialize##PASS_CLASS##PassFlag,                          \
                    initialize##PASS_CLASS##PassOnce, std::ref(Registry));     \
  }

#endif