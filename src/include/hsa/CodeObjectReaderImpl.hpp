//===-- CodeObjectReaderImpl.hpp ------------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This file defines the concrete implementation of the \c CodeObjectReader
/// class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_CODE_OBJECT_READER_IMPL_HPP
#define LUTHIER_HSA_CODE_OBJECT_READER_IMPL_HPP
#include "hsa/CodeObjectReader.hpp"

namespace luthier::hsa {

/// \brief the concrete implementation of the \c CodeObjectReader interface
class CodeObjectReaderImpl : public CodeObjectReader {

public:
  CodeObjectReaderImpl() : CodeObjectReader() {};

  explicit CodeObjectReaderImpl(hsa_code_object_reader_t Handle)
      : CodeObjectReader(Handle) {};

  [[nodiscard]] std::unique_ptr<CodeObjectReader> clone() const override;

  llvm::Error createFromMemory(llvm::StringRef Elf) override;

  llvm::Error destroy() override;
};

} // namespace luthier::hsa

#endif