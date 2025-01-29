//===-- ROCmLibraryErrorDefine.h --------------------------------*- C++ -*-===//
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
/// This file describes the \c LUTHIER_DEFINE_ROCM_LIBRARY_ERROR macro,
/// used to create \c llvm::Errors for ROCm libraries. It does not have an
/// include guard, therefore this macro must be undef-ed after usage inside
/// the target header file.
//===----------------------------------------------------------------------===//

/// \brief Macro used to create errors for each ROCm library used in Luthier
#define LUTHIER_DEFINE_ROCM_LIBRARY_ERROR(LIBRARY_NAME, LIBRARY_ERROR_TYPE,    \
                                          DEFAULT_SUCCESS_VALUE)               \
  class LIBRARY_NAME##Error final                                              \
      : public llvm::ErrorInfo<LIBRARY_NAME##Error> {                          \
  public:                                                                      \
    static char ID;         /* ID of the error */                              \
    const std::string File; /* Path of the file the error was encountered */   \
    const int                                                                  \
        LineNumber; /* Line number of the file the error was encountered */    \
    const std::string                                                          \
        StackTrace; /* Stack trace of where the error occurred */              \
    const std::string Expression;   /* Expression that caused the error */     \
    const LIBRARY_ERROR_TYPE Error; /* Encapsulated library error */           \
                                                                               \
    /* Public constructor for the library Error; */                            \
    /* Not meant to be used directly; */                                       \
    /* Use the error-checking macros instead */                                \
    LIBRARY_NAME##Error(const llvm::StringRef FileName, const int LineNumber,  \
                        const llvm::StringRef StackTrace,                      \
                        const LIBRARY_ERROR_TYPE Error,                        \
                        const llvm::StringRef Expression)                      \
        : File(FileName), LineNumber(LineNumber), StackTrace(StackTrace),      \
          Expression(Expression), Error(Error){};                              \
    /* Factory function used by macros to check for COMGR errors */            \
    /* \c llvm::ErrorSuccess is returned; Otherwise, the library error is      \
     * returned */                                                             \
    static llvm::Error LIBRARY_NAME##ErrorCheck(                               \
        llvm::StringRef FileName, int LineNumber, LIBRARY_ERROR_TYPE Expr,     \
        llvm::StringRef ExprStr,                                               \
        LIBRARY_ERROR_TYPE Expected = (DEFAULT_SUCCESS_VALUE));                \
                                                                               \
    void log(llvm::raw_ostream &OS) const override;                            \
                                                                               \
    [[nodiscard]] std::error_code convertToErrorCode() const override {        \
      llvm_unreachable("Not implemented");                                     \
    }                                                                          \
  }