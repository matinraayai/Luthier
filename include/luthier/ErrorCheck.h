//===-- ErrorCheck.h - Luthier Error Checking Macros  -----------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains useful macros to check for \c llvm::Error in Luthier.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_ERROR_CHECK_H
#define LUTHIER_ERROR_CHECK_H

/// \brief Reports a fatal error if the passed \p llvm::Error argument is not
/// equal to \c llvm::Error::success()
#define LUTHIER_REPORT_FATAL_ON_ERROR(Error)                                   \
  do {                                                                         \
    if (Error) {                                                               \
      llvm::report_fatal_error(std::move(Error), true);                        \
    }                                                                          \
  } while (0)

/// \brief returns from the function if the passed \p llvm::Error argument is
/// not equal to \c llvm::Error::success()
#define LUTHIER_RETURN_ON_ERROR(Error)                                         \
  do {                                                                         \
    if (Error) {                                                               \
      return (Error);                                                          \
    }                                                                          \
  } while (0)

#endif