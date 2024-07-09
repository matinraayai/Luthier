//===-- log.hpp - Luthier Logging Utilities -------------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility functions and macros used for logging in Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_SRC_LOG_HPP
#define LUTHIER_SRC_LOG_HPP

#include <llvm/Support/FormatVariadic.h>

#define LUTHIER_LOG_FUNCTION_CALL_START                                        \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("<< Function call to {0} >>\n",     \
                                           __PRETTY_FUNCTION__));
#define LUTHIER_LOG_FUNCTION_CALL_END                                          \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("<< Return from function {0}>>\n",  \
                                           __PRETTY_FUNCTION__));

#endif // LUTHIER_SRC_LOG_HPP
