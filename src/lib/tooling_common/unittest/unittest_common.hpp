//===-- unittest_common.hpp - Luthier Unit Test Utilities -----------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Contains headers, functions, and macros to be used in all unit tests.
/// This file MUST be included in every unit test.
//===----------------------------------------------------------------------===//
#ifndef UNITTEST_COMMON_HPP
#define UNITTEST_COMMON_HPP

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <system_error>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace unittest {

// enum EXIT_CODE {
//   SUCCESS = 0,
//   FAILED = 1
// };

// void parseUnittestError(llvm::Error Err) {
// }

/// TODO: Make this not crappy
/// Define a new return on error macro for the unit tests. This is done because
/// the LUTHIER_RETURN_ON_ERROR macro returns from the current function with an
/// llvm::Error type. However, this macro is meant to be used in the main()
/// function of any given unit test, where we of course must return an int.
#define UNITTEST_RETURN_ON_ERROR(Error)                                        \
  do {                                                                         \
    if (Error) {                                                               \
      return -1;                                                               \
    }                                                                          \
  } while (0)

inline std::string getMIRFileName(const std::string& Path) {
  size_t SlashPos = Path.find_last_of('/');
  size_t DotPos   = Path.find_last_of('.');
  return Path.substr(SlashPos + 1, DotPos - SlashPos - 1) + ".mir";
}

/// For some reason, I can't convert llvm::raw_fd_ostream to llvm::Expected<...>
// llvm::Expected<llvm::raw_fd_ostream> 
// getOutFile(const std::string& code_obj_name, const std::string& dir) {
//   std::string OutFileName;
//   dir == "-" ? OutFileName = unittest::getMIRFileName(code_obj_name)      :
//                OutFileName = dir + unittest::getMIRFileName(code_obj_name);
//   std::error_code EC;
//   llvm::raw_fd_ostream 
//       OutFile(llvm::StringRef(OutFileName), EC);
//   if (EC) {
//     llvm::errs() << "Error when opening output file\n\n";
//     LUTHIER_RETURN_ON_ERROR(llvm::errorCodeToError(EC));
//   }
//   return OutFile;
// }

static int toolInit(rocprofiler_client_finalize_t FiniFunc, void* ToolData) {
  std::cout << "Rocprofiler initialization was called" << std::endl;
  return 0;
}

void toolFini(void* ToolData) { 
  std::cout << "Tool was finalized\n";
}

/// Callback function called during rocprofiler intercept table registration.
/// Saves the intercepted HSA API Table to *Data.
static void saveHsaApiTable(rocprofiler_intercept_table_t Type,
                            uint64_t LibVersion, uint64_t LibInstance,
                            void **Tables, uint64_t NumTables,
                            void *Data) {
  if(Type != ROCPROFILER_HSA_TABLE)
    throw std::runtime_error{"unexpected library type: " +
                             std::to_string(static_cast<int>(Type))};
  if(LibInstance != 0) 
    throw std::runtime_error{"multiple instances of HSA runtime library"};
  if(NumTables != 1) 
    throw std::runtime_error{"expected only one table of type HsaApiTable"};

  std::cout << "Save captured table to tool data" << std::endl;
  memcpy(Data, Tables[0], sizeof(HsaApiTable));
}

} // namespace unittest 

#endif
