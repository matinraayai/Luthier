//
// Created by User on 3/17/2025.
//

#ifndef TESTING_UTILS_H
#define TESTING_UTILS_H

#include "hip/HipCompilerApiInterceptor.hpp"
#include <string>

llvm::Error compile_and_link(std::string filepath);

#endif //TESTING_UTILS_H
