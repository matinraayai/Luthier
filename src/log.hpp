#ifndef SIBIR_SRC_LOG_HPP
#define SIBIR_SRC_LOG_HPP

#include <amd_comgr/amd_comgr.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <fmt/core.h>
#include <stdexcept>
//TODO: implement a proper logger


#ifdef SIBIR_LOG_ENABLE_DEBUG

#define SibirLogDebug(format, ...) fmt::println(stdout, format, __VA_ARGS__)

#else

#define SibirLogDebug

#endif


/**
#define SibirInfo(msg) \
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "%-5d: [%zx] %p %s: " msg, \
          getpid(), std::this_thread::get_id(), this, __func__)

#define SibirWarning(msg) \
  ClPrint(amd::LOG_WARNING, amd::LOG_CODE, "%-5d: [%zx] %p %s: " msg, \
          getpid(), std::this_thread::get_id(), this, __func__)

#define SibirDebug(format, ...) \
  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, \
          getpid(), std::this_thread::get_id(), this, __func__, __VA_ARGS__)

#define SibirWarning(format, ...) \
  ClPrint(amd::LOG_WARNING, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, \
          getpid(), std::this_thread::get_id(), this, __func__, __VA_ARGS__)

#define SibirInfo(format, ...) \
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, \
          getpid(), std::this_thread::get_id(), this, __func__, __VA_ARGS__)
**/


// Borrowed from ROCclr (for now)

#define SibirErrorMsg(msg) \
  fprintf(stderr, msg)

#define SibirErrorFmt(format, ...) \
  fmt::print(stderr, fmt::runtime(format), __VA_ARGS__)

#endif//SIBIR_SRC_LOG_HPP
