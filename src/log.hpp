#ifndef SIBIR_SRC_LOG_HPP
#define SIBIR_SRC_LOG_HPP

#include <fmt/color.h>

//TODO: implement a proper logger


#ifdef SIBIR_LOG_ENABLE_INFO

#define SIBIR_LOG_FUNCTION_CALL_START fmt::print(stdout, fmt::emphasis::underline | fg(fmt::color::burly_wood), "<< Sibir function call to {} >>\n", __PRETTY_FUNCTION__);
#define SIBIR_LOG_FUNCTION_CALL_END fmt::print(stdout, fmt::emphasis::underline | fg(fmt::color::burly_wood), "<< Return from function {}>>\n", __PRETTY_FUNCTION__);

#else

#define SIBIR_LOG_FUNCTION_CALL_START
#define SIBIR_LOG_FUNCTION_CALL_END

#endif

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
