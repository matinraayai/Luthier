#ifndef LUTHIER_SRC_LOG_HPP
#define LUTHIER_SRC_LOG_HPP

#include <fmt/color.h>

//TODO: implement a proper logger


#ifdef LUTHIER_LOG_ENABLE_INFO

#define LUTHIER_LOG_FUNCTION_CALL_START fmt::print(stdout, fmt::emphasis::underline | fg(fmt::color::burly_wood), "<< Function call to {} >>\n", __PRETTY_FUNCTION__);
#define LUTHIER_LOG_FUNCTION_CALL_END fmt::print(stdout, fmt::emphasis::underline | fg(fmt::color::burly_wood), "<< Return from function {}>>\n", __PRETTY_FUNCTION__);

#else

#define LUTHIER_LOG_FUNCTION_CALL_START
#define LUTHIER_LOG_FUNCTION_CALL_END

#endif

#ifdef LUTHIER_LOG_ENABLE_DEBUG

#define LuthierLogDebug(format, ...) fmt::println(stdout, format, __VA_ARGS__)

#else

#define LuthierLogDebug

#endif


/**
#define LuthierInfo(msg) \
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "%-5d: [%zx] %p %s: " msg, \
          getpid(), std::this_thread::get_id(), this, __func__)

#define LuthierWarning(msg) \
  ClPrint(amd::LOG_WARNING, amd::LOG_CODE, "%-5d: [%zx] %p %s: " msg, \
          getpid(), std::this_thread::get_id(), this, __func__)

#define LuthierDebug(format, ...) \
  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, \
          getpid(), std::this_thread::get_id(), this, __func__, __VA_ARGS__)

#define LuthierWarning(format, ...) \
  ClPrint(amd::LOG_WARNING, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, \
          getpid(), std::this_thread::get_id(), this, __func__, __VA_ARGS__)

#define LuthierInfo(format, ...) \
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, \
          getpid(), std::this_thread::get_id(), this, __func__, __VA_ARGS__)
**/


// Borrowed from ROCclr (for now)

#define LuthierErrorMsg(msg) \
  fprintf(stderr, msg)

#define LuthierErrorFmt(format, ...) \
  fmt::print(stderr, fmt::runtime(format), __VA_ARGS__)

#endif//LUTHIER_SRC_LOG_HPP
