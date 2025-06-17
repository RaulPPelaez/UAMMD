#ifndef UAMMD_EXCEPTION_H
#define UAMMD_EXCEPTION_H
#include "System/Log.h"
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <system_error>

namespace uammd {
using exception = std::exception;

void backtrace_nested_exception(const uammd::exception &e, int level = 0) {

  Logging::log<Logging::EXCEPTION>(std::string(level, ' ') + "level " +
                                   std::to_string(level) +
                                   " exception: " + e.what());

  try {
    std::rethrow_if_nested(e);
  } catch (const std::exception &e) {
    backtrace_nested_exception(e, level + 1);
  } catch (...) {
  }
}

}; // namespace uammd

#endif
