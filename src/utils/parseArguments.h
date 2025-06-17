/*Raul P. Pelaez 2019
  Some utilities to process argv and argc
*/
#ifndef PARSEARGUMENTS_H
#define PARSEARGUMENTS_H

#include "exception.h"
#include <cstring>
#include <iterator>
#include <sstream>
#include <vector>

namespace uammd {
namespace detail {

template <class T, class StringIterator>
static std::vector<T> stringsTovalues(StringIterator first, int numberValues) {
  std::stringstream ss;

  try {
    std::copy(first, first + numberValues,
              std::ostream_iterator<std::string>(ss, " "));
  } catch (...) {
    std::throw_with_nested(std::runtime_error("Not enough arguments in argv"));
  }

  std::vector<T> vals(std::istream_iterator<T>(ss), {});

  if (!ss.eof())
    throw std::runtime_error("Invalid comand line argument");

  return vals;
}

} // namespace detail

class CommandLineArgumentParser {
  int m_argc = 0;
  char **m_argv = nullptr;

public:
  CommandLineArgumentParser(int argc, char **argv)
      : m_argc(argc), m_argv(argv) {}

  template <class T> T getFlagArgument(std::string flag) {
    auto val = getFlagArgumentMany<T>(flag, 1);
    return *(std::begin(val));
  }

  template <class T>
  std::vector<T> getFlagArgumentMany(std::string flag, int numberArguments) {
    if (numberArguments < 1)
      throw std::invalid_argument("Not enough arguments in argv");

    for (int i = 1; i < m_argc; i++) {
      if (flag.compare(m_argv[i]) == 0) {
        auto firstArgument = m_argv + i + 1;
        return detail::stringsTovalues<T>(firstArgument, numberArguments);
      }
    }

    throw std::runtime_error("Flag not found");
  }

  bool isFlagPresent(std::string flag) {
    for (int i = 1; i < m_argc; i++) {
      if (flag.compare(m_argv[i]) == 0) {
        return true;
      }
    }
    return false;
  }
};
} // namespace uammd
#endif
