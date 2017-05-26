/*Raul P.Pelaez 2017.
  A magic function that returns the demangled name of a type,
  I got it from http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
  From an answer by Howard Hinnant.
*/
#ifndef TYPE_NAMES_H
#define TYPE_NAMES_H
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

template <class T>
std::string type_name() {
  typedef typename std::remove_reference<T>::type TR;
  std::unique_ptr<char, void(*)(void*)> own
    (
#ifndef _MSC_VER
     abi::__cxa_demangle(typeid(TR).name(), nullptr,
			 nullptr, nullptr),
#else
     nullptr,
#endif
     std::free
     );
  std::string r = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const<TR>::value)
    r += " const";
  if (std::is_volatile<TR>::value)
    r += " volatile";
  if (std::is_lvalue_reference<T>::value)
    r += "&";
  else if (std::is_rvalue_reference<T>::value)
    r += "&&";
  return r;
}
#endif
