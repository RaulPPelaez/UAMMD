/*Raul P. Pelaez 2017. C++ useful constructions.
  
  Code to ease working with variadic templates, C++14+ features not present in C++11...
  
References:
[1] https://www.murrayc.com/permalink/2015/12/05/modern-c-variadic-template-parameters-and-tuples  

*/

#ifndef CXX_UTILS_H
#define CXX_UTILS_H
#include<cstring>
namespace uammd{

  namespace stringUtils{
    //Strip a certain pattern from a string
    inline std::string removePattern(std::string str, std::string pattern){
      std::string::size_type i = str.find(pattern);
      while (i != std::string::npos) {
	str.erase(i, pattern.length());
	i = str.find(pattern, i);
      }
      return str;
    }

  }
  namespace printUtils{
    //Transform size in bytes to a pretty string in B, KB, MB...
    std::string prettySize(size_t size) {
      static const char *SIZES[] = { "B", "KB", "MB", "GB" };
      uint div = 0;
      size_t rem = 0;

      while (size >= 1024u && div < (sizeof(SIZES)/ sizeof (*SIZES))) {
	rem = (size % 1024u);
	div++;
	size /= 1024;
      }

      double size_d = (float)size + (float)rem / 1024.0;
      std::string result = std::to_string(size_d) + " " + std::string(SIZES[div]);
      return result;
    }
  }


  //These lines replace the std::make_index_sequence functions from C++14
  template <std::size_t ...> struct index_sequence {};

  template <std::size_t N, std::size_t ...Is>
  struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...> {};

  template <std::size_t ... Is>
  struct make_index_sequence<0, Is...> : index_sequence<Is...> {};

namespace SFINAE{
  //This magic macro creates a struct such that has_X<T>::value will be true if
  // T has a callable member called X and false otherwise (at compile time).
#define SFINAE_DEFINE_HAS_MEMBER(X)			\
  template <typename T>  \
  class has_##X  \
  {  \
    using one = char;  \
    using two = long;  \
  \
    template <class C> static constexpr inline one test( decltype(&C::X) ) ;	\
    template <class C> static constexpr inline two test(...);  \
  \
  public:  \
    static constexpr bool value = std::is_same<decltype(test<T>(0)),one>::value; \
  };  \


}




  
}
#endif
