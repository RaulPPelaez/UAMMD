









#include<iostream>
#include<tuple>

using namespace std;





template <std::size_t ...> struct index_sequence {};

template <std::size_t N, std::size_t ...Is>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...> {};

template <std::size_t ... Is>
struct make_index_sequence<0, Is...> : index_sequence<Is...> {};
  

struct functor{
  float *a;
  char *b;
  int *c;
  functor(float *a, char *b, int *c):a(a), b(b), c(c){}
  
  
};






template<class Functor, class ...T, size_t ...Is>
inline void unpackTupleAndCall_impl(Functor &f, std::tuple<T...> &values, int i, index_sequence<Is...>){
  printf("%.3f %c %d\n", *(i+std::get<Is>(values))...);    
}

template<class Functor, class ...T>
inline void unpackTupleAndCall(Functor &f, int i, std::tuple<T...> &values){
  constexpr int n= sizeof...(T);//std::tuple_size<decltype(values)>::value;
  unpackTupleAndCall_impl(f, values, i, make_index_sequence<n>());
}


template<class Functor, class ...T>
void compute(Functor f,
	std::tuple<T...>  arrays){

  for(int i=0; i<5; i++)
    unpackTupleAndCall(f, i, arrays);
}




int main(){
  int n = 5;
  float f[] = {1.1, 2.2, 3.3, 4.4, 5.5}; 
  char  c[] = {'a', 'b', 'c', 'd', 'e'}; 
  int   i[] = {1,2,3,4,5}; 


  functor fun(f,c,i);

  compute(fun, make_tuple(f,c,i));
  



  return 0;
}


