/*Raul P. Pelaez 2021. Useful containers for UAMMD
 */
#ifndef UAMMD_CONTAINER_H
#define UAMMD_CONTAINER_H
#include"global/defines.h"
#include"System/System.h"
#include"misc/allocator.h"
#include<thrust/device_vector.h>
namespace uammd{
  namespace detail{
    //This is a very barebones container. Its purpose is only to avoid the unnecessary unninitialized_fill kernel that thrust issues on device_vector creation. Thus it mascarades as a thrust::device_vector.
    template<class T, class Allocator = System::allocator<T>>
    class UninitializedCachedContainer{
      using Container = std::shared_ptr<T>;
      using Ptr = T*;
      Container m_data;
      size_t m_size, capacity;

      Container create(size_t s){
	if(s>0){
	  try{
	    return Container(Allocator().allocate(s), [](T* ptr){Allocator().deallocate(ptr);});
	  }
	  catch(...){
	    System::log<System::EXCEPTION>("[UninitializedCachedContainer] Could not allocate buffer of size %zu", s);
	    throw;
	  }
	}
	else{
	  return Container();
	}
      }

    public:
      using iterator = typename thrust::device_vector<T>::iterator;

      UninitializedCachedContainer(size_t i_size = 0)
	: m_size(0), capacity(0), m_data() {
	this->resize(i_size);
      }

      UninitializedCachedContainer(const std::vector<T> &other):
	UninitializedCachedContainer(other.size()){
	thrust::copy(other.begin(), other.end(), begin());
      }

      UninitializedCachedContainer(const UninitializedCachedContainer<T> &other):
	UninitializedCachedContainer(other.size()){
	thrust::copy(other.begin(), other.end(), begin());
      }

      iterator begin() const{ return iterator(data()); }

      iterator end() const{ return begin() + m_size; }

      size_t size() const{
	return m_size;
      }

      void resize(size_t newsize){
	if(newsize > capacity){
	  auto data2 = create(newsize);
	  if(size()>0){
	    thrust::copy(thrust::cuda::par, begin(), end(), data2.get());
	  }
	  m_data.swap(data2);
	  capacity = newsize;
	  m_size = newsize;
	}
	else{
	  m_size = newsize;
	}
      }

      thrust::device_ptr<T> data() const{
	return thrust::device_ptr<T>(m_data.get());
      }

      void clear(){
	resize(0);
	m_data = create(0);
	capacity = 0;
      }

      void swap(UninitializedCachedContainer<T> & another){
        m_data.swap(another.m_data);
      }

      // auto operator=(UninitializedCachedContainer<T> &other){
      // 	return UninitializedCachedContainer<T>(other);
      // }
    };
  }

  template<class T>  using uninitialized_cached_vector = detail::UninitializedCachedContainer<T>;
}
#endif
