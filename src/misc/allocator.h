/*Raul P. Pelaez 2019. Allocators, this file defines some memory_resources and allocators compatible with c++17's std::pmr system. Thrust also provides similar ones with very recent versions (1.8.1+) but atm they are quite experimental and have some downsides. In theory these classes could be reduced to some aliases when thrust provides stable memory_resource options.

 */
#ifndef UAMMD_ALLOCATOR_H
#define UAMMD_ALLOCATOR_H


#include<thrust/system/cuda/pointer.h>
#include<thrust/system/cuda/memory.h>
#include<map>

namespace uammd{

  //A memory_resource interface to be used with thrust device pointers
  class device_memory_resource{
  public:
    using pointer = thrust::cuda::pointer<void>;
    virtual pointer allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)){
      return do_allocate(bytes, alignment);
    }
    virtual void deallocate(pointer p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)){
      return do_deallocate(p, bytes, alignment);
    }

    bool is_equal(const device_memory_resource &other) const noexcept{
      return do_is_equal(other);
    }

    virtual pointer do_allocate(std::size_t bytes, std::size_t alignment) = 0;
    virtual void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) = 0;
  
    virtual bool do_is_equal(const device_memory_resource &other) const noexcept{
      return this == &other;
    }
  
  };

  //A pool device memory_resource, stores previously allocated blocks in a cache
  // and retrieves them fast when similar ones are allocated again (without calling malloc everytime).
  struct device_pool_memory_resource: public device_memory_resource{
    // using value_type = typename std::pointer_traits<pointer>::element_type;
    using pointer = device_memory_resource::pointer;
    ~device_pool_memory_resource(){
      free_all();
    }
    using FreeBlocks =  std::multimap<std::ptrdiff_t, void*>;
    using AllocatedBlocks =  std::map<void*, std::ptrdiff_t>;
    FreeBlocks free_blocks;
    AllocatedBlocks allocated_blocks;

    virtual pointer do_allocate( std::size_t bytes, std::size_t alignment) override{
      pointer result;
      std::ptrdiff_t blockSize = 0;    
      auto available_blocks = free_blocks.equal_range(bytes);
      auto available_block = available_blocks.first;
      //Look for a block of the same size
      if(available_block == free_blocks.end()) available_block = available_blocks.second;
      //Try to find a block greater than requested size
      if(available_block != free_blocks.end() ){
	result = pointer(available_block -> second);
	blockSize = available_block -> first;	      
	free_blocks.erase(available_block);
      }
      else{
	try{
	  result = thrust::cuda::malloc<char>(bytes);
	  blockSize = bytes;
	}
	catch(...){
	  throw;
	}
      }
      allocated_blocks.insert(std::make_pair(result.get(), blockSize));
      return result;


    }

    virtual void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) override{
      auto block = allocated_blocks.find(p.get());
  
      if(block == allocated_blocks.end())
	throw  std::system_error(EFAULT, std::generic_category(), "Address is not handled by this instance.");
   
      std::ptrdiff_t num_bytes = block->second;
   
      allocated_blocks.erase(block);
      free_blocks.insert(std::make_pair(num_bytes, p.get()));
    }

    void free_all(){
      try{
	for(auto &i: free_blocks) thrust::cuda::free(thrust::cuda::pointer<void>(i.second));
	for(auto &i: allocated_blocks) thrust::cuda::free(thrust::cuda::pointer<void>(i.first));
      }
      catch(...){
	throw;
      }
    }

  };

  //Takes a pointer type (including smart pointers) anr returns a reference to the underlying type
  template<class T> struct pointer_to_lvalue_reference{
  private:
    using element_type = typename std::pointer_traits<T>::element_type;
  public:
    using type = typename std::add_lvalue_reference<element_type>::type;
  };

  //Specialization for special thrust pointer/reference types...
  template<class T> struct pointer_to_lvalue_reference<thrust::cuda::pointer<T>>{
    using type = thrust::system::cuda::reference<T>;
  };

  //An allocator that can be used for any type using the same underlying memory_resource
  template<class T, class MR = device_memory_resource>
  class polymorphic_allocator{
    //using MR = device_memory_resource;
  public:
    //C++17 definitions for allocator interface
    using size_type = std::size_t;  
    using value_type = T;
    using differente_type = std::ptrdiff_t;

    //All of the traits below are deprecated in C++17, but thrust counts on them
    using void_pointer = typename MR::pointer;  
    using pointer = typename std::pointer_traits<void_pointer>::template rebind<value_type>;
  
    //using reference = thrust::system::cuda::reference<element_type>;
    using reference = typename pointer_to_lvalue_reference<pointer>::type;
    using const_reference = typename pointer_to_lvalue_reference<std::add_const<pointer>>::type;
  
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    polymorphic_allocator(MR * resource) : res(resource){}
    MR* resource(){return res;}
    pointer allocate(size_type n){
      return pointer(res->do_allocate(n * sizeof(T), alignof(T)));
    }
    void deallocate(pointer p, size_type n){
      return res->do_deallocate(p, n * sizeof(T), alignof(T));
    }
  private:
    MR * res;
  };


  //thrust versions are not reliable atm, std::pmr is C++17 which CUDA 10.1 does not support yet and is not thrust::cuda compatible (thrust expects device allocators to return thurst::cuda::pointers)
  //template<class T> using memory_resource = thrust::mr::memory_resource<T>;
  //template<class T> using polymorphic_resource_adaptor = thrust::mr::polymorphic_adaptor_resource<T>;
  //template<typename T, class MR> using allocator = thrust::mr::allocator<T, MR>;
  //template<class T> using polymorphic_allocator = thrust::mr::polymorphic_allocator<T>;

}

#endif
