/*Raul P. Pelaez 2019. Property, a container for a particle property (pos, force, vel...)

  Maintains a GPU and CPU version, has two copies of the GPU version for fast swapping.
  Each copy is allocated when first requested.

  Requesting a property reference in read mode prevents other users from requesting a write reference and viceversa.

*/
#ifndef PROPERTY_CUH
#define PROPERTY_CUH

#include"System/System.h"
#include"utils/GPUUtils.cuh"
#include"utils/debugTools.cuh"
#include<thrust/device_vector.h>
#include <thrust/iterator/iterator_adaptor.h>
#include"third_party/managed_allocator.h"

namespace uammd{
  //Forward declaration for friend attribute
  class ParticleData;
  template<class T> struct Property;

  template<class T>
  class property_ptr:
    public thrust::iterator_adaptor<property_ptr<T>, T*>
  {

  public:
    using Iterator = T*;
    using super_t = thrust::iterator_adaptor<property_ptr<T>, Iterator>;
  private:
    T *ptr;
    size_t m_size;
    bool *isBeingRead, *isBeingWritten;
    bool isCopy = false; //true if this instance was created when passed to a cuda kernel
    friend class thrust::iterator_core_access;

    void unlockProperty(){
      *isBeingWritten = false;
      *isBeingRead = false;
    }

  public:

    property_ptr():
      super_t(nullptr),
      ptr(nullptr),
      m_size(0),
      isBeingRead(nullptr), isBeingWritten(nullptr)
      {}

    property_ptr(T* ptr,
		 bool *isBeingWritten, bool *isBeingRead,
		 size_t in_size):
      super_t(ptr),
      ptr(ptr),
      m_size(in_size),
      isBeingWritten(isBeingWritten),
      isBeingRead(isBeingRead)
    {}

    __host__ __device__ property_ptr(const property_ptr& _orig ):super_t(_orig.ptr) { *this = _orig; isCopy = true; }

    __host__ __device__ ~property_ptr(){
#ifdef __CUDA_ARCH__
      return;
#else
      if(isCopy) return;
      if(ptr)
	unlockProperty();
#endif
    }

    __host__ __device__ T* raw() const { return ptr;}

    __host__ __device__ T* get() const { return raw();}

    __host__ __device__ Iterator end() const{
      if(ptr)
	return begin()+size();
      else
	return nullptr;
    }

    __host__ __device__ Iterator begin() const{ return get();}

    __host__ __device__ size_t size() const{ return m_size;}
  };

  struct illegal_property_access: public std::runtime_error{
    using std::runtime_error::runtime_error;
  };

  template<class T>
  struct Property{
    friend class ParticleData;
  public:
    using valueType = T;
    using iterator = property_ptr<valueType>;
    Property(): Property(0, "noName", nullptr){}
    Property(std::string name, shared_ptr<System> sys): Property(0, name, sys){}
    Property(int N, std::string name, shared_ptr<System> sys):N(N), name(name), sys(sys)
    {
      sys->log<System::DEBUG>("[Property] Property %s created with size %d", name.c_str(), N);
      CudaCheckError();
      //There is no benefit from using managed memory for Properties, at least for now (arch <=600)
      //I would like to leave the possibility here though.
      if(false and sys->getSystemParameters().managedMemoryAvailable){
	sys->log<System::DEBUG1>("[Property] Property %s is using managed memory");
	this->isManaged = true;
      }
    }
    ~Property() = default;

    void resize(int Nnew){
      N = Nnew;
    }

    void swapInternalBuffers(){
      try{
	tryToResizeAndSwapInternalContainers();
      }
      catch(...){
	sys->log<System::ERROR>("[Property] Exception raised during internal container swap");
	throw;
      }
    }

    void swapCPUContainer(std::vector<T> &outsideHostVector){
      sys->log<System::DEBUG1>("[Property] Swapping internal CPU container of property (%s)", name.c_str());
      swapWithExternalContainer(hostVector, outsideHostVector);
      forceUpdate(access::location::gpu);
    }

    void swapGPUContainer(thrust::device_vector<T> &outsideDeviceVector){
      sys->log<System::DEBUG1>("[Property] Swapping internal CPU container of property (%s)", name.c_str());
      swapWithExternalContainer(deviceVector, outsideDeviceVector);
      forceUpdate(access::location::cpu);
    }

    iterator data(access::location dev, access::mode mode){
      sys->log<System::DEBUG5>("[Property] %s requested from %d (0=cpu, 1=gpu, 2=managed) with access %d (0=r, 1=w, 2=rw)",
			       name.c_str(), dev, mode);
      try{
	return tryToGetData(dev,mode);
      }
      catch(...){
	sys->log<System::ERROR>("[Property] Exception raised in data request for property "+name);
	throw;
      }
    }

    iterator begin(access::location dev, access::mode mode){
      return data(dev, mode);
    }

    void forceUpdate(access::location dev){
      switch(dev){
      case access::location::cpu:
	this->hostVectorNeedsUpdate = true;
	break;
      case access::location::gpu:
	this->deviceVectorNeedsUpdate = true;
	break;
      }
    }

    std::string getName() const{ return this->name;}

    int size() const{ return this->N;}

    bool isAllocated() const{ return this->N>0;}

  private:

    thrust::device_vector<T> deviceVector, deviceVector_alt;
    std::vector<T> hostVector;
    uammd::managed_vector<T> managedVector, managedVector_alt;

    uint N = 0;
    bool deviceVectorNeedsUpdate = false, hostVectorNeedsUpdate = false;
    string name;
    bool isBeingWritten = false, isBeingRead= false;
    bool isManaged = false;
    shared_ptr<System> sys;

    T* getAltGPUBuffer(){
      if(isManaged){
	managedVector_alt.resize(N);
	CudaSafeCall(cudaDeviceSynchronize());
	return thrust::raw_pointer_cast(managedVector_alt.data());
      }
      else{
	deviceVector_alt.resize(N);
	return thrust::raw_pointer_cast(deviceVector_alt.data());
      }
    }

    property_ptr<T> tryToGetData(access::location dev, access::mode mode){
      const bool requestedForWriting = (mode==access::mode::write or mode==access::mode::readwrite);

      throwIfIllegalDataRequest(mode);

      lockIfNecesary(mode);

      switch(dev){
      case access::location::cpu:
	if(hostVector.size()!= N){
	  sys->log<System::DEBUG1>("[Property] Resizing host version of " +
				   name + " to " + std::to_string(N)+ " elements");
	  hostVector.resize(N);
	}
	updateHostData();
	if(requestedForWriting)
	  deviceVectorNeedsUpdate=true;
	return property_ptr<T>(hostVector.data(), &this->isBeingWritten, &this->isBeingRead, size());

      case access::location::gpu:
	if(deviceVector.size()!= N){
	  sys->log<System::DEBUG1>("[Property] Resizing device version of " +
				   name + " to " + std::to_string(N)+ " elements");
	deviceVector.resize(N);
	}
	updateDeviceData();
	if(requestedForWriting)
	  hostVectorNeedsUpdate=true;
	return property_ptr<T>(thrust::raw_pointer_cast(deviceVector.data()),
			       &this->isBeingWritten, &this->isBeingRead, size());

      case access::location::managed:
	  if(!isManaged){
	    throw std::runtime_error("[Property] Current system does not accept Managed memory requests.");
	  }
	  if(sys->getSystemParameters().cuda_arch < 600)
	    CudaSafeCall(cudaDeviceSynchronize());
	  return property_ptr<T>(thrust::raw_pointer_cast(managedVector.data()),
				 &this->isBeingWritten, &this->isBeingRead, size());

      default:
	throw std::runtime_error("[Property] Invalid location requested");
      }
    }

    void throwIfIllegalDataRequest(access::mode mode){
      const bool requestedForWriting = (mode==access::mode::write or mode==access::mode::readwrite);
      const bool requestedForReading = (mode==access::mode::read);
      {
	const bool isIllegalRequestForWriting = (this->isBeingWritten or this->isBeingRead) and requestedForWriting;
	const bool isIllegalRequestForReading = (this->isBeingWritten and requestedForReading);
	if(isIllegalRequestForWriting or isIllegalRequestForReading){
	  sys->log<System::ERROR>("[Property] You cant request " + name + " property for " +
				  (this->isBeingWritten?"writing":"reading") + " while its locked!");
	  throw illegal_property_access("Property "+name+" requested while locked");
	}
      }
    }

    void lockIfNecesary(access::mode request_mode){
      const bool requestedForWritting = (request_mode==access::mode::write or request_mode==access::mode::readwrite);
      const bool requestedForReading = (request_mode==access::mode::read);
      if(requestedForWritting)
	this->isBeingWritten = true;
      if(requestedForReading)
	this->isBeingRead = true;
    }
    void updateHostData(){
      if(hostVectorNeedsUpdate){
	hostVector.resize(N);
	CudaSafeCall(cudaMemcpy(hostVector.data(),
				thrust::raw_pointer_cast(deviceVector.data()),
				N*sizeof(T), cudaMemcpyDeviceToHost));
	hostVectorNeedsUpdate=false;
      }
    }

    void updateDeviceData(){
      if(deviceVectorNeedsUpdate){
	deviceVector = hostVector;
	deviceVectorNeedsUpdate=false;
      }
    }

    void tryToResizeAndSwapInternalContainers(){
      if(isManaged){
	sys->log<System::DEBUG1>("[Property] Swapping internal managed references of %s", name.c_str());
	managedVector_alt.resize(N);
	managedVector.swap(managedVector_alt);
      }
      else{
	sys->log<System::DEBUG1>("[Property] Swapping internal device references of %s", name.c_str());
        deviceVector_alt.resize(N);
	managedVector.swap(managedVector_alt);
	hostVectorNeedsUpdate = true;
      }
    }

    template<class InternalContainer, class ExternalContainer>
    void tryToSwapWithExternalContainer(InternalContainer &myContainer, ExternalContainer &outsideContainer){
      throwIfnotInSwappableState();
      if(outsideContainer.size() != N) {
	sys->log<System::DEBUG1>("[Property] Resizing input container, had %d elements, should have %d",
				 outsideContainer.size(), N);
	outsideContainer.resize(N);
      }
      myContainer.swap(outsideContainer);
    }

    void throwIfnotInSwappableState(){
      if(isManaged){
	throw std::runtime_error("[Property] Cannot swap the container of a Managed property (" + name+")");
      }
      if(this->isBeingRead || this->isBeingWritten){
	sys->log<System::ERROR>("[Property] Cannot swap property %s while it is locked for writing/reading",
				    name.c_str());
	throw illegal_property_access("Property "+name+" requested while locked");
      }
    }

  };

}
#endif