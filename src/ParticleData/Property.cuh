/*Raul P. Pelaez 2017. Property, a container for a particle property (pos, force, vel...)

  Stores a GPU and CPU version of the information, has two copies of the GPU version for swapping.
  The CPU and the GPU copie arrays are only created when asked for them.
  
  Controls the acces of the data in a way such as that a property that is currently being read cannot be written to, or one being written to cannot be read.
  

*/
#ifndef PROPERTY_CUH
#define PROPERTY_CUH

#include"System/System.h"
#include"utils/GPUUtils.cuh"
#include"utils/debugTools.cuh"
#include<thrust/device_vector.h>


namespace uammd{
  //Forward declaration for friend attribute
  class ParticleData;
  template<class T> struct Property;
  
  template<class T>
  class property_ptr{
    T *ptr;
    bool *isBeingRead, *isBeingWritten;
    access::mode mode;
    access::location location;
    Property<T> *father_property;
  public:
    property_ptr():
      ptr(nullptr),
      isBeingRead(nullptr), isBeingWritten(nullptr),      
      mode(access::mode::nomode),
      location(access::location::nodevice),
      father_property(nullptr){}
    property_ptr(T* ptr, bool *isBeingWritten, bool *isBeingRead, access::mode mode, access::location loc,
		 Property<T> *father_property = nullptr):
      ptr(ptr),
      isBeingWritten(isBeingWritten),      
      isBeingRead(isBeingRead),
      mode(mode), location(loc),
      father_property(father_property){
    
      if(mode==access::mode::write || mode==access::mode::readwrite){
	*isBeingWritten = true;
      }
      else{
	*isBeingRead = true;
      }
    }
    ~property_ptr(){
      if(ptr){
	if(mode==access::mode::write || mode==access::mode::readwrite) *isBeingWritten = false;
	else{
	  *isBeingRead = false;
	}
      }
    }
    T* raw(){ return ptr;}

    //Swap one of the Property containers for an external one
    void swapContainer(thrust::device_vector<T> &swappable){
      father_property->swapGPUContainer(swappable);
    }
    void swapContainer(std::vector<T> &swappable){      	        
      father_property->swapCPUContainer(swappable);
    }
    
    
  };


  template<class T>
  struct Property{
    friend class ParticleData;
  private:  
    thrust::device_vector<T> deviceVector, deviceVector_alt;
    std::vector<T> hostVector;
    uint N = 0;
    bool deviceVectorNeedsUpdate = false, hostVectorNeedsUpdate= true;
    string name;
    bool isBeingWritten = false, isBeingRead= false;
    const shared_ptr<const System> &sys;

    T* getAltGPUBuffer(){
      if(deviceVector_alt.size() != N) deviceVector_alt.resize(N);
      return thrust::raw_pointer_cast(deviceVector_alt.data()); 
    }
  public:
    typedef T valueType;
    Property(): Property(0, "noName", nullptr){}
    Property(std::string name, const shared_ptr<const System> &sys): Property(0, name, sys){}
    Property(int N, std::string name, const shared_ptr<const System> &sys):N(N), name(name), sys(sys)
    {
      sys->log<System::DEBUG>("[Property] Property %s created with size %d", name.c_str(), N);
      CudaCheckError();
      if(N==0) return;
      deviceVector.resize(N);    
    }
    ~Property() = default;
    void resize(int Nnew){
      sys->log<System::DEBUG>("[Property] Resizing GPU version of %s to %d particles", name.c_str(), Nnew);
      this->N = Nnew;
      try{
	deviceVector.resize(Nnew);
      }
      catch(const thrust::system_error &e){
	sys->log<System::CRITICAL>("[Property] Thrust failed at deviceVector.resize(%d) with address %p with the message: %s",
				   Nnew, thrust::raw_pointer_cast(deviceVector.data()), e.what());
      }
      if(deviceVector_alt.size()>0){
	sys->log<System::DEBUG1>("[Property] Resizing alt GPU version of %s", name.c_str(), Nnew);
	deviceVector_alt.resize(Nnew);
      }
      //Only resize CPU memory if it has been created
      if(hostVector.size() > 0){
	hostVector.resize(Nnew);
	sys->log<System::DEBUG>("[Property] Resizing CPU version of %s", name.c_str());
      }
    }
    void swapInternalBuffers(){
      sys->log<System::DEBUG1>("[Property] Swapping GPU references of %s", name.c_str());
      if(deviceVector_alt.size() != N) deviceVector_alt.resize(N);
      deviceVector.swap(deviceVector_alt);
    }
    void swapCPUContainer(std::vector<T> &outsideHostVector){
      sys->log<System::DEBUG1>("[Property] Swapping internal CPU container of property (%s)", name.c_str());      
      if(this->isBeingRead || this->isBeingWritten)
	sys->log<System::CRITICAL>("[Property] You cannot swap the container of a property (%s) while is being read or written!", name.c_str());

      if(outsideHostVector.size() != N) {
	sys->log<System::DEBUG1>("[Property] Resizing input container, had %d elements, should have %d", outsideHostVector.size(), N);      
	outsideHostVector.resize(N);
      }
    
      hostVector.swap(outsideHostVector);
    }
    void swapGPUContainer(thrust::device_vector<T> &outsideDeviceVector){
      sys->log<System::DEBUG1>("[Property] Swapping internal GPU container of property (%s)", name.c_str());      
      if(this->isBeingRead || this->isBeingWritten)
	sys->log<System::CRITICAL>("[Property] You cannot swap the container of a property (%s) while is being read or written!", name.c_str());

      if(outsideDeviceVector.size() != N) {
	sys->log<System::DEBUG1>("[Property] Resizing input container, had %d elements, should have %d", outsideDeviceVector.size(), N);      
	outsideDeviceVector.resize(N);
      }
    
      deviceVector.swap(outsideDeviceVector);
    }

    property_ptr<T> data(access::location dev, access::mode mode){
      sys->log<System::DEBUG5>("[Property] %s requested from %d (0=cpu, 1=gpu) with access %d (0=r, 1=w, 2=rw)", name.c_str(), dev, mode);  
      if(this->isBeingWritten){
	sys->log<System::CRITICAL>("[Property] You cant request %s property while its locked for writing!", name.c_str());
      }
      if(mode==access::mode::write || mode==access::mode::readwrite){
	if(dev==access::location::gpu) hostVectorNeedsUpdate=true;
	else if(dev==access::location::cpu) deviceVectorNeedsUpdate=true;
	else { sys->log<System::CRITICAL>("[Property] Invalid access location in %s", name.c_str());}
	if(this->isBeingRead){
	  sys->log<System::CRITICAL>("[Property] You cant write to %s property while its being read!", name.c_str()); 
	}
      }

      T *devicePtr = thrust::raw_pointer_cast(deviceVector.data());
      T *hostPtr = hostVector.data();    
      switch(dev){
      case access::location::cpu:
	//Allocate CPU memory only if asked for it
	if(this->hostVector.size() == 0){
	  hostVector.resize(N);
	  hostPtr = hostVector.data();
	}
	if(hostVectorNeedsUpdate){
	  CudaSafeCall(cudaMemcpy(hostPtr, devicePtr, N*sizeof(T), cudaMemcpyDeviceToHost));
	  hostVectorNeedsUpdate=false;
	}
	return property_ptr<T>(hostPtr, &this->isBeingWritten, &this->isBeingRead, mode, dev);
      case access::location::gpu:
	if(deviceVectorNeedsUpdate){
	  CudaSafeCall(cudaMemcpy(devicePtr, hostPtr, N*sizeof(T), cudaMemcpyHostToDevice));
	  deviceVectorNeedsUpdate=false;
	}
	return property_ptr<T>(devicePtr, &this->isBeingWritten, &this->isBeingRead, mode, dev);
      case access::location::nodevice:
      default:
	return property_ptr<T>(nullptr, &this->isBeingWritten, &this->isBeingRead, mode, dev);
      }
    }
    void forceUpdate(access::location dev){
      switch(dev){
      case access::location::cpu:
	this->hostVectorNeedsUpdate = true;
	break;
      case access::location::gpu:
	this->deviceVectorNeedsUpdate = true;
	break;
      case access::location::nodevice:
      default:
	this->deviceVectorNeedsUpdate = false;
	
      }
    }
    string getName(){ return this->name;}
    int size(){return this->N;}
    bool isAllocated(){ return this->N>0;}
  };

}
#endif