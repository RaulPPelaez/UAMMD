/*Raul P. Pelaez 2017. ParticleData.
  Handles and stores all properties a particle can have. 
  However they are only initialized when they are asked for the first time.
  Offers a way to access this properties.

  Can change in size and periodically sorts the particles to increase spatial locality.
  
  All changes in the particle data are announced using boost signals. 
  You can suscribe to this signals by asking for them with get*Signal()


  Entities using this class must take into account that the addresses of the properties and the order/number of the particles can change at any point. The former is solved by asking ParticleData for the address of a property each time it is going to be used, the latter are informed through signals so any needed computation can be performed.


  CREATION:

  auto pd = make_shared<ParticleData>(numberParticles, system);

  USAGE:

  To get a certain property:
  
  You can get a property both in GPU or CPU memory
  and must specify the kind of access (read, write, readwrite)
 
  If the mode is set to write, the handle will gain exclusivity and no one else will be able to 
  access it until it is realeased (the handle is deleted). 
  You cannot write to an array that is currently being read.
  For this it is important to control the scope of the property handles.

  //Get a handle to it
  auto pos_handle = pd->getPos(access::location::cpu, access::mode::read);
  //Get a raw memory pointer if needed
  real4* pos_ptr = pos_handle.raw();

  To get the indices of particles in the original order (ordered by ID):
  int * originalOrder = pd->getIndexArrayById(access::location::cpu);
  particle zero would be: pos.raw()[originalOrder[0]];

  //To get a property only if it has been asked for before (i.e if the mass has been set)
  auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
  //mass.raw() will be nullptr if mass has not been asked for before. 
  //Note that this call will never allocate the property
    
  CONNECT TO A SIGNAL:

  When the particles are reordered, or the number of them changes a signal will be thrown.
  In order to hear this signals a user class must:

  class User{
    connection reorderConnection, numParticlesChangedConnection;
    public:
     User(std::shared_ptr<ParticleData> pd){
       reorderConnection = pd->getReorderSignal()->
         connect([this](){this->handle_reorder();});

       numParticlesChangedConnection = pd->getNumParticlesChangedSignal()->
         connect([this](int Nnew){this->handle_numChanged(Nnew);});
     }
     ~User(){
     //Remember to disconnect when the signal is not needed anymore!
       reorderConnection.disconnect();
       numParticlesChangedConnection.disconnect();
     }
     void handle_reorder(){
       std::cout<<"A reorder occured!!"<std::endl;
     }  
     void handle_numChanged(int Nnew){
       std::cout<<"Particle number changed, now it is: "<<Nnew<<std::endl;
     }
  };

  LIST OF SIGNALS:

  numParticlesChangedSignal() -> int : Triggered when the total number of particles changes
  reorderSignal() -> void : Triggered when the global sorting of particles changes
  [PROPERTY]WrittenSignal() -> void: Triggered when PROPERTY has been requested with the write or readwrite flag (notice that the signal is emitted at requesting of the property, so the requester has writing rights

  TODO:
  100- Try libsigc++ if boost becomes a problem
  
*/
#ifndef PARTICLEDATA_CUH
#define PARTICLEDATA_CUH
#include"System/System.h"

#include"ParticleData/Property.cuh"
#include"utils/ParticleSorter.cuh"

//#include<boost/signals2.hpp>
#include<third_party/nod/nod.hpp>
//#include<boost/signals2/signal_type.hpp>
#include<third_party/boost/preprocessor.hpp>
#include<third_party/boost/preprocessor/stringize.hpp>
#include<third_party/boost/preprocessor/seq/for_each.hpp>
#include<third_party/boost/preprocessor/tuple/elem.hpp>
#include<thrust/device_vector.h>
#include <thrust/system_error.h>

#include"utils/vector.cuh"


//List here all the properties with this syntax:
/*       ((PropertyName, propertyName, TYPE))				\      */
//The preprocessor ensures that they are included wherever is needed
#define ALL_PROPERTIES_LIST ((Pos, pos, real4))     \
                            ((Id, id, int))	       \
                            ((Mass, mass, real))       \
			    ((Force, force, real4))    \
  			    ((Energy, energy, real))   \
			    ((Vel, vel, real3))        \
  			    ((Radius, radius, real))
/*
			    ((Torque, torque, real4))  \
  			    ((AngVel, angVel, real4))  \
  			    ((Dir, dir, real4))        \
    			    ((Charge, charge, real))        
*/

namespace uammd{

  template<class T>
  using signal = typename nod::unsafe_signal<T>;

  using connection = nod::connection;

  // template<class T>
  // using signal = typename boost::signals2::signal_type
  //   <
  //   T,
  //     boost::signals2::keywords::mutex_type<boost::signals2::dummy_mutex>
  //     >::type;

  // using connection = boost::signals2::connection;
  
  
  //Get the Name (first letter capital) from a tuple in the property list
#define PROPNAME_CAPS(tuple) BOOST_PP_TUPLE_ELEM(3, 0 ,tuple)
  //Get the name (no capital) from a tuple in the property list
#define PROPNAME(tuple) BOOST_PP_TUPLE_ELEM(3, 1 ,tuple)
  //Get the type from a tuple in the property list
#define PROPTYPE(tuple) BOOST_PP_TUPLE_ELEM(3, 2 ,tuple)

//This macro iterates through all properties applying some macro
#define PROPERTY_LOOP(macro)  BOOST_PP_SEQ_FOR_EACH(macro, _, ALL_PROPERTIES_LIST)


  
  class ParticleData{
  public:
    //Hints to ParticleData about how to perform different task. Mainly how to sort the particles.
    struct Hints{
      bool orderByHash = false;
      Box hash_box = Box(make_real3(128));
      real3 hash_cutOff = make_real3(10.0);
      bool orderByType = false;      

    };

  private:
    shared_ptr<System> sys;
#define DECLARE_PROPERTIES_T(type, name) Property<type> name;
#define DECLARE_PROPERTIES(r,data, tuple) DECLARE_PROPERTIES_T(PROPTYPE(tuple), PROPNAME(tuple))

    //Declare all property containers
    PROPERTY_LOOP(DECLARE_PROPERTIES)

    int numberParticles;
    shared_ptr<signal<void(void)>> reorderSignal = std::make_shared<signal<void(void)>>();
    shared_ptr<signal<void(int)>> numParticlesChangedSignal = std::make_shared<signal<void(int)>>();

//Declare write access signals for all properties
#define DECLARE_SIGNAL_PROPERTIES_T(type, name) shared_ptr<signal<void(void)>> BOOST_PP_CAT(name,WriteRequestedSignal = std::make_shared<signal<void(void)>>();)
#define DECLARE_SIGNAL_PROPERTIES(r,data, tuple) DECLARE_SIGNAL_PROPERTIES_T(PROPTYPE(tuple), PROPNAME(tuple))
    //Declare all property write signals
    PROPERTY_LOOP(DECLARE_SIGNAL_PROPERTIES)
    
    
    
    std::shared_ptr<ParticleSorter> particle_sorter;
    thrust::host_vector<int> originalOrderIndexCPU;
    bool originalOrderIndexCPUNeedsUpdate;
    Hints hints;
  public:
    ParticleData(int numberParticles, shared_ptr<System> sys);
    ~ParticleData(){
      sys->log<System::DEBUG>("[ParticleData] Destroyed");
      CudaCheckError();
    }

    
    //Generate getters for all properties except ID
#define GET_PROPERTY_T(Name,name)  GET_PROPERTY_R(Name,name)
#define GET_PROPERTY_R(Name, name)					\
  inline auto get ## Name(access::location dev, access::mode mode) -> decltype(name.data(dev,mode)){ \
    if(!name.isAllocated()) name.resize(numberParticles);		\
    if(!name.isAllocated() or mode==access::mode::write or mode==access::mode::readwrite){ \
    (*name ## WriteRequestedSignal)();				\
    }									\
      return name.data(dev,mode);	                            	\
    }									\
    
#define GET_PROPERTY(r, data, tuple) GET_PROPERTY_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))

    //Define getProperty() functions for all properties in list
    PROPERTY_LOOP(GET_PROPERTY)
    

        //Generate getters for all properties except ID
#define GET_PROPERTY_IF_ALLOC_T(Name,name)  GET_PROPERTY_IF_ALLOC_R(Name,name)
#define GET_PROPERTY_IF_ALLOC_R(Name, name)					\
    inline auto get ## Name ## IfAllocated(access::location dev, access::mode mode) -> decltype(name.data(dev,mode)){ \
      if(!name.isAllocated()){                    \
	decltype(name.data(dev,mode)) tmp;        \
	return tmp;	                          \
      }						  \
      return this->get ## Name(dev,mode);	  \
    }						  \
    
#define GET_PROPERTY_IF_ALLOC(r, data, tuple) GET_PROPERTY_IF_ALLOC_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))

    //Define getProperty() functions for all properties in list
    PROPERTY_LOOP(GET_PROPERTY_IF_ALLOC)
    


    
    //Generate isPropAllocated for all properties
#define IS_ALLOCATED_T(Name, name) IS_ALLOCATED_R(Name, name)    
#define IS_ALLOCATED_R(Name, name)					\
    inline bool is##Name##Allocated(){return name.isAllocated();}	\
    
#define IS_ALLOCATED(r, data, tuple) IS_ALLOCATED_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))
    
    PROPERTY_LOOP(IS_ALLOCATED)

    //Sort the particles to improve a certain kind of access pattern.
    void sortParticles();

    //Return an array that allows to access the particles in an ID ordered manner (as they started)
    const int * getIdOrderedIndices(access::location dev){
      sys->log<System::DEBUG5>("[ParticleData] Id order requested for %d (0=cpu, 1=gpu)", dev);
      auto id = getId(access::location::gpu, access::mode::read);
      int *sortedIndex = particle_sorter->getIndexArrayById(id.raw(), numberParticles);
      if(!sortedIndex) sortedIndex = id.raw();
      sys->log<System::DEBUG6>("[ParticleData] Id reorder completed.");  
      if(dev == access::location::gpu){	
	return sortedIndex;
      }
      else{
	if(originalOrderIndexCPUNeedsUpdate){
	  sys->log<System::DEBUG1>("[ParticleData] Updating CPU original order array");  
	  originalOrderIndexCPU.resize(numberParticles);
	  int * sortedIndexCPU = thrust::raw_pointer_cast(originalOrderIndexCPU.data());
	  CudaSafeCall(cudaMemcpy(sortedIndexCPU,
				  sortedIndex,
				  numberParticles*sizeof(int),
				  cudaMemcpyDeviceToHost));
	  originalOrderIndexCPUNeedsUpdate = false;
	  return sortedIndexCPU;
	}
	else{
	  return thrust::raw_pointer_cast(originalOrderIndexCPU.data());
	}
	  
      }
      

    }
    //Apply newest order to a certain iterator
    template<class InputIterator, class OutputIterator>
    void applyCurrentOrder(InputIterator in, OutputIterator out, int numElements){
      particle_sorter->applyCurrentOrder(in, out, numElements);
    }
    
    const int * getCurrentOrderIndexArray(){
      return particle_sorter->getSortedIndexArray(numberParticles);      
    }
  

    shared_ptr<signal<void(void)>> getReorderSignal(){
      sys->log<System::DEBUG>("[ParticleData] Reorder signal requested");  
      return this->reorderSignal;
    }


    //Generate getters for all properties except ID
#define GET_PROPERTY_SIGNAL_T(Name,name)  GET_PROPERTY_SIGNAL_R(Name,name)
#define GET_PROPERTY_SIGNAL_R(Name, name)				\
  inline shared_ptr<signal<void(void)>> get ## Name ## WriteRequestedSignal(){ \
    return this->name ## WriteRequestedSignal;				\
    }									
#define GET_PROPERTY_SIGNAL(r, data, tuple) GET_PROPERTY_SIGNAL_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))    
    PROPERTY_LOOP(GET_PROPERTY_SIGNAL)

          
    void emitReorder(){
      sys->log<System::DEBUG>("[ParticleData] Emitting reorder signal...");
      (*this->reorderSignal)();
    }

    shared_ptr<signal<void(int)>> getNumParticlesChangedSignal(){
      return this->numParticlesChangedSignal;
    }

    void changeNumParticles(int Nnew);
    int getNumParticles(){ return this->numberParticles;}


    
  void hintSortByHash(Box hash_box, real3 hash_cutOff){
    hints.orderByHash = true;
    hints.hash_box = hash_box;
    hints.hash_cutOff = hash_cutOff;

  }

  private:
    void emitNumParticlesChanged(int Nnew){
      (*numParticlesChangedSignal)(Nnew);
    }  

  };


#define INIT_PROPERTIES_T(NAME, name) ,  name(BOOST_PP_STRINGIZE(NAME), sys)
#define INIT_PROPERTIES(r,data, tuple) INIT_PROPERTIES_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))
  
  ParticleData::ParticleData(int numberParticles, shared_ptr<System> sys):
    numberParticles(numberParticles),
    originalOrderIndexCPUNeedsUpdate(true),
    sys(sys)
    PROPERTY_LOOP(INIT_PROPERTIES)
  {
    sys->log<System::MESSAGE>("[ParticleData] Created with %d particles.", numberParticles);    
    id.resize(numberParticles);
    CudaCheckError();
    auto id_prop = id.data(access::location::gpu, access::mode::write);

    //Fill Ids with 0..numberParticle (id[i] = i)
    cub::CountingInputIterator<int> ci(0);
    try{
      thrust::copy(ci, ci + numberParticles, thrust::device_ptr<int>(id_prop.raw()));
    }
    catch(thrust::system_error &e){
      sys->log<System::CRITICAL>("[ParticleData] Thrust could not copy ID vector. Error: %s", e.what());
    }
  }

  //Sort the particles to improve a certain kind of access pattern.
  void ParticleData::sortParticles(){
    sys->log<System::DEBUG>("[ParticleData] Sorting particles...");
    //Orders according to positions
    {      
      auto posPtr     = pos.data(access::gpu, access::write);
      if(hints.orderByHash || !hints.orderByType){
	int3 cellDim = make_int3(hints.hash_box.boxSize/hints.hash_cutOff);
	particle_sorter->updateOrderByCellHash(posPtr.raw(), numberParticles, hints.hash_box, cellDim);
      }
      
    }
  //This macro reorders to the newest order a property given its name 
#define APPLY_CURRENT_ORDER(r, data, tuple) APPLY_CURRENT_ORDER_R(PROPNAME(tuple))
#define APPLY_CURRENT_ORDER_R(name) {					\
      if(name.isAllocated()){						\
	auto devicePtr     = name.data(access::gpu, access::write);	\
	auto device_altPtr = name.getAltGPUBuffer();			\
	particle_sorter->applyCurrentOrder(devicePtr.raw(), device_altPtr, numberParticles); \
	name.swapInternalBuffers();						\
      }									\
    }    
    //Apply current order to all allocated properties. See APPLY_CURRENT_ORDER macro
    PROPERTY_LOOP(APPLY_CURRENT_ORDER)

    originalOrderIndexCPUNeedsUpdate = true;
    //Notify all connected entities of the reordering
    this->emitReorder();
    
  }



  void ParticleData::changeNumParticles(int Nnew){
    sys->log<System::CRITICAL>("[ParticleData] CHANGE PARTICLES FUNCTIONALITY NOT IMPLEMENTED YET!!!");
    sys->log<System::DEBUG>("[ParticleData] Adding/Removing particles...");
    this->numberParticles = Nnew;
    pos.resize(Nnew);
#define RESIZE_PROPERTY_R(name) {if(this->name.isAllocated()){this->name.resize(this->numberParticles);}}
#define RESIZE_PROPERTY(r, data, tuple) RESIZE_PROPERTY_R(PROPNAME(tuple))
    
    PROPERTY_LOOP(RESIZE_PROPERTY)

    originalOrderIndexCPUNeedsUpdate = true;
    this->emitNumParticlesChanged(Nnew);
  }
}

#undef ALL_PROPERTIES_LIST
#undef PROPNAME_CAPS
#undef PROPNAME
#undef PROPTYPE
#undef PROPERTY_LOOP
#undef DECLARE_PROPERTIES_T
#undef DECLARE_PROPERTIES
#undef DECLARE_SIGNAL_PROPERTIES_T
#undef DECLARE_SIGNAL_PROPERTIES
#undef GET_PROPERTY_T
#undef GET_PROPERTY_R
#undef GET_PROPERTY
#undef GET_PROPERTY_SIGNAL_T
#undef GET_PROPERTY_SIGNAL_R
#undef GET_PROPERTY_SIGNAL
#undef IS_ALLOCATED_T
#undef IS_ALLOCATED_R
#undef IS_ALLOCATED
#undef GET_PROPERTY_IF_ALLOC
#undef GET_PROPERTY_IF_ALLOC_T
#undef GET_PROPERTY_IF_ALLOC_R
#undef APPLY_CURRENT_ORDER
#undef APPLY_CURRENT_ORDER_R
#undef RESIZE_PROPERTY_R
#undef RESIZE_PROPERTY




#endif

