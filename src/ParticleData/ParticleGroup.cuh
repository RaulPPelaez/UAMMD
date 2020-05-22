/*Raul P. Pelaez 2017. ParticleGroup

  ParticleGroup allows to track a certain subset of the particles in the system.

  Usage:


From a ParticleData pd and a System sys:

Create a group:

  //Different selectors offer different criteria
  //In this case, it will result in a group with particles whose ID lies between 4 and 8

  particle_selector::IDRange selector(4,8);
  auto pg = make_shared<ParticleGroup>(selector, pd, sys, name);

  OR

  auto pg = make_shared<ParticleGroup>(pd, sys, name); //All particles in system

  OR

  auto pg = make_shared<ParticleGroup>(ids.begin(), ids.end(), pd, sys, name); //From a container with a list of particle IDs

........................

You can request the current indices of the particles in a group with:

  pg->getIndexIterator(access::location::gpu); //or cpu, (RECOMMENDED)

//Or get a plain array with the indices directly, if it exists.

  pg->getIndicesRawPtr(access::location::gpu); //or cpu, it will be nullptr if all particles are in the group


//You can also request an iterator that will read a ParticleData array using the group indices directly.
//Optionally a cub load modifier can be passed, which will cause the iterator to be read through a certain GPU cache
//This allows to iterate over the particles in a group without worrying about groups, particle sorting, etc...

  auto pos = pd->getPos(access::location::gpu, access::mode::read);
  auto posIter = pg->getPropertyInputIterator<cub::LOAD_LDG>(pos.raw(), access::location::gpu); //or cpu

*/

#ifndef PARTICLEGROUP_CUH
#define PARTICLEGROUP_CUH

#include"System/System.h"
#include"ParticleData.cuh"
#include<thrust/device_vector.h>
#include<vector>
#include<third_party/cub/cub.cuh>
#include"third_party/type_names.h"
#include<memory>
namespace uammd{
  /*Small structs that encode different ways of selecting a certain set of particles,
    i.e by type, spatial location, ID...
    A particle selector must have an isSelected method that takes a particle index (not ID) and
    a ParticleData reference.
    It will be called for all particles except when not needed (i.e with All).*/
  namespace particle_selector{
    //Select all the particles
    class All{
    public:
      All(){}
      static constexpr bool isSelected(int particleIndex, std::shared_ptr<ParticleData> &pd){
	return true;
      }
    };

    class None{
    public:
      None(){}
      static constexpr bool isSelected(int particleIndex, std::shared_ptr<ParticleData> &pd){
	return false;
      }
    };

    //Select particles with ID in a certain range
    class IDRange{
      int firstID, lastID;
    public:
      IDRange(int first, int last):
	firstID(first),
	lastID(last){
      }

      bool isSelected(int particleIndex, std::shared_ptr<ParticleData> &pd){
	int particleID = (pd->getId(access::cpu, access::read).raw())[particleIndex];
	return particleID>=firstID && particleID<=lastID;
      }

    };

    //Select particles inside a certain rectangular region of the simulation box.
    class Domain{
      Box domain, simulationBox;
      real3 origin;
    public:
      Domain(real3 origin, Box domain, Box simulationBox):
	origin(origin),
	domain(domain),
	simulationBox(simulationBox){

      }
      bool isSelected(int particleIndex, std::shared_ptr<ParticleData> &pd){
	real3 pos = make_real3(pd->getPos(access::cpu, access::read).raw()[particleIndex]);
	pos = simulationBox.apply_pbc(pos);
	pos += origin;
	return domain.isInside(pos);
      }
    };

    //Select particles by type (pos.w)
    class Type{
      std::vector<int> typesToSelect;
    public:
      Type(int type): typesToSelect({type}){
      }

      Type(std::vector<int> typesToSelect): typesToSelect(typesToSelect){
      }
      bool isSelected(int particleIndex, std::shared_ptr<ParticleData> &pd){
	int type_i = int(pd->getPos(access::cpu, access::read).raw()[particleIndex].w);
	for(auto type: typesToSelect){
	  if(type_i==type) return true;
	}
	return false;
      }




    };
  };

  namespace ParticleGroup_ns{
    //Updates the indices of the particles in a group using pd->getIdOrderedIndices()
    __global__ void updateGroupIndices(//An array that stores the indices of the particles in the group per id.
				       const int * __restrict__ id2index,
				       //Out: the current ParticleData indices of the particles in the group
				       int * __restrict__ particlesIndices,
				       //In: Ids of the particle sin the group
				       const int * __restrict__ particlesIds,
				       int numberParticles
				       ){
      int tid = blockIdx.x*blockDim.x + threadIdx.x;
      if(tid >= numberParticles) return;
      int id = particlesIds[tid];
      int index = id2index[id];
      particlesIndices[tid] = index;
    }

  }
    // Keeps track of a certain subset of particles in a ParticleData entity
    // You can ask ParticleGroup to return you:
    // -The particle IDs of its members.
    // -The current indices of its members in the ParticleData arrays.

    // You can ask for the indices as a raw memory pointer or as a custom iterator.
    // Asking for the raw memory is a risky bussiness, as this array may not even exists (i.e if all particles in the system are in the group, it might decide to not create it, as it would be unnecessary). In this case, you will get a nullptr.

  class ParticleGroup{
    std::shared_ptr<ParticleData> pd;
    std::shared_ptr<System> sys;
    cudaStream_t st = 0;
    //A list of the particle indices and ids of the group (updated to current order)
    thrust::device_vector<int> myParticlesIndicesGPU, myParticlesIdsGPU;
    thrust::host_vector<int>   myParticlesIndicesCPU;

    bool updateHostVector = true;
    bool needsIndexListUpdate = false;

    //number of particles in group and in all system (pd)
    int numberParticles, totalParticles;

    bool allParticlesInGroup = false;

    std::string name;

    connection reorderConnection;

  public:
    /*Defaults to all particles in group*/
    ParticleGroup(std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys, std::string name = std::string("noName"));

    /*Create the group from a selector*/
    template<class ParticleSelector>
    ParticleGroup(ParticleSelector selector,
		  std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys, std::string name = std::string("noName"));

    /*Create the group from a list of particle IDs*/
    template<class InputIterator>
    ParticleGroup(InputIterator begin, InputIterator end,
		  std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys,
		  std::string name = std::string("noName"));

    ~ParticleGroup(){
      sys->log<System::DEBUG>("[ParticleGroup] Group %s destroyed", name.c_str());
      CudaCheckError();
      reorderConnection.disconnect();
      if(st) CudaSafeCall(cudaStreamDestroy(st));
    }

    //Remove all particles from the group
    void clear(){
      this->numberParticles = 0;
    }
    //Add particles to the group via an array with ids
    void addParticlesById(access::location loc, const int *ids, int N);
    //Add particles to the group via an array with the current indices of the particles in pd (faster)
    void addParticlesByCurrentIndex(access::location loc, const int *indices, int N);
    //Update index list if needed
    void computeIndexList(bool forceUpdate = false);

    void handleReorder(){
      sys->log<System::DEBUG>("[ParticleGroup] Handling reorder signal in group %s", this->name.c_str());
      if(!allParticlesInGroup && numberParticles > 0){
	needsIndexListUpdate = true;
      }
    }

    //Access the index array only if it is not a nullptr (AKA if the group does not contain all particles)
    struct IndexAccess{
      IndexAccess(const int * indices):indices(indices){}
      inline __host__ __device__ int operator()(const int &i) const{
	if(!indices) return i;
	else return indices[i];
      }
    private:
      const int * indices;
    };

    //Transform sequential indexing to indices of particle sin group
    using IndexIterator = cub::TransformInputIterator<int, IndexAccess, cub::CountingInputIterator<int>>;

    static IndexIterator make_index_iterator(const int *indices){
      return IndexIterator(cub::CountingInputIterator<int>(0), IndexAccess(indices));
    }

    //Get a raw memory pointer to the index list if it exists
    inline const int * getIndicesRawPtr(access::location loc){
      if(this->allParticlesInGroup || numberParticles == 0 ) return nullptr;
      this->computeIndexList();
      int *ptr;
      switch(loc){
      case access::location::cpu:
	if(updateHostVector){
	  myParticlesIndicesCPU = myParticlesIndicesGPU;
	  updateHostVector = false;
	}
	ptr = thrust::raw_pointer_cast(myParticlesIndicesCPU.data());
	break;
      case access::location::gpu:
	ptr = thrust::raw_pointer_cast(myParticlesIndicesGPU.data());
	break;
      default:
	ptr = nullptr;
      }
      return ptr;
    }

    //Get an iterator with the indices of particles in this group
    inline IndexIterator getIndexIterator(access::location loc){
      auto ptr = getIndicesRawPtr(loc);
      return make_index_iterator(ptr);
    }

    //Simply reads an iterator, optionally a cub cache mode can be selected
    template<class Iterator, cub::CacheLoadModifier modifier = cub::LOAD_DEFAULT>
    struct TransformIndex{
      TransformIndex(const Iterator &it):it(it){}
      using value_type =  typename std::iterator_traits<Iterator>::value_type;
      inline __host__ __device__ value_type operator()(const int &i) const{
	return it[i];
      }
    private:
      cub::CacheModifiedInputIterator<modifier, value_type> it;
    };

    //Reads an iterator transforming sequential indexing to indices of the particles in the group
    template<class Iterator,
	     cub::CacheLoadModifier modifier = cub::LOAD_DEFAULT>
    using accessIterator = cub::TransformInputIterator<typename std::iterator_traits<Iterator>::value_type,
						       TransformIndex<Iterator, modifier>,
						       IndexIterator>;
  private:
    template<cub::CacheLoadModifier modifier = cub::LOAD_DEFAULT, class Iterator>
    accessIterator<Iterator, modifier> make_access_iterator(const Iterator &it, access::location loc){
      return accessIterator<Iterator, modifier>(this->getIndexIterator(loc),
						TransformIndex<Iterator, modifier>(it));
    }

  public:

    //Returns an iterator that will have size pg->getNumberParticles() and will iterate over the
    // particles in the group.
    //For example, If a group contains only the particle with id=10, passing pd->getPos(...).raw() to this function
    // will return an iterator so that iterator[0] = pos[10]; and it will take into account any possible reordering of the pos array.
    template<cub::CacheLoadModifier modifier = cub::LOAD_DEFAULT, class Iterator>
    accessIterator<Iterator, modifier> getPropertyInputIterator(const Iterator & property,
								access::location loc){
      return this->make_access_iterator<modifier>(property, loc);
    }

    template<cub::CacheLoadModifier modifier = cub::LOAD_DEFAULT, class PropertyIterator>
    accessIterator<typename PropertyIterator::Iterator, modifier> getPropertyInputIterator(const PropertyIterator & property){
      using T = typename PropertyIterator::value_type;
      static_assert(std::is_same<PropertyIterator, typename Property<T>::iterator>::value,
		    "You must specify a location or call this function with a Property::iterator argument");
      auto loc = property.location();
      if(loc == access::location::managed){
	loc = access::location::cpu;
      }
      return this->make_access_iterator<modifier>(property.raw(), loc);
    }


    int getNumberParticles(){
      return this->numberParticles;
    }

    std::string getName(){ return this->name;}
  };

  template<class ParticleSelector>
  ParticleGroup::ParticleGroup(ParticleSelector selector,
			       std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys, std::string name):
    pd(pd), sys(sys), name(name){
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s created with selector %s",
			      name.c_str(), type_name<ParticleSelector>().c_str());
    totalParticles = pd->getNumParticles();
    std::vector<int> ids;
    for(int i=0;i<totalParticles;i++){
      if(selector.isSelected(i, pd))
	ids.push_back(i);
    }
    numberParticles = ids.size();
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s contains %d particles.", name.c_str(), numberParticles);
    if(numberParticles==totalParticles){
      allParticlesInGroup = true;
    }
    else{
      myParticlesIdsGPU = ids;
      reorderConnection = pd->getReorderSignal()->connect([this](){this->handleReorder();});
      myParticlesIndicesGPU.resize(numberParticles);
      this->computeIndexList(true);
    }
  }

  //Specialization of a particle group with an All selector
  template<>
  ParticleGroup::ParticleGroup(particle_selector::All selector,
			       std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys,
			       std::string name):
  pd(pd), sys(sys), name(name){
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s created with All selector",name.c_str());
    this->allParticlesInGroup = true;
    this->totalParticles = pd->getNumParticles();
    this->numberParticles = totalParticles;

    sys->log<System::MESSAGE>("[ParticleGroup] Group %s contains %d particles.",
			      name.c_str(), numberParticles);
  }
  //Specialization of an empty particle group
  template<>
  ParticleGroup::ParticleGroup(particle_selector::None selector,
			       std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys,
			       std::string name):
  pd(pd), sys(sys), name(name){
    this->allParticlesInGroup = false;
    this->totalParticles = pd->getNumParticles();
    this->numberParticles = 0;
    reorderConnection = pd->getReorderSignal()->connect([this](){this->handleReorder();});
  }


  //Constructor of ParticleGroup when an ID list is provided
  template<class InputIterator>
  ParticleGroup::ParticleGroup(InputIterator begin, InputIterator end,
			       std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys,
			       std::string name):
    pd(pd), sys(sys), name(name){
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s created from ID list.", name.c_str());
    numberParticles = std::distance(begin, end);
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s contains %d particles.", name.c_str(), numberParticles);
    this->totalParticles = pd->getNumParticles();

    if(numberParticles == totalParticles){
      this->allParticlesInGroup = true;
    }
    else{
      reorderConnection = pd->getReorderSignal()->connect([this](){this->handleReorder();});

      //Create ID list in CPU
      myParticlesIdsGPU.assign(begin, end);
      myParticlesIndicesGPU.resize(numberParticles);
      /*Force update (creation) of the index list)*/
      this->computeIndexList(true);
    }
  }



  //If no selector is provided, All is assumed
  ParticleGroup::ParticleGroup(std::shared_ptr<ParticleData> pd, std::shared_ptr<System> sys,
			       std::string name):
    ParticleGroup(particle_selector::All(), pd, sys, name){}


  //This is trivial with  pd->getIdOrderedIndices()!
  //Handle a reordering of the particles (which invalids the previous relation between IDs and indices)
  void ParticleGroup::computeIndexList(bool forceUpdate){
    if(numberParticles==0) return;
    if(this->needsIndexListUpdate || forceUpdate){//Update only if needed
      sys->log<System::DEBUG>("[ParticleGroup] Updating group %s after last particle sorting", name.c_str());
      const int *id2index = pd->getIdOrderedIndices(access::location::gpu);
      int Nthreads=(numberParticles>=512)?512:numberParticles;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      int *myParticlesIndicesGPU_ptr = thrust::raw_pointer_cast(myParticlesIndicesGPU.data());
      int *myParticlesIdsGPU_ptr = thrust::raw_pointer_cast(myParticlesIdsGPU.data());
      ParticleGroup_ns::updateGroupIndices<<<Nblocks, Nthreads>>>(id2index,
						myParticlesIndicesGPU_ptr,
						myParticlesIdsGPU_ptr,
						numberParticles);
      this->needsIndexListUpdate = false;
      updateHostVector = true;
      sys->log<System::DEBUG1>("[ParticleGroup] Updating group %s DONE!", name.c_str());
    }
  }
  //Add particles to the group via an array with ids
  void ParticleGroup::addParticlesById(access::location loc, const int *ids, int N){
    sys->log<System::DEBUG1>("[ParticleGroup] Adding %d particles to group %s via ids!", N, name.c_str());
    int numberParticlesPrev = numberParticles;
    numberParticles += N;
    myParticlesIndicesGPU.resize(numberParticles);
    myParticlesIdsGPU.resize(numberParticles);

    const int *id2index = pd->getIdOrderedIndices(access::location::gpu);
    int Nthreads=(N>=128)?128:N;
    int Nblocks=N/Nthreads + ((N%Nthreads)?1:0);

    int *myParticlesIndicesGPU_ptr = thrust::raw_pointer_cast(myParticlesIndicesGPU.data());
    int *myParticlesIdsGPU_ptr = thrust::raw_pointer_cast(myParticlesIdsGPU.data());
    if(!st) CudaSafeCall(cudaStreamCreate(&st));

    auto copyKind = cudaMemcpyDeviceToDevice;
    if(loc==access::location::cpu)  copyKind = cudaMemcpyHostToDevice;

    CudaSafeCall(cudaMemcpyAsync(myParticlesIdsGPU_ptr+numberParticlesPrev, ids,
				 N*sizeof(int), copyKind, st));

    const int *d_ids = ids;
    cudaStream_t upSt = st;
    if(loc==access::location::cpu){
      d_ids = myParticlesIdsGPU_ptr + numberParticlesPrev;
      upSt = 0;
    }

    ParticleGroup_ns::updateGroupIndices<<<Nblocks, Nthreads, 0, upSt>>>(id2index,
					      myParticlesIndicesGPU_ptr + numberParticlesPrev,
					      d_ids,
					      N);
    CudaSafeCall(cudaStreamSynchronize(st));

  }

  namespace ParticleGroup_ns{
    __global__  void IdsFromIndices(const int *indices, const int *index2Id, int* groupParticleIds, int N){
      int tid = blockIdx.x*blockDim.x + threadIdx.x;
      if(tid>=N) return;
      int index = indices[tid];
      int id = index2Id[index];
      groupParticleIds[tid] = id;
    }

  }
  //Add particles to the group via an array with the current indices of the particles in pd (faster)
  void ParticleGroup::addParticlesByCurrentIndex(access::location loc, const int *indices, int N){
    sys->log<System::DEBUG1>("[ParticleGroup] Adding %d particles to group %s via indices!", N, name.c_str());
    if(N==0) return;
    int numberParticlesPrev = numberParticles;
    numberParticles += N;
    myParticlesIndicesGPU.resize(numberParticles);
    myParticlesIdsGPU.resize(numberParticles);

    const int *id2index = pd->getIdOrderedIndices(access::location::gpu);
    int Nthreads=(N>=128)?128:N;
    int Nblocks=N/Nthreads + ((N%Nthreads)?1:0);

    int *myParticlesIndicesGPU_ptr = thrust::raw_pointer_cast(myParticlesIndicesGPU.data());
    int *myParticlesIdsGPU_ptr = thrust::raw_pointer_cast(myParticlesIdsGPU.data());

    if(!st) CudaSafeCall(cudaStreamCreate(&st));
    auto copyKind = cudaMemcpyDeviceToDevice;
    if(loc==access::location::cpu)  copyKind = cudaMemcpyHostToDevice;

    CudaSafeCall(cudaMemcpyAsync(myParticlesIndicesGPU_ptr+numberParticlesPrev, indices,
				 N*sizeof(int), copyKind, st));
    auto index2id = pd->getId(access::location::gpu, access::mode::read);

    const int *d_indices = indices;
    cudaStream_t upSt = st;
    if(loc == access::location::cpu){
      d_indices = myParticlesIndicesGPU_ptr+numberParticlesPrev;
      upSt = 0;
    }
    ParticleGroup_ns::IdsFromIndices<<<Nblocks, Nthreads, 0, upSt>>>(d_indices,
				     index2id.raw(),
				     myParticlesIdsGPU_ptr+numberParticlesPrev,
				     N);
    CudaSafeCall(cudaStreamSynchronize(st));
  }

}
#endif
