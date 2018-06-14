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

  pg->getIndexIterator(); (RECOMMENDED)

  OR

  pg->getIndicesRawPtr();

*/

#ifndef PARTICLEGROUP_CUH
#define PARTICLEGROUP_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include<thrust/device_vector.h>
#include<vector>
#include<third_party/cub/cub.cuh>

#include"third_party/type_names.h"
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
      static constexpr bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
	return true;
      }    
    };

    class None{
    public:
      None(){}    
      static constexpr bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
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

      bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
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
      bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
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
      bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
	int type_i = int(pd->getPos(access::cpu, access::read).raw()[particleIndex].w);
	for(auto type: typesToSelect){
	  if(type_i==type) return true;
	}
	return false;
      }    

    
    

    };
  };

  namespace ParticleGroup_ns{  
    // Transforms from a particle ID based array of flags (member/not-member) to an index based array (current indices of particles ParticleData).
    // This transformation is needed when the particles are sorted.
  
  /*
  __global__ void updateIndexFlags(int* particleId, bool *id_flags, bool *index_flags, int N){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index>=N) return;

    //ID of this index...
    int id = particleId[index];
    //If flag of this index is a member of the group
    if(id_flags[id]){
      //This index is a member of the group
      index_flags[index] = true;
    }
    else{
      index_flags[index] = false;
    }
  }
*/
  //Updates the indices of the particles in a group using pd->getIdOrderedIndices()
  __global__ void updateGroupIndices(const int * __restrict__ id2index,
				     int * __restrict__ particlesIndices,
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
    shared_ptr<ParticleData> pd;
    shared_ptr<System> sys;
    cudaStream_t st = 0;
    //ID (particle name) and index (particle index in pd arrays) based membership flags
    //thrust::device_vector<bool> IDFlagsGPU, indexFlagsGPU;

    //A list of the particle indices of the group (updated to current order)
    thrust::device_vector<int> myParticlesIndicesGPU, myParticlesIdsGPU;
    thrust::host_vector<int>   myParticlesIndicesCPU;

    bool updateHostVector = true;
    bool needsIndexListUpdate = false;

    //number of particles in group and in all system (pd)
    int numberParticles, totalParticles;

    bool allParticlesInGroup = false;
  
    std::string name;

    //Cub additional storage
    // thrust::device_vector<char> temporaryStorage;
    // size_t temporaryStorageSize = 0;
    // int* cubNumSelectedGPU;

    connection reorderConnection;

  public:
    /*Defaults to all particles in group*/
    ParticleGroup(shared_ptr<ParticleData> pd, shared_ptr<System> sys, std::string name = std::string("noName"));

    /*Create the group from a selector*/
    template<class ParticleSelector>
    ParticleGroup(ParticleSelector selector,
		  shared_ptr<ParticleData> pd, shared_ptr<System> sys, std::string name = std::string("noName"));

    /*Create the group from a list of particle IDs*/
    template<class InputIterator>
    ParticleGroup(InputIterator begin, InputIterator end,
		  shared_ptr<ParticleData> pd, shared_ptr<System> sys,
		  std::string name = std::string("noName"));

    ~ParticleGroup(){
      sys->log<System::DEBUG>("Group %s destroyed", name.c_str());
      CudaCheckError();
      reorderConnection.disconnect();
      if(st) cudaStreamDestroy(st);
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

    //Custom iterator for indices in group,
    //falls back to a counting iterator if all particles are in group
    struct IndexIterator{
      const int *indices;
      IndexIterator(const int *indices):
      indices(indices){ }

      inline __host__ __device__ int operator()(const int &i) const{
	return this->operator[](i);
      }
      inline __host__ __device__ int operator[](const int &i) const{	
	if(indices == nullptr) return i;
	else{
	  return indices[i];
	}
      }
    
    };

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
      const int *ptr = nullptr;
      if(!this->allParticlesInGroup && this->numberParticles != 0){
	ptr = getIndicesRawPtr(loc);
      }
      return IndexIterator(ptr);
			   
    }

    template<class Iterator>
    struct accessIterator{
      Iterator ptr;
      using return_type =  typename std::add_lvalue_reference<decltype(ptr[0])>::type;
      IndexIterator it;
      accessIterator(Iterator ptr, IndexIterator it): ptr(ptr),it(it){

      }
      inline __host__ __device__ return_type operator[](const int &i) const{
	return ptr[it[i]];
      }

    };
    template<class Iterator>
    accessIterator<Iterator> getPropertyInputIterator(Iterator property,
						      access::location loc){
      return accessIterator<Iterator>(property, this->getIndexIterator(loc));      
    }
              
    int getNumberParticles(){
      return this->numberParticles;
    }

    std::string getName(){ return this->name;}
  };



  template<class ParticleSelector>
  ParticleGroup::ParticleGroup(ParticleSelector selector,
			       shared_ptr<ParticleData> pd, shared_ptr<System> sys, std::string name):
    pd(pd), sys(sys), name(name){
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s created with selector %s",
			      name.c_str(), type_name<ParticleSelector>().c_str());  
    totalParticles = pd->getNumParticles();
    /*Create ID list in CPU*/
    //thrust::host_vector<bool> IDFlagsCPU(totalParticles);
    std::vector<int> ids;
    for(int i=0;i<totalParticles;i++){
      // IDFlagsCPU[i] = selector.isSelected(i, pd);
      // if(IDFlagsCPU[i]) numberParticles++;
      if(selector.isSelected(i, pd))
	ids.push_back(i);    
    }
    numberParticles = ids.size();
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s contains %d particles.",
			      name.c_str(), numberParticles);    
    /*Handle the case in which all particles belong to the group*/
    if(numberParticles==totalParticles){
      allParticlesInGroup = true;
      //IDFlagsCPU.clean();
    }
    else{
      myParticlesIdsGPU = ids;
      
      //Connect to reorder signal, index list needs to be updated each time a reorder occurs
      reorderConnection = pd->getReorderSignal()->connect([this](){this->handleReorder();});      
      //Allocate
      myParticlesIndicesGPU.resize(numberParticles);
      // IDFlagsGPU = IDFlagsCPU;
      // indexFlagsGPU.resize(totalParticles);    
      // cudaMalloc(&cubNumSelectedGPU, sizeof(int));
      
      //Force update (creation) of the index list)
      this->computeIndexList(true);
    }
  }

  //Specialization of a particle group with an All selector
  template<>
  ParticleGroup::ParticleGroup(particle_selector::All selector,
			       shared_ptr<ParticleData> pd, shared_ptr<System> sys,
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
			       shared_ptr<ParticleData> pd, shared_ptr<System> sys,
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
			       shared_ptr<ParticleData> pd, shared_ptr<System> sys,
			       std::string name):
    pd(pd), sys(sys), name(name){
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s created from ID list.", name.c_str());   
    numberParticles = std::distance(begin, end);
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s contains %d particles.", name.c_str(), numberParticles);
    this->totalParticles = pd->getNumParticles();
  
    if(numberParticles = totalParticles){
      this->allParticlesInGroup = true;    
    }
    else{
      reorderConnection = pd->getReorderSignal()->connect([this](){this->handleReorder();});      
      
      //Create ID list in CPU
      //thrust::host_vector<bool> IDFlagsCPU(totalParticles);
      //Turn on given member ID's flags
      // for(auto i=begin; i != end; ++i){
      // 	IDFlagsCPU[*i] =  true;
      // }
      myParticlesIdsGPU.assign(begin, end);
      myParticlesIndicesGPU.resize(numberParticles);
      //IDFlagsGPU = IDFlagsCPU;
      //indexFlagsGPU.resize(totalParticles);
    
      //cudaMalloc(&cubNumSelectedGPU, sizeof(int));
      /*Force update (creation) of the index list)*/
      this->computeIndexList(true);
    
    }    
  }



  //If no selector is provided, All is assumed
  ParticleGroup::ParticleGroup(shared_ptr<ParticleData> pd, shared_ptr<System> sys,
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
      
      /*Without using getIdOrderedIndices()
      //Get needed arrays
      bool *IDFlags_ptr = thrust::raw_pointer_cast(IDFlagsGPU.data());
      bool *indexFlags_ptr = thrust::raw_pointer_cast(indexFlagsGPU.data());

      auto allParticlesIDs_handle = pd->getId(access::location::gpu, access::mode::read);
      int *allParticlesIDs = allParticlesIDs_handle.raw();
    

      int Nthreads=512;
      int Nblocks=totalParticles/Nthreads + ((totalParticles%Nthreads)?1:0);
      //fill indexFlags
      ParticleGroup_ns::updateIndexFlags<<<Nblocks, Nthreads>>>(allParticlesIDs, IDFlags_ptr, indexFlags_ptr, totalParticles);

      int *myParticlesIndices_ptr = thrust::raw_pointer_cast(myParticlesIndicesGPU.data());

      if(temporaryStorage.size()==0){
	char *d_tmp = thrust::raw_pointer_cast(temporaryStorage.data());
	cub::CountingInputIterator<int> flag2index(0); 
	cub::DeviceSelect::Flagged(d_tmp, temporaryStorageSize,
				   flag2index, indexFlags_ptr,
				   myParticlesIndices_ptr,
				   cubNumSelectedGPU, totalParticles);
	temporaryStorage.resize(temporaryStorageSize);
      }

      //Fill an array of size numberParticles with the indices of the 'true' flags in indexFlags
      char *d_tmp = thrust::raw_pointer_cast(temporaryStorage.data());
      cub::CountingInputIterator<int> flag2index(0); 
      cub::DeviceSelect::Flagged(d_tmp, temporaryStorageSize,
				 flag2index, indexFlags_ptr,
				 myParticlesIndices_ptr,
				 cubNumSelectedGPU, totalParticles);
      this->needsIndexListUpdate = false;
      updateHostVector = true;
      */
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
    if(!st) cudaStreamCreate(&st);
    
    auto copyKind = cudaMemcpyDeviceToDevice;
    if(loc==access::location::cpu)  copyKind = cudaMemcpyHostToDevice;
    
    cudaMemcpyAsync(myParticlesIdsGPU_ptr+numberParticlesPrev, ids,
		    N*sizeof(int), copyKind, st);

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
    cudaStreamSynchronize(st);

  }

  namespace ParticleGroup_ns{
    __global__  void IdsFromIndices(const int *indices, const int *indexToId, int* groupParticleIds, int N){
      int tid = blockIdx.x*blockDim.x + threadIdx.x;
      if(tid>=N) return;
      int index = indices[tid];
      int id = indexToId[index];
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
    
    if(!st) cudaStreamCreate(&st);
    auto copyKind = cudaMemcpyDeviceToDevice;
    if(loc==access::location::cpu)  copyKind = cudaMemcpyHostToDevice;
    
    cudaMemcpyAsync(myParticlesIndicesGPU_ptr+numberParticlesPrev, indices,
		    N*sizeof(int), copyKind, st);
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
    cudaStreamSynchronize(st);
  }

}
#endif
