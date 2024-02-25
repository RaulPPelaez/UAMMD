/*Raul P. Pelaez 2017-2022. ParticleGroup

  ParticleGroup allows to track a certain subset of the particles in the system.

  Usage:

Create a group:

  //Different selectors offer different criteria
  //In this case, it will result in a group with particles whose ID lies between 4 and 8

  particle_selector::IDRange selector(4,8);
  auto pg = make_shared<ParticleGroup>(selector, pd, name);

  OR

  auto pg = make_shared<ParticleGroup>(pd, name); //All particles in system

  OR

  auto pg = make_shared<ParticleGroup>(ids.begin(), ids.end(), pd, name); //From a container with a list of particle IDs

........................

You can request the current indices of the particles in a group with:

  pg->getIndexIterator(access::location::gpu); //or cpu, (RECOMMENDED)

//Or get a plain array with the indices directly, if it exists.

  pg->getIndicesRawPtr(access::location::gpu); //or cpu, it will be nullptr if all particles are in the group


//You can also request an iterator that will read a ParticleData array using the group indices directly.
//Optionally a cub load modifier can be passed, which will cause the iterator to be read through a certain GPU cache
//This allows to iterate over the particles in a group without worrying about groups, particle sorting, etc...

  auto pos = pd->getPos(access::location::gpu, access::mode::read);
  auto posIter = pg->getPropertyIterator<cub::LOAD_LDG>(pos.begin());

*/

#ifndef PARTICLEGROUP_CUH
#define PARTICLEGROUP_CUH

#include"System/System.h"
#include"ParticleData.cuh"
#include<thrust/iterator/constant_iterator.h>
#include<thrust/device_vector.h>
#include<vector>
#include<third_party/uammd_cub.cuh>
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
    thrust::device_vector<int> myParticlesIndexMaskGPU;
    thrust::host_vector<int>   myParticlesIndexMaskCPU;

    bool updateHostVector = true;
    bool needsIndexListUpdate = false;
    bool updateMask = true;

    //number of particles in group and in all system (pd)
    int numberParticles, totalParticles;

    bool allParticlesInGroup = false;

    std::string name;

    connection reorderConnection;

    //Update index list if needed
    void computeIndexList(bool forceUpdate = false);

  public:
    //Defaults to all particles in group
    ParticleGroup(std::shared_ptr<ParticleData> pd, std::string name = std::string("noName"));

    //Create the group from a selector
    template<class ParticleSelector>
    ParticleGroup(ParticleSelector selector,
		  std::shared_ptr<ParticleData> pd,
		  std::string name = std::string("noName"));

    //Create the group from a list of particle IDs
    template<class InputIterator>
    ParticleGroup(InputIterator begin, InputIterator end,
		  std::shared_ptr<ParticleData> pd,
		  std::string name = std::string("noName"));

    ~ParticleGroup(){
      sys->log<System::DEBUG>("[ParticleGroup] Group %s destroyed", name.c_str());
      CudaCheckError();
      reorderConnection.disconnect();
      if(st) CudaSafeCall(cudaStreamDestroy(st));
    }

    //Returns a reference to the internal ParticleData instance of this group
    auto getParticleData(){return this->pd;}

    //Remove all particles from the group
    void clear(){
      this->numberParticles = 0;
    }
    //Add particles to the group via an array with ids
    void addParticlesById(access::location loc, const int *ids, int N);
    //Add particles to the group via an array with the current indices of the particles in pd (faster)
    void addParticlesByCurrentIndex(access::location loc, const int *indices, int N);

    void handleReorder(){
      sys->log<System::DEBUG>("[ParticleGroup] Handling reorder signal in group %s", this->name.c_str());
      if(!allParticlesInGroup && numberParticles > 0){
	needsIndexListUpdate = true;
  updateMask = true;
      }
    }

    const int* getGroupIndexMaskRawPtr(access::location loc) {
      int* ptr;

      if(this->allParticlesInGroup and updateMask){
        myParticlesIndexMaskCPU.resize(totalParticles);
        myParticlesIndexMaskGPU.resize(totalParticles);
        std::fill(myParticlesIndexMaskCPU.begin(),myParticlesIndexMaskCPU.end(),1);

        myParticlesIndexMaskGPU  = myParticlesIndexMaskCPU;
        updateMask = false;
      }

      if(updateMask){
        myParticlesIndexMaskCPU.resize(totalParticles);
        myParticlesIndexMaskGPU.resize(totalParticles);
        std::fill(myParticlesIndexMaskCPU.begin(),myParticlesIndexMaskCPU.end(),0);

        const int* indexRaw = this->getIndicesRawPtr(access::location::cpu);

        fori(0,numberParticles){
          myParticlesIndexMaskCPU[indexRaw[i]]=1;
        }

        myParticlesIndexMaskGPU  = myParticlesIndexMaskCPU;
        updateMask = false;
      }

      switch(loc){
        case access::location::cpu:
          ptr = thrust::raw_pointer_cast(myParticlesIndexMaskCPU.data());
          break;
        case access::location::gpu:
          ptr = thrust::raw_pointer_cast(myParticlesIndexMaskGPU.data());
          break;
        default:
          ptr = nullptr;
      }
      return ptr;
    }

    //Get a raw memory pointer to the index list if it exists
    const int * getIndicesRawPtr(access::location loc){
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

    //Access the index array only if it is not a nullptr (AKA if the group does not contain all particles)
    struct IndexAccess{
      IndexAccess(const int * indices):indices(indices){}

      inline __host__ __device__ int operator()(int i) const{
	return indices?indices[i]:i;
      }

    private:
      const int * indices;
    };

    using IndexIterator = thrust::transform_iterator<IndexAccess, thrust::counting_iterator<int>>;

    template<class Iterator>
    using accessIterator = thrust::permutation_iterator<Iterator, IndexIterator>;

    //Get an iterator with the indices of particles in this group
    IndexIterator getIndexIterator(access::location loc){
      return IndexIterator(thrust::make_counting_iterator<int>(0), IndexAccess(this->getIndicesRawPtr(loc)));
    }

    //Access the mask array only if it is not a nullptr (AKA if the group does not contain all particles)
    //Otherwise always return true
    struct MaskAccess{
      MaskAccess(const int * masks):masks(masks){}

      inline __host__ __device__ bool operator()(int i) const{
	return masks[i];
      }

    private:
      const int * masks;
    };

    using MaskIterator = thrust::transform_iterator<MaskAccess, thrust::counting_iterator<int>>;

    //Get an iterator of size pd->getNumParticles() returning 1 if the particle in that index is
    // part of the group and 0 otherwise.
    //loc specifies the memory location of the underlying array
    MaskIterator getGroupIndexMask(access::location loc){
      return MaskIterator(thrust::make_counting_iterator<int>(0), MaskAccess(this->getGroupIndexMaskRawPtr(loc)));
    }

  private:

    template<class Iterator>
    auto make_permutation_iterator(Iterator it, access::location loc){
      return accessIterator<Iterator>(it, getIndexIterator(loc));
    }

  public:

    //Returns an iterator that will have size pg->getNumberParticles() and will iterate over the
    // particles in the group.
    //For example, If a group contains only the particle with id=10, passing pd->getPos(...).begin() to this function
    // will return an iterator so that iterator[0] = pos[10]; and it will take into account any possible reordering of the pos array.
    template<class Iterator>
    auto getPropertyIterator(Iterator iter, access::location loc){
      return this->make_permutation_iterator(iter, loc);
    }

    //Overload when using a ParticleData::Property
    template<class Iterator>
    auto getPropertyIterator(Iterator property){
      // using value_type = typename std::iterator_traits<Iterator>::element_type;
      // static_assert(std::is_same<Iterator, typename Property<value_type>::iterator>::value,
      // 	       "You must specify a location or call this function with a Property::iterator argument");
      auto loc = property.location();
      if(loc == access::location::managed){
	loc = access::location::cpu;
      }
      return this->getPropertyIterator(property.raw(), loc);
    }

    int getNumberParticles(){
      return this->numberParticles;
    }

    std::string getName(){ return this->name;}
  };

  template<class ParticleSelector>
  ParticleGroup::ParticleGroup(ParticleSelector selector,
			       std::shared_ptr<ParticleData> pd, std::string name):
    pd(pd), sys(pd->getSystem()), name(name){
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
			       std::shared_ptr<ParticleData> pd,
			       std::string name):
    pd(pd), sys(pd->getSystem()), name(name){
    sys->log<System::MESSAGE>("[ParticleGroup] Group %s created with All selector",name.c_str());
    this->allParticlesInGroup = true;
    this->totalParticles = pd->getNumParticles();
    this->numberParticles = totalParticles;

    sys->log<System::MESSAGE>("[ParticleGroup] Group %s contains %d particles.",
			      name.c_str(), numberParticles);
  }

  ParticleGroup::ParticleGroup(std::shared_ptr<ParticleData> pd,
			       std::string name):
    ParticleGroup(particle_selector::All(), pd, name){}

  //Specialization of an empty particle group
  template<>
  ParticleGroup::ParticleGroup(particle_selector::None selector,
			       std::shared_ptr<ParticleData> pd,
			       std::string name):
    pd(pd), sys(pd->getSystem()), name(name){
    this->allParticlesInGroup = false;
    this->totalParticles = pd->getNumParticles();
    this->numberParticles = 0;
    reorderConnection = pd->getReorderSignal()->connect([this](){this->handleReorder();});
  }


  //Constructor of ParticleGroup when an ID list is provided
  template<class InputIterator>
  ParticleGroup::ParticleGroup(InputIterator begin, InputIterator end,
			       std::shared_ptr<ParticleData> pd,
			       std::string name):
    pd(pd), sys(pd->getSystem()), name(name){
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
