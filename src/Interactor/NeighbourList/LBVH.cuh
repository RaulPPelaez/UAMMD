/* Raul P. Pelaez 2019. Linear Bounding Volume Hierarchy neighbour list.

   This neighbour list works by partitioning the system by particles instead of partitioning space (as CellList does). This means that it handles better systems with great cut off or density disparities.


Here is a little break down of the algorithm, although it is beautifully explained in the references.

   1-Sorts the particles by assigning a hash to each particle such that close particles of the same type tend to be close in memory(based on a Z-order curve and the types).
      Particles are sorted by type and then by Z-order hash

   2-Partitions the sorted hashes in a binary tree structure following Karra's algorithm [2].
      The special hashing ensures that a certain node exists in the tree that is the father of all particles of a certain type. And which has no leaf of a different type under it.
      This allows to construct a single tree and then find the root nodes of every type subtree. Which scales great with the number of types. As opposed to generate an entirely new tree for every type as in [1], [4].

   3-Assigns an Axis Aligned Bounding Box (AABB) to each node of the tree that joins the AABBs of the nodes below it (with the particles, AKA leaf nodes, being at the innermost level). This is done using Karra's algorithm in [2].
      The AABBs are stored in a "quantized" manner that allows to store a node in a single int4, which increases traversal time [4].
      The joining of boxes stops at the root of every type subtree.

   4-The neighbours of a given particle are found by traversing the AABB subtrees of every type.
      The traversal is done with a top-down approach, each particle starts at the root of a given subtree. If the particle overlaps the node with a given cut off, the algorithm goes to the next child node, otherwise it skips to the next node/periodic image/tree. The algorithm is detailed in [1] and [3].
      For a given particle, overlap with the 27 (in 3D) periodic images of the current subtree is computed before traversal of a tree and encoded in a single integer to reduce divergence (except the main box, which is traversed first by default) (see [4]).
      This process is repeated for every type subtree.

   References:
   [1] Efficient neighbor list calculation for molecular simulation of colloidal systems using graphics processing units. Michael P. Howard, et.al. 2014. https://doi.org/10.1016/j.cpc.2016.02.003
   [2] Maximizing Parallelism in the Construction of BVHs,Octrees, andk-d Trees, Tero Karras. 2012. https://devblogs.nvidia.com/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
   [3] Ray casting using a roped BVH with CUDA. R. Torres, et. al. 2009. https://doi.org/10.1145/1980462.1980483
pp. 95–102.
   [4] Quantized bounding volume hierarchies for neighbor search in molecular simulations on graphics processing units. Michael P. Howard, et. al. https://doi.org/10.1016/j.commatsci.2019.04.004

   TODO:
   100- Handle non-periodic boxes
   100- Handle 2D case
   100- Find a way to inform neighbour lists of cut-offs per type pair. Currently kind of hackish.
   90- NeighbourContainer is twice as slow as transverseLBVHList, this should be improved.
   80- Better type subtree heuristics (currently every type spawns a subtree)
 */
#ifndef LBVH_CUH
#define LBVH_CUH
#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"utils/ParticleSorter.cuh"

#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"utils/cxx_utils.h"

#include"utils/TransverserUtils.cuh"

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/gather.h>
#include<thrust/for_each.h>

#include<cub/cub.cuh>

#include<limits>
#include<fstream>
#include<bits/stdc++.h>
#include<thrust/iterator/permutation_iterator.h>
#include"third_party/managed_allocator.h"
namespace uammd{
  namespace LBVH_ns{

    struct alignas(int2) Node{
      int parent;
      int sibiling;
    };

    //The quantized AABB data structure developed in [4]
    struct alignas(int4) QuantizedAABB{
      int4 data;
      //Pack three 10 bit numbers in an integer
      __host__ __device__ int pack(int3 bin){ return ((bin.x)<<20) + ((bin.y) << 10) + (bin.z);}

      __host__ __device__ void setlo_c(int3 bin){data.x = pack(bin);}
      __host__ __device__ void sethi_c(int3 bin){data.y = pack(bin);}

      __host__ __device__ int& left(){ return data.z;}
      __host__ __device__ int& rope(){ return data.w;}

      __host__ __device__ void setlo(real3 low, real3 L){
    	int3 bin = make_int3( ( low/L + 0.5f)*1023.0f-0.5f  );
	bin.x = max(bin.x, 0);
	bin.y = max(bin.y, 0);
	bin.z = max(bin.z, 0);
	setlo_c(bin);
      }
      __host__ __device__ void sethi(real3 high, real3 L){
	int3 bin = make_int3( ( high/L + 0.5f)*1023.0f + 0.5f );
	bin.x = min(bin.x, 1023);
	bin.y = min(bin.y, 1023);
	bin.z = min(bin.z, 1023);

	sethi_c(bin);
      }

      __host__ __device__ int3 getlo(){return unpack((data.x));}
      __host__ __device__ int3 gethi(){return unpack((data.y));}

      //True if the minimum distance between a sphere of radius r centered at c and this box is less or equal than r.
      __device__ bool overlaps(float3 c, float r2, float3 L){
	constexpr float inc = 1.0f/1023.0f;
	const float3 lo = (make_float3(getlo())*inc-0.5f) * L;
	const float3 hi = (make_float3(gethi())*inc-0.5f) * L;

	const float3 y = make_float3(fmin(fmax(c.x,   lo.x), hi.x)-c.x,
				     fmin(fmax(c.y,   lo.y), hi.y)-c.y,
				     fmin(fmax(c.z,   lo.z), hi.z)-c.z);

	const float dmin = fma(y.x,y.x, fma(y.y,y.y, y.z*y.z)); //dot(y, y);
	return dmin <= r2;
      }

      //Join this box with another
      __host__ __device__ QuantizedAABB bubble(QuantizedAABB &other){
	auto res = *this;
	{//lower left corner
	  auto olo = this->getlo();
	  auto low = other.getlo();
	  if(olo.x < low.x) low.x = olo.x;
	  if(olo.y < low.y) low.y = olo.y;
	  if(olo.z < low.z) low.z = olo.z;
	  res.setlo_c(low);
	}
	{//upper right corner
	  auto ohi = this->gethi();
	  auto high = other.gethi();
	  if(ohi.x > high.x) high.x = ohi.x;
	  if(ohi.y > high.y) high.y = ohi.y;
	  if(ohi.z > high.z) high.z = ohi.z;
	  res.sethi_c(high);
	}
	return res;
      }
    private:
      //Unpack three 10 bit numbers from an integer.
      __host__ __device__ int3 unpack(int h){ return make_int3((h >> 20) & 0x3ffu,
							       (h >> 10) & 0x3ffu,
							       (h      ) & 0x3ffu);}


    };

    //Non quantized AABB
    struct alignas(int4) AABB{
      int2 data;
      real3 lo;
      real3 hi;
      __host__ __device__ int& left(){ return data.x;}
      __host__ __device__ int& rope(){ return data.y;}

      __host__ __device__ void setlo(real3 low, real3 L){
	lo = low;
      }
      __host__ __device__ void sethi(real3 high, real3 L){
	hi = high;
      }
      __host__ __device__ real3 getlo(){return lo;}
      __host__ __device__ real3 gethi(){return hi;}

      __device__ bool overlaps(float3 c, float r2, float3 L){

	const float3 y = make_float3(fmin(fmax(c.x,   lo.x), hi.x)-c.x,
				     fmin(fmax(c.y,   lo.y), hi.y)-c.y,
				     fmin(fmax(c.z,   lo.z), hi.z)-c.z);

	const float dmin = fma(y.x,y.x, fma(y.y,y.y, y.z*y.z)); //dot(y, y);
	return dmin <= r2;
      }
      __host__ __device__ AABB bubble(AABB &other){
	auto res = *this;
	{//lower left corner
	  auto olo = res.getlo();
	  auto low = other.getlo();
	  if(olo.x < low.x) low.x = olo.x;
	  if(olo.y < low.y) low.y = olo.y;
	  if(olo.z < low.z) low.z = olo.z;
	  res.lo = low;
	}
	{//upper right corner
	  auto ohi = res.gethi();
	  auto high = other.gethi();
	  if(ohi.x > high.x) high.x = ohi.x;
	  if(ohi.y > high.y) high.y = ohi.y;
	  if(ohi.z > high.z) high.z = ohi.z;
	  res.hi = high;
	}
	return res;
      }

    };

    class Delta{
      const uint* keys;
      const int size;
      const uint ki;
      const int i;
    public:
      __device__ __host__ Delta(const uint* keys, int size, int i): keys(keys), size(size), ki(keys[i]), i(i){ }
      __device__ int operator()(int j) const{
	if((j<0 or j>=size)) return -1;
	//const auto ki = keys[i];
	const auto kj = keys[j];
	if(ki == kj) return 32 + __clz(i^j);
	else         return __clz(ki^kj);
      }

    };


    //Given a root node, this function returns an integer encoding which of the 27 periodic images the position pib
    // overlaps.
    template<class AABB, class LBVHList>
    __device__ int treeFlags(AABB root, real rc2, LBVHList nl, real3 pib){
      int flag = 0;
      {
	const real3 L = nl.box.boxSize;
	fori(0, 27){
	  real3 origin;
	  origin.x =(i%3-1)*L.x;
	  origin.y =((i/3)%3-1)*L.y;
	  origin.z =(i/9-1)*L.z;
	  if(!(origin.x == 0 and origin.y == 0 and origin.z == 0) and root.overlaps(pib+origin, rc2, L)){
	    flag |= 1<<i;
	  }
	}
      }
      return flag;
    }


    template<class Transverser, class LBVHList>
    __global__ void transverseLBVHList(Transverser tr,
				       ParticleGroup::IndexIterator globalIndex, //Transforms a group index into a global index
				       LBVHList nl,
  				       int numberParticles){
      const int i = blockIdx.x*blockDim.x + threadIdx.x;
      if(i >= numberParticles) return;

      const auto pi = cub::ThreadLoad<cub::LOAD_LDG>(nl.sortPos + i);
      const auto pib = nl.box.apply_pbc(make_real3(pi));

      auto quantity = tr.zero();
      const int ori = globalIndex[nl.groupIndex[i]];
      SFINAE::Delegator<Transverser> del;
      del.getInfo(tr, ori);
      //For every type tree
      for(int treecount = 0; treecount<nl.Ntrees; treecount++){
	//Reset p_shifted to main box
	real3 p_shifted = pib;
	//Get cut off for current type combination
	real rc2;
	{
	  const int ti= int(pi.w+0.5);
	  const int tj = treecount;
	  rc2 = tr.getCutOff2BetweenTypes(ti, tj);
	}
	//Get periodic overlap flags
	int periodicFlags = treeFlags(nl.aabbs[nl.roots[treecount]], rc2, nl, pib);

	do{//For all overlapping periodic images
	  //Start at the root node
	  int current_node = nl.roots[treecount];
	  do{//Traverse the tree until reaching an escape (encoded as rope<0)
	    auto node = cub::ThreadLoad<cub::LOAD_LDG>(nl.aabbs + current_node);
	    const int left = node.left();
	    current_node = node.rope();
	    if(node.overlaps(p_shifted, rc2, nl.box.boxSize)){
	      if(left < 0){
		const int j = -left-numberParticles+1;
		const auto pj = cub::ThreadLoad<cub::LOAD_LDG>(nl.sortPos + j);
		const int global_index =  globalIndex[nl.groupIndex[j]];
		tr.accumulate(quantity,
			      del.compute(tr,
					  global_index,
					  pi, pj));
	      }
	      else current_node = left;
	    }
	  }while(current_node>=0);
	  //When the tree is processed, go to next image or fo to next tree if none remain.
	  if(periodicFlags != 0){
	    const int next_image = __ffs(periodicFlags)-1; //least significant non zero bit
	    const auto L = nl.box.boxSize;
	    p_shifted.x = pib.x + (next_image%3-1)*L.x;
	    p_shifted.y = pib.y + ((next_image/3)%3-1)*L.y;
	    p_shifted.z = pib.z + (next_image/9-1)*L.z;
	    periodicFlags ^= 1<<next_image; //Flip bit to 0
	  }else break;

	}while(true);
      }

      tr.set(ori, quantity);
    }


    //Count leading zeros
    int clz(uint n){
      n |= (n >>  1);
      n |= (n >>  2);
      n |= (n >>  4);
      n |= (n >>  8);
      n |= (n >> 16);
      return 32-ffs(n - (n >> 1));
    }


    struct MortonPlusTypeHash{
      uint lastMortonBit;
      MortonPlusTypeHash(Grid grid, const real4 *pos):grid(grid), pos(pos){
	//The last bit set by the morton hash, there are 32-lastMortonBit bits available to encode types.
	lastMortonBit = 32-clz(interleave3(grid.cellDim-1));
      }
      static inline __host__ __device__ uint encodeMorton(const uint &i){
	uint x = min(i,1023);
	x &= 0x3ff;
	x = (x | x << 16) & 0x30000ff;
	x = (x | x << 8)  & 0x300f00f;
	x = (x | x << 4)  & 0x30c30c3;
	x = (x | x << 2)  & 0x9249249;
	return x;
      }
      static inline __host__ __device__ uint interleave3(const int3 &cell){
	return encodeMorton(cell.x) | (encodeMorton(cell.y) << 1) | (encodeMorton(cell.z) << 2);
      }
      inline __host__ __device__ uint hash(const int3 &cell, int type) const{
	return interleave3(cell) | type << lastMortonBit;
      }

      inline __host__ __device__ uint getLastMortonBit() const{
	return lastMortonBit;
      }

      int getMaxNumberTypes(){
	return (1<<(32-lastMortonBit))-1;
      }

      inline __host__ __device__ uint operator()(int i) const{
	real4 p = pos[i];
	return hash(grid.getCell(p), int(p.w));
      }

    private:
      Grid grid;
      const real4 *pos;
    };

    //Compute the range of primitives that internal node i cover
    inline __device__ int2 findRange(int i, Delta delta_i){

      int d; //-1 if key i is the right end of the range, 1 if it is the left
      int deltamin; //The common prefix between the hashes of key i and its leaf sibiling
      {//Determine if key i is the left or right end of the range
	const int delta_ip1 = delta_i(i+1);
	const int delta_im1 = delta_i(i-1);
	d = (delta_ip1 > delta_im1)?1:-1;
	deltamin = min(delta_ip1, delta_im1);
      }
      //Compute an upper limit for the other end of the range
      int lmax = 2;
      while( delta_i(i+lmax*d) > deltamin ) lmax = lmax << 1;
      //Binary search for the other end of the range, j
      int l = 0;
      {
	int t = lmax;
	do{
	  t = t >> 1;
	  l += (delta_i(i+ (l+t)*d) > deltamin)?t:0;
	}while (t>1);
      }

      const int j = i+l*d;
      return {i, j};
    }
    //Find the index of the primitive between first and last with the first different bit.
    // Such that delta(split, split+1) = delta(first,last)
    //This is were internal node "i" is split in two
    inline __device__ int findSplit(int first, int last, Delta delta_i){
      //Length of the longest common prefix between the first and last keys in the range
      const int deltanode = delta_i(last);
      int s = 0;
      const int d = (last>first)?1:-1; //Direction of the range
      int t = (last-first)*d;    //Upper limit for the range
      //Binary search for split position
      do{
	t = (t+1) >> 1;
	s += (delta_i(first+(s+t)*d) > deltanode)?t:0;
      }while(t>1);
      //Shift according to direction
      const int split = first + s*d + min(d,0);
      return split;
    }
    //Generates the tree hierarchy and stores it in nodes and lefts. See [2]
    __global__ void genHierarchy(Node* nodes,
				 int* lefts,
				 const uint* sortedHashes,
				 int typeEncodeFirstBit,
				 int numberParticles){
      const int i = blockIdx.x*blockDim.x + threadIdx.x;
      if(i >= numberParticles-1) return;

      //For a given particle delta_i(j) computes the length of the longest common bit prefix between
      // the hashes of i and j.
      Delta delta_i(sortedHashes, numberParticles, i);
      const int2 range = findRange(i, delta_i);
      const int first = range.x;
      const int last = range.y;
      const int split = findSplit(first, last, delta_i);

      //Leaf nodes are placed after internal nodes
      const int leafOffset = numberParticles-1;
      //split is the left child node, which is a leaf if the split corresponds to the first primitive of the range.
      //Similar with the right child
      const int left = split + (split == min(first, last)?leafOffset:0);
      const int right = split + 1 + (split + 1 == max(first, last)?leafOffset:0);

      int parent = i;
      {
       	const int tf= sortedHashes[first]>>typeEncodeFirstBit;
       	const int tl= sortedHashes[last]>>typeEncodeFirstBit;
       	if(tf != tl){
	  //Nodes above a type tree are discarded labeling them with a -2 parent.
       	  parent = -2;
       	}
      }
      nodes[left] = {.parent = parent,
		     .sibiling = right};

      nodes[right] = {.parent = parent,
		      .sibiling = -left};

      lefts[i] = left;

      if(i==0){
	nodes[0] = {.parent = -1,
		    .sibiling = -1};
      }
    }

    //Find the root of every type subtree. Which are the innermost nodes with parent = -2
    __global__ void findRoots(const Node* nodes,
			      int *treeRoots,
			      const uint* sortedHashes,
			      int lastMortonBit,
			      int numberParticles
			      ){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      //Each particle ascends the tree, but only one will find its type root node.
      if(id>=numberParticles) return;

      int node = numberParticles-1 + id;
      int ti = sortedHashes[id]>>lastMortonBit;
      if(id==numberParticles-1){
	int parent = nodes[node].parent;

	int root;
	while(parent >= 0){
	  root = parent;
	  parent = nodes[root].parent;
	}
	treeRoots[ti] = root;
	return;
      }
      int tip1 = sortedHashes[id+1]>>lastMortonBit;
      if(ti == tip1) return;

      int parent = nodes[node].parent;

      int root;
      while(parent >= 0){
	root = parent;
	parent = nodes[root].parent;
      }
      treeRoots[ti] = root;
    }


    //Assign an Axis Aligned Bounding Box to each node on the tree. See [1].
    template<class AABB>
    __global__ void genAABB(const Node* nodes,
			    AABB* aabb, //Compressed aabb format, see [4].
			    float4* uncompressedBoxes,
			    int* visitCount,
			    Box box,
			    const real4 *sortPos,
			    int numberParticles){
      const int i = blockIdx.x*blockDim.x + threadIdx.x;
      if(i >= numberParticles) return;
      const int Nnodes = 2*numberParticles-1;
      const float3 L = make_float3(box.boxSize);

      //Start from a leaf node and ascend the tree
      int node = i + numberParticles - 1;

      const float3 p = box.apply_pbc(make_real3(sortPos[i]));

      {//Store aabb of leaf node
	AABB aabb_i;
	aabb_i.setlo(p, L);
	aabb_i.sethi(p, L);
	aabb_i.left() = -1;
	aabb_i.rope() = -1;
	aabb[node] = aabb_i;
      }

      uncompressedBoxes[node]   = make_float4(p);
      uncompressedBoxes[Nnodes+node] = make_float4(p);

      auto low = p;
      auto high = p;

      int parent = nodes[node].parent;
      while(parent>=0){
	{ //Only the second thread to arrive to a given node actually processes it, this
	  //ensures that the sibiling has already been processed.
	  const int node_visits = atomicAdd(visitCount+parent, 1);
	  if(node_visits != 1) break;
	}
	//Join the aabb of my sibiling with mine and store them in compressed format
	const int mergefrom = abs(nodes[node].sibiling);
	{//lower left corner
	  auto olo = make_float3(uncompressedBoxes[mergefrom]);
	  if(olo.x < low.x) low.x = olo.x;
	  if(olo.y < low.y) low.y = olo.y;
	  if(olo.z < low.z) low.z = olo.z;
	  aabb[parent].setlo(low, L);
	}
	{//upper right corner
	  auto ohi = make_float3(uncompressedBoxes[Nnodes + mergefrom]);
	  if(ohi.x > high.x) high.x = ohi.x;
	  if(ohi.y > high.y) high.y = ohi.y;
	  if(ohi.z > high.z) high.z = ohi.z;
	  aabb[parent].sethi(high, L);
	}

	//Store them uncompressed also
	uncompressedBoxes[parent] = make_float4(low);
	uncompressedBoxes[Nnodes + parent] = make_float4(high);

	//Ensure global memory writes are available to other threads
	__threadfence();

	//Go to next level
	node = parent;
	parent = nodes[node].parent;
      }

    }

    struct Bubble{
      template<class AABB>
      __device__ AABB operator()(AABB a, AABB b){
	auto res = a.bubble(b);
	return res;
      }
    };

    __device__ int nextTree(int root, const int* treeRoots, int Ntypes){
      fori(0, Ntypes){
	if(treeRoots[i] > root) return treeRoots[i];
      }
      return -1;
    }
    //Compute traversal skips for every node.
    //The rope is the next node to process after a particular branch is processed. See [3]
    template<class AABB>
    __global__ void computeRopes(const Node* nodes,
				 const int* lefts,
				 const int* treeRoots,
				 int Ntypes,
				 AABB* aabb,
				 int numberParticles){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if(i>=2*numberParticles-1) return;
      //rope=-1 means the tree is finished
      int rope = -1;
      {
	int node = i;
	while(rope < 0 and node > 0 ){
	  rope = nodes[node].sibiling;
	  if(nodes[node].parent == -2){rope = -1;break;}//-nextTree(node, treeRoots, Ntypes); break;}
	  node = nodes[node].parent;
	}
      }
      auto b = aabb[i];
      int left = lefts[i];
      if(i >= numberParticles-1) left = -i;
      b.left() = left;
      b.rope() = rope;
      aabb[i] = b;
    }


    struct CompareType{
      __host__ __device__ bool operator()(real4 pi, real4 pj){
	return pi.w<pj.w;
      }
    };


    template<class IT, class T>
    __global__ void storeValue(IT it, T* it2){
      *it2 = *it;
    }

  }

  class LBVHList{
  private:
    shared_ptr<ParticleData> pd;
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;

    //If true, the next issued neighbour list update will always result in a reconstruction of the list.
    bool force_next_update = true;
    bool rebuildNlist;

    //The box and cut off used in the current state of the cell list/neighbour list
    Box currentBox;


    connection numParticlesChangedConnection, posWriteConnection;

    void handleNumParticlesChanged(int Nnew){
      sys->log<System::DEBUG>("[LBVHList] Number particles changed signal handled.");
      force_next_update = true;
    }
    void handlePosWriteRequested(){
      sys->log<System::DEBUG1>("[LBVHList] Issuing a list update after positions were written to.");
      force_next_update = true;
    }

    using Node = LBVH_ns::Node;
    using AABB = LBVH_ns::QuantizedAABB;
    //using AABB = LBVH_ns::AABB;

    // template<class T>
    // using temporal_device_vector = thrust::device_vector<T,
    //   polymorphic_allocator<T,
    //   managed_memory_resource,
    //   thrust::cuda::pointer<T>>>;
    //using temporal_device_vector = thrust::device_vector<T>;

    thrust::device_vector<real4> sortPos;
    thrust::device_vector<AABB> aabbs;
    thrust::device_vector<int> treeRoots;

    ParticleSorter ps;
    template<class T>
    std::shared_ptr<T> allocate_temporary_vector(int numberElements){
      auto alloc = sys->getTemporaryDeviceAllocator<T>();
      return std::shared_ptr<T>(alloc.allocate(numberElements),
				[=](T* ptr){ alloc.deallocate(ptr);});

    }


    int countParticleTypes(cudaStream_t st){
      auto alloc = sys->getTemporaryDeviceAllocator<char>();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto groupPos = pg->getPropertyIterator(pos.begin(), access::location::gpu);
      int numberParticles = pg->getNumberParticles();
      auto maxElement = thrust::max_element(thrust::cuda::par(alloc).on(st),
					    groupPos, groupPos + numberParticles,
					    LBVH_ns::CompareType());

      auto temp = allocate_temporary_vector<real4>(1);
      LBVH_ns::storeValue<<<1,1,0,st>>>(maxElement, temp.get());
      real4 h_temp;
      CudaSafeCall(cudaMemcpy(&h_temp, temp.get(), sizeof(real4), cudaMemcpyDeviceToHost));
      int Ntypes = int(h_temp.w+0.5)+1;
      sys->log<System::DEBUG3>("[LBVHList] Found %d particle types", Ntypes);
      return Ntypes;
    }

    int sortParticlesWithMortonPlusTypesHash(int Ntypes, cudaStream_t st){
      int numberParticles = sortPos.size();
      //Choose a morton grid with a number of cells that allows to encode all the possible particle types
      int maxNtypes = 0;
      int maxNcells = 2048;
      int3 ncellsMorton;
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      const real maxL = std::max({currentBox.boxSize.x, currentBox.boxSize.y, currentBox.boxSize.z});
      do{
	maxNcells = maxNcells/2;
	real h = maxL/maxNcells;
	ncellsMorton = make_int3(currentBox.boxSize/h+0.5);

	Grid grid(currentBox, ncellsMorton);
	LBVH_ns::MortonPlusTypeHash hasher(grid, pos.raw());
	maxNtypes = hasher.getMaxNumberTypes()-1;
      }while(maxNtypes < Ntypes or maxNcells > 1024);

      sys->log<System::DEBUG3>("[LBVHList] Choosing (%d, %d, %d) morton cells",
			       ncellsMorton.x, ncellsMorton.y, ncellsMorton.z);

      Grid grid(currentBox, ncellsMorton);
      LBVH_ns::MortonPlusTypeHash hasher(grid, pos.raw());
      auto maxHash = hasher.hash(grid.cellDim-1, Ntypes-1);
      auto it = thrust::make_counting_iterator<int>(0);
      auto hashIterator = thrust::make_transform_iterator(it, hasher);
      int lastMortonBit = hasher.getLastMortonBit();
      ps.updateOrderWithCustomHash(hashIterator,
				   numberParticles,
				   maxHash,
				   st);

      ps.applyCurrentOrder(pos.begin(), sortPos.begin(), numberParticles, st);

      return lastMortonBit;
    }
    void generateHierarchy(Node* d_nodes, int * d_lefts, int lastMortonBit, cudaStream_t st){
      int numberParticles = sortPos.size();
      int Nthreads=128;
      int Nblocks=(numberParticles-1)/Nthreads + (((numberParticles-1)%Nthreads)?1:0);
      auto mhsh = ps.getSortedHashes();
      LBVH_ns::genHierarchy<<<Nblocks, Nthreads, 0, st>>>(d_nodes,
							  d_lefts,
							  mhsh,
							  lastMortonBit,
							  numberParticles);
      CudaCheckError();
      sys->log<System::DEBUG3>("[LBVHList] Find roots");
      LBVH_ns::findRoots<<<Nblocks, Nthreads, 0, st>>>(d_nodes,
						       thrust::raw_pointer_cast(treeRoots.data()),
						       mhsh,
						       lastMortonBit,
						       numberParticles);
      CudaCheckError();
    }
    void generateAABB(Node* d_nodes, cudaStream_t st){
      int numberParticles = sortPos.size();
      int NInternalNodes= numberParticles-1;
      int NtotalNodes = NInternalNodes + numberParticles;

      auto visitCount = allocate_temporary_vector<int>(NtotalNodes);
      fillWithGPU<<<NtotalNodes/128+1, 128, 0, st>>>(visitCount.get(), 0, NtotalNodes);
      auto d_aabbs = thrust::raw_pointer_cast(aabbs.data());
      {
	//AABB of each node in uncompressed format, real4 for faster memory access.
	auto uncompBoxes = allocate_temporary_vector<real4>(2*NtotalNodes);
	int Nthreads = 128;
	int Nblocks= NInternalNodes/Nthreads + ((NInternalNodes%Nthreads)?1:0);
	LBVH_ns::genAABB<<<Nblocks, Nthreads, 0, st>>>(d_nodes,
						       d_aabbs,
						       uncompBoxes.get(),
						       visitCount.get(),
						       currentBox,
						       thrust::raw_pointer_cast(sortPos.data()),
						       numberParticles);
	CudaCheckError();
	if(treeRoots.size()>1){
	  auto tr = thrust::make_permutation_iterator(aabbs.begin(), treeRoots.begin());
	  auto alloc = sys->getTemporaryDeviceAllocator<char>();
	  AABB root = thrust::reduce(thrust::cuda::par(alloc).on(st),
				     tr, tr+treeRoots.size(),
				     AABB(),
				     LBVH_ns::Bubble());
	  aabbs[0] = root;
	  CudaCheckError();
	}
      }
    }
  public:

    LBVHList(shared_ptr<ParticleData> pd,
	     shared_ptr<System> sys): LBVHList(pd, std::make_shared<ParticleGroup>(pd, sys), sys){
    }

    LBVHList(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys): pd(pd), pg(pg), sys(sys), ps(sys){
      sys->log<System::MESSAGE>("[LBVHList] Created");


      pd->getNumParticlesChangedSignal()->connect([this](int Nnew){this->handleNumParticlesChanged(Nnew);});
      pd->getPosWriteRequestedSignal()->connect([this](){this->handlePosWriteRequested();});

    }
    ~LBVHList(){
      sys->log<System::DEBUG>("[LBVHList] Destroyed");
      numParticlesChangedConnection.disconnect();
      posWriteConnection.disconnect();
      CudaCheckError();
    }

    //Use a transverser to transverse the tree structure directly (without constructing a neighbour list)
    template<class Transverser>
    void transverseList(Transverser &tr, cudaStream_t st = 0){
      int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG2>("[LBVHList] Transversing Cell List with %s", type_name<Transverser>().c_str());

      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);

      auto globalIndex = pg->getIndexIterator(access::location::gpu);

      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);

      auto nl = this->getLBVHList();
      LBVH_ns::transverseLBVHList<<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
									   globalIndex,
									   nl,
									   numberParticles);
      CudaCheckError();
    }

    template<class Transverser>
    void transverseListWithNeighbourList(Transverser &tr, cudaStream_t st = 0){
      int numberParticles = pg->getNumberParticles();
      sys->log<System::CRITICAL>("[LBVHList] The transverseListWithNeighbourList capability is not available yet");
      CudaCheckError();
    }

    //Check if the cell list needs updating
    bool needsRebuild(Box box, real cutOff = 0){
      pd->hintSortByHash(box, box.boxSize/1023.f);
      if(force_next_update){
	force_next_update = false;
	return true;
      }
      if(box != currentBox) return true;
      return false;

    }

    void update(Box box, real cutOff = 0, cudaStream_t st = 0){
      if(this->needsRebuild(box) == false) return;
      sys->log<System::DEBUG2>("[LBVHList] Updating list");
      currentBox = box;

      const int numberParticles = pg->getNumberParticles();

      sortPos.resize(numberParticles);

      sys->log<System::DEBUG3>("[LBVHList] Hashing and sorting");
      const int Ntypes = countParticleTypes(st);
      treeRoots.resize(Ntypes);

      int lastMortonBit = sortParticlesWithMortonPlusTypesHash(Ntypes, st);

      sys->log<System::DEBUG3>("[LBVHList] Generate hierarchy");
      //Generate tree
      const int NLeafNodes = numberParticles;
      const int NInternalNodes = NLeafNodes-1;
      const int NtotalNodes = NInternalNodes + NLeafNodes;

      auto nodes = allocate_temporary_vector<Node>(NtotalNodes);
      auto lefts = allocate_temporary_vector<int>(NtotalNodes);
      generateHierarchy(nodes.get(), lefts.get(), lastMortonBit, st);

      sys->log<System::DEBUG3>("[LBVHList] Compute AAABBs");
      //Compute AABB
      aabbs.resize(NtotalNodes);
      generateAABB(nodes.get(), st);

      sys->log<System::DEBUG3>("[LBVHList] Compute ropes");
      {//Compute traversal ropes between nodes. see [3]
	int Nthreads = 128;
	int Nblocks=(NtotalNodes)/Nthreads + (((NtotalNodes)%Nthreads)?1:0);

	LBVH_ns::computeRopes<<<Nblocks, Nthreads, 0, st>>>(nodes.get(),
							    lefts.get(),
							    thrust::raw_pointer_cast(treeRoots.data()),
							    Ntypes,
							    thrust::raw_pointer_cast(aabbs.data()),
							    numberParticles);
	CudaCheckError();

      }
      sys->log<System::DEBUG1>("[LBVHList] Tree built");
      CudaCheckError();
      rebuildNlist = true;
    }

    struct LBVHListData{
      AABB *aabbs;
      real4* sortPos;
      int *groupIndex;
      Box box;
      int numberParticles;
      int *roots;
      int Ntrees;
      //real *cutOffPerType;
    };
    LBVHListData getLBVHList(){
      sys->log<System::DEBUG2>("[LBVHList] List requested");
      this->update(currentBox, 0);
      LBVHListData cl;
      cl.aabbs = thrust::raw_pointer_cast(aabbs.data());
      cl.sortPos =  thrust::raw_pointer_cast(sortPos.data());
      cl.box = currentBox;
      cl.numberParticles = sortPos.size();
      cl.groupIndex =  ps.getSortedIndexArray(cl.numberParticles);
      cl.roots = thrust::raw_pointer_cast(treeRoots.data());
      cl.Ntrees = treeRoots.size();
      //cl.cutOffPerType = thrust::raw_pointer_cast(cutOffPerType.data());
      return cl;
    }

#if 0 //This code is just dectivated for now, it is twice as slow than transverseLBVHList and I do not know how to
    // communicate cut offs per type yet.
    class NeighbourContainer; //forward declaration for befriending
  private:
    class NeighbourIterator; //forward declaration for befriending

    //Neighbour is a small accesor for NeighbourIterator
    //Represents a particle, you can ask for its index and position
    struct Neighbour{
      __device__ Neighbour(const Neighbour &other):
      internal_i(other.internal_i){
	groupIndex = other.groupIndex;
	sortPos = other.sortPos;
      }
      //Index in the internal sorted index of the cell list
      __device__ int getInternalIndex(){return internal_i;}
      //Index in the particle group
      __device__ int getGroupIndex(){return groupIndex[internal_i];}
      __device__ real4 getPos(){return cub::ThreadLoad<cub::LOAD_LDG>(sortPos+internal_i);}

    private:
      int internal_i;
      const int* groupIndex;
      const real4* sortPos;
      friend class NeighbourIterator;
      __device__ Neighbour(int i, const int* gi, const real4* sp):
	internal_i(i), groupIndex(gi), sortPos(sp){}
    };

    //This forward iterator must be constructed by NeighbourContainer,
    // Provides a list of neighbours for a certain particle by traversing the trees, using the cut off type for eac one.
    //A neighbour is provided as a Neighbour instance

    class NeighbourIterator:
      public thrust::iterator_adaptor<
 NeighbourIterator,
   int,   Neighbour,
   thrust::any_system_tag,
   thrust::forward_device_iterator_tag,
   Neighbour,   int
   >{
      friend class thrust::iterator_core_access;
      const int i; //Particle index
      int typei;
      int j; //Current neighbour index
      const LBVHListData nl;
      int b;
      int current_node;
      real3 p_shifted;
      real3 pi;
      real current_rc2;
      int current_tree;
      __device__ void start(){
	{
	  const real4 piw = nl.sortPos[i];
	   pi = nl.box.apply_pbc(make_real3(piw));
	   typei = int(piw.w+0.5);
	   //pi = nl.sortPos[i];
	}
	current_tree = -1;
	nextTree();
	j = 0;
      }

      template<class AABB>
      __device__ int treeFlags(AABB root){
	int flag = 0;
	{
	  const real3 L = nl.box.boxSize;
	  fori(0, 27){
	    const real3 origin = make_real3(i%3-1,
				      (i/3)%3-1,
				      i/9-1
				      )*L;
	    if(i != 13
	       and root.overlaps(pi + origin, current_rc2, L)){
	      flag |= 1<<i;
	    }
	  }
	}

	return flag;
      }
      __device__ inline bool nextImage(){
	if(b != 0){
	  p_shifted = pi;
	  const int next_image = __ffs(b)-1; //least significant non zero bit
	  const auto L = nl.box.boxSize;
	  p_shifted.x = pi.x + (next_image%3-1)*L.x;
	  p_shifted.y = pi.y + ((next_image/3)%3-1)*L.y;
	  p_shifted.z = pi.z + (next_image/9-1)*L.z;
	  b ^= 1<<next_image; //Clear bit
	  current_node = nl.roots[current_tree];
	  return true;
	}
	else return nextTree();
      }
      __device__ inline bool nextTree(){
	current_tree++;
	if(current_tree < nl.Ntrees){
	  {
	    //const int typei = int(pi.w + 0.5f);
	    const real rc = nl.cutOffPerType[typei + nl.Ntrees*current_tree];
	    current_rc2 = rc*rc;
	  }
	  current_node = nl.roots[current_tree];
	  p_shifted = pi;//nl.box.apply_pbc(make_real3(pi));
	  b = treeFlags(nl.aabbs[current_node]);
	  return true;
	}
	else return false;
      }

      //Take j to the next neighbour
      __device__  void increment(){
	while(j>=0){
	  while(current_node>=0){
	    auto node = cub::ThreadLoad<cub::LOAD_LDG>(nl.aabbs + current_node);
	    const int left = node.left();
	    current_node = node.rope();
	    if(node.overlaps(p_shifted, current_rc2, nl.box.boxSize)){
	      if(left < 0){
		j = -left-nl.numberParticles+1;
		return;
	      }
	      else current_node = left;
	    }
	  }
	  j = (!nextImage())?-1:j;
	}


      }

      __device__ Neighbour dereference() const{
	return Neighbour(j, nl.groupIndex, nl.sortPos);
      }

      //Can only be advanced
      __device__  void decrement() = delete;
      __device__  Neighbour operator[](int i) = delete;

      __device__  bool equal(NeighbourIterator const& other) const{
	return other.i == i and other.j==j;
      }

    public:
      //j==-1 means there are no more neighbours and the iterator is invalidated
      __device__  operator bool(){ return j!= -1;}
    private:
      friend class NeighbourContainer;
      __device__ NeighbourIterator(int i, LBVHListData nl, bool begin):
	i(i),j(-1), nl(nl){
	if(begin) start();
      }
    };
  public:
    //This is a pseudocontainer which only purpose is to provide begin() and end() NeighbourIterators for a certain particle
    struct NeighbourContainer{
      int my_i = -1;
      LBVHListData nl;
      NeighbourContainer(LBVHListData nl): nl(nl){}
      __device__ void set(int i){this->my_i = i;}
      __device__ NeighbourIterator begin(){return NeighbourIterator(my_i, nl, true);}
      __device__ NeighbourIterator end(){  return NeighbourIterator(my_i, nl, false);}
    };

    NeighbourContainer getNeighbourContainer(){
      auto nl = getLBVHList();
      sys->log<System::DEBUG1>("[LBVHList] Neighbour container issued");
      return NeighbourContainer(nl);
    }
#endif

    const real4* getSortedPositionIterator(){
      return thrust::raw_pointer_cast(sortPos.data());
    }
    const int* getGroupIndexIterator(){
      auto nl = getLBVHList();
      return nl.groupIndex;
    }

  };

}
#endif


