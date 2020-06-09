/*Raul P. Pelaez 2018-2020. Adapted from Pablo Ibañez Freire's MonteCarlo Anderson code.

  This module implements Anderson's Monte Carlo NVT GPU algorithm [1].
  The algorithm is presented for hard spheres in two dimensions, but suggests it should be valid for any interparticle potential or dimensionality.

  Works by partitioning the system into a checkerboard like pattern in which the cells of a given subgrid are separated a distance greater or equal than the cut off distance of the potential.
  This allows for the cells of a subgrid to be updated independently. In 3D there are 8 subgrids (4 in 2D).
  Each one of the 8 subgrids is processed sequentially.

  The algorithm can be summarized as follows:


  1- The checkerboard is placed with a random origin
  2- For each subgrid perform a certain prefixed number of trials on the particles inside each cell

Certain details must be taken into account to ensure detailed-balance:
  1- Any origin in the simulation box must be equally probable
  2- The different subgrids must be processed in a random order
  3- The particles in each cell must be processed in a random order*
  4- Any particle attempt to leave the cell is directly rejected


  * We believe that detailed balance is mantained even if the particles are selected randomly (as opposed to traverse a random permutation of the particle list)
References:

[1] Massively parallel Monte Carlo for many-particle simulations on GPUs. Joshua A. Anderson et. al. https://arxiv.org/pdf/1211.1646.pdf
*/
#ifndef MC_NVT_CUH
#define MC_NVT_CUH

#include"uammd.cuh"

#include"Interactor/NeighbourList/CellList/CellListBase.cuh"
#include"Integrator/Integrator.cuh"
#include"utils/Box.cuh"

namespace uammd {

  namespace MC_NVT{
    template<class Pot>
    class Anderson: public Integrator{
    public:
      struct Parameters{
	Box box;                          //Box to work on
	real temperature = -1;	                  //System temperature
	int triesPerCell = 10;          //Each cell will attempt attemptsPerCell movements
	real initialJumpSize = 1.0;	  //Starting size of the random position displacement
	real acceptanceRatio = 0.5;//The parameters will be auto tuned to reach this acceptance ratio
	int tuneSteps = 10;	          //Check the acceptance ratio and tune parameters every this steps during thermalization
	int seed = 0;                  //0 means draw a random number from the system generator
      };

      Anderson(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       shared_ptr<Pot> pot,
	       Parameters par);

      ~Anderson(){
	cudaStreamDestroy(st);
      }

      void updateSimulationBox(Box box);

      virtual void forwardTime() override;

      virtual real sumEnergy() override;

      real getCurrentStepSize(){
	return jumpSize;
      }

      real getCurrentAcceptanceRatio(){
	return currentAcceptanceRatio;
      }

    private:

      shared_ptr<Pot> pot;
      shared_ptr<CellListBase> cl;
      Parameters par;
      bool is2D;
      int steps;
      int seed;
      Grid grid;
      real3 currentOrigin;
      real maxOriginDisplacement;
      real jumpSize;
      real currentAcceptanceRatio;
      thrust::device_vector<uint> triedChanges;
      thrust::device_vector<uint> acceptedChanges;
      thrust::device_vector<real4> sortPos;
      int totalTries, totalChanges;
      cudaStream_t st;
      const std::array<int3,8> offset3D{{{0,0,0},
	                                 {1,0,0},
	        		         {0,1,0},
	        		         {1,1,0},
	        		         {0,0,1},
	        		         {1,0,1},
	        		         {0,1,1},
	        			 {1,1,1}}};
      void updateListWithCurrentOrigin();
      void updateOrigin();
      void storeCurrentSortedPositions();
      void performStep();
      void updateParticlesInSubgrid(int3 subgrid);
      void updateGlobalPositions(){
	int numberParticles = pg->getNumberParticles();
	auto pos = pd->getPos(access::location::gpu, access::mode::write);
	auto posGroup = pg->getPropertyIterator(pos);
	auto clData = cl->getCellList();
	auto posGroup_tr = thrust::make_permutation_iterator(posGroup, clData.groupIndex);
	thrust::copy(thrust::cuda::par, sortPos.begin(), sortPos.end(), posGroup_tr);
      }
      void updateAcceptanceRatio();
      void updateJumpSize();
      void resetAcceptanceCounters();
    };

  }
}
#include"Anderson.cu"

#endif

