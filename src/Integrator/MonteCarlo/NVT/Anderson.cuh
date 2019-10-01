/*Raul P. Pelaez 2018. Adapted from Pablo Ibañez Freire's MonteCarlo Anderson code.

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

#include"Interactor/Potential/Potential.cuh"
#include"Interactor/Potential/PotentialUtils.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/ExternalForces.cuh"

namespace uammd {

  namespace MC_NVT{
    template<class Pot, class ExternalPot = BasicExternalPotential>
    class Anderson{
    public:
      struct Parameters{
	Box box;                          //Box to work on
	real kT = -1;	                  //System temperature
	int attempsPerCell = 10;          //Each cell will attempt attemptsPerCell movements
	real initialJumpSize = 1.0;	  //Starting size of the random position displacement
	int thermalizationSteps = 1000;	  //Number of thermalization steps
	real desiredAcceptanceRatio = 0.5;//The parameters will be auto tuned to reach this acceptance ratio
	real acceptanceRatioRate = 1.2;   //The rate of change in the acceptance ratio each time it is revised
	int tuneSteps = 10;	          //Check the acceptance ratio and tune parameters every this steps during thermalization
	ullint seed = 0;                  //0 means draw a random number from the system generator
      };
    protected:


      shared_ptr<Pot> pot;
      shared_ptr<ExternalPot> eP;
      shared_ptr<CellList> cl;

      Parameters par;

      bool is2D;  //True if box.z = 0
      int steps; //Number of steps performed since start

      ullint seed;

      Grid grid;
      real3 currentCutOff;

      real3 currentOrigin; //Current origin of the checkerboard
      real maxOriginDisplacement; //Range of the checkerboard origin

      real jumpSize;

      thrust::device_vector<uint> triedChanges;    //Number of attempts per cell
      thrust::device_vector<uint> acceptedChanges; //Number of accepted steps per cell

      //Temporal storage for GPU/CPU communication
      thrust::device_vector<char> tmpStorage;

      //Current measure of total number of tries and accepted moves
      uint totalTries, totalChanges;

      thrust::device_vector<char> cubTempStorage;

      //Positions sorted in the internal CellList order, allow for a faster traversal
      thrust::device_vector<real4> sortPos;
      cudaStream_t st;

      //The different subgrids
      const std::array<int3,8> offset3D{{{0,0,0},
	                                 {1,0,0},
	        		         {0,1,0},
	        		         {1,1,0},
	        		         {0,0,1},
	        		         {1,0,1},
	        		         {0,1,1},
	        			 {1,1,1}}};
      shared_ptr<System> sys;
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
    public:

      Anderson(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       shared_ptr<Pot> pot,
	       shared_ptr<ExternalPot> eP,
	       Parameters par);

      Anderson(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       shared_ptr<Pot> pot,
	       Parameters par): Anderson(pd, pg, sys, pot, shared_ptr<ExternalPot>(), par){}

      ~Anderson();

      void updateSimulationBox(Box box);
      void updateAccRatio();

      real computeInternalEnergy(bool resetEnergy = true);
      real computeExternalEnergy(bool resetEnergy = true);

      virtual void forwardTime();

      template<bool countTries> void step();

      uint2 getNumberTriesAndNumberAccepted();
      void resetAcceptanceCounters();

    };

  }
}
#include"Anderson.cu"

#endif

