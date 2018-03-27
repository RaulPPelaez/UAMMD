/* Raul P. Pelaez 2018, HydroGrid [1] analyzer UAMMD wrapper 


   HydroGrid computes static and dynamic structure factors of a 3D particle simulation (2D if cellDim.z = 1).
   It can also output vtk files with the concentration of the system.

   This class wraps the calculateConcentration function in HydroGrid, transforming the UAMMD data format to the one HG (HydroGrid) uses.


   HG is configured in an input file called hydroGridOptions.nml, see [1].

   HG can label some particles in the system as "green", being the rest "red" in order to compute cross spectra.

USAGE:

  Assuming there is a vlaid hydroGridOptions.nml file in the current folder:


  HydroGrid::Parameters par;
  par.box = Box(make_real3(Lx, Ly, 0));         //Simulation box
  par.cellDim = make_int3(ncellsx, ncellsy, 1);    //number of cells to perform the analysis on
  par.dt = dt;                                  //Time between two steps
  par.outputName = "run";                       //Prefix name of HG output files
  par.fistGreenParticle = 0;                    //Index of first green particle (default: 0)
  par.lastGreenParticle = numberParticles/2;    //Last green particle (default: -1, means all particles)

  HydroGrid hg(pd, sys, parameters); //pd-> ParticleData instance, sys-> System instance
  hg.init();
  ...
  //Simulation loop
  ...
  hg.update(step); //Feed HG with data from step "step".
  ...
  hg.write(step);  //Write current results to disk
  ...
  //End of the simulation

  //HydroGrid will free memory once the object is destroyed
 

COMPILATION:

You will need to link agains libCallHydroGrid.so, located at $(HYDROGRID_ROOT)/src/libCallHydroGrid.so
See examples/hydroGridAnalysis.cu and the corresponding Makefile entrance "hg".

References:

[1] https://github.com/stochasticHydroTools/HydroGrid
*/


#include "System/System.h"
#include "ParticleData/ParticleData.cuh"
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include<fstream>
#include<cstring>

// #ifndef HYDROGRID_SRC
// #error "HYDROGRID_SRC must be defined if HydroGrid.cuh is included!"
// #endif

// #define HG_HEADER HydroGrid.h

// #define HG_INCLUDE_FILE <HYDROGRID_SRC##HG_HEADER>

// extern "C" {
// #include HG_INCLUDE_FILE
// }

// #undef HG_HEADER
// #undef HG_INCLUDE_FILE
extern "C" {
#include"HydroGrid.h"
}
namespace uammd{

  class HydroGrid{
  protected:
    shared_ptr<ParticleData> pd;
    shared_ptr<System> sys;
    std::string name;
    std::vector<double> concentration, density, velocity;
    std::vector<double3> posOld;
    //int step;
    double velDimension; //Number of velocity coordinates 
  public:

    struct Parameters{
      Box box;                      //Simulation box
      int3 cellDim;                 //Number of cells for the analysis
      std::string outputName;       //Name prefix of HG output files
      double dt;                    //Simulation time between calls to HG
      int firstGreenParticle = 0;   //Index of first green particle
      int lastGreenParticle  = -1;  //-1 means all particles are green, default behavior
    } par;  
    HydroGrid(shared_ptr<ParticleData> pd,

	      shared_ptr<System> sys,
	      Parameters par,
	      std::string name="noName"):
      pd(pd), sys(sys), par(par), name(name){
      sys->log<System::MESSAGE>("[HydroGrid] Created.");
      //step = 0;

    }
    void init(){
      sys->log<System::MESSAGE>("[HydroGrid] Initializing.");
      sys->log<System::MESSAGE>("[HydroGrid] Box: %f %f %f",par.box.boxSize.x, par.box.boxSize.y, par.box.boxSize.z);
      sys->log<System::MESSAGE>("[HydroGrid] cells: %d %d %d", par.cellDim.x, par.cellDim.y, par.cellDim.z);
      sys->log<System::MESSAGE>("[HydroGrid] dt: %f ", par.dt);
      sys->log<System::MESSAGE>("[HydroGrid] N: %d ", pd->getNumParticles());

      std::ifstream fileinput ("hydroGridOptions.nml");
      std::string word, wordfile;
      while(!fileinput.eof()){
	getline(fileinput,word);
	wordfile += word + "\n";
      }
      fileinput.close();
      std::string fileOutName = par.outputName + ".hydroGridOptions.nml";
      std::ofstream fileout(fileOutName.c_str());
      fileout << wordfile << std::endl;
      fileout.close();

      velDimension = (par.cellDim.z>1)?3:2;
      double3 boxSize = make_double3(par.box.boxSize);
      if(par.cellDim.z<=1) boxSize.z = 1;
      createHydroAnalysis_C((int*)&par.cellDim.x,
			    3 /*nSpecies*/,
			    velDimension /*nVelocityDimensions*/,
			    1 /*isSingleFluid*/,
			    (double*)&boxSize,
			    NULL /*heatCapacity*/,
			    par.dt /*time step*/,
			    0 /*nPassiveScalars*/,
			    1 /*structFactMultiplier*/,
			    1 /*project2D*/);

      
    }
    void update(int step){
      sys->log<System::DEBUG>("[HydroGrid] Update.");
      int ncells = par.cellDim.x*par.cellDim.y*par.cellDim.z;
      
      density.resize(ncells);
      velocity.resize(velDimension*ncells);
      concentration.resize(2*ncells); //Red and green particles

      //Compute concentration      
      int numberParticles = pd->getNumParticles();
      if(par.lastGreenParticle == -1) par.lastGreenParticle = numberParticles;

      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

      
      particlesToGrid();
      
      bool firstUpdate = posOld.size() == 0;
      if(firstUpdate) posOld.resize(numberParticles, make_double3(0));
      
      fori(0, numberParticles){
	posOld[i] = make_double3(pos.raw()[sortedIndex[i]]);
      }


      updateHydroAnalysisMixture_C(velocity.data(), density.data(), concentration.data());
      
      //This fixes the first step not having the velocity
      if(firstUpdate){
	resetHydroAnalysis_C(); // Write to files
      }
      
    }
    void write(int step){
      sys->log<System::DEBUG>("[HydroGrid] Writing.");
      writeToFiles_C(step);
    }
    ~HydroGrid(){
      sys->log<System::DEBUG>("[HydroGrid] Destroying.");
      writeToFiles_C(-1);
      destroyHydroAnalysis_C();      
    }

    //Spread particle properties to the grid
    void particlesToGrid(){
      int N = pd->getNumParticles();
      
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      Grid grid(par.box, par.cellDim);
      int ncells = par.cellDim.x*par.cellDim.y*par.cellDim.z;      

      std::fill(concentration.begin(), concentration.end(), 0);
      std::fill(density.begin(), density.end(), 0);
      std::fill(velocity.begin(), velocity.end(), 0);

      double invCellVolume = grid.invCellSize.x*grid.invCellSize.y;
      if(grid.cellDim.z > 1)
	invCellVolume *= grid.invCellSize.z;
      
      fori(0,N){
	double3 vi;
	double3 pi = make_double3(pos.raw()[sortedIndex[i]]);
	if(posOld.data())
	  vi = (pi-posOld[i])/sqrt(par.dt);
	else
	  vi = make_double3(0);

	//My particle's cell index in the grid
	int icell = grid.getCellIndex(grid.getCell(pi));
	bool isGreen = i>=par.firstGreenParticle && i<=par.lastGreenParticle;
	concentration[icell + ncells*(isGreen?0:1)] += invCellVolume;
	density[icell] += invCellVolume;

	velocity[           icell] += invCellVolume*vi.x;
	velocity[  ncells + icell] += invCellVolume*vi.y;
	if(velDimension==3)
	  velocity[2*ncells + icell] += invCellVolume*vi.z;
      }

    }


  };

}
