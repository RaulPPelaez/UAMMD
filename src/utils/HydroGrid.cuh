/* Raul P. Pelaez 2018, HydroGrid [1] analyzer UAMMD wrapper 


   HydroGrid computes static and dynamic structure factors of a 2D particle simulation.
   It can also output vtk files with the concentration of the system.

   This class wraps the calculateConcentration function in HydroGrid, transforming the UAMMD data format to the one HG (HydroGrid) uses.


   HG is configured in an input file called hydroGridOptions.nml, see [1].

   HG can label some particles in the system as "green", being the rest "red" in order to compute cross spectra.

USAGE:

  Assuming there is a vlaid hydroGridOptions.nml file in the current folder:


  HydroGrid::Parameters par;
  par.box = Box(make_real3(Lx, Ly, 0));         //Simulation box
  par.cellDim = make_int2(ncellsx, ncellsy);    //number of cells to perform the analysis on
  par.dt = dt*sampleHydrogridSteps;             //Time between calls to hg.update()
  par.outputName = "run";                       //Prefix name of HG output files
  par.fistGreenParticle = 0;                    //Index of first green particle (default: 0)
  par.lastGreenParticle = numberParticles/2;    //Last green particle (default: -1, means all particles)

  HydroGrid hg(pd, sys, parameters); //pd-> ParticleData instance, sys-> System instance
  hg.init();
  ...
  //Simulation loop
  ...
  hg.update(); //each sampleHydroGridSteps
  ...
  hg.write();  //Write current results to disk
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



void calculateConcentration(std::string outputname,
			      double lx, // Domain x length
			      double ly, // Domain y length
			      int green_start, // Start of "green" particles
			      int green_end, // End of "green" particles
			      int mx, // Grid size x
			      int my, // Grid size y
			      int step, // Step of simulation
			      double dt, // Time interval between successive snapshots (calls to updateHydroGrid)
			      int np, // Number of particles
			      int option, // option = 0 (initialize), 1 (update), 2 (save), 3 (save+finalize), 4 (finalize only),
			      double *x_array, double *y_array);

namespace uammd{

  class HydroGrid{
  protected:
    shared_ptr<ParticleData> pd;
    shared_ptr<System> sys;
    std::string name;
    std::vector<double> rx, ry;
    int step;
  public:

    struct Parameters{
      Box box;                      //Simulation box
      int2 cellDim;                 //Number of cells for the analysis
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
      step = 0;

    }


    void callCalculateConcentration(int option){
      int numberParticles = pd->getNumParticles();
      if(par.lastGreenParticle == -1) par.lastGreenParticle = numberParticles;
      rx.resize(numberParticles);
      ry.resize(numberParticles);
      
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

      fori(0, numberParticles){
	real4 pi = pos.raw()[sortedIndex[i]];
	rx[i] = pi.x;
	ry[i] = pi.y;
      }

      calculateConcentration(par.outputName,
			     par.box.boxSize.x,    // Domain x length
			     par.box.boxSize.y,    // Domain y length
			     par.firstGreenParticle,  
			     par.lastGreenParticle,
			     par.cellDim.x,          // Grid size x
			     par.cellDim.y,          // Grid size y
			     step,                 // Step of simulation
			     par.dt, // Time interval between successive snapshots (calls to updateHydroGrid)
			     numberParticles,                   // Number of particles
			     option,            // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
			     rx.data(),
			     ry.data());
    }
    void init(){
      sys->log<System::DEBUG>("[HydroGrid] Initializing.");
      sys->log<System::DEBUG>("[HydroGrid] Box: %f %f",par.box.boxSize.x, par.box.boxSize.y);
      sys->log<System::DEBUG>("[HydroGrid] cells: %d %d.", par.cellDim.x, par.cellDim.y);
      sys->log<System::DEBUG>("[HydroGrid] dt: %f ", par.dt);
      sys->log<System::DEBUG>("[HydroGrid] N: %d ", pd->getNumParticles());

      callCalculateConcentration(0);
    }
    void update(){
      step++;
      sys->log<System::DEBUG>("[HydroGrid] Update.");
      callCalculateConcentration(1);
    }
    void write(){
      sys->log<System::DEBUG>("[HydroGrid] Writing.");
      callCalculateConcentration(2);
    }
    ~HydroGrid(){
      sys->log<System::DEBUG>("[HydroGrid] Destroying.");
      callCalculateConcentration(3);
      
    }
      
    


  };

}