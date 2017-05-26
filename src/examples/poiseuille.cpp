/* Raul P.Pelaez 2017- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration and constructs the simulation.


   This is a simulation of a Poiseuille flow using Positively Split Ewald BDHI method.

   Two walls are fixed to the top and center of the box, and tracer particles are placed between them.
   An external force is applied to the tracer particles so the move in the x direction. Particles near the walls are slowed, creating a poiseuille flow.
    

 */

#include "SimulationConfig.h"
#include<random>
#include<iomanip>
#include<string.h>
using namespace std;



/*External forces*/
struct ExtTor{

  ExtTor(int Nx,int Ny):Nx(Nx), Ny(Ny){}
  inline __device__ real4 operator()(real4 pos, int i){

    /*The walls are fixed by fixed point bonds*/
    if(i<4*Nx*Ny){      
      return make_real4(0);
    }
    /*Tracers are forced left or right according to their z*/
    else{
      if(pos.z>0)
	return make_real4(0.3f, 0, 0, 0);
      else
	return make_real4(-0.3f, 0, 0, 0);
    }
    
  }
  int Nx, Ny;
};





SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
    
  Timer tim; tim.tic();
  
  /*No fluctuations*/
  gcnf.T = 0.0;
  
  gcnf.L = make_real3(100);

  gcnf.dt = 0.001;
  
  gcnf.nsteps1 = 100000;
  gcnf.nsteps2 = 0;
  gcnf.print_steps = 100;
  gcnf.measure_steps = -1;


  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);


  int Nx = int(gcnf.L.x+0.5);
  int Ny = int(gcnf.L.y+0.5);
  pos = Vector4(240e3);
 
  real K_fixedPoint = 70.0;

  
  Vector4 wall = initLattice(make_real3(Nx,Ny,0), Nx*Ny, tri);
  
  /*Top and bottom walls*/
  fori(0, Nx*Ny) pos[i] = wall[i] + make_real4(0, 0, -gcnf.L.z/2.0+0.5, 0);
  fori(0, Nx*Ny) pos[i+Nx*Ny] = wall[i] + make_real4(0, 0, gcnf.L.z/2.0-0.5, 0);
  /*Central walls*/
  fori(0, Nx*Ny) pos[i+Nx*Ny*2] = wall[i] - make_real4(0, 0, 0.5, 0);
  fori(0, Nx*Ny) pos[i+Nx*Ny*3] = wall[i] + make_real4(0, 0, 0.5, 0);
  wall.freeMem();

  
  /*Create Fixed point bonds for the walls*/
  ofstream fpbonds("bonds.dat");
  fpbonds<<0<<endl;
  fpbonds<<int(4*Nx*Ny)<<endl;
  fori(0, 4*Nx*Ny){
    fpbonds<<i<<" "<<pos[i].x<<" "<<pos[i].y<<" "<<pos[i].z<<" "<<K_fixedPoint<<" 0\n";
  }
  fpbonds.close();

  /*Place tracer particles*/
  real Lx = gcnf.L.x;
  real Ly = gcnf.L.y;
  real Lz = gcnf.L.z;
  int Nideal = pos.size()-4*Nx*Ny;
  fori(0, Nideal/2){

    pos[i+4*Nx*Ny] = make_real4(
				grng.uniform(-Lx/2, Lx/2),
				grng.uniform(-Ly/2, Ly/2),
				grng.uniform(-Lz/2+1.5, -1.5),				
				0);    
  }
  fori(Nideal/2, Nideal){

    pos[i+4*Nx*Ny] = make_real4(
				grng.uniform(-Lx/2, Lx/2),
				grng.uniform(-Ly/2, Ly/2),
				grng.uniform(1.5, Lz/2-1.5),				
				0);    
  }
    
  
  
  setParameters();

  pos.upload();

  Matrixf K(3,3);
  K.fill_with(0);
  real vis = 1.0;
  real rh = 1.0;

  
  /*Set integrator to PSE*/
  integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(K, vis, rh, PSE);

  // auto lj = make_shared<PairForces<CellList, Potential::LJ>>(powf(2,1/6.));
  // fori(0,5)
  //   forj(0,i+1)
  //   lj->setPotParams(i,j,
  // 		     {1.0f /*epsilon*/, 1.0f /*sigma*/, powf(2,1/6.) /*rcut*/, true/*shift?*/});
  // integrator->addInteractor(lj);

  /*Put BondedForces and ExternalForces modules as interactors*/
  
  auto bonds = make_shared<BondedForces<BondedType::HarmonicPBC>>("bonds.dat");
  integrator->addInteractor(bonds);

  auto ext = make_shared<ExternalForces<ExtTor>>(ExtTor(Nx, Ny));
  integrator->addInteractor(ext);
    
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  
  tim.tic();

  run(gcnf.nsteps1);  
  
  
  /*********************************End of simulation*****************************/
  double total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;  

}

void Writer::write_concurrent(){
  real3 L = gcnf.L;
  uint N = gcnf.N;
  real4 *posdata = pos.data;
  cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
  fori(0,N){    
    uint type = i>40e3?5:0;
    real r = i>40e3?0.2:0.5;
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<r<<" "<<type<<"\n";
  }
  cout<<flush;
}
