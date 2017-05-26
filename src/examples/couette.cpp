/* Raul P.Pelaez 2017- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration and constructs the simulation.


   This is a simulation of a couette flow using Positively Split Ewald BDHI method.

   Two walls are put on the top and the center of the box, the rest of the box is randomly filled
     with tracer particles. The walls are held together with Elastic network model.

   The central wall is forced to move in the x direction, while the top one is forced to move on the oposite, draggin the tracer particles and creating a coulette flow.

 */

#include "SimulationConfig.h"
#include<random>
#include<iomanip>
#include<string.h>
using namespace std;

/*External forces*/
struct ExtTor{

  ExtTor(int Nx,int Ny, real kwall, real3 L):Nx(Nx), Ny(Ny), kwall(kwall), L(L){}
  inline __device__ real4 operator()(real4 pos, int i){

    /*First wall is fixed to the top and forced to move left*/
    if(i<2*Nx*Ny){      
      real h = (pos.z-L.z/2.0f);
      h -= floorf(h/L.z+0.5f)*L.z;	
      return make_real4(-0.3f, 0, -kwall*(h), 0);
    }
    /*Second wall is fixed to the center and forced to move right*/
    else if(i>2*Nx*Ny && i<4*Nx*Ny){
      return make_real4(0.3f, 0, -kwall*(pos.z), 0);
    }
    /*Tracer particles have no force*/
    else{
      return make_real4(0,0, 0, 0);
    }
    
  }
  int Nx, Ny;
  real kwall;
  real3 L;
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
 
  real K_central = 70.0;
  real kwall = 10.0;



  
  Vector4 wall = initLattice(make_real3(Nx,Ny,0), Nx*Ny, tri);
  /*Top and bottom walls*/
  fori(0, Nx*Ny) pos[i] = wall[i] + make_real4(0, 0, -gcnf.L.z/2.0+0.5, 0);
  fori(0, Nx*Ny) pos[i+Nx*Ny] = wall[i] + make_real4(0, 0, gcnf.L.z/2.0-0.5, 0);
  /*Central walls*/
  fori(0, Nx*Ny) pos[i+Nx*Ny*2] = wall[i] - make_real4(0, 0, 0.5, 0);
  fori(0, Nx*Ny) pos[i+Nx*Ny*3] = wall[i] + make_real4(0, 0, 0.5, 0);

  wall.freeMem();

  ofstream out("central.walls");
  out<<Nx*Ny*4<<endl;
  fori(0, 4*Nx*Ny){
    real3 pi = make_real3(pos[i]);
    out<<pi<<" 0"<<endl;
  }
  out.close();
  int pid = system(("../tools/Elastic_Network_Model central.walls 1.5 "+to_string(K_central)+" "
		    +to_string(gcnf.L.x)+" "+to_string(gcnf.L.y)+
		    " -1 | awk '{if(NF==1) print; else print $1"
		    + ", $2" +
		    ", $3,$4}' > central.bonds").c_str());

  pid = system("cat central.bonds > bonds.dat; rm -f central.bonds central.walls");

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

  
  //integrator = make_shared<VerletNVT>();
  integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(K, vis, rh, PSE);

  // auto lj = make_shared<PairForces<CellList, Potential::LJ>>(powf(2,1/6.));
  // fori(0,5)
  //   forj(0,i+1)
  //   lj->setPotParams(i,j,
  // 		     {1.0f /*epsilon*/, 1.0f /*sigma*/, powf(2,1/6.) /*rcut*/, true/*shift?*/});
  // integrator->addInteractor(lj);

  auto bonds = make_shared<BondedForces<BondedType::HarmonicPBC>>("bonds.dat");
  integrator->addInteractor(bonds);

  auto ext = make_shared<ExternalForces<ExtTor>>(ExtTor(Nx, Ny, kwall, gcnf.L));
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
