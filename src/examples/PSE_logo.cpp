/*Raul P. Pelaez 2017

  UAMMD input file example.

  Creates an initial configuration of ~400e3 particles from an image containing the word UAMMD.

  Then makes a simulation using PSE of those particles falling due to a gravity like force.

  The particles fall until they meet a potential wall, see GravityFunctor.
  
*/
#include "SimulationConfig.h"

/*This functor is passed to ExternalForces and adds to each particle a force in the z direction, each step*/
struct GravityFunctor{
  inline __device__ real4 operator()(const real4 &pos, int i){
    real4 f = make_real4(0);
    if(pos.z> 80.0f){
      f.z -= 0.2f;
    }
    f.z += 0.1f;
    return f;
    
  }
};

#include"res/logo.c"
/*Make a particle configuration from the logo image*/
std::vector<real4> spawn_uammd_logo();

SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
  Timer tim; tim.tic();

  /*Set parameters*/
  gcnf.T = 1.0;
  
  gcnf.L = make_real3(140, 140, 128);

  gcnf.dt = 0.01;


  gcnf.nsteps2 = 2000;
  gcnf.print_steps = 10;  

  /*Seed for the random number generation*/
  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);
  grng = Xorshift128plus(gcnf.seed);

  std::vector<real4> p = spawn_uammd_logo();
  pos = Vector4(gcnf.N);
  fori(0,gcnf.N){
    pos[i] = p[i];
  }
  p.clear();
  
  // ofstream out("uammd.pos");
  // out<<gcnf.N<<endl;  
  // fori(0,gcnf.N)
  //   out<<p[i].x<<" "<<p[i].y<<" "<<p[i].z<<" 0\n";

  // out.close();
  // return;
  //  pos = readFile("uammd.pos");

    
  /*Fix parameters*/
  setParameters();

  /*Upload positions to gpu*/
  pos.upload();
  
  
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;

  tim.tic();
  
  /*Change the integrator and run the simulation*/
  real rh = 1.0; /*Hydrodynamic radius*/
  real vis = 1.0;/*viscosity*/
  Matrixf K(3,3);/*Shear matrix*/
  K.fill_with(0);

  auto gravity = make_shared<ExternalForces<GravityFunctor>>(GravityFunctor());
  
  integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(K, vis, rh, PSE);  
  integrator->addInteractor(gravity);
  
  /*Run the simulation*/
  run(gcnf.nsteps2);
  
  /*********************************End of simulation*****************************/
  double total_time = tim.toc();

  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;

}
/*Change the output format here, in this function, the updated positions will be available 
 in CPU. Writing is done in parallel, while the simulation is still running.
*/
void Writer::write_concurrent(){
  real3 L = gcnf.L;
  uint N = gcnf.N;
  real4 *posdata = pos.data;
  cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
  fori(0,N){
    uint type = (uint)(posdata[i].w+0.5)+5;
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<.25f<<" "<<type<<"\n";
  }
  cout<<flush;
}

std::vector<real4> spawn_uammd_logo(){
  std::vector<real4> p;
  int w = uammd_image.width;
  int h = uammd_image.height;
  fori(0,w*h){

    if(uammd_image.pixel_data[i*4] == '\0' && grng.uniform(0,1) < 1.0){
      forj(0,20){
   	if(grng.uniform(0,1) < 0.6)
   	  //cout<<i%w<<" "<<i/w<<" "<<grng.uniform(0,20)<<" 1.5 0\n";
   	  p.push_back(make_real4(i%w + grng.uniform(-2,2),
   				 grng.uniform(-30,30),
   				 i/w + grng.uniform(-2,2), 0));	
      }

    }

  }  
  //cout<<flush;
  
  gcnf.N = p.size();

  real4 cm = make_real4(0);
  fori(0, gcnf.N){
    cm += p[i];
  }
  cm /= (double)gcnf.N;

  fori(0, gcnf.N){
    p[i] -= cm;
  }
  
  real4 max = make_real4(0);
  real4 min = make_real4(10000000);
  
  fori(0, gcnf.N){
    if(p[i].x>max.x) max.x = p[i].x;
    if(p[i].y>max.y) max.y = p[i].y;
    if(p[i].z>max.z) max.z = p[i].z;

    if(p[i].x<min.x) min.x = p[i].x;
    if(p[i].y<min.y) min.y = p[i].y;
    if(p[i].z<min.z) min.z = p[i].z;
  }

  real size = 130;
  real scale = size/(abs(min.x)+abs(max.x));
  fori(0, gcnf.N){
    p[i] *= scale;
  }
  return p;
}
