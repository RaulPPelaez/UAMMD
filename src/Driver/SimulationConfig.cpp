/* Raul P.Pelaez 2016- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration, constructs and runs the simulation.   


   Currently configured to create a Brownian Dynamics with hydrodynamic interactions using the Positively Split Edwald method.


   This input file implements a bond type called ChemicalBond and makes a BondedForces interactor out of it. 
   It also implements an ExternalForces functor that puts all particles in an harmonic trap according to their z position.

   Although this functionality is not commented out and a simulation with ideal particles is created.

   For a simpler input file example see WCA.cpp.

References:
   [1] https://github.com/RaulPPelaez/UAMMD/wiki

*/






#include "SimulationConfig.h"


/*Random generators for the Chemical Bonds*/
curandState_t *rngst_tmp = nullptr, *curngst_cpu = nullptr;
__global__ void initcurng(curandState_t *cust, ullint seed, int N){
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id>N) return;
  curand_init(seed, id, 0, &cust[id]);

}

/*A bond that can activate or deactivate according to a random number*/
struct ChemicalBond{
  struct BondInfo{
    real r0, k, range;
    bool active;
    curandState_t curng;
  };
  ChemicalBond(int nbonds):nbonds(nbonds){
    cudaMalloc(&rngst_tmp, nbonds*sizeof(curandState_t));
    curngst_cpu = new curandState_t[nbonds];
    initcurng<<<nbonds/128+1,128>>>(rngst_tmp, gcnf.seed, nbonds);

    cudaMemcpy(curngst_cpu, rngst_tmp, nbonds*sizeof(curandState_t), cudaMemcpyDeviceToHost);
    cudaFree(rngst_tmp);
      
  }
  inline __device__ real force(int i, int j, const real3 &r12, BondInfo &bi){
    real r2 = dot(r12, r12);
    float Z = curand_uniform(&bi.curng);
          
    if(bi.active){
      if(Z>0.999f) bi.active = false;
      //if(r2>bi.range*bi.range){return 0;}	
      real invr = rsqrtf(r2);
      real f = -bi.k*(real(1.0)-bi.r0*invr); //F = -k·(r-r0)·rvec/r
      return f;
      
    }
    else{
      if(Z>0.999f) bi.active = true;
      return 0;
    }

  }
        
  static __host__ BondInfo readBond(std::istream &in){
    static int ibond =0;
    BondInfo bi;
    in>>bi.k>>bi.r0;
    bi.range = 2.0;
    bi.active = true;

    bi.curng = curngst_cpu[ibond];
      
    ibond++;
    return bi;
  }
  int nbonds;
};



/*This struct is passed to ExternalForces and its () operator is called for every particle*/
struct ExtTor{
  inline  __device__ real4 operator()(const real4 &pos, int i){
    return make_real4(0,0,-10.0f*(pos.z-20.0f),0);
  }

};

SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){    
  Timer tim; tim.tic();

  gcnf.T = 1.0;
  
  gcnf.L = make_real3(64);
  
  gcnf.N = pow(2, 14);
  gcnf.dt = 0.01;

  gcnf.rcut = 2.5;
   
  gcnf.nsteps1 = 1000000;
  gcnf.print_steps = 500;
  gcnf.measure_steps = -1;

  
  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);  
  
  //pos = initLattice(gcnf.L, gcnf.N, sc);

  pos = Vector4(gcnf.N);
  fori(0,gcnf.N){
    pos[i] = make_real4( grng.uniform3(-gcnf.L.x*0.5, gcnf.L.x*0.5), 1);

  }
  
    
  setParameters();
  /*Upload positions to GPU once the initial conditions are set*/
  pos.upload();
  

  Matrixf K(3,3); //Shear matrix
  K.fill_with(0.0);
  
  real vis = 1.0;//Viscosity
  real rh = 1.0; //Hydrodynamic radius
  
  integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(K, vis, rh, PSE);
  //integrator = make_shared<VerletNVT>();
  
   // ifstream in("kk.2bonds");
   // int nbonds;
   // in>>nbonds;
   // in.close();
   // ChemicalBond cb(nbonds);
   // /*Short range forces, using LJ potential*/
   //auto interactor = make_shared<PairForces<CellList, Potential::LJ>>();
   // interactor->setPotParams(0, 0,
   // 			   {1/*epsilon*/, 1/*sigma*/, 2.5f/*rcut*/, 0/*shift?*/});

  //auto interactor = make_shared<BondedForces<ChemicalBond>>("kk.2bonds");
   
  //auto interactor2 = make_shared<ExternalForces<ExtTor>>(ExtTor());

  
  //integrator->addInteractor(interactor);
  //integrator->addInteractor(interactor2);


  //measurables.push_back(make_shared<EnergyMeasure>(integrator->getInteractors(), integrator)); 

  
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  
  tim.tic();

  run(5000, true);  
  run(gcnf.nsteps1);  

  
  /*********************************End of simulation*****************************/
  double total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;  

}
/*Change the output format here, in this function, the updated positions will be available 
 in CPU. Writing is done in parallel.
*/
void Writer::write_concurrent(){
  real3 L = gcnf.L;
  uint N = gcnf.N;
  real4 *posdata = pos.data;
  cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
  fori(0,N){    
    uint type = gcnf.color2name[posdata[i].w];
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<0.5<<" "<<type<<"\n";
  }
  cout<<flush;
}
