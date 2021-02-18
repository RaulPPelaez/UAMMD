/*Raul P. Pelaez 2020. Doubly periodic Poisson example.
Computes the electric field between two opposite charges placed in a doubly periodic box (periodic in XY, open in Z).
See the wiki page for more information on the Doubly Periodic Poisson module.
USAGE:
./poisson [L] [r] [gw]

gw: Gaussian width of the charges
L: Box size (cubic box)
r: distance between the charges

In the limit when L->inf the field between the two particles should be:
Ex =  exp(-r**2/(4.0*gw**2))/(4*pi**1.5*gw*r) - erf(r/(2.0*gw))/(4*pi*r**2);

*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
#include"Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include<fstream>

using namespace uammd;
using std::make_shared;
using std::endl;


DPPoissonSlab::Parameters createDPPoissonSlabParameters(real Lxy, real gw){  
  DPPoissonSlab::Parameters par;    
  par.Lxy = make_real2(Lxy);
  par.H = 50;
  DPPoissonSlab::Permitivity perm;
  perm.inside = 1;
  perm.top = 1;
  perm.bottom = 1;
  par.permitivity = perm;
  par.gw = gw;
  par.split = gw*0.1;
  return par;
}

int main(int argc, char *argv[]){
  int N = 2;
  real gw = std::stod(argv[3]);
  real L = std::stod(argv[1]);
  real r = std::stod(argv[2]);
  auto sys = make_shared<System>(argc, argv);
  auto pd = make_shared<ParticleData>(N, sys);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto charge = pd->getCharge(access::location::cpu, access::mode::write);
    pos[0] = make_real4(-r*0.5,0,0,0);
    pos[1] = make_real4( r*0.5,0,0,0);
    charge[0] = 1;
    charge[1] = -1;
  }
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  auto par = createDPPoissonSlabParameters(L, gw);
  auto poisson = make_shared<DPPoissonSlab>(pd, pg, sys, par);
  {
    auto force = pd->getForce(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
  }
  poisson->sumForce(0);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    auto force = pd->getForce(access::location::cpu, access::mode::read);
    auto charge = pd->getCharge(access::location::cpu, access::mode::read);
    real3 p;
    fori(0,N){
      real4 pc = pos[i];
      p = make_real3(pc);
      int type = charge[i];
      std::cout<<std::setprecision(15)<<p<<" q: "<<charge[i]<<" F: "<<force[i]<<endl;
    }
  }
  sys->finish();
  return 0;
}


