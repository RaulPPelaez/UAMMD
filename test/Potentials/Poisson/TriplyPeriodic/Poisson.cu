/*Raul P. Pelaez 2019-2022. Poisson test
Computes the electric field between two opposite charges placed in a periodic box.
USAGE:
./poisson [gw] [L] [r]

gw: Gaussian width of the charges
L: Box size (cubic box)
r: distance between the charges

In the limit when L->inf the field between the two particles should be:
Ex =  exp(-r**2/(4.0*gw**2))/(4*pi**1.5*gw*r) - erf(r/(2.0*gw))/(4*pi*r**2);
*/
#include"uammd.cuh"
#include"Interactor/SpectralEwaldPoisson.cuh"
#include<fstream>
#include<iomanip>
using namespace uammd;
using std::make_shared;
using std::endl;

int main(int argc, char *argv[]){
  int N = 2;
  real L = std::stod(argv[1]);
  real r = std::stod(argv[2]);
  real tolerance = std::stod(argv[3]);
  real gw = std::stod(argv[4]);
  real split = std::stod(argv[5]);
  auto sys = make_shared<System>(argc, argv);
  auto pd = make_shared<ParticleData>(N, sys);
  Box box(L);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto charge = pd->getCharge(access::location::cpu, access::mode::write);
    auto ori = make_real4(make_real3(sys->rng().uniform3(-0.5, 0.5))*L, 0);
    pos[0] = make_real4(-r*0.5,0,0,0)+ori;
    pos[1] = make_real4( r*0.5,0,0,0)+ori;
    charge[0] = 1;
    charge[1] = -1;
  }
  Poisson::Parameters par;
  par.box = box;
  par.epsilon = 1;
  par.gw = gw;
  par.tolerance = tolerance;
  par.split = split;
  //par.upsampling = 1.0;
  auto poisson = make_shared<Poisson>(pd, par);
  {
    auto force = pd->getForce(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
    auto energy = pd->getEnergy(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par, energy.begin(), energy.end(), real());
  }
  poisson->sum({.force=true, .energy=true, .virial=false});
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    auto force = pd->getForce(access::location::cpu, access::mode::read);
    auto energy = pd->getEnergy(access::location::cpu, access::mode::read);
    auto charge = pd->getCharge(access::location::cpu, access::mode::read);
    real3 p;
    fori(0,N){
      real4 pc = pos[i];
      p = make_real3(pc);
      int type = charge[i];
      std::cout<<std::setprecision(15)<<p<<" q: "<<charge[i]<<" F: "<<force[i]<<" E: "<<energy[i]<<endl;
    }
  }
  sys->finish();
  return 0;
}
