/*Raul P. Pelaez 2019. Poisson
*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
#include"Interactor/IBM_Poisson.cuh"
#include<fstream>

using namespace uammd;
using std::make_shared;
using std::endl;

int main(int argc, char *argv[]){

  int N = 2;
  real gw = std::stod(argv[3]);
  real L = std::stod(argv[1]);
  real r = std::stod(argv[2]);
  auto sys = make_shared<System>(argc, argv);

  auto pd = make_shared<ParticleData>(N, sys);

  Box box(L);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto charge = pd->getCharge(access::location::cpu, access::mode::write);

    pos[0] = make_real4(-r*0.5,0,0,0);
    pos[1] = make_real4( r*0.5,0,0,0);
    // fori(0, 100){
    //   pos[1+i] = make_real4(0.25*L*i/100.0,0,0,0);
    //   charge[1+i] = 0;
    // }
    charge[0] = 1;
    charge[1] = -1;

  }

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");



  Poisson::Parameters par;

  par.box = box;
  par.epsilon = 1;
  par.gw = gw;
  par.tolerance = 1e-8;
  par.split = std::stod(argv[4]);
  //par.upsampling = 4.0;
  auto poisson = make_shared<Poisson>(pd, pg, sys, par);
  {
    auto force = pd->getForce(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par,
		 force.begin(), force.end(), real4());
    auto energy = pd->getEnergy(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par,
		 energy.begin(), energy.end(), real());
  }

  poisson->sumForce(0);
  //poisson->sumEnergy();

  {
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      auto energy = pd->getEnergy(access::location::cpu, access::mode::read);
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


