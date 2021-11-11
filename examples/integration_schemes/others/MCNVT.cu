/*Raul P. Pelaez 2018
  A Monte Carlo NVT example with the Anderson module.

  Will perform a simulation of a LJ liquid at a certain temperature and output positions and total internal energy (without the thermal energy).

  Look for argv[ to see what each parameter does.


  Try running:

      ./mcnvt 14 32 100000 50 1 20 0.0525 300 0.9 2>&1

 */
#include"uammd.cuh"

#include"Interactor/Potential/Potential.cuh"
#include"Integrator/MonteCarlo/NVT/Anderson.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>
using namespace uammd;
using namespace std;
void resetEnergy(std::shared_ptr<ParticleData> pd){
  auto energy = pd->getEnergy(access::location::gpu, access::mode::write);
  thrust::fill(thrust::cuda::par,energy.begin(), energy.end(), 0);
}
int main(int argc, char *argv[]) {
  auto sys = make_shared<System>();
  if(argc<10){
    sys->log<System::CRITICAL>("[System] Not enough input arguments!. Look for argv[ in the source code");
  }
    int N = pow(2,atoi(argv[1]));
    ullint seed = 0xf31337Bada55D00dULL^time(NULL);
    sys->rng().setSeed(seed);
    auto pd = make_shared<ParticleData>(N, sys);
    auto pg = make_shared<ParticleGroup>(pd, sys, "All");
    Box box(std::stod(argv[2]));
    {
        auto pos = pd->getPos(access::location::cpu, access::mode::write);
        fori(0,N) {
	  pos.raw()[i] = make_real4(sys->rng().uniform3(-box.boxSize.x*0.5,box.boxSize.x*0.5), 0);
            pos.raw()[i].w = 0;
	}
    }
    ofstream outPos("pos.dat"),outEnergy("energy.dat");
    using LJ=Potential::LJ;
    auto pot = make_shared<LJ>(sys);
    {
      LJ::InputPairParameters par;
      par.epsilon = 1.0;
      par.shift = false;

      par.sigma = 1;
      par.cutOff = 2.5*par.sigma;
      pot->setPotParameters(0, 0, par);
    }
    using MC = MC_NVT::Anderson<LJ>;
    MC::Parameters par;
    par.box = box;
    par.temperature = std::stod(argv[5]);
    par.triesPerCell = std::atoi(argv[6]);
    par.initialJumpSize = std::stod(argv[7]); //0.05
    par.acceptanceRatio = std::stod(argv[9]); //0.8
    par.tuneSteps = 10;
    auto mc = make_shared<MC>(pd, pg, sys, pot, par);
    sys->log<System::MESSAGE>("RUNNING!!!");
    pd->sortParticles();
    Timer tim;
    tim.tic();
    int nsteps = std::atoi(argv[3]);
    int printSteps = std::atoi(argv[4]);
    forj(0,nsteps) {
      mc->forwardTime();
      if(j%printSteps==1) {
	sys->log<System::DEBUG>("[System] Writing to disk...");
	real3 p;
	auto pos = pd->getPos(access::location::cpu, access::mode::read);
	resetEnergy(pd);
	outEnergy << mc->sumEnergy() <<" "<<mc->getCurrentAcceptanceRatio()<<" "<<mc->getCurrentStepSize()<<std::endl;
	outPos<<"#"<<endl;
	fori(0,N) {
	  p = make_real3(pos[i]);
	  int type = pos[i].w;
	  outPos<<p<<" 0.5 "<<type<<"\n";
	}
	outPos<<flush;
      }
      if(j%500 == 1){
	pd->sortParticles();
      }
    }
    auto totalTime = tim.toc();
    sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
    sys->finish();
    return 0;
}
