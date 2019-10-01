
#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"Interactor/AngularBondedForces.cuh"
#include"Interactor/BondedForces.cuh"

#include<fstream>

using namespace std;
using namespace uammd;

int main(int argc, char * argv[]){
  auto sys = make_shared<System>(argc, argv);
  int N;

  ifstream in("init.pos");

  in>>N;
  auto pd = make_shared<ParticleData>(N, sys);
  {
    auto ps = pd->getPos(access::location::cpu, access::mode::write);
    real4 * pos = ps.raw();
    fori(0,N){
      in>>pos[i].x>>pos[i].y>>pos[i].z;
      pos[i].w = 0;
    }
  }



  using AngularBondType = AngularBondedForces_ns::AngularBond;
  using Angular = AngularBondedForces<AngularBondType>;

  Angular::Parameters ang_params;
  Box box (128);
  ang_params.readFile = "angular.bonds";
  auto abf = make_shared<Angular>(pd, sys, ang_params, AngularBondType(box));

  using BondType = BondedType::Harmonic;
  using BondedForces = BondedForces<BondType>;

  BondedForces::Parameters params;
  params.file = "harmonic.bonds";
  auto bf = make_shared<BondedForces>(pd, sys, params);

  BD::EulerMaruyama::Parameters par;
  par.temperature = 0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = 0.1;

  auto bd = make_shared<BD::EulerMaruyama>(pd, sys, par);

  bd->addInteractor(bf);
  bd->addInteractor(abf);

  ofstream out("pos.dat");


  forj(0,10000){

    bd->forwardTime();
    if(j%10==0){
    auto ps = pd->getPos(access::location::cpu, access::mode::read);
    real4 * pos = ps.raw();
    out<<"#"<<endl;
    fori(0,N){
      out<<pos[i].x<<" "<<pos[i].y<<" "<<pos[i].z<<"\n";
    }
    }
  }



  return 0;
}