
#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"Interactor/TorsionalBondedForces.cuh"
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
  BD::EulerMaruyama::Parameters par;
  par.temperature = 0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = 0.1;
  auto bd = make_shared<BD::EulerMaruyama>(pd, par);
  {
    using TorsionalBondType = TorsionalBondedForces_ns::TorsionalBond;
    using Torsional = TorsionalBondedForces<TorsionalBondType>;
    Torsional::Parameters ang_params;
    real3 box =make_real3(128,128,128);
    ang_params.readFile = "torsional.bonds";
    auto bondType = std::make_shared<TorsionalBondType>(box);
    auto abf = make_shared<Torsional>(pd, sys, ang_params, bondType);
    bd->addInteractor(abf);
  }
  {
    using BondType = BondedType::Harmonic;
    using BondedForces = BondedForces<BondType>;
    BondedForces::Parameters params;
    params.file = "harmonic.bonds";
    auto bf = make_shared<BondedForces>(pd, params);
    bd->addInteractor(bf);
  }
  {
    using AngularBondType = AngularBondedForces_ns::AngularBond;
    using Angular = AngularBondedForces<AngularBondType>;
    Angular::Parameters ang_params;
    real3 box =make_real3(128,128,128);
    ang_params.readFile = "angular.bonds";
    auto bondType = std::make_shared<AngularBondType>(box);
    auto angbf = make_shared<Angular>(pd, ang_params, bondType);
    bd->addInteractor(angbf);
  }
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
