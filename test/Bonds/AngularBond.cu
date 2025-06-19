
#include "Integrator/BrownianDynamics.cuh"
#include "Interactor/AngularBondedForces.cuh"
#include "Interactor/BondedForces.cuh"
#include "uammd.cuh"

#include <fstream>

using namespace std;
using namespace uammd;

int main(int argc, char *argv[]) {
  auto sys = make_shared<System>(argc, argv);
  int N;
  ifstream in("init.pos");
  in >> N;
  auto pd = make_shared<ParticleData>(N, sys);
  {
    auto ps = pd->getPos(access::location::cpu, access::mode::write);
    real4 *pos = ps.raw();
    fori(0, N) {
      in >> pos[i].x >> pos[i].y >> pos[i].z;
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
    using AngularBondType = AngularBondedForces_ns::AngularBond;
    using Angular = AngularBondedForces<AngularBondType>;
    Angular::Parameters ang_params;
    ang_params.file = "angular.bonds";
    auto bondCompute =
        std::make_shared<AngularBondType>(make_real3(128, 128, 128));
    auto abf = make_shared<Angular>(pd, ang_params, bondCompute);
    bd->addInteractor(abf);
  }
  {
    using BondType = BondedType::Harmonic;
    using BondedForces = BondedForces<BondType, 2>;
    BondedForces::Parameters params;
    params.file = "harmonic.bonds";
    auto bf = make_shared<BondedForces>(pd, params);
    bd->addInteractor(bf);
  }
  ofstream out("pos.dat");
  forj(0, 10000) {
    bd->forwardTime();
    if (j % 10 == 0) {
      auto ps = pd->getPos(access::cpu, access::read);
      real4 *pos = ps.raw();
      out << "#" << endl;
      fori(0, N) {
        out << pos[i].x << " " << pos[i].y << " " << pos[i].z << "\n";
      }
    }
  }

  return 0;
}
