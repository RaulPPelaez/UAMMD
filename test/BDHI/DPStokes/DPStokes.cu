//Raul P.Pelaez 2021, DPStokes test
#include"uammd.cuh"
#include"Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"

using namespace uammd;

int main(int argc, char* argv[]){

  int N = 2;

  double Lx = std::stod(argv[1]);
  double H = Lx;

  double h = 1.0;
  int Nxy = int(Lx/h+0.5);
  int nz = Nxy;
  real gw = 1;
  int support = 13;
  real viscosity = 1/(6*M_PI);
  
  Box box(make_real3(Lx, Lx, H));

  auto sys = std::make_shared<System>(argc, argv);
  auto pd = std::make_shared<ParticleData>(N, sys);
  using Scheme = DPStokesSlab_ns::DPStokes;

  Scheme::Parameters par;

  par.cells = make_int3(Nxy, Nxy, nz);
  par.gw = gw;
  box.setPeriodicity(1,1,0);
  par.box = box;
  par.support = support;  
  par.viscosity = viscosity;

  auto stokes = std::make_shared<Scheme>(par);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos[0] = make_real4(0,0,1,0);
    pos[1] = make_real4(0,0,-1,0);
    auto force = pd->getForce(access::location::cpu, access::mode::write);
    force[0] = make_real4(1,1,1,0);
    force[1] = make_real4(-1,-1,-1,0);
  }
  thrust::host_vector<real4> res(N);
  {
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto force = pd->getForce(access::location::gpu, access::mode::read);
    auto res_gpu = stokes->Mdot(pos.begin(), force.begin(), N, 0);
    thrust::copy(res_gpu.begin(), res_gpu.end(), res.begin());
  }
  {
    double m0 = 1.0/(sqrt(M_PI));
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    std::cerr<<std::setprecision(17)<<res[0].x<<" "<<res[0].y<<" "<<res[0].z<<std::endl;
    std::cerr<<m0<<" "<<(1-abs(res[0].x/m0))<<std::endl;
  }


  return 0;
}

