/*Raul P. Pelaez 2019-2020. Spectral Poisson solver with Ewald splitting.
Computes forces and energies acting on a group of charges due to their electrostatic interaction.

USAGE:

Poisson is created as the typical UAMMD module:

```c++
#include<uammd.cuh>
#include<Interactor/SpectralEwaldPoisson.cuh>
using namespace uammd;
...
int main(int argc, char *argv[]){
...
  int N = 1<<14;
  auto sys = make_shared<System>(arc, argv);
  auto pd = make_shared<ParticleData>(N, sys);          
  auto pg = make_shared<ParticleGroup>(pd, sys, "All"); //A group with all the particles
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto charge = pd->getCharge(access::location::cpu, access::mode::write);
   ...
  }
  Poisson::Parameters par;
  par.box = Box(128);
  par.epsilon = 1; //Permittivity
  par.gw = 1.0;
  par.tolerance = 1e-4;
  par.split = 1.0;
  auto poisson = make_shared<Poisson>(pd, pg, sys, par);
...
  myintegrator->addInteractor(poisson);
...
return 0;
}
```
The tolerance parameter is the maximum relative error allowed in the potential for two charges. The potential for L->inf is extrapolated and compared with the analytical solution. Also in Ewald split mode the relative error between two different splits is less than the tolerance. See test/Potential/Poisson  

See the wiki page for more information.
 */
#ifndef SPECTRALEWALDPOISSON_CUH
#define SPECTRALEWALDPOISSON_CUH
#include"Interactor/Interactor.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include "misc/IBM.cuh"
#include "utils/utils.h"
#include "global/defines.h"
#include"utils/Grid.cuh"
#include"utils/cufftPrecisionAgnostic.h"
#include"utils/cufftComplex3.cuh"
#include"utils/cufftComplex4.cuh"
#include"third_party/managed_allocator.h"
#include"misc/TabulatedFunction.cuh"

namespace uammd{
  namespace Poisson_ns{
    struct Gaussian{
      int support;
      Gaussian(real tolerance, real width, real h){
	this-> prefactor = cbrt(pow(2*M_PI*width*width, -1.5));
	this-> tau = -1.0/(2.0*width*width);
	real rmax = sqrt(log(tolerance*sqrt(2*M_PI*width*width))/tau);
	support = std::max(3, int(2*rmax/h+0.5));
      }

      inline __device__ real phi(real r) const{
	return prefactor*exp(tau*r*r);
      }
    private:
      real prefactor;
      real tau;
    };
  }

  class Poisson: public Interactor{
  public:
    using Kernel = Poisson_ns::Gaussian;
    using cufftComplex3 = cufftComplex3_t<real>;
    using cufftComplex4 = cufftComplex4_t<real>;
    using cufftComplex = cufftComplex_t<real>;
    using cufftReal = cufftReal_t<real>;

    using NeighbourList = CellList;

    struct Parameters{
      real upsampling = -1.0;
      int3 cells = make_int3(-1, -1, -1); //Number of Fourier nodes in each direction
      Box box;
      real epsilon = -1;
      real tolerance = 1e-5;
      real gw = -1;
      int support = -1;
      real split = -1;
    };

    Poisson(shared_ptr<ParticleData> pd,
	    shared_ptr<System> sys,
	    Parameters par):
      Poisson(pd, std::make_shared<ParticleGroup>("All"), sys, par){}

    Poisson(shared_ptr<ParticleData> pd,
	    shared_ptr<ParticleGroup> pg,
	    shared_ptr<System> sys,
	    Parameters par);
    ~Poisson();

    void sumForce(cudaStream_t st) override;

    real sumEnergy() override;

    real sumForceEnergy(cudaStream_t st) override{
      sys->log<System::DEBUG2>("[Poisson] Sum Force and Energy");
      farField(st);
      nearFieldForce(st);
      nearFieldEnergy(st);
      return 0;
    }

  private:
    shared_ptr<Kernel> kernel;
    shared_ptr<NeighbourList> nl;

    cufftHandle cufft_plan_forward, cufft_plan_inverse;

    template<class T> using managed_vector = thrust::device_vector<T>;
    template<class T> using temporal_vector = thrust::device_vector<T, System::allocator_thrust<T>>;
    //using managed_vector = thrust::device_vector<T, managed_allocator<T>>;

    managed_vector<char> cufftWorkArea;
    std::shared_ptr<TabulatedFunction<real>> nearFieldGreensFunction;
    managed_vector<real> nearFieldGreensFunctionTable;

    std::shared_ptr<TabulatedFunction<real>> nearFieldPotentialGreensFunction;
    managed_vector<real> nearFieldPotentialGreensFunctionTable;

    cudaStream_t st;

    void initCuFFT();

    void farField(cudaStream_t st);
    void nearFieldForce(cudaStream_t st);
    void nearFieldEnergy(cudaStream_t st);

    void spreadCharges(real* gridCharges);
    void forwardTransformCharge(real* gridCharges, cufftComplex* gridChargesFourier);
    void convolveFourier(cufftComplex* gridChargesFourier, cufftComplex4* gridFieldPotentialFourier);
    void inverseTransform(cufftComplex* gridFieldPotentialFourier, real* gridFieldPotential);
    void interpolateFields(real4* gridFieldPotential);

    Box box;
    Grid grid;

    real epsilon;
    real split;
    real gw;
    real nearFieldCutOff;
    real tolerance;
  };

}

#include"SpectralEwaldPoisson.cu"
#endif

