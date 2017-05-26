/*Raul P. Pelaez 2016. Brownian Euler Maruyama Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*Tdt)·dW·B
   Being:
     X - Positions
     M - Diffusion matrix
     K - Shear matrix
     dW- Noise vector
     B - chol(M)
*/
#ifndef BROWNIANEULERMARUYAMAINTEGRATOR_CUH
#define BROWNIANEULERMARUYAMAINTEGRATOR_CUH
#include "globals/defines.h"
#include "utils/utils.h"
#include "Integrator.h"
#include<curand.h>

#ifndef SINGLE_PRECISION
#define curandGenerateNormal curandGenerateNormalDouble
#endif

class BrownianEulerMaruyama: public Integrator{
public:
  //Constructor, you have to provide D and K.
  BrownianEulerMaruyama(Matrixf M, Matrixf K);
  BrownianEulerMaruyama(Matrixf M, Matrixf K, int N, real3 L, real dt);
  ~BrownianEulerMaruyama();

  void update() override;
  real sumEnergy() override;
private:
  Matrixf M, K, B;
  Vector3 noise;
  real sqrt2Tdt;
  curandGenerator_t rng;
  real T;
};



#endif
