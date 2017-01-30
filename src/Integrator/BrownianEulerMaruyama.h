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
#ifndef BROWNIANEULERMARUYAMAINTEGRATOR_H
#define BROWNIANEULERMARUYAMAINTEGRATOR_H
#include "globals/defines.h"
#include "utils/utils.h"
#include "Integrator.h"
#include "BrownianEulerMaruyamaGPU.cuh"
#include<curand.h>

#ifndef SINGLE_PRECISION
#define curandGenerateNormal curandGenerateNormalDouble
#endif

class BrownianEulerMaruyama: public Integrator{
public:
  //Constructor, you have to provide D and K.
  BrownianEulerMaruyama(Matrixf D, Matrixf K);
  ~BrownianEulerMaruyama();

  void update() override;
  real sumEnergy() override;
private:
  Matrixf D, K, B;
  Vector3 noise;
  
  curandGenerator_t rng;
  brownian_euler_maruyama_ns::Params params;

};



#endif
