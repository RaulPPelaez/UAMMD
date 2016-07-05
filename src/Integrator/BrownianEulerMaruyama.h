/*Raul P. Pelaez 2016. Brownian Euler Maruyama Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(dt)·dW·B
   Being:
     X - Positions
     D - Diffusion matrix
     K - Shear matrix
     dW- Noise vector
     B - sqrt(D)
*/
#ifndef BROWNIANEULERMARUYAMAINTEGRATOR_H
#define BROWNIANEULERMARUYAMAINTEGRATOR_H
#include "utils/utils.h"
#include "Integrator.h"
#include "BrownianEulerMaruyamaGPU.cuh"
#include<curand.h>

class BrownianEulerMaruyama: public Integrator{
public:
  //Constructor, you have to provide D and K.
  BrownianEulerMaruyama(Vector4Ptr D,
			Vector4Ptr K);
  ~BrownianEulerMaruyama();

  void update() override;
  float sumEnergy() override;
private:
  Vector3 noise;
  
  curandGenerator_t rng;
  BrownianEulerMaruyamaParameters params;
  
  Vector4Ptr D, K;
  Vector4 B;
};



#endif
