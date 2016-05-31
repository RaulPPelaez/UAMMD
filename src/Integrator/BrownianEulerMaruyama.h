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
#include<thread>



class BrownianEulerMaruyama: public Integrator{
public:
  //Constructor, you have to provide D and K.
  BrownianEulerMaruyama(shared_ptr<Vector<float4>> pos,
			shared_ptr<Vector<float4>> force,
			shared_ptr<Vector<float4>> D,
			shared_ptr<Vector<float4>> K,
			uint N, float L, float dt);
  ~BrownianEulerMaruyama();

  void update() override;

private:
  Vector<float3> noise;
  curandGenerator_t rng;
  BrownianEulerMaruyamaParameters params;
  shared_ptr<Vector<float4>> D, K, B;
};



#endif
