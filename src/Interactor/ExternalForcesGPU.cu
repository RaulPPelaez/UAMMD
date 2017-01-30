
#include"globals/defines.h"
#include"ExternalForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"

namespace external_forces_ns{

  /*This function takes the position of a particle and returns the force on it*/
  inline __device__ real4 ForceFunction(real4 pos, uint i){
    real4 f = make_real4(0);
    real *F = &(f.x);
    real *r = &(pos.x);
    real L = 7.2f;
    real K = 100.0f;
    for(int i=0; i<3; i++){
      if(r[i]> L*0.5f)
	F[i] -= K*(r[i]-L*0.5f);
      else if(r[i]<-L*0.5f)
	F[i] -= K*(r[i]+L*0.5f);
    }
    
    return f;

  }
  
  //Parameters in constant memory, super fast access
  __constant__ Params params; 

  
  void initGPU(Params m_params){
    m_params.invL = 1.0/m_params.L;
    /*Upload parameters to constant memory*/
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }

  __global__ void externalForcesD(const __restrict__ real4 *pos, __restrict__ real4* force, uint N){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=N) return;
    force[i] += ForceFunction(pos[i], i);   
  }

  
  
  void computeExternalForce(real4 *force, real4 *pos, uint N){    
    uint nthreads = 128<N?128:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0); 

    externalForcesD<<< nblocks , nthreads>>>(pos, force, N);
  }
}
