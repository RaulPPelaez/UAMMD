/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
   
  Solves the following stochastich differential equation:
      X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(dt)·B·dW
   Being:
     X - Positions
     D - Diffusion matrix
     K - Shear matrix
     dW- Brownian noise vector
     B - sqrt(D)

 The Diffusion matrix is computed via the Rotne Prager Yamakawa tensor

 The module offers several ways to compute and sovle the different terms.

 The brownian Noise can be computed by:
     -Computing sqrt(D)·dW explicitly performing a Cholesky decomposition on D.
     -Through a Lanczos iterative method to reduce D to a smaller Krylov subspace and performing the operation there.

  On the other hand the mobility(diffusion) matrix can be handled in several ways:
     -Storing and computing it explicitly as a 3Nx3N matrix.
     -Not storing it and recomputing it when a product D·v is needed.

REFERENCES:

1- Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
        J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347

*/

#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_H
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_H
#include"globals/defines.h"
#include "utils/utils.h"
#include "Integrator.h"
#include "BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<curand.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<cuda_runtime.h>
#include"utils/cuda_lib_defines.h"

/*How the diffusion matrix will be handled,
  -MATRIXFULL stores a 3Nx3N matrix, computes it once and performs a matrix vector multiplication when needed
  -MATRIXFREE doesnt store a D matrix, and recomputes it on the fly when asked to multiply it by a vector
  -DEFAULT is FULL
*/
enum DiffusionMatrixMode{DEFAULT,MATRIXFULL, MATRIXFREE};

/*Method of obtaining the Brownian noise vector y = sqrt(D)·z
 -Cholesky Performs a Choesky decomposition on D and explicitly multiplies it by z, needs FULL matrix mode.
 -LANCZOS Performs a Krylov subspace reduction on D, and computes y in a much smaller subspace.
 */
enum StochasticNoiseMethod{CHOLESKY, LANCZOS};

namespace brownian_hy_euler_maruyama_ns{
  /*------------------------------DIFFUSION------------------------------------*/
  /*Diffusion matrix handler*/
  /*Takes care of computing the mobility(diffusion) matrix,
    store it (if needed) and, most importantly, computing D·v*/
  class Diffusion{
    uint N;/*number of particles*/
    Matrixf D;/*In a Matrix-Free method this is an empty matrix*/
    Matrixf D0;/*D0, self diffusion matrix, 3x3*/
    brownian_hy_euler_maruyama_ns::RPYParams params;
    DiffusionMatrixMode mode;
  public:
    Diffusion(Matrixf D0, uint N, DiffusionMatrixMode mode=DEFAULT);

    /*Fills the diffusion matrix, in a matrix-free method does nothing*/
    void compute();

    /*res = D·v *//*D(3N,3N), v(3N), res(3N)*/
    void dot(real *v, real *res, cublasHandle_t handle=0);

    /*Returns nullptr in a Matriz-Free method*/
    real* getMatrix();

  };
  
  /*------------------------------BROWNIAN NOISE----------------------------------*/
  /*This virtual class takes D and computes a brownian noise array for each particle*/
  /*BrownianNoseComputer has at least a curand generator*/
  class BrownianNoiseComputer{
  public:
    BrownianNoiseComputer(uint N);/*Initialize curand*/
    ~BrownianNoiseComputer();
    /*Initialize whatever you need according to D and N*/
    virtual bool init(Diffusion &D, uint N) = 0;
    /*Returns a pointer to the Brownian Noise vector, can be a pointer to noise i.e*/
    virtual real* compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream=0) = 0;
  protected:
    curandGenerator_t rng;
    Vector3 noise;
  };

  /*-------------------------------Cholesky---------------------------------------*/
  /*Computes the brownian noise using a cholesky decomposition on D, defined in cpp*/
  class BrownianNoiseCholesky: public BrownianNoiseComputer{
  public:
    BrownianNoiseCholesky(uint N): BrownianNoiseComputer(N){}
    /*Initialize cuSolver*/
    bool init(Diffusion &D, uint N) override;
    /*Perform sqrt(D)·z by Cholesky decomposition and trmv multiplication*/
    real* compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream = 0) override;
  private:
    /*BdW is stored in the parents noise Vector3*/
    /*Cholesky decomposition through cuSolver*/
    cusolverDnHandle_t solver_handle;
    /*Cusolver temporal storage*/
    int h_work_size;    
    real *d_work;
    int *d_info;
  };
  /*--------------------------------Lanczos--------------------------------------*/
  /*Computes the brownian noise using a Krylov subspace approximation from D \ref{1}, defined in cpp*/
  class BrownianNoiseLanczos: public BrownianNoiseComputer{
  public:
    BrownianNoiseLanczos(uint N, uint max_iter=100);
    bool init(Diffusion &D, uint N) override;
    real* compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream = 0) override;
  protected:
    void compNoise(real z2, uint N, uint iter); //computes the noise in the current iteration
    /*BdW is stored in the parents noise Vector3*/    
    uint max_iter; //~100
    Vector3 w; //size N; v in each iteration
    Matrixf V; // 3Nxmax_iter; Krylov subspace base
    /*Matrix D in Krylov Subspace*/
    Matrixf H, Htemp; //size max_iter * max_iter;
    Matrixf P,Pt; //Transformation matrix to diagonalize H
    /*upper diagonal and diagonal of H*/
    Vector<real> hdiag, hdiag_temp, hsup; //size max_iter

    cusolverDnHandle_t solver_handle;
    cublasHandle_t cublas_handle;
    int h_work_size;
    real *d_work;
    int *d_info;   
  };

}
/*-----------------------------INTEGRATOR CLASS----------------------------------*/


class BrownianHydrodynamicsEulerMaruyama: public Integrator{
public:
  BrownianHydrodynamicsEulerMaruyama(Matrixf D0, Matrixf K,
				     StochasticNoiseMethod stochMethod = CHOLESKY,
				     DiffusionMatrixMode mode=DEFAULT, int max_iter = 0);
				     
  ~BrownianHydrodynamicsEulerMaruyama();

  void update() override;
  real sumEnergy() override;
  
private:

  Vector3 force3; /*Cublas needs a real3 array instead of real4 to multiply matrices*/
  
  Vector3 DF;  /*Result of D·F*/
  Matrixf K, D0; /*Shear and self diffusion matrices*/

  brownian_hy_euler_maruyama_ns::Diffusion D;   /*Mobility handler*/
  brownian_hy_euler_maruyama_ns::Params params; /*GPU parameters (CPU version)*/
  
  cudaStream_t stream, stream2;

  /*Brownian noise computer, a shared pointer to the virtual base class*/
  shared_ptr<brownian_hy_euler_maruyama_ns::BrownianNoiseComputer> cuBNoise;
  
  cublasStatus_t status;
  cublasHandle_t handle;
};



#endif
