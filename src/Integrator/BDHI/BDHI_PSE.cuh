/*Raul P. Pelaez 2017. Positively Split Edwald BDHI Module
  
  As this is a BDHI module. BDHI_PSE computes the terms M·F and B·dW in the differential equation:
            dR = K·R·dt + M·F·dt + sqrt(2Tdt)· B·dW
  
  The mobility, M, is computed according to the Rotne-Prager-Yamakawa (RPY) tensor.

  The computation uses periodic boundary conditions (PBC) 
  and partitions the RPY tensor in two, positively defined contributions [1], so that: 
      M = Mr + Mw
       Mr - A real space short range contribution.
       Mw - A wave space long range contribution.

  Such as:
     M·F = Mr·F + Mw·F
     B·dW = sqrt(Mr)·dWr + sqrt(Mw)·dWw
####################      Short Range     #########################

 
  Mr·F: The short range contribution of M·F is computed using a neighbour list (this is like a sparse matrix-vector product in which each element is computed on the fly), see PSE_ns::RPYNearTransverser.
        The RPY near part function (see Apendix 1 in [1]) is precomputed and stored in texture memory,
	see PSE_ns::RPYPSE_nearTextures.

  sqrt(Mr)·dW: The near part stochastic contribution is computed using the Lanczos algorithm (see misc/LanczosAlgorithm.cuh), the function that computes M·v is provided via a functor called PSE_ns::Dotctor, the logic of M·v itself is the same as in M·F (see PSE_ns::RPYNearTransverser) and is computed with the same neighbour list.

###################        Far range     ###########################



  Mw·F:  Mw·F = σ·St·FFTi·B·FFTf·S · F. The long range wave space part.
         -σ: The volume of a grid cell
	 -S: An operator that spreads each element of a vector to a regular grid using a gaussian kernel.
	 -FFT: Fast fourier transform operator.
	 -B: A fourier scaling factor in wave space to transform forces to velocities, see eq.9 in [1].
	 
        Related functions: 
	FFT: cufftExecR2C (forward), cufftC2R(inverse)
	S: PSE_ns::particles2Grid (S), PSE_ns::grid2Particles (σ·St)
	B: PSE_ns::fillFourierScalingFactor, PSE_ns::forceFourier2vel

  sqrt(Mw)·dWw: The far range stochastic contribution is computed in fourier space along M·F as:
               Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw = 
                            = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	        Only one St·FFTi is needed, the stochastic term is added as a velocity in fourier space.                dWw is a gaussian random vector of complex numbers, special care must be taken to ensure the correct conjugacy properties needed for the FFT. See PSE_ns::fourierBrownianNoise

Therefore, in the case of Mdot_far, for computing M·F, Bw·dWw is also summed.

computeBdW computes only the real space stochastic contribution.
 
References:

[1]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations
           -  https://arxiv.org/pdf/1611.09322.pdf
[2]  Spectral accuracy in fast Ewald-based methods for particle simulations
           -  http://www.sciencedirect.com/science/article/pii/S0021999111005092

 */
#ifndef BDHI_PSE_CUH
#define BDHI_PSE_CUH

#include "BDHI.cuh"
#include "utils/utils.h"
#include "globals/globals.h"
#include "globals/defines.h"
#include"NeighbourList/CellList.cuh"
#include"misc/LanczosAlgorithm.cuh"
#include<cufft.h>
#include<curand_kernel.h>
#include<thread>

#ifndef SINGLE_PRECISION

#define cufftComplex cufftDoubleComplex
#define cufftReal cufftDoubleReal
#define cufftExecR2C cufftExecD2Z
#define cufftExecC2R cufftExecZ2D 

#define curand_normal2 curand_normal2_double
#endif

namespace BDHI{


  struct RPYPSE_nearTextures{
  private:
    cudaArray *FGPU, *GGPU;
    cudaTextureObject_t texF, texG;
    real vis, rh, psi;
    real rcut;
    int ntab;
    real M0;
  public:
    cudaTextureObject_t getFtex(){ return texF;}
    cudaTextureObject_t getGtex(){ return texG;}
    RPYPSE_nearTextures(){}
    RPYPSE_nearTextures(real vis, real rh, real psi, real M0, real rcut, int ntab = 4096);

    double2 FandG(double r);

  private:

    double params2FG(double r,
		     double f0, double f1, double f2, double f3,
		     double f4, double f5, double f6, double f7);
  };

  
  class PSE: public BDHI_Method{
  public:
    PSE(real vis, real T, real rh, real psi, int N, int max_iter = 100);
    ~PSE();
    void setup_step(              cudaStream_t st = 0) override;
    void computeMF(real3* MF,     cudaStream_t st = 0) override;    
    void computeBdW(real3* BdW,   cudaStream_t st = 0) override;  
    void computeDivM(real3* divM, cudaStream_t st = 0) override;
    void finish_step(             cudaStream_t st = 0) override;


    template<typename vtype>
    void Mdot_near(real3 *Mv, vtype *v, cudaStream_t st);
    template<typename vtype>
    void Mdot_far(real3 *Mv, vtype *v, cudaStream_t st);
    template<typename vtype>
    void Mdot(real3 *Mv, vtype *v, cudaStream_t st);

    
  private:
    std::thread Mdot_nearThread, Mdot_farThread, NearNoise_Thread;
    
    real T;
    /*Kernel launch parameters*/
    int Nthreads, Nblocks;
    real psi; /*Splitting factor*/
    
    /****Near (real space) part *****/    
    /*Rodne Prager Yamakawa PSE near real space part textures*/
    BDHI::RPYPSE_nearTextures nearTexs;
    real rcut;
    CellList cl;
    LanczosAlgorithm lanczos;

    /****Far (wave space) part) ******/
    real kcut; /*Wave space cutoff and corresponding real space grid size */
			
    Mesh mesh; /*Grid parameters*/
    
    /*Grid interpolation kernel parameters*/
    int3 P; //Gaussian spreading/interpolation kernel support points*/
    real3 m, eta; // kernel width and gaussian splitting in each direction


    cufftHandle cufft_plan_forward, cufft_plan_inverse;
    GPUVector<real> cufftWorkArea;
    
    Vector3 gridVels;    //Interpolated grid velocities in real space

    Vector<cufftComplex> gridVelsFourier;     //Interpolated grid velocities in fourier space
    GPUVector<real3> fourierFactor;  // Fourier scaing factors to go from F to V in wave space

    GPUVector<curandState> farNoise;

    cudaStream_t stream, stream2;
  };
}
#endif
