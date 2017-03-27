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
namespace BDHI{


  struct RPYPSE_nearTextures{
  private:
    cudaArray *FGPU, *GGPU;
    cudaTextureObject_t texF, texG;
    real rh, psi;
    real rcut;
    int ntab;
    real M0;
  public:
    cudaTextureObject_t getFtex(){ return texF;}
    cudaTextureObject_t getGtex(){ return texG;}
    RPYPSE_nearTextures(){}
    RPYPSE_nearTextures(real rh, real psi, real M0, real rcut, int ntab = 4096);

    double2 FandG(double r);

  private:

    double params2FG(double r,
		     double f0, double f1, double f2, double f3,
		     double f4, double f5, double f6, double f7);
  };

  
  class PSE: public BDHI_Method{
  public:
    PSE(real M0, real T, real rh, real psi, int N, int max_iter = 100);
    ~PSE();
    void setup_step(              cudaStream_t st = 0) override;
    void computeMF(real3* MF,     cudaStream_t st = 0) override;    
    void computeBdW(real3* BdW,   cudaStream_t st = 0) override;  
    void computeDivM(real3* divM, cudaStream_t st = 0) override;


    template<typename vtype>
    void Mdot_near(real3 *Mv, vtype *v, cudaStream_t st);
    template<typename vtype>
    void Mdot_far(real3 *Mv, vtype *v, cudaStream_t st);
    template<typename vtype>
    void Mdot(real3 *Mv, vtype *v, cudaStream_t st);

    
  private:
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
