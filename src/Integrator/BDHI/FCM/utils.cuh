/* Raul P. Pelaez and Pablo Palacios Alonso 2021

 */

#ifndef FCM_UTILS_CUH
#define FCM_UTILS_CUH
#include"uammd.cuh"
namespace uammd{
  namespace BDHI{
#ifdef UAMMD_DEBUG
    template<class T> using gpu_container = thrust::device_vector<T>;
    template<class T>  using cached_vector = detail::UninitializedCachedContainer<T>;
#else
    template<class T> using gpu_container = thrust::device_vector<T, managed_allocator<T>>;
    template<class T> using cached_vector = thrust::device_vector<T, managed_allocator<T>>;
#endif    
    
    namespace fcm_detail{
      using cufftComplex = cufftComplex_t<real>;
      using cufftComplex3 = cufftComplex3_t<real>;

      __device__ int3 indexToWaveNumber(int i, int3 nk){
	int ikx = i%(nk.x/2+1);
	int iky = (i/(nk.x/2+1))%nk.y;
	int ikz = i/((nk.x/2+1)*nk.y);
	ikx -= nk.x*(ikx >= (nk.x/2+1));
	iky -= nk.y*(iky >= (nk.y/2+1));
	ikz -= nk.z*(ikz >= (nk.z/2+1));
	return make_int3(ikx, iky, ikz);
      }

      __device__ real3 waveNumberToWaveVector(int3 ik, real3 L){
	return (real(2.0)*real(M_PI)/L)*make_real3(ik.x, ik.y, ik.z);
      }

      /*Apply the projection operator to a wave number with a certain complex factor.
	res = (I-\hat{k}^\hat{k})·factor*/
      __device__ cufftComplex3 projectFourier(real3 k, cufftComplex3 factor){
	const real invk2 = real(1.0)/dot(k,k);
	cufftComplex3 res;
	{//Real part
	  const real3 fr = make_real3(factor.x.x, factor.y.x, factor.z.x);
	  const real kfr = dot(k,fr)*invk2;
	  const real3 vr = (fr-k*kfr);
	  res.x.x = vr.x;
	  res.y.x = vr.y;
	  res.z.x = vr.z;
	}
	{//Imaginary part
	  const real3 fi = make_real3(factor.x.y, factor.y.y, factor.z.y);
	  const real kfi = dot(k,fi)*invk2;
	  const real3 vi = (fi-k*kfi);
	  res.x.y = vi.x;
	  res.y.y = vi.y;
	  res.z.y = vi.z;
	}
	return res;
      }
      
      /*Compute gaussian complex noise dW, std = prefactor -> ||z||^2 = <x^2>/sqrt(2)+<y^2>/sqrt(2) = prefactor*/
      /*A complex random number for each direction*/
      __device__ cufftComplex3 generateNoise(real prefactor, uint id, uint seed1, uint seed2){	  //Uncomment to use uniform numbers instead of gaussian
	Saru saru(id, seed1, seed2);
	cufftComplex3 noise;
	const real complex_gaussian_sc = real(0.707106781186547)*prefactor; //1/sqrt(2)
	//const real sqrt32 = real(1.22474487139159)*prefactor;
	// = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	noise.x = make_real2(saru.gf(0, complex_gaussian_sc));
	// = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	noise.y = make_real2(saru.gf(0, complex_gaussian_sc));
	// = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	noise.z = make_real2(saru.gf(0, complex_gaussian_sc));
	return noise;
      }

      __device__ bool isNyquistWaveNumber(int3 cell, int3 ncells){
	/*Beware of nyquist points! They only appear with even cell dimensions
	  There are 8 nyquist points at most (cell=0,0,0 is excluded at the start of the kernel)
	  These are the 8 vertex of the inferior left cuadrant. The O points:
	  +--------+--------+
	  /|       /|       /|
	  / |      / |      / |
	  +--------+--------+  |
	  /|  |    /|  |    /|  |
	  / |  +---/-|--+---/-|--+
	  +--------+--------+  |	/|
	  |  |/ |  |  |/ |  |  |/ |
	  |  O-----|--O-----|--+	 |
	  | /|6 |  | /|7 |  | /|	 |
	  |/ |  +--|/-|--+--|/-|--+
	  O--------O--------+  |	/
	  |5 |/    |4 |/    |  |/
	  |  O-----|--O-----|--+
	  ^   | / 3    | / 2    | /  ^
	  |   |/       |/       |/  /
	  kz  O--------O--------+  ky
	  kx ->     1
	*/
	//Is the current wave number a nyquist point?
	const bool isXnyquist = (cell.x == ncells.x - cell.x) and (ncells.x%2 == 0);
	const bool isYnyquist = (cell.y == ncells.y - cell.y) and (ncells.y%2 == 0);
	const bool isZnyquist = (cell.z == ncells.z - cell.z) and (ncells.z%2 == 0);
	const bool nyquist =  (isXnyquist and cell.y==0    and cell.z==0)  or  //1
	  (isXnyquist and isYnyquist and cell.z==0)  or  //2
	  (cell.x==0    and isYnyquist and cell.z==0)  or  //3
	  (isXnyquist and cell.y==0    and isZnyquist) or  //4
	  (cell.x==0    and cell.y==0    and isZnyquist) or  //5
	  (cell.x==0    and isYnyquist and isZnyquist) or  //6
	  (isXnyquist and isYnyquist and isZnyquist);    //7
	return nyquist;
      }
    }
  }
}
#endif
