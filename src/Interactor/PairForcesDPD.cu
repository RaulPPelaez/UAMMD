/*Raul P. Pelaez 2017. PairForcesDPD implementation.

  Computes short range pair forces (see PairForces Module) 
  and implements a Dissipative Particle Dynamics thermostat.

  TODO

  This module is an example of how to implement and use a general transverser to use in a Neighbour List.
*/
#include"PairForcesDPD.cuh"
#include<curand_kernel.h>
#include <thrust/swap.h>
template<class NL>
PairForcesDPD<NL>::PairForcesDPD(std::function<real(real,real)> Ffoo,
				 std::function<real(real,real)> Efoo):
  PairForcesDPD<NL>(gcnf.rcut, gcnf.L, gcnf.N, Ffoo, Efoo){}
template<class NL>
PairForcesDPD<NL>::PairForcesDPD(real rcut, real3 L, int N,
				 std::function<real(real,real)> Ffoo,
				 std::function<real(real,real)> Efoo):
  Interactor(128, L, N), nl(rcut), sortVel(N){
  if(vel.size()!=gcnf.N){
    vel = Vector3(N);
  }
  gamma = gcnf.gamma;
  noiseAmp = sqrt(2.0*gamma*gcnf.T/gcnf.dt);
  
  /*Warmup rng*/
  fori(0, 1000) seed = grng.next();

  
  pot = TablePotential(Ffoo, Efoo, 4096*rcut/real(2.5)+1, rcut);
}


namespace PairForcesDPD_ns{
  //Random number, the seed is used to recover a certain number in the random stream
  //TODO: This is a bit awkward, probably it will be best to manually generate the number
  inline __device__ real randGPU(const ullint &seed, curandState *rng){
    curand_init(seed, 0, 0, rng);
    #if defined SINGLE_PRECISION
    return curand_normal(rng);
    #else
    return curand_normal_double(rng);
    #endif
  }

  //tags: forceijDPD forceDPDij
  /*An example of a more general transverser, aside from the position, 
    the force compute needs the velocity and particle index*/
  /*See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces-DPD*/
  class forceDPDTransverser{
  public:
    /*A struct with all the information the force compute needs*/
    /*A function getInfo has to be created which fills and returns this type*/
    /*Later, compute gets the ParticleInfo of both particles, aside from the position*/
    struct ParticleInfo{
      real3 vel;
      uint pi; /*Ordered index*/
    };

    /*The constructor holds all necesary parameters, textures, and device references*/
    /*Once inside the kernel, all threads will have access to them*/
    forceDPDTransverser(real4 *newForce,
			float invrc, float invrc2, /*Precomputed numbers for the force computation*/
			real gamma, real noiseAmp, 
			cudaTextureObject_t texForce,/*Texture holding the short-range force*/
			TexReference texSortVel,/*Ordered particle velocities*/
			ullint seed, ullint N, /*A random seed and the number of particles*/
			BoxUtils box): /*The box information, apply_pbc*/
      newForce(newForce),
      invrc(invrc), invrc2(invrc2),
      gamma(gamma), noiseAmp(noiseAmp),
      texForce(texForce),
      texSortVel(texSortVel),
      seed(seed), N(N), box(box){};

    /*Get the information of particle index, in sorted order!*/
    inline __device__ ParticleInfo getInfo(uint index){
      return {make_real3(tex1Dfetch<real4>(texSortVel, index)),  index};
    }

    /*Besides the position, compute now takes the ParticleInfo of both particles*/
    inline __device__ real4 compute(const real4 &R1, const real4 &R2,
				    const ParticleInfo &P1,const ParticleInfo &P2){
      real3 r12 = make_real3(R1)-make_real3(R2);  
      box.apply_pbc(r12);

      const real r2 = dot(r12,r12);
      /*Squared distance between 0 and 1*/
      const real r2c = r2*invrc2;
      
      real w; //The intensity of the DPD thermostat 
      real rinv;
      if(r2c<real(1.0)){ /*Closer than the cut off*/
	if(r2c==real(0.0)) return make_real4(real(0.0));
	//w = r-rc -> linear
	rinv = rsqrt(r2);
	w = rinv-invrc;
      }
      else return make_real4(real(0.0));
            
      const real3 v12 = P1.vel-P2.vel;

      uint i0 = P1.pi;
      uint j0 = P2.pi;
      /*Prepare the seed for the RNG, it must be the same seed
	for pair ij and ji!*/
      if(i0>j0)
	thrust::swap(i0,j0);
      
      curandState rng;
      real randij = randGPU(i0+(ullint)N*j0 + seed, &rng);

      real fmod = -(real) tex1D<float>(texForce, r2c);
      
      fmod -= gamma*w*w*dot(r12,v12); //Damping
      fmod += noiseAmp*randij*w; //Random force
      return make_real4((real)fmod*r12);
    }

    /*Accumulate the result for each neighbour*/
    inline __device__ void accumulate(real4 &total, const real4 &current){
      total += current;
    }    
    /*Update the force acting on particle pi, pi is in the normal order*/
    inline __device__ void set(uint pi, const real4 &totalForce){
      newForce[pi] += totalForce;
    }

    inline __device__ real4 zero(){
      return make_real4(real(0.0));
    }

    
  private:
    real4* newForce;
    TexReference texSortVel; /*particle velocities in sorted order*/
    cudaTextureObject_t texForce; /*Texture containing the potential*/
    float invrc,invrc2;    /*Precompute this expensive number*/
    ullint seed, N;
    real gamma, noiseAmp;
    BoxUtils box;
  };
  
  /*Transform a real3 into a real4, which is better for memory fetches!*/
  struct changeFunctor{
    inline __device__ real4 operator()(real3 t) const{ return make_real4(t);}
  };
}

template<class NL>
void PairForcesDPD<NL>::sumForce(){
  nl.makeNeighbourList();
  /*Put real3 vel into a real4 sorted array, to access as texture later*/
  /*vel.getTexture() could be used if texures worked for real3...*/
  nl.reorderTransformProperty(vel.d_m, sortVel.d_m, PairForcesDPD_ns::changeFunctor(), N);
  /*Move the seed to the next step*/
  seed = grng.next();
  /*Create an instance of the transverser*/
  auto ft = PairForcesDPD_ns::forceDPDTransverser(force.d_m,
				1.0/(nl.getRcut()), 1.0/(nl.getRcut()*nl.getRcut()),
				gamma, noiseAmp,
				pot.getForceTexture().tex,
				sortVel.getTexture(),
						  seed, N, BoxUtils(L));
  nl.transverse(ft);
}


template class PairForcesDPD<CellList>;
