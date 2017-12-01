/*Raul P. Pelaez 2017.
  BDHI Lanczos submodule. 
  
  Computes the mobility matrix on the fly when needed, so it is a mtrix free method.

  M·F is computed as an NBody interaction (a dense Matrix vector product).

  BdW is computed using the Lanczos algorithm [1].


  References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations.
  -http://dx.doi.org/10.1063/1.4742347
  [2] J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347

*/
#include"BDHI_Lanczos.cuh"
#include"utils/GPUUtils.cuh"
#include"Interactor/NBody.cuh"


namespace uammd{
  namespace BDHI{

    Lanczos::Lanczos(shared_ptr<ParticleData> pd,
		     shared_ptr<ParticleGroup> pg,
		     shared_ptr<System> sys,		       
		     BDHI::Parameters par):
      pd(pd), pg(pg), sys(sys),
      hydrodynamicRadius(par.hydrodynamicRadius),
      temperature(par.temperature),
      tolerance(par.tolerance),
      rpy(par.hydrodynamicRadius){
      
      sys->log<System::MESSAGE>("[BDHI::Lanczos] Initialized");  

      //Lanczos algorithm computes,
      //given an object that computes the product of a Matrix(M) and a vector(v), sqrt(M)·v
      lanczosAlgorithm = std::make_shared<LanczosAlgorithm>(sys, par.tolerance);

      this->selfMobility = 1.0/(6*M_PI*par.viscosity*par.hydrodynamicRadius);
      
      sys->log<System::MESSAGE>("[BDHI::Lanczos] Self Mobility: %f", selfMobility);

      //Init rng
      curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
    
      curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next());

      thrust::device_vector<real> noise(30000);
      auto noise_ptr = thrust::raw_pointer_cast(noise.data());
      //Warm cuRNG
      curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0);
      curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0);

    }

  
    Lanczos::~Lanczos(){

    }

    namespace Lanczos_ns{
      /*Compute the product Mv = M·v, computing M on the fly when needed, without storing it*/
      /*This critital compute is the 99% of the execution time in a BDHI simulation*/
      /*Each thread handles one particle with the other N, including itself*/
      /*That is 3 full lines of M, or 3 elements of M·v per thread, being the x y z of ij with j=0:N-1*/
      /*In other words. M is made of NxN boxes of size 3x3,
	defining the x,y,z mobility between particle pairs, 
	each thread handles a row of boxes and multiplies it by three elements of v*/
      /*vtype can be real3 or real4*/
      template<class vtype>
      struct NbodyMatrixFreeMobilityDot{    
	typedef real3 computeType;
	typedef real3 infoType;
	NbodyMatrixFreeMobilityDot(vtype* v,
				   real3 *Mv,
				   BDHI::RotnePragerYamakawa rpy,
				   real M0):
	  v(v), Mv(Mv), rpy(rpy), M0(M0){}
	/*Start with 0*/
	inline __device__ computeType zero(){ return make_real3(0);}

	inline __device__ infoType getInfo(int pi){
	  return make_real3(v[pi]);
	}
	/*Just count the interaction*/
	inline __device__ computeType compute(const real4 &pi, const real4 &pj,
					      const infoType &vi, const infoType &vj){
	  /*Distance between the pair*/
	  const real3 rij = make_real3(pi)-make_real3(pj);
	  const real r = sqrtf(dot(rij, rij));
	  /*Self mobility*/
	  if(r==real(0.0))
	    return vj;
	  /*Compute RPY coefficients, see more info in BDHI::RPYutils::RPY*/
	  real2 c12 = rpy.RPY(r);

	  const real f = c12.x;
	  const real g = c12.y;

	  real3 Mv_t;
	  /*Update the result with Dij·vj, the current box dot the current three elements of v*/
	  /*This expression is a little obfuscated, Mij*vj*/
	  /*
	    M = f(r)*I+g(r)*r(diadic)r - > (M·v)_ß = f(r)·v_ß + g(r)·v·(r(diadic)r)
	    Where f and g are the RPY coefficients
	  */
      
	  const real gv = g*dot(rij, vj);
	  /*gv = g(r)·( vx·rx + vy·ry + vz·rz )*/
	  /*(g(r)·v·(r(diadic)r) )_ß = gv·r_ß*/
	  Mv_t = f*vj + gv*rij/(r*r);
	  return Mv_t;
	}
	/*Sum the result of each interaction*/
	inline __device__ void accumulate(computeType &total, const computeType &cur){total += cur;}

	/*Write the final result to global memory*/
	inline __device__ void set(int id, const computeType &total){
	  Mv[id] = M0*total;
	}
	vtype* v;
	real3* Mv;
	real M0;
	BDHI::RotnePragerYamakawa rpy;
      };

      /*A functor to pass to LanczosAlgorithm the operation Mv = M·v*/
      template<typename vtype>
      struct Dotctor{
	using myTransverser = Lanczos_ns::NbodyMatrixFreeMobilityDot<vtype>;
	myTransverser Mv_tr;
    	shared_ptr<NBody> nbody;
	cudaStream_t st;
	Dotctor(BDHI::RotnePragerYamakawa rpy,
		real M0, shared_ptr<NBody> nbody, cudaStream_t st):
	  Mv_tr(nullptr, nullptr, rpy, M0),
	  nbody(nbody),
	  st(st)
	  {}

	inline void operator()(real3* Mv, vtype *v){
	  Mv_tr.v  = v; /*src*/
	  Mv_tr.Mv = Mv; /*Result*/
	  nbody->transverse(Mv_tr, st);
	}

      };
    }

    void Lanczos::computeMF(real3* MF, cudaStream_t st){
      /*For M·v product. Being M the Mobility and v an arbitrary array. 
	The M·v product can be seen as an Nbody interaction Mv_j = sum_i(Mij*vi)
	Where Mij = RPY( |rij|^2 ).
    
	Although M is 3Nx3N, it is treated as a Matrix of NxN boxes of size 3x3,
	and v is a vector3.
      */
      sys->log<System::DEBUG1>("[BDHI::Lanczos] MF");
      using myTransverser = Lanczos_ns::NbodyMatrixFreeMobilityDot<real4>;

      auto force = pd->getForce(access::location::gpu, access::mode::read);
      myTransverser Mv_tr(force.raw(), MF, rpy, selfMobility);
 
      NBody nbody(pd, pg, sys);
  
      nbody.transverse(Mv_tr, st);
    }


    void Lanczos::computeBdW(real3 *BdW, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::Lanczos] BdW");
      if(temperature > real(0.0)){
	st = 0;
	int numberParticles = pg->getNumberParticles();
	auto nbody = std::make_shared<NBody>(pd, pg, sys);
	/*Lanczos Algorithm needs a functor that provides the dot product of M and a vector*/
	Lanczos_ns::Dotctor<real3> Mdot(rpy, selfMobility, nbody, st);

	//Filling V instead of an external array (for v in sqrt(M)·v) is faster 
	real *noise = lanczosAlgorithm->getV(numberParticles);
	curandGenerateNormal(curng, noise,
			     3*numberParticles + (3*numberParticles)%2,
			     real(0.0), real(1.0));
	
	lanczosAlgorithm->solve(Mdot, (real*) BdW, noise, numberParticles, tolerance, st);
      }
    }

    namespace Lanczos_ns{
      /*This Nbody Transverser computes the analytic divergence of the RPY tensor*/
      struct divMTransverser{
	divMTransverser(real3* divM, real M0, real rh): divM(divM), M0(M0), rh(rh){
	  this->invrh = 1.0/rh;
	}
    
	inline __device__ real3 zero(){ return make_real3(real(0.0));}
	inline __device__ real3 compute(const real4 &pi, const real4 &pj){
	  /*Work in units of rh*/
	  const real3 r12 = (make_real3(pi)-make_real3(pj))*invrh;
	  const real r2 = dot(r12, r12);
	  if(r2==real(0.0))
	    return make_real3(real(0.0));
	  real invr = rsqrtf(r2);
	  /*Just the divergence of the RPY tensor in 2D, taken from A. Donev's notes*/
	  /*The 1/6pia is in M0, the factor kT is in the integrator, and the factor 1/a is in set*/
	  if(r2>real(4.0)){
	    real invr2 = invr*invr;
	    return real(0.75)*(r2-real(2.0))*invr2*invr2*r12*invr;
	  }
	  else{
	    return real(0.09375)*r12*invr;
	  }
	}
	inline __device__ void accumulate(real3 &total, const real3 &cur){total += cur;}
    
	inline __device__ void set(int id, const real3 &total){
	  divM[id] = M0*total*invrh;
	}
      private:
	real3* divM;
	real M0;
	real rh, invrh;
      };

    }

    void Lanczos::computeDivM(real3* divM, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::Lanczos] divM");
      Lanczos_ns::divMTransverser divMtr(divM, selfMobility, hydrodynamicRadius);
  
      NBody nbody(pd, pg, sys);
  
      nbody.transverse(divMtr, st);
    }
  }
}