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
#include"misc/Transform.cuh"
#include"Interactor/NBodyForces.cuh"
#include<fstream>
using namespace std;
using namespace BDHI;

Lanczos::Lanczos(real M0, real rh, int N, real tolerance):
  BDHI_Method(M0, rh, N),
  utilsRPY(rh), lanczosAlgorithm(N, tolerance){
  cerr<<"\tInitializing Lanczos subsystem...";  
 
  lanczosAlgorithm.init();

  BLOCKSIZE = 1024;
  Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  Nblocks  =  N/Nthreads +  ((N%Nthreads!=0)?1:0); 


  cerr<<"DONE!!"<<endl;  
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
  struct NbodyFreeMatrixMobilityDot{    
    typedef real3 computeType;
    typedef real3 infoType;
    NbodyFreeMatrixMobilityDot(vtype* v,
			       real3 *Mv,
			       BDHI::RPYUtils utilsRPY,
			       real M0):
      v(v), Mv(Mv), utilsRPY(utilsRPY), M0(M0){}
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
      real2 c12 = utilsRPY.RPY(r);

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
    BDHI::RPYUtils utilsRPY;
  };

  /*A functor to pass to LanczosAlgorithm the operation Mv = M·v*/
  template<typename vtype>
  struct Dotctor{
    typedef typename Lanczos_ns::NbodyFreeMatrixMobilityDot<vtype> myTransverser;
    myTransverser Mv_tr;
 
    NBodyForces<myTransverser> nbody_Mdot;
    
    Dotctor(BDHI::RPYUtils utilsRPY, real M0, cudaStream_t st):
      Mv_tr(nullptr, nullptr, utilsRPY, M0),
      nbody_Mdot(Mv_tr, st){}

    inline void operator()(real3* Mv, vtype *v){
      Mv_tr.v  = v; /*src*/
      Mv_tr.Mv = Mv; /*Result*/
      nbody_Mdot.transverse(Mv_tr);
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
  
  typedef typename Lanczos_ns::NbodyFreeMatrixMobilityDot<real4> myTransverser;
  
  myTransverser Mv_tr(force.d_m, MF, utilsRPY, M0);
 
  NBodyForces<myTransverser> nbody_Mdot(Mv_tr, st);
  
  nbody_Mdot.sumForce();
}


void Lanczos::computeBdW(real3 *BdW, cudaStream_t st){
  if(gcnf.T==real(0.0)) return;
  st = 0;
  /*Lanczos Algorithm needs a functor that provides the dot product of M and a vector*/
  Lanczos_ns::Dotctor<real3> Mdot(utilsRPY, M0, st);

  lanczosAlgorithm.solveNoise(Mdot, (real*) BdW, st);
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
  Lanczos_ns::divMTransverser divMtr(divM, M0, utilsRPY.rh);
  
   NBodyForces<Lanczos_ns::divMTransverser> nbody_divM(divMtr,st);
  
   nbody_divM.sumForce();
}
