/*Raul P. Pelaez 2017-2020. Positively Split Edwald Rotne-Prager-Yamakawa Brownian Dynamics with Hydrodynamic interactions.

Near field

*/

#ifndef BDHI_PSE_NEARFIELD_CUH
#define BDHI_PSE_NEARFIELD_CUH
#include "Integrator/BDHI/BDHI_PSE.cuh"
#include "utils.cuh"
#include"misc/TabulatedFunction.cuh"
#include"RPY_PSE.cuh"
#include"third_party/saruprng.cuh"
#include"utils/debugTools.h"
#include "Interactor/NeighbourList/CellList.cuh"
//#include"Interactor/NeighbourList/VerletList.cuh"
#include"misc/LanczosAlgorithm.cuh"
#include"utils/container.h"
namespace uammd{
  namespace BDHI{
    namespace pse_ns{

      class NearField{
      public:
	using NeighbourList = CellList;
	NearField(Parameters par, std::shared_ptr<ParticleGroup> pg):
	  box(par.box),
	  pg(pg),
	  tolerance(par.tolerance)
	{
	  initializeDeterministicPart(par);
	  this->seed = pg->getParticleData()->getSystem()->rng().next32();
	  CudaCheckError();
	}

	~NearField(){}

	//Computes the deterministic part of the hydrodynamic displacements
	void Mdot(real4* forces, real3 *MF, cudaStream_t st);

	//Computes the stochastic part of the hydrodynamic displacements as prefactor*sqrt(2*T*M)*dW
	void computeStochasticDisplacements(real3* BdW, real temperature, real prefactor, cudaStream_t st);

      private:
	shared_ptr<ParticleGroup> pg;
	Box box;
	real rcut;
	//Rodne Prager Yamakawa PSE real space part textures
	thrust::device_vector<real2> tableDataRPY;
	shared_ptr<TabulatedFunction<real2>> RPY_near;
	uint seed;
	shared_ptr<NeighbourList> cl;
	real tolerance;
	shared_ptr<lanczos::Solver> lanczos;
	real lanczosTolerance;
	void initializeDeterministicPart(Parameters par){
	  const double split = par.psi;
	  /*Near neighbour list cutoff distance, see sec II:C in [1]*/
	  this->rcut = sqrt(-log(par.tolerance))/split;
	  if(0.5*box.boxSize.x < rcut){
	    System::log<System::WARNING>("[BDHI::PSE] A real space cut off (%e) larger than half the box size (%e) can result in artifacts!, try increasing the splitting parameter (%e)", rcut, 0.5*box.boxSize.x, split);
	    rcut = box.boxSize.x*0.5;
	  }
	  this->cl = std::make_shared<NeighbourList>(pg);
	  const double a = par.hydrodynamicRadius;
	  RPYPSE_near rpy(a, split, (6*M_PI*a*par.viscosity), rcut);
	  const real textureTolerance = a*par.tolerance; //minimum distance described
	  constexpr uint maximumTextureElements = 1<<22;
	  uint nPointsTable = std::min((rcut/textureTolerance + 0.5), 2e30); //2e30 to avoid overflow
	  nPointsTable = std::min(maximumTextureElements, std::max(1u<<14, nPointsTable));
	  tableDataRPY.resize(nPointsTable+1);
	  RPY_near = std::make_shared<TabulatedFunction<real2>>(thrust::raw_pointer_cast(tableDataRPY.data()),
								nPointsTable,
								0.0, //minimum distance
								rcut,//maximum distance
								rpy //Function to tabulate
								);
	  System::log<System::MESSAGE>("[BDHI::PSE] Number of real RPY texture points: %d", nPointsTable);
	  System::log<System::MESSAGE>("[BDHI::PSE] Close range distance cut off: %f", rcut);
	}
      };

      namespace pse_ns{
	/*Compute the product M_nearv = M_near·v by transversing a neighbour list

	  This operation can be seen as an sparse MatrixVector product.
	  Mv_i = sum_j ( Mr_ij·vj )
	*/
	/*Each thread handles one particle with the other N, including itself*/
	/*That is 3 full lines of M, or 3 elements of M·v per thread, being the x y z of ij with j=0:N-1*/
	/*In other words. M is made of NxN boxes of size 3x3,
	  defining the x,y,z mobility between particle pairs,
	  each thread handles a row of boxes and multiplies it by three elements of v*/
	/*vtype can be real3 or real4*/
	/*Very similar to the NBody transverser in Lanczos, but this time the RPY tensor
	  is read from a texture due to its complex form and the transverser is handed to a neighbourList*/
	template<class vtype>
	struct RPYNearTransverser{
	  typedef real3 computeType;
	  typedef real3 infoType;

	  RPYNearTransverser(vtype *v, real3 *Mv,
			     /*RPY_near(r) = F(r)·(I-r^r) + G(r)·r^r*/
			     TabulatedFunction<real2> FandG, real rcut,
			     Box box):
	    v(v), Mv(Mv), FandG(FandG), box(box){
	    this->rcut2 = (rcut*rcut);
	  }

	  inline __device__ infoType getInfo(int pi){
	    return make_real3(v[pi]);
	  }

	  /*Compute the dot product Mr_ij(3x3)·vj(3)*/
	  inline __device__ computeType compute(const real4 &pi, const real4 &pj,
						const infoType &vi, const infoType &vj){
	    real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
	    const real r2 = dot(rij, rij);
	    /*Fetch RPY coefficients from a table, see RPYPSE_near*/
	    /* Mreal(r) = (F(r)·I + (G(r)-F(r))·rr)/(6*pi*vis*a) */
	    //f and g are divided by 6*pi*vis*a in the texture
	    const real2 fg = FandG.get<cub::LOAD_LDG>(sqrt(r2));
	    const real f = fg.x;
	    const real g = fg.y;
	    /*If i==j */
	    if(r2==real(0.0)){
	      /*M_ii·vi = F(0)*I·vi/(6*pi*vis*a) */
	      return f*make_real3(vj);
	    }
	    /*Update the result with Mr_ij·vj, the current box dot the current three elements of v*/
	    /*This expression is a little obfuscated, Mr_ij·vj*/
	    /*
	      Mr = (f(r)*I+(g(r)-f(r))*r(diadic)r)/(6*pi*vis*a) - > (M·v)_ß = (f(r)·v_ß + (g(r)-f(r))·v·(r(diadic)r))/(6*pi*vis*a)
	      Where f and g are the RPY coefficients, which are already divided by 6*pi*vis*a in the table.
	    */
	    const real invr2 = real(1.0)/r2;
	    const real gmfv = (g-f)*dot(rij, vj)*invr2;
	    /*gmfv = (g(r)-f(r))·( vx·rx + vy·ry + vz·rz )*/
	    /*((g(r)-f(r))·v·(r(diadic)r) )_ß = gmfv·r_ß*/
	    return make_real3(f*vj + gmfv*rij);
	  }

	  inline __device__ void set(int id, const computeType &total){
	    Mv[id] += make_real3(total);
	  }
	  vtype* v;
	  real3* Mv;
	  TabulatedFunction<real2> FandG;
	  real rcut2;
	  Box box;
	};

	/*LanczosAlgorithm needs a functor that computes the product M·v*/
	/*Dotctor takes a list transverser and a cell list on construction,
	  and the operator () takes an array v and returns the product M·v*/
	struct Dotctor: lanczos::MatrixDot{
	  /*Dotctor uses the same transverser as in Mr·F*/
	  using myTransverser = RPYNearTransverser<real3>;
	  myTransverser Mv_tr;
	  shared_ptr<NearField::NeighbourList> cl;
	  int numberParticles;
	  cudaStream_t st;

	  Dotctor(myTransverser Mv_tr, shared_ptr<NearField::NeighbourList> cl, int numberParticles, cudaStream_t st):
	    Mv_tr(Mv_tr), cl(cl), numberParticles(numberParticles), st(st){ }

	  inline void operator()(real* v, real *Mv){
	    Mv_tr.v = (real3*)v;
	    Mv_tr.Mv = (real3*)Mv;
	    thrust::fill(thrust::cuda::par.on(st), Mv, Mv + 3*numberParticles, real());
	    cl->transverseList(Mv_tr, st);
	  }
	};

	struct SaruTransform{
	  uint seed1, seed2;
	  real variance;
	  SaruTransform(real variance, uint s1, uint s2):
	    variance(variance), seed1(s1), seed2(s2){}

	  __device__ real3 operator()(uint i){
	    Saru rng(i, seed1, seed2);
	    return make_real3(rng.gf(0,1), rng.gf(0,1).x)*variance;
	  }
	};

      }

      void NearField::Mdot(real4* forces, real3 *MF, cudaStream_t st){
	//The deterministic part can be skipped if there are no forces
	if(forces){
	  cl->update(box, rcut, st);
	  System::log<System::DEBUG1>("[BDHI::PSE] Computing MF real space...");
	  pse_ns::RPYNearTransverser<real4> tr(forces, MF, *RPY_near, rcut, box);
	  cl->transverseList(tr, st);
	}
      }

      void NearField::computeStochasticDisplacements(real3* BdW, real temperature, real prefactor, cudaStream_t st){
	//Compute stochastic term only if T>0
	if(temperature == real(0.0)) return;
	if(not lanczos){
	  //It appears that this tolerance is unnecesary for lanczos, but I am not sure so better leave it like this.
	  this->lanczosTolerance = this->tolerance; //std::min(0.05f, sqrt(par.tolerance));
	  this->lanczos = std::make_shared<lanczos::Solver>();
	}
	cl->update(box, rcut, st);
	pse_ns::RPYNearTransverser<real3> tr(nullptr, nullptr, *RPY_near, rcut, box);
	int numberParticles = pg->getNumberParticles();
	pse_ns::Dotctor Mvdot_near(tr, cl, numberParticles, st);
	/*Lanczos algorithm to compute M_near^1/2 · noise. See LanczosAlgorithm.cuh*/
	uninitialized_cached_vector<real3> noise(numberParticles);
	const auto id_tr = thrust::make_counting_iterator<uint>(0);
	const uint seed2 = pg->getParticleData()->getSystem()->rng().next32();
	real noise_prefactor = prefactor*sqrt(2*temperature);
	thrust::transform(thrust::cuda::par.on(st), id_tr, id_tr + numberParticles, noise.begin(),
			  pse_ns::SaruTransform(noise_prefactor, seed, seed2));
	lanczos->run(Mvdot_near, (real*) BdW, (real*)noise.data().get(),
		     tolerance, 3*numberParticles, st);
      }
    }
  }
}
#endif
