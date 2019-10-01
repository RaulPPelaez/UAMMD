/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation

  Solves the following stochastich differential equation:
  X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*kb*T*dt)·B·dW
  Being:
  X - Positions
  M - Mobility matrix -> M = D/kT
  K - Shear matrix
  dW- Brownian noise vector
  B - B*B^T = M -> i.e Cholesky decomposition B=chol(M) or Square root B=sqrt(M)
  divM - Divergence of the mobility matrix, zero in 3D and 2D, but non zero in q2D, which is turned off for the moment.

  The Mobility matrix is computed via the Rotne Prager Yamakawa tensor.

  The module offers several ways to compute and solve the different terms.

  BDHI::Cholesky:
  -Computing M·F and B·dW  explicitly storing M and performing a Cholesky decomposition on M.

  BDHI::Lanczos:
  -A Lanczos iterative method to reduce M to a smaller Krylov subspace and performing the operation B·dW there, the product M·F is performed in a matrix-free way, recomputing M every time M·v is needed.

  BDHI::PSE:
  -The Positively Split Edwald Method, which takes the computation to fourier space. [2]

  REFERENCES:

  1- Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
  J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347
  2- Rapid sampling of stochastic displacements in Brownian dynamics simulations
  The Journal of Chemical Physics 146, 124116 (2017); doi: http://dx.doi.org/10.1063/1.4978242

  TODO:
  100- Optimize streams
*/
#include"BDHI_EulerMaruyama.cuh"

namespace uammd{
  namespace BDHI{
    template<class Method>
    EulerMaruyama<Method>::EulerMaruyama(shared_ptr<ParticleData> pd,
					 shared_ptr<ParticleGroup> pg,
					 shared_ptr<System> sys,
					 Parameters par):
      Integrator(pd, pg, sys, "BDHI::EulerMaruyama/"+type_name<Method>()),
      K(par.K),
      par(par),
      steps(0)
    {
      bdhi = std::make_shared<Method>(pd, pg, sys, par);
      sys->log<System::MESSAGE>("[BDHI::EulerMaruyama] Initialized");

      int numberParticles = pg->getNumberParticles();

      sys->log<System::MESSAGE>("[BDHI::EulerMaruyama] Temperature: %f", par.temperature);
      sys->log<System::MESSAGE>("[BDHI::EulerMaruyama] Viscosity: %f", par.viscosity);
      sys->log<System::MESSAGE>("[BDHI::EulerMaruyama] Time step: %f", par.dt);
      if(par.hydrodynamicRadius>0)
	sys->log<System::MESSAGE>("[BDHI::EulerMaruyama] Hydrodynamic Radius: %f", par.hydrodynamicRadius);
      if(par.K.size()==3){
	real3 Kx = par.K[0];
	real3 Ky = par.K[1];
	real3 Kz = par.K[2];
	sys->log<System::MESSAGE>("[BDHI::EulerMaruyama] Shear Matrix: [ %f %f %f; %f %f %f; %f %f %f ]",
				  Kx.x, Kx.y, Kx.z,
				  Ky.x, Ky.y, Ky.z,
				  Kz.x, Kz.y, Kz.z);
      }

      cudaStreamCreate(&stream);
      cudaStreamCreate(&stream2);

      /*Result of multiplyinf M·F*/
      MF.resize(numberParticles, real3());
      BdW.resize(numberParticles+1, real3());
      //if(par.is2D) divM.resize(numberParticles, real3());

    }
    template<class Method>
    EulerMaruyama<Method>::~EulerMaruyama(){
      sys->log<System::MESSAGE>("[BDHI::EulerMaruyama] Destroyed");
      cudaStreamDestroy(stream);
      cudaStreamDestroy(stream2);
    }

    namespace EulerMaruyama_ns{
      /*
	dR = dt(KR+MF) + sqrt(2*T*dt)·BdW +T·divM·dt -> divergence is commented out for the moment
      */
      /*With all the terms computed, update the positions*/
      /*T=0 case is templated*/
      template<class IndexIterator>
      __global__ void integrateGPUD(real4* __restrict__ pos,
				    IndexIterator indexIterator,
				    const real3* __restrict__ MF,
				    const real3* __restrict__ BdW,
				    const real3* __restrict__ K,
				    //const real3* __restrict__ divM,
				    int N,
				    real sqrt2Tdt, real T, real dt, bool is2D){
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	int i = indexIterator[id];
	/*Position and color*/
	real4 pc = pos[i];
	real3 p = make_real3(pc);
	real c = pc.w;

	/*Shear stress*/
	if(K){
	  real3 KR = make_real3(0);
	  KR.x = dot(K[0], p);
	  KR.y = dot(K[1], p);
	  /*2D clause. Although K[2] should be 0 in 2D anyway...*/
	  if(!is2D)
	    KR.z = dot(K[2], p);
	  p += KR*dt;
	}
	/*Update the position*/
	p += MF[id]*dt;
	/*T=0 is treated specially, there is no need to produce noise*/
	if(BdW){
	  real3 bdw  = BdW[id];
	  if(is2D)
	    bdw.z = 0;
	  p += sqrt2Tdt*bdw;
	}
	/*If we are in q2D and the divergence term exists*/
	// if(divM){
	//   real3 divm = divM[id];
	//   //divm.z = real(0.0);
	//   //p += params.T*divm*params.invDelta*params.invDelta*params.dt; //For RFD
	//   p += T*dt*divm;
	// }
	/*Write to global memory*/
	pos[i] = make_real4(p,c);
      }
    }


    /*Advance the simulation one time step*/
    template<class Method>
    void EulerMaruyama<Method>::forwardTime(){
      sys->log<System::DEBUG1>("[BDHI::EulerMaruyama] Performing integration step %d", steps);
      /*
	dR = dt(KR+MF) + sqrt(2*T*dt)·BdW +T·divM·dt
      */
      steps++;

      for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*par.dt);

      if(steps==1){
	for(auto forceComp: interactors){
	  forceComp->updateTimeStep(par.dt);
	  forceComp->updateTemperature(par.temperature);
	  forceComp->updateBox(par.box);
	}
      }

      int numberParticles = pg->getNumberParticles();

      int BLOCKSIZE = 128; /*threads per block*/
      int nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int nblocks = numberParticles/nthreads +  ((numberParticles%nthreads!=0)?1:0);

      auto indexIter = pg->getIndexIterator(access::location::gpu);
      {
	auto force = pd->getForce(access::location::gpu, access::mode::write);
	/*Reset force*/
	fillWithGPU<<<nblocks, nthreads, 0, stream>>>(force.raw(),
						     indexIter, make_real4(0), numberParticles);
      }
      /*Compute new force*/
      for(auto forceComp: interactors) forceComp->sumForce(stream);

      bdhi->setup_step(stream);

      auto d_MF = thrust::raw_pointer_cast(MF.data());
      bdhi->computeMF(d_MF, stream);


      if(par.temperature>0){
	auto d_BdW = thrust::raw_pointer_cast(BdW.data());
	bdhi->computeBdW(d_BdW, stream);
      }

      // if(par.is2D){
      // 	auto d_divM = thrust::raw_pointer_cast(divM.data());
      // 	bdhi->computeDivM(d_divM, stream2);
      // }

      real sqrt2Tdt = sqrt(2*par.dt*par.temperature);

      bdhi->finish_step(stream);

      /*Update the positions*/
      /* R += KR + MF + sqrt(2dtT)BdW + kTdivM*/

      real3* d_BdW = nullptr;
      if(par.temperature > 0) d_BdW = thrust::raw_pointer_cast(BdW.data());

      real3* d_K = nullptr;
      if(par.K.size() > 0) d_K = thrust::raw_pointer_cast(K.data());

      //real3* d_divM = nullptr;
      //if(par.is2D) d_divM = thrust::raw_pointer_cast(divM.data());

      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);

      //cudaStreamSynchronize(stream2);

      EulerMaruyama_ns::integrateGPUD<<<nblocks, nthreads, 0, stream>>>(pos.raw(),
									indexIter,
									d_MF,
									d_BdW,
									d_K,
									//d_divM,
									numberParticles,
									sqrt2Tdt,
									par.temperature,
									par.dt, par.is2D);


    }
    template<class Method>
    real EulerMaruyama<Method>::sumEnergy(){
      return real(0.0);
    }
  }
}