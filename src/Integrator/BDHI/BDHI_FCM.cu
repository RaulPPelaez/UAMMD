/*Raul P. Pelaez 2018-2020. Force Coupling Method BDHI Module.
  See BDHI_FCM.cuh for information.
*/
#include"BDHI_FCM.cuh"
namespace uammd{
  namespace BDHI{
    auto FCMIntegrator::computeHydrodynamicDisplacements(){
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto torque = pd -> getTorqueIfAllocated(access::location::gpu, access::mode::read);
      auto dir = pd->getDirIfAllocated(access::location::gpu, access::mode::readwrite);
      int numberParticles = pg->getNumberParticles();
      return fcm->computeHydrodynamicDisplacements(pos.raw(), force.raw(), torque.raw(),
						   numberParticles,
						   temperature, 1.0/sqrt(dt), st);
    }

    void FCMIntegrator::updateInteractors(){
      for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*dt);
      if(steps==1){
	for(auto forceComp: interactors){
	  forceComp->updateTimeStep(dt);
	  forceComp->updateTemperature(temperature);
	  forceComp->updateBox(fcm->getBox());
	}
      }
    }

    void FCMIntegrator::resetForces(){
      int numberParticles = pg->getNumberParticles();
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      auto forceGroup = pg->getPropertyIterator(force);
      thrust::fill(thrust::cuda::par.on(st), forceGroup, forceGroup + numberParticles, real4());
      CudaCheckError();
    }

    void FCMIntegrator::resetTorques(){
      int numberParticles = pg->getNumberParticles();
      auto torque = pd->getTorque(access::location::gpu, access::mode::write);
      auto torqueGroup = pg->getPropertyIterator(torque);
      thrust::fill(thrust::cuda::par.on(st), torqueGroup, torqueGroup + numberParticles, real4());
      CudaCheckError();
    }

    void FCMIntegrator::computeCurrentForces(){
      resetForces();
      if (pd->isDirAllocated()) resetTorques();
      for(auto forceComp: interactors) forceComp->sum({.force=true},st);
      CudaCheckError();
    }

    namespace FCM_ns{
      /*
	dR = dt(KR+MF) + sqrt(2*T*dt)·BdW +T·divM·dt -> divergence is commented out for the moment
      */
      /*With all the terms computed, update the positions*/
      /*T=0 case is templated*/
      template<class IndexIterator>
      __global__ void integrateEulerMaruyamaD(real4* pos,
					      real4* dir,
					      IndexIterator indexIterator,
					      const real3* linearV,
					      const real3* angularV,
					      int N,
					      real dt){
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	int i = indexIterator[id];
	/*Position and color*/
	real4 pc = pos[i];
	real3 p = make_real3(pc);
	real c = pc.w;
	/*Update the position*/
	p += linearV[id]*dt;
	/*Write to global memory*/
	pos[i] = make_real4(p,c);
	/*Update the orientation*/
	if(dir){ 
	  Quat dirc = dir[i];
	  //printf("W %f %f %f\n", angularV[id].x, angularV[id].y, angularV[id].z);
	  //printf("V %f %f %f\n", linearV[id].x, linearV[id].y, linearV[id].z);
	  real3 dphi = angularV[id]*dt;
	  dirc = rotVec2Quaternion(dphi)*dirc;
	  dir[i] = make_real4(dirc);
	}
      }
    }

    void FCMIntegrator::forwardTime(){
      steps++;
      sys->log<System::DEBUG1>("[BDHI::FCM] Performing integration step %d", steps);
      updateInteractors();
      computeCurrentForces(); //Compute forces and torques
      int numberParticles = pg->getNumberParticles();
      auto disp = computeHydrodynamicDisplacements();
      auto linearVelocities = disp.first;
      auto angularVelocities = disp.second;
      auto indexIter = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto dir = pd->getDirIfAllocated(access::location::gpu, access::mode::readwrite);      
      real3* d_linearV = thrust::raw_pointer_cast(linearVelocities.data());
      real3* d_angularV = dir.raw()?thrust::raw_pointer_cast(angularVelocities.data()):nullptr;
      int BLOCKSIZE = 128; /*threads per block*/
      int nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int nblocks = numberParticles/nthreads + ((numberParticles%nthreads!=0)?1:0);      
      FCM_ns::integrateEulerMaruyamaD<<<nblocks, nthreads, 0, st>>>(pos.raw(),
								    dir.raw(),
								    indexIter,
								    d_linearV,
								    d_angularV,
								    numberParticles,
								    dt);
      CudaCheckError();
    }
  }
}

