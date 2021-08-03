#ifndef INITIALORIENTATIONS_CUH
#define INITIALORIENTATIONS_CUH
#include "quaternion.cuh"
namespace uammd{
  namespace extensions{
    
    enum Option{
      Aligned,
      Random
    };

    //Set the initial orientations of each particle. Options: Random and Aligned
    std::vector<real4> initOrientations(int nParticles, uint seed, Option option){
      std::vector<real4> orientations(nParticles);
      Saru rng(seed,nParticles);
      auto randomQuaternion = [&] (){
	real x0 = rng.f();
	real r1 = sqrt(1.0-x0);
	real r2 = sqrt(x0);
	real ang1 = 2*M_PI*rng.f();
	real ang2 = 2*M_PI*rng.f();
	return uammd::make_real4(r2*cos(ang2),r1*sin(ang1),r1*cos(ang1),r2*sin(ang2));
      };
      switch(option){
      case Aligned: //All the particles are aligned with the laboratory frame
	std::fill(orientations.begin(), orientations.end(),uammd::make_real4(1,0,0,0));
	break;
      case Random:
	// The quaternions are generated randomly uniformly distributed
	// http://refbase.cvc.uab.es/files/PIE2012.pdf	
	std::generate(orientations.begin(), orientations.end(),randomQuaternion);
	break;
      }
      return orientations;
    }

    /*Generates a quaternion that encodes a basis which have the z axis
      aligned with axis*/
    std::vector<real4> initOrientations(int nParticles, real3 axis){
      std::vector<real4> orientations(nParticles);
      axis /= sqrt(dot(axis,axis));
      real3 vrot = -cross(axis,make_real3(0,0,1));
      real theta = asin(sqrt(dot(vrot,vrot)));
      Quat qdir = rotVec2Quaternion(vrot,theta);
      real4 orientation = qdir.to_real4();
      std::fill(orientations.begin(), orientations.end(),orientation);
      return orientations;
    }
    
    std::vector<real3> initAxis(int nParticles, uint seed){
      std::vector<real3> axis(nParticles);
      Saru rng = Saru(seed, nParticles);
      auto randomQuaternion = [&] (){
	real x0 = rng.f();
	real r1 = sqrt(1.0-x0);
	real r2 = sqrt(x0);
	real ang1 = 2*M_PI*rng.f();
	real ang2 = 2*M_PI*rng.f();
	Quat q= Quat(r2*cos(ang2),r1*sin(ang1),r1*cos(ang1),r2*sin(ang2));	
	return q.getVz();
      };
      std::generate(axis.begin(), axis.end(),randomQuaternion);
      return axis;
      }

    std::vector<real3> initAxis(int nParticles, Saru rng){
      std::vector<real3> axis;
      auto randomQuaternion = [&] (){
	real x0 = rng.f();
	real r1 = sqrt(1.0-x0);
	real r2 = sqrt(x0);
	real ang1 = 2*M_PI*rng.f();
	real ang2 = 2*M_PI*rng.f();
	Quat q= Quat(r2*cos(ang2),r1*sin(ang1),r1*cos(ang1),r2*sin(ang2));	
	return q.getVz();
      };
      std::generate(axis.begin(), axis.end(),randomQuaternion);
      return axis;
    }
  }
}
#endif
