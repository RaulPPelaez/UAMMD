#ifndef INITIALORIENTATIONS_CUH
#define INITIALORIENTATIONS_CUH

#include "quaternion.cuh"
namespace uammd{
  namespace extensions{
    std::vector<real4> initOrientations(int nParticles, uint seed, std::string option){
      //Set the initial orientations of each particle
      std::vector<real4> orientations(nParticles);
      if (option=="aligned"){
	//All the particles are aligned with the laboratory frame
	std::fill(orientations.begin(), orientations.end(),uammd::make_real4(1,0,0,0));
      } else if (option=="random"){
	// The quaternions are generated randomly uniformly distributed
	// http://refbase.cvc.uab.es/files/PIE2012.pdf
	Saru rng(seed,nParticles);
	auto randomQuaternion = [&] (){
	  real x0 = rng.f();
	  real r1 = sqrt(1.0-x0);
	  real r2 = sqrt(x0);
	  real ang1 = 2*M_PI*rng.f();
	  real ang2 = 2*M_PI*rng.f();
	  return uammd::make_real4(r2*cos(ang2),r1*sin(ang1),r1*cos(ang1),r2*sin(ang2));
	};
	std::generate(orientations.begin(), orientations.end(),randomQuaternion);
      } else {
	System::log<System::ERROR>("[initOrientations] The initialization option %s is not valid", option.c_str());
	System::log<System::ERROR>("[initOrientations] Valid options: aligned and random");
	throw std::runtime_error("Invalid initialization option");
      }
      return orientations;
    }

    std::vector<real4> initOrientations(int nParticles, real3 axis){
      //Generates a quaternion that encodes a basis which have the z axis
      //aligned with axis
      std::vector<real4> orientations(nParticles);
      axis /= sqrt(dot(axis,axis));
      real3 vrot = -cross(axis,make_real3(0,0,1));
      real theta = asin(sqrt(dot(vrot,vrot)));
      Quat qdir = rotVec2Quaternion(vrot,theta);
      //std::cout<<qdir.getVz()<<"\n";
      real4 orientation = qdir.to_real4();
      std::fill(orientations.begin(), orientations.end(),orientation);
      return orientations;
    }

        std::vector<real4> initOrientations(int nParticles, Saru rng, std::string option){
      //Set the initial orientations of each particle
      std::vector<real4> orientations(nParticles);
      if (option=="aligned"){
	//All the particles are aligned with the laboratory frame
	std::fill(orientations.begin(), orientations.end(),uammd::make_real4(1,0,0,0));
      } else if (option=="random"){
	// The quaternions are generated randomly uniformly distributed
	// http://refbase.cvc.uab.es/files/PIE2012.pdf
	auto randomQuaternion = [&] (){
	  real x0 = rng.f();
	  real r1 = sqrt(1.0-x0);
	  real r2 = sqrt(x0);
	  real ang1 = 2*M_PI*rng.f();
	  real ang2 = 2*M_PI*rng.f();
	  return make_real4(r2*cos(ang2),r1*sin(ang1),r1*cos(ang1),r2*sin(ang2));
	};
	std::generate(orientations.begin(), orientations.end(),randomQuaternion);
      } else {
	System::log<System::ERROR>("[initOrientations] The initialization option %s is not valid", option.c_str());
	System::log<System::ERROR>("[initOrientations] Valid options: aligned and random");
	throw std::runtime_error("Invalid initialization option");
      }
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
