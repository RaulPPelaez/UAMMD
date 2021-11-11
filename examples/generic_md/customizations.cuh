/*Raul P. Pelaez 2021
  Customizations for generic_simulation.cu
  You can modify this file to tweak the different interactions available there.

  USAGE:
  Compile generic_simulation.cu (use the Makefile) and run it.
  This will generate a data.main file with information about what this code can do.
  When you run the program again, data.main will be read and the simulation it describes will take place.
  A lot of systems can be simulated just by modifying data.main, without having to code anything. 
  The auto generated data.main will describe a LJ liquid MD simulation.
  If you need anything that cannot be done via the data.main, such as using a potential different from LJ you can encode it in this file.
  Down below you will encounter a few functions and classes, inspect the comments above each one to learn more about what part of the simulation each one affects and instructions on how to modify them.
  I recommend to use some sort of code folding capability to glance at this code for the first time, since you will be able to fit every function and class, including their accompanying comments, inside a single screen.
  Then you can quickly see where each thing is.
  If by ay chance you are using emacs, you can fold the code by enabling hs mode:
    M-x hs-minor-mode
  And then running:
    M-x hs-hide-level
  You can then unfold each section by placing the cursor on top of it and running:
    M-x hs-show-block

    Note that in order to understand this file, some knowledge of C++ and CUDA is needed.
    The most important thing to take into account is that a function marked by __device__ is a function that runs on the GPU.
     __device__ functions  must abide to a series of restrictions and special rules which explanation is outside the scope of this header. 
     Mainly, keep in mind that device functions cannot read global variables and can only read memory allocated for the gpu (so do not try to read a pointer allocated with malloc or an std::vector).
     If this is not something you are familiar with it is highly advisable that you go through a basic CUDA tutorial before attempting to make heavy modifications to this file.
 */
#include"uammd.cuh"
#include "utils/InputFile.h"
using namespace uammd;

//Parameters that will be read from the data.main
//You can add new parameters here and read them by adding a corresponding line in the readCustomParameters function below
struct Parameters{
  int numberParticles;
  real3 L;
  int numberSteps, printSteps, relaxSteps;  
  real dt, viscosity, hydrodynamicRadius;
  real friction; //Friction coefficient for VerletNVT
  real temperature;
  real  sigma, epsilon, cutOff;  
  std::string readFile, chargeReadFile;
  std::string bondFile, angularBondFile, torsionalBondFile;
  std::string outfile, outfileVelocities, outfileEnergy;
  std::string integrator;
  //Electrostatics
  bool useElectrostatics = false;
  real permittivity;
  real gaussianWidth;
  //DPD
  real gamma_dpd, A_dpd, cutOff_dpd;
  //SPH
  real support_sph;
  real gasStiffness_sph;
  real restDensity_sph;
  //External potential, in this case gravity and a wall
  bool enableGravity = false;
  real gravity;
  real kwall;
  //Imagine you need an additional parameter
  //real myCustomParameter;
  
};

//All members in the default Parameters struct (defined above) are read from data.main elsewhere.
//If you want to read and additional parameter you can add it to Parameters and then read it here
void readCustomParameters(InputFile &in, Parameters &par){
  //Populate the members of Parameters that you added like this
  //in.getOption("myCustomParameter", InputFile::Required)>>par.myCustomParameter;
  //This line will prompt an error if the data.main does not contain the option myCustomParameter
  //You guessed it right, you may change Required by Optional and the program will not halt if the data.main line is not found
}

//Positions are initialized elsewhere, either by reading from the file "readFile" in data.main or starting in an fcc lattice
//In case electrostatics are turned on, charge will also be initialized elsewhere (all charges to 1) before this call.
//In case SPH is the chosen integrator, masses will be initialized to 1.
//You can initialize and/or modify any properties at start-up here.
//If your interactions are going to need something extra, like mass or radius, you can initialize it here.
//You are free to use the fourth element in the positions to encode whatever you want, you will have access to this when computing the different interactions
void furtherParticleInitialization(std::shared_ptr<ParticleData> pd, Parameters par){
  //For example, say you want to modify charges in the CPU:
  //auto charges = pd->getCharge(access::cpu, access::write);
  //Set every charge to 2
  //std::fill(charges.begin(), charges.end(), 2);
  //or equivalently:
  //for(auto &m: mass) m=2;

  //Or maybe you want to use the mass in the short range potential
  //Note that mass=1 for all particles unless you modify it here
  //Lets say you also want to perform this operation in the GPU via thrust
  //auto mass = pd->getMass(access::gpu, access::write);
  //thrust::fill(thrust::cuda::par, mass.begin(), mass.end(), 10);

  //Another example, say you want to read particle radius from a file because you want to use it in the short range potential.
  //std::ifstream in(par.myRadiusReadFile);
  //std::istream_iterator<real3> begin(in), end;
  //std::vector<real> radius_in_file(begin, end);
  //if(radius_in_file.size() != par.numberParticles){
  // std::cerr<<"ERROR, I expect "<<par.numberParticles<<" lines in "<<par.myRadiusReadFile<<". Found "<<radius_in_file.size()<<std::endl;
  // exit(1);
  //}
  //auto radius = pd->getRadius(access::cpu, access::write);
  //std::copy(radius_in_file.begin(), radius_in_file.end(), radius.begin());

}

//Some helper functions to compute forces/energies
__device__ real lj_force(real r2){
  const real invr2 = real(1.0)/r2;
  const real invr6 = invr2*invr2*invr2;
  real fmoddivr = (real(-48.0)*invr6 + real(24.0))*invr6*invr2;
  return fmoddivr;
}

__device__ real lj_energy(real r2){
  const real invr2 = real(1.0)/r2;
  const real invr6 = invr2*invr2*invr2;
  real E = real(4.0)*invr6*(invr6-real(1.0));
  return E;
}

//A simple LJ Potential for the short range interaction
//As an example, this LJ potential will read a couple of unnecessary particle properties; the names or id's and the masses
//You can use this to learn how to customize it.
//Note that albeit its called ShortRange, this potential will work the same if the cut off is really large (a.i the box size)
//If you need a more complex potential that what you can do with this see the example advanced/customPotential.cu
//If you would rather get a neighbour list and work with that see the example advanced/neighbourList.cu or uammd_as_a_library/neighbourList.cu
struct ShortRangePotential{
  real rc;
  real epsilon, sigma;
  
  ShortRangePotential(Parameters par){
    this->rc = par.cutOff;
    this->epsilon = par.epsilon;
    this->sigma = par.sigma;
  }
  
  //A host function that must return the maximum required cut off for the interaction
  real getCutOff(){
    return rc;
  }
  //This object takes care of the interaction in the GPU, this interface is much more generic than shown here, if you need
  // something that is not easy to encode by changing this, see advanced/customPotential.cu
  struct ForceEnergy{
    real4 *force;
    real* energy;
    Box box;
    real rc;
    real ep, s;
    real *mass;
    int *id;
    ForceEnergy(Box i_box, real i_rc, real4* i_force, real* i_energy, real ep,  real s, real*mass, int* id):
      box(i_box),
      rc(i_rc),
      force(i_force),
      energy(i_energy),mass(mass), id(id), ep(ep), s(s){
      //All members will be available in the device functions
    }
    //Place in this struct any per-particle data that you need to compute the interaction between a particle pair.
    struct Info{
      real mass;
      int id;
    };
    //Fetch the required additional data for a particle at a ceratin index (note that the position is always available)
    __device__ Info getInfo(int index){
      return {mass[index], id[index]};
    }
    //For each pair of close particles computes and returns the LJ force and/or energy based only on the positions
    __device__ real4 compute(real4 pi, real4 pj, Info infoi, Info infoj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      //mass of particle i is in: infoi.mass
      //name of particle j is in: infoj.id
      //Note that a particle is considered to be a neighbour of itself
      if(r2>0 and r2< rc*rc){
	return make_real4(ep/s*lj_force(r2/(s*s))*rij,
			  energy?(ep/s*(lj_energy(r2/(s*s))-lj_energy(rc*rc/(s*s)))):real(0.0));
      }
      return real4();
    }
    //The "set" function will be called with the accumulation of the result of "compute" for all neighbours.
    __device__ void set(int id, real4 total){
      //Write the total result to memory if the pointer was provided
      force[id] += make_real4(total.x, total.y, total.z, 0);
      if(energy) energy[id] += real(0.5)*total.w;
    }
  };

  //This function must construct and return an instance of the struct above, the name of the type or where it is defined does not matter, but the
  // functions implemented in it must be present
  ForceEnergy getForceEnergyTransverser(Box box, std::shared_ptr<ParticleData> pd){
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();
    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite).raw();    
    auto mass = pd->getMass(access::location::gpu, access::mode::read).raw();
    auto id = pd->getId(access::location::gpu, access::mode::read).raw();
    return ForceEnergy(box, rc, force, energy, epsilon, sigma, mass, id);
  }
  
};

//External potential acting on each particle independently. In this example particles experience gravity
//and there is a wall at the bottom
//If you want a more in depth example of an external potential, see the example interaction_modules/external.cu
struct GravityAndWall{
  real g;
  real zwall;
  real kwall;
  GravityAndWall(Parameters par):
    g(par.gravity),
    zwall(-par.L.z*0.5),
    kwall(par.kwall){}

  //This function will be called for each particle
  //The arguments will be modified according to what was returned by getArrays below
  __device__ real3 force(real4 pos /*, real mass */){
    real3 f = {0,0,-g};
    //A soft wall that prevents particles from crossing the wall (well they will cross it if determined enough)
    real dist = pos.z - zwall;
    //If particle is near the wall
    if(fabs(dist)<real(10.0)){
      //If particle is above the wall:
      if(dist<0){
	real d2 = dist*dist;
	f += {0,0, kwall/(d2 + real(0.1))};
      }//If particle has crossed the wall lets flip the gravity so it crosses again
      else{
	f.z = g;
      }
    }
    return f;
  }
  
  //Energy can be ommited in the integrators this example use. It defaults to 0.
  //__device__ real energy(real4 pos){ return 0;}
  
  auto getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::gpu, access::read);    
    return pos.begin();
    //If more than one property is needed this would be the way to do it:
    //auto mass = pd->getMass(access::gpu, access::read);
    //return std::make_tuple(pos.begin(), mass.begin());
  }
  
};

//Harmonic bond, a good example on how to implement a bonded force
struct HarmonicBond{

  HarmonicBond(Parameters par){
    //In this case no parameter is needed beyond whats in the bond file.
  }
  //Place in this struct whatever static information is needed for a given bond
  //In this case spring constant and equilibrium distance
  //the function readBond below takes care of reading each BondInfo from the file
  struct BondInfo{
    real k, r0;
  };
  //This function will be called for every bond read in the bond file
  //In the case of a Fixed Point bond, j will be -1
  //i,j: id of particles in bond
  //r12: ri-rj
  //bi: bond information.
  inline __device__ real3 force(int i, int j, real3 r12, BondInfo bi){
    real r2 = dot(r12, r12);
    if(r2==real(0.0)) return make_real3(0.0);
    real invr = rsqrt(r2);
    real f = -bi.k*(real(1.0)-bi.r0*invr); //F = -k·(r-r0)·rvec/r
    return f*r12;
  }
  inline __device__ real energy(int i, int j, real3 r12, BondInfo bi){
    real r2 = dot(r12, r12);
    if(r2==real(0.0)) return real(0.0);
    real r = sqrt(r2);
    const real dr = r-bi.r0;
    return real(0.5)*bi.k*dr*dr;
  }

  //This function will be called for each bond in the bond file
  //It must use the stream that is handed to it to construct a BondInfo.  
  static __host__ BondInfo readBond(std::istream &in){
    /*BondedForces will read i j, readBond has to read the rest of the line*/
    BondInfo bi;
    in>>bi.k>>bi.r0;
    return bi;
  }

};

//This angular potential is similar to the HarmonicBond above, the difference is that
//Now three particles are involved in each bond instead of two
struct Angular{
  Box box;
  Angular(Parameters par): box(Box(par.L)){}
  
  struct BondInfo{
    real ang0, k;
  };
  
  inline __device__ real3 force(int i, int j, int k,
				int bond_index,
				real3 posi,
				real3 posj,
				real3 posk,
				BondInfo bond_info){
    const real ang0 = bond_info.ang0;
    const real kspring = bond_info.k;
    //         i -------- j -------- k
    //             rij->     rjk ->
    //Compute distances and vectors
    //---rij---
    const real3 rij =  box.apply_pbc(posj - posi);
    const real rij2 = dot(rij, rij);
    const real invsqrij = rsqrt(rij2);
    //---rkj---
    const real3 rjk =  box.apply_pbc(posk - posj);
    const real rjk2 = dot(rjk, rjk);
    const real invsqrjk = rsqrt(rjk2);
    const real a2 = invsqrij * invsqrjk;
    real cijk = dot(rij, rjk)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)
    //Cos must stay in range
    if(cijk>real(1.0)) cijk = real(1.0);
    else if (cijk<real(-1.0)) cijk = -real(1.0);
    real ampli;
    // //Approximation for small angle displacements
    // real sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
    // //sijk cant be zero to avoid division by zero
    // if(sijk<std::numeric_limits<real>::min()) sijk = std::numeric_limits<real>::min();
    // ampli = -kspring * (acos(cijk) - ang0)/sijk; //The force amplitude -k·(theta-theta_0)
    //ampli = -kspring*(-sijk*cos(ang0)+cijk*sin(ang0))+ang0; //k(1-cos(ang-ang0))
    if(ang0 == real(0.0)){
      ampli = -real(2.0)*kspring;
    }
    else{
      const real theta = acos(cijk);
      if(theta==real(0.0))  return make_real3(0);
      const real sinthetao2 = sin(real(0.5)*theta);
      ampli = -real(2.0)*kspring*(sinthetao2 - sin(ang0*real(0.5)))/sinthetao2;
    }
    //Magical trigonometric relations to infere the direction of the force
    const real a11 = ampli*cijk/rij2;
    const real a12 = ampli*a2;
    const real a22 = ampli*cijk/rjk2;
    //Sum according to my position in the bond
    // i ----- j ------ k
    if(bond_index==i){
      return make_real3(a12*rjk -a11*rij); //Angular spring
    }
    else if(bond_index==j){
      //Angular spring
      return real(-1.0)*make_real3((-a11 - a12)*rij + (a12 + a22)*rjk);
    }
    else if(bond_index==k){
      //Angular spring
      return real(-1.0)*make_real3(a12*rij -a22*rjk);
    }
    return make_real3(0);
  }

  inline __device__ real energy(int i, int j, int k,
				int bond_index,
				real3 posi,
				real3 posj,
				real3 posk,
				BondInfo bond_info){   
    const real ang0 = bond_info.ang0;
    const real kspring = bond_info.k;
    //         i -------- j -------- k
    //             rij->     rjk ->
    //Compute distances and vectors
    //---rij---
    const real3 rij =  box.apply_pbc(posj - posi);
    const real rij2 = dot(rij, rij);
    const real invsqrij = rsqrt(rij2);
    //---rkj---
    const real3 rjk =  box.apply_pbc(posk - posj);
    const real rjk2 = dot(rjk, rjk);
    const real invsqrjk = rsqrt(rjk2);
    const real a2 = invsqrij * invsqrjk;
    real cijk = dot(rij, rjk)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)
    //Cos must stay in range
    if(cijk>real(1.0)) cijk = real(1.0);
    else if (cijk<real(-1.0)) cijk = -real(1.0);
    real ampli;
    // //Approximation for small angle displacements
    // real sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
    // //sijk cant be zero to avoid division by zero
    // if(sijk<std::numeric_limits<real>::min()) sijk = std::numeric_limits<real>::min();
    // ampli = -kspring * (acos(cijk) - ang0)/sijk; //The force amplitude -k·(theta-theta_0)
    //ampli = -kspring*(-sijk*cos(ang0)+cijk*sin(ang0))+ang0; //k(1-cos(ang-ang0))
    const real theta = acos(cijk);
    if(theta==real(0.0))  return real(0);
    const real sinthetao2 = sin(real(0.5)*theta);
    const real stmst= (sinthetao2 - sin(ang0*real(0.5)));
    ampli = real(4.0)*kspring*stmst*stmst;
    //Split the bond energy between the three particles
    return ampli/real(3.0);
  }

  static BondInfo readBond(std::istream &in){
    BondInfo bi;
    in>>bi.k>>bi.ang0;
    return bi;
  }  
};

//This torsional potential is similar to the HarmonicBond above, the difference is that
//Now four particles are involved in each bond instead of two
struct Torsional{
private:

  __device__ real3 cross(real3 a, real3 b){
    return make_real3(a.y*b.z - a.z*b.y, (-a.x*b.z + a.z*b.x), a.x*b.y - a.y*b.x);
  }

public:
  Box box;
  Torsional(Parameters par): box(Box(par.L)){}

  struct BondInfo{
    real phi0, k;
  };

  inline __device__ real3 force(int j, int k, int m, int n,
				int bond_index,
				real3 posj,
				real3 posk,
				real3 posm,
				real3 posn,
				BondInfo bond_info){
    const real3 rjk = box.apply_pbc(posk - posj);
    const real3 rkm = box.apply_pbc(posm - posk);
    const real3 rmn = box.apply_pbc(posn - posm);
    real3 njkm = cross(rjk, rkm);
    real3 nkmn = cross(rkm, rmn);
    const real n2 = dot(njkm, njkm);
    const real nn2 = dot(nkmn, nkmn);
    if(n2 > 0 and nn2 > 0) {
      const real invn = rsqrt(n2);
      const real invnn = rsqrt(nn2);
      const real cosphi = dot(njkm, nkmn)*invn*invnn;
      real Fmod = 0;
      // #define SMALL_ANGLE_BENDING
      // #ifdef SMALL_ANGLE_BENDING
      const real phi = acos(cosphi);
      if(cosphi*cosphi <= 1 and phi*phi > 0){
	Fmod = -bond_info.k*(phi - bond_info.phi0)/sin(phi);
      }
      //#endif
      njkm *= invn;
      nkmn *= invnn;
      const real3 v1 = (nkmn - cosphi*njkm)*invn;
      const real3 fj = Fmod*cross(v1, rkm);
      if(bond_index == j){
	return real(-1.0)*fj;
      }
      const real3 v2 = (njkm - cosphi*nkmn)*invnn;
      const real3 fk = Fmod*cross(v2, rmn);
      const real3 fm = Fmod*cross(v1, rjk);
      if(bond_index == k){
	return fm + fj - fk;
      }
      const real3 fn = Fmod*cross(v2, rkm);
      if(bond_index == m){
	return fn + fk - fm;
      }
      if(bond_index == n){
	return real(-1.0)*fn;
      }
    }
    return real3();
  }

  inline __device__ real energy(int j, int k, int m, int n,
				int bond_index,
				real3 posj,
				real3 posk,
				real3 posm,
				real3 posn,
				BondInfo bond_info){
    return 0;
  }

  static BondInfo readBond(std::istream &in){
    BondInfo bi;
    in>>bi.k>>bi.phi0;
    return bi;
  }

};

