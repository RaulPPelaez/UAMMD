/*
Raul P. Pelaez 2016. Potentials

TODO:
100- Implement a custom interpolation table instead of using tex1D.
 */
#ifndef POTENTIAL_H
#define POTENTIAL_H
#include<functional>
#include<cuda.h>
#include"globals/defines.h"
#include"globals/globals.h"
#include"utils/utils.h"



/*Tabulates the values given by two functions into a texture.
  The two functions must return a real and take (real r2, real rcut)
*/
class TablePotential{
public:
  
  TablePotential(){};
  TablePotential(std::function<real(real,real)> Ffoo,
		 std::function<real(real,real)> Efoo,
		 int Nsamples, real rc);
  ~TablePotential();
  size_t getSize(){ return N;}
  float *getForceData(){ return F.data;}
  float *getEnergyData(){ return E.data;}
  TexReference getForceTexture(){ return {(void*)F.d_m, this->texForce};}
  TexReference getEnergyTexture(){ return {(void*)E.d_m, this->texEnergy};}

  void print();

private:
  uint N;
  Vector<float> F;
  Vector<float> E;

  cudaArray *FGPU, *EGPU;

  cudaTextureObject_t texForce, texEnergy;
  
  std::function<real(real, real)> forceFun;
  std::function<real(real, real)> energyFun;

  real rc;
};


/*Potential classes
  The main purpose of these classes is to provide getForceFunctor and getEnergyFunctor methods returning functors that compute either force or energy given a particle pair. See Potential::LJ for an example.
  The handling of types and any other potential specific feature is handled here (i.e using a table instead of computing the potential) 
*/

real forceLJ(real r2, real rcut);
real energyLJ(real r2, real rcut);
namespace Potential{

  /*A minimal example of what a potential needs to implement*/
  class TinyPotential{
  public:
    /*The type of the parameters of a certain pair*/
    typedef real TypeParams;
    /*A default constructor*/
    TinyPotential(){};
    /*A constructor that takes the maximum distance two particles will have (the range of the neighbour list)*/
    TinyPotential(real rmax){};

    /*A functor that takes a squared distance and the types of a particle pair*/
    template<bool many_types>
    struct computeFunctor{
      inline __device__ real operator()(real r2, int typei, int typej){
	return 0;
      }      
    };
    /*Return a functor that computes f/r*/
    template<bool many_types>
    computeFunctor<many_types> getForceFunctor(){
      return computeFunctor<many_types>();
    }
    /*Return a functor that computes energy*/
    template<bool many_types>
    computeFunctor<many_types> getEnergyFunctor(){
      return computeFunctor<many_types>();
    }
    
    /*What should i do with the parameters of a type pair*/
    void setPotParams(int namei, int namej, TypeParams tp){
      return; 
    }
       
  };
  
  /*Computes the Lennard-Jonnes potential
  */  
  class LJ{
  public:
    /*A pair has the following properties: { epsilon, sigma, rcut, shift} , all of them in natural system units, so WCA would have rcut = 2^(1/6)*sigma*/
    typedef real4 TypeParams;
    LJ(){}
    LJ(real rmax):invrc2(1.0/(rmax*rmax)){
      int ntypes = gcnf.color2name.size(); 
      potParams = Vector<real4>(ntypes*ntypes);
      potParams[0] = {1,1,rmax*rmax, 0};
      potParams.upload();    
    }

    /*This compute functor can return either the energy or the force depending on the second template parameter provided*/
    template<bool many_types, bool energy_mode>
    struct computeFunctor{
      computeFunctor(real invrc2,
		   cudaTextureObject_t potParams, int ntypes):
	invrc2(invrc2),
	potParams(potParams),
	ntypes(ntypes){}
      inline __device__ real operator ()(real r2, int ti, int tj){
	/*Squared distance between 0 and 1*/
	if(r2==real(0.0)) return real(0.0);      	

	/*Reescale for current type pair*/
	real4 tp;
	if(many_types)  tp = tex1Dfetch<real4>(potParams, ti+ntypes*tj);	
	else tp = tex1Dfetch<real4>(potParams, 0);
	
	const real rc2 = tp.z;
	if(r2 >= rc2) return real(0.0);
		
	const real epsilon = tp.x;
	const real sigma2 = tp.y;	


	/*Compute force/r or energy*/
	const real invr2 = sigma2/r2;
	const real invr6 = invr2*invr2*invr2;
	real lj_rc = tp.w; /*tp.w contains the shift, that is F_LJ(rcut)*/

	/******Compute energy*******/
	if(energy_mode){
	  
	  if(lj_rc != real(0.0)){
	    /*With shift, u(r) = lj(r)-lj(rc)  -(r-rc)·(dlj(r)/dr|_rc) */
	    real rc = sqrtf(rc2);
	    real invrc2 = real(sigma2)/(rc2);
	    real invrc6 = invrc2*invrc2*invrc2;
	    lj_rc = -(sqrtf(r2)-rc)*lj_rc - real(2.0)*epsilon*invrc6*(invrc6-real(1.0));
	  }
	  return real(2.0)*(epsilon*invr6*(invr6-real(1.0))) + lj_rc;
	}
	/******Compute Force/r*******/
	else{
	  if(lj_rc != real(0.0)) lj_rc = lj_rc*sqrtf(invr2);
	  return  -real(48.0)*epsilon*invr2*(invr6*invr6-real(0.5)*invr6) + lj_rc;
	}
      }
      cudaTextureObject_t potParams;
      int ntypes;
      real invrc2;
    };

    /*Create and return an instance of computeFunctor, in force or energy mode*/
    template<bool many_types>
    computeFunctor<many_types, false> getForceFunctor(){
      int ntypes = gcnf.color2name.size();
      return computeFunctor<many_types, false>(invrc2, potParams.getTexture().tex, ntypes);
    }
    template<bool many_types>
    computeFunctor<many_types, true> getEnergyFunctor(){
      int ntypes = gcnf.color2name.size();
      return computeFunctor<many_types, true>(invrc2, potParams.getTexture().tex, ntypes);
    }

    /*Handle and store the parameters or a certain type pair*/
    void setPotParams(int namei, int namej, TypeParams tp){
      /*Get the color (type) from the names*/
      int ci=-1,cj=-1;
      int ntypes = gcnf.color2name.size(); 
      fori(0,ntypes){
	int c = gcnf.color2name[i];
	if(c == namei) ci = i;
	if(c == namej) cj = i;
      }
      if(cj==-1 || ci ==-1 ||ci>=ntypes || cj>=ntypes){  
	cerr<<"WARNING: Cannot set particle names "<<namei<<","<<namej<<". No pairs with that name in the system"<<endl;
	return;
      }
    
      real epsilon = tp.x;
      real sigma = tp.y;
      real rc = tp.z;

      real shift = 0;
      if(tp.w != real(0.0)){/*Store f_lj(rcut) in shift*/
	real invrc7 = pow(sigma/rc, 7);
	real invrc13 = pow(sigma/rc, 13);
	shift = epsilon*(48.0*invrc13 - 24.0*invrc7);
      }
          
      potParams[ci+ntypes*cj] = {epsilon, (sigma*sigma), rc*rc, shift};
      potParams[cj+ntypes*ci] = {epsilon, (sigma*sigma), rc*rc, shift};
      potParams.upload();
    }    
  private:
    Vector<real4> potParams;
    real invrc2;
  }; 


  /*Computes the Lennard-Jonnes potential using a texture table
    It is very basic and doesnt allow for different cut off radius between particle pairs (in units of each pair sigma, so every pair must have the same rcut/sigma).
    You can use Potential::LJ for more flexible functionality.
  */
  class TableLJ{
  public:
    /*A pair has the following properties: { epsilon, sigma, rcut, NOT_USED} , all of them in natural system units, so WCA would have rcut = 2^(1/6)*sigma*/
    typedef real4 TypeParams;
    TableLJ(){}
    /*With the maximum distance, create a table containing the force/r and energy of the LJ potential from 0 to rmax (with ep=1 sigma=1)*/
    TableLJ(real rmax):invrc2(1.0/(rmax*rmax)),   
      tab(forceLJ, energyLJ, 4096*rmax/real(2.5)+1, rmax){
      
      
      int ntypes = gcnf.color2name.size();       
      potParams = Vector<real4>(ntypes*ntypes);
      potParams.fill_with({0,0,0,0});
      potParams.upload();        
    }

    /*This compute functor can return either the energy or the force depending on the texture provided*/
    template<bool many_types>
    struct computeFunctor{
      /*I need the pair type information and the texture table*/
      computeFunctor(real invrc2, cudaTextureObject_t texForceEnergy,
		   cudaTextureObject_t potParams, int ntypes):
	invrc2(invrc2),
	texForceEnergy(texForceEnergy),
	potParams(potParams),
	ntypes(ntypes){ }
      inline __device__ real operator ()(real r2, int ti, int tj){
	/*Squared distance between 0 and 1*/
	float r2c = r2*invrc2;
	if(r2c==0) return 0;
	if(r2c>=real(1.0)) return 0;
	
	real2 tp;
	/*Reescale for current type pair*/
	if(many_types) tp = make_real2(tex1Dfetch<real4>(potParams, ti+ntypes*tj));	
	else           tp = make_real2(tex1Dfetch<real4>(potParams, 0));
	
	real epsilon = tp.x;
	real invSigma2 = tp.y;
	r2c *= invSigma2;
	/*Get the force/r or energy from the texture*/
	return epsilon*((real) tex1D<float>(texForceEnergy, r2c));
      }
      cudaTextureObject_t texForceEnergy, potParams;
      int ntypes;
      real invrc2;
    };

    /*Create an instance of the computeFunctor with the force/r texture and return it*/
    template<bool many_types>
    computeFunctor<many_types> getForceFunctor(){
      int ntypes = gcnf.color2name.size();
      return computeFunctor<many_types>(invrc2,				    
				      tab.getForceTexture().tex,
				      potParams.getTexture().tex, ntypes);
    }
    /*Create an instance of the computeFunctor with the energy texture and return it*/
    template<bool many_types>
    computeFunctor<many_types> getEnergyFunctor(){
      int ntypes = gcnf.color2name.size();
      return computeFunctor<many_types>(invrc2,				    
				      tab.getEnergyTexture().tex,
				      potParams.getTexture().tex, ntypes);
    }

    /*Handle and store the parameters for a particular type pair*/
    void setPotParams(int namei, int namej, TypeParams tp){
      /*Get color from name*/
      int ci=-1,cj=-1;
      int ntypes = gcnf.color2name.size(); 
      fori(0,ntypes){
	int c = gcnf.color2name[i];
	if(c == namei) ci = i;
	if(c == namej) cj = i;
      }
      
      if(cj==-1 || ci ==-1 ||ci>=ntypes || cj>=ntypes){      
	cerr<<"WARNING: Cannot set particle names "<<namei<<","<<namej<<". No pairs with that name in the system"<<endl;
	return;
      }
    
      real epsilon = tp.x;
      real sigma = tp.y;
      if(tp.y>real(1.0)){
	cerr<<"ERROR: Biggest Sigma has to be 1.0 with TableLJ, use Potential::LJ"<<endl;
	exit(1);
      }
      real rc = tp.z;
      fori(0, ntypes){
	if(rc/sigma != potParams[i].z*sqrt(potParams[i].y) && potParams[i].z != real(0.0)){
	  cerr<<rc/sigma<< " "<<potParams[i].z<<endl;
	  cerr<<"ERROR: Different cut off radius are not supported in TableLJ! Use Potential::LJ instead!"<<endl;
	  exit(1);
	}
      }
      
      if(tp.w != real(0.0)) cerr<<"WARNING: TableLJ will ignore the shift flag, it will always shift the potential!\n\tUse Potential::LJ for this."<<endl;
      
      potParams[ci+ntypes*cj] = {epsilon, real(1.0)/(sigma*sigma), rc, 0};
      potParams[cj+ntypes*ci] = {epsilon, real(1.0)/(sigma*sigma), rc, 0};
      potParams.upload();
    }
  
  private:
    TablePotential tab;
    Vector<real4> potParams;
    real invrc2;
  };
}







#endif
