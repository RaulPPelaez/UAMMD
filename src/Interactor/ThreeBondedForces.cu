/*Raul P. Pelaez 2017


 */

/********************************THREE BONDED FORCES**********************************/

#include"ThreeBondedForces.cuh"
#include"globals/defines.h"
#include<cmath>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<set>
#include"GPUutils.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"



ThreeBondedForces::ThreeBondedForces(const char *readFile):
ThreeBondedForces(readFile, gcnf.L, gcnf.N){}

ThreeBondedForces::~ThreeBondedForces(){}

ThreeBondedForces::ThreeBondedForces(const char * readFile, real3 L, int N):
  Interactor(64, L, N), TPP(64){
  name = "ThreeBondedForces";
  cerr<<"Initializing Three Bonded Forces..."<<endl;

  nbonds = 0;
  /*Read the bond list from the file*/
  ifstream in(readFile);
  vector<vector<uint>> isInBonds(N);
  in>>nbonds;
  vector<ThreeBond> blst(nbonds); //Temporal storage for the bonds in the file

  
  cerr<<"\tDetected: "<<nbonds<<" particle-particle-particle bonds"<<endl;
  if(nbonds>0){
    
    for(uint b=0; b<nbonds; b++){
      uint i, j, k;
      in>>i>>j>>k;
      
      isInBonds[i].push_back(b);
      isInBonds[j].push_back(b);
      isInBonds[k].push_back(b);      
      
      blst[b].i = i;
      blst[b].j = j;
      blst[b].k = k;
      
      in>>blst[b].kspring>>blst[b].r0>>blst[b].ang;
      
      if(blst[b].r0>=0.5f*L.x || blst[b].r0>=0.5f*L.y || blst[b].r0>=0.5f*L.z){
	cerr<<"The equilibrium distance of bond "<<b<<" is too large!!"<<endl;
	cerr<<"\t This will cause unexpected behavior when computing distance using PBC!"<<endl;
      }
    }
  }
  
  
  bondList  = Vector<ThreeBond>(nbonds*3);//Allocate 3*nbonds
  bondStart = Vector<uint>(N); bondStart.fill_with(0xffFFffFF);
  bondEnd   = Vector<uint>(N); bondEnd.fill_with(0);

  
  fori(0,N){
    int nbondsi;
    nbondsi = isInBonds[i].size();
    if(nbondsi==0) continue;
    
    int offset;
    if(i>0)
      offset = bondEnd[i-1];
    else
      offset = 0;
    
    forj(0,nbondsi){
      bondList[offset+j] = blst[isInBonds[i][j]];
    }
    bondEnd[i] = offset+nbondsi;
    bondStart[i] = offset;
  }

  vector<uint> pwb; //Particles with bonds
  fori(0,N){
    if(bondStart[i]!=0xffFFffFF){
      pwb.push_back(i);
    }
  }

  bondParticleIndex.assign(pwb.begin(), pwb.end());
  
  nbonds *= 3; //We store all the bonds in which every particle is involved, per particle.


  bondList.upload();
  bondStart.upload();
  bondEnd.upload();
  bondParticleIndex.upload();
 
  
  cerr<<pwb.size()<<" particles are involved in at least one bond."<<endl;
  
  cerr<<"Three Bonded Forces\t\tDONE!!\n\n";

}




namespace Bonded_ns{
  //Custom kernel to compute and sum the force in a three particle angle spring
  /*
    Computes the potential: V(theta) = 0.5 K(theta-theta_0)^2
    F(\vec{ri}) = d(V(theta))/d(cos(theta))·d(cos(theta))/d(\vec{ri})
   */
  __global__ void computeThreeBondedForce(real4* __restrict__ force, const real4* __restrict__ pos,
					   const uint* __restrict__ bondStart,
					   const uint* __restrict__ bondEnd,
					   const uint* __restrict__ bondedParticleIndex,
					  const ThreeBondedForces::ThreeBond* __restrict__ bondList,
					   BoxUtils box){
    extern __shared__ real4 forceTotal[];
    /*A block per particle, as in computeBondedForcesD*/
    uint p = bondedParticleIndex[blockIdx.x];
  
    real4 posp = pos[p];
  
    uint first = bondStart[p];
    uint last = bondEnd[p];
   
    real4 f = make_real4(real(0.0));

    uint i,j,k;             //The bond indices
    real4 posi,posj, posk; //The bond particles
    real r0, kspring, ang0; //The bond info

    /*         i -------- j -------- k*/
    /*             rij->      <-rkj  */
    
    real3 rij, rkj; //rij = ri - rj
  
    real invsqrij, invsqrkj; //1/|rij|
    real rij2, rkj2;  //|rij|^2

    
    real a2; 
    real cijk, sijk;
    real a, a11, a12, a22;
    real ampli;

    /*Go through my bonds*/
    for(int b = first+threadIdx.x; b<last; b+=blockDim.x){
      /*Recover bond info*/
      auto bond = bondList[b];
      i = bond.i;
      j = bond.j;
      k = bond.k;

      kspring = bond.kspring;
      r0 = bond.r0;
      ang0 = bond.ang;

      
      //TODO Texture memory target
      /*Store the positions of the three particles*/
      /*We already got one of them, p*/
      /*Differentiate between the three particles in the bond*/
      if(p==i){
	posi = posp;
	posj = pos[j];
	posk = pos[k];
      }
      else if(p==j){
	posi = pos[i];
	posj = posp;
	posk = pos[k];
      }
      else{
	posi = pos[i];
	posj = pos[j];
	posk = posp;
      }

      /*Compute distances and vectors*/
      /***rij***/
      rij =  make_real3(posi-posj);
      box.apply_pbc(rij);
      rij2 = dot(rij, rij);
      invsqrij = rsqrt(rij2);
      /***rkj***/
      rkj =  make_real3(posk-posj);
      box.apply_pbc(rkj);
      rkj2 = dot(rkj, rkj);
      invsqrkj = rsqrt(rkj2);
      /********/
      
      a2 = invsqrij * invsqrkj;
      cijk = dot(rij, rkj)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)

      /*Cos must stay in range*/
      if(cijk>1.0f) cijk = real(1.0);
      else if (cijk<-1.0f) cijk = -real(1.0);
      
      sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
      /*sijk cant be zero to avoid division by zero*/
      if(sijk<real(0.000001)) sijk = real(0.000001);


      ampli = -real(100.0)*kspring * (acosf(cijk) - ang0); //The force amplitude -k·(theta-theta_0)

      /*

	ampli = -kang*(-sijk*cos(ang0)+cijk*sin(ang0))+ang0; //k(1-cos(ang-ang0))
	
      */

      //Magical trigonometric relations to infere the direction of the force
      a = ampli/sijk;
      a11 = a*cijk/rij2;
      a12 = -a*a2;
      a22 = a*cijk/rkj2;
      
      /*Sum according to my position in the bond*/
      // i ----- j ------ k
      if(p==i){
	f += make_real4(a11*rij + a12*rkj); //Angular spring

	f += make_real4(-kspring*(real(1.0) - r0*invsqrij)*rij ); //Harmonic spring
	
	// real rep = 0;
	// if(1.0f/invsqrij >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrij,14) + 24.0f*pow(invsqrij,8);
	//f += make_real4((-kspring/(1.0f-rij2/(r0*r0)) )*rij); //fene spring
      }
      else if(p==j){
	//Angular spring
	f -= make_real4(a11*rij + a12*rkj + a22*rkj + a12*rij);
	

	f += make_real4(kspring*(real(1.0) - r0*invsqrij)*rij); //First harmonic spring
	f += make_real4(kspring*(real(1.0) - r0*invsqrkj)*rkj); //Second harmonic spring

	// real rep = 0;
	// if(1.0f/invsqrij >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrij,14) + 24.0f*pow(invsqrij,8);
	// f -= make_real4((-kspring/(1.0f-rij2/(r0*r0)) + rep)*rij); //fene spring

	// if(1.0f/invsqrkj >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrkj,14) + 24.0f*pow(invsqrkj,8);
	// f -= make_real4((-kspring/(1.0f-rkj2/(r0*r0)) + rep)*rkj); //fene spring


	
	// f -= make_real4((-kspring/(1.0f-rij2/(r0*r0)))*rij); // first fene spring
	//f -= make_real4((-kspring/(1.0f-rkj2/(r0*r0)))*rkj); //second fene spring
      }
      else if(p==k){
	//Angular spring
	f += make_real4(a22*rkj + a12*rij);
	//Harmonic spring
	f += make_real4(-kspring*(real(1.0)-r0*invsqrkj)*rkj);

	// real rep = 0;
	// if(1.0f/invsqrkj >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrkj,14) + 24.0f*pow(invsqrkj,8);

	//f += make_real4((-kspring/(1.0f-rkj2/(r0*r0)))*rkj); //fene spring
      }
    }

    //The fisrt thread sums all the contributions
    forceTotal[threadIdx.x] = f;
    __syncthreads();
    //TODO Implement a warp reduction
    if(threadIdx.x==0){
      real4 ft = make_real4(real(0.0));
      for(int i=0; i<blockDim.x; i++){
	ft += forceTotal[i];
      }
      force[p] += ft;
    }

  }



}
void ThreeBondedForces::sumForce(){
  if(nbonds>0){
    BoxUtils box(L);
    int Nparticles_with_bonds = bondParticleIndex.size();
    Bonded_ns::computeThreeBondedForce<<<Nparticles_with_bonds, TPP>>>(force.d_m, pos.d_m,
							    bondStart, bondEnd,
							    bondParticleIndex,  bondList,
							    box);

  }

}


real ThreeBondedForces::sumEnergy(){
  return 0;
}

real ThreeBondedForces::sumVirial(){
  return 0;
}


