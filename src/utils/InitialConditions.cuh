/*Raul P. Pelaez 2017.
  Some utilities to create initial configurations of particles.


 */
#ifndef INITIALCONDITIONS_CUH
#define INITIALCONDITIONS_CUH

#include"utils/vector.cuh"
#include"global/defines.h"
#include"third_party/bravais/bravais.h"


//Available lattices:
//typedef enum {sc, bcc, fcc, dia, hcp, sq, tri} lattice;

std::vector<real4> initLattice(real3 L, uint N, BRAVAISLAT lat){
  
  // cerr<<"Starting in a ";

  // switch(lat){
  // case sc:   cerr<<"cubic";     break;
  // case fcc:  cerr<<"FCC";       break;
  // case bcc:  cerr<<"BCC";       break;
  // case hcp:  cerr<<"HCP";       break;
  // case tri:  cerr<<"triangular";break;
  // case sq:   cerr<<"square";    break;
  // case dia:  cerr<<"zincblende";break;
  // }
  
  // cerr<<" Lattice...";
  std::vector<float4> pos(N, make_real4(0));


  Bravais((float *) pos.data(),
	  lat,/*lattice type*/
 	  N,
 	  L.x, L.y, L.z,
	  0.0f, /*Color*/
 	  NULL, NULL, /*Basis and vector files*/
 	  false); /*Keep aspect ratio*/
  for(int i = 0; i<N; i++){
    pos[i] += make_float4(0.56f, 0.56f, 0.56f, 0.0f);
    if(L.z==real(0.0)) pos[i].z = 0.0f;
  }

  std::vector<real4> pos_real(N);
  for(int i = 0; i<N; i++)
    pos_real[i] = make_real4(pos[i].x, pos[i].y, pos[i].z, real(0.0));

  //  cerr<<"\tDONE!"<<endl;
  return pos_real;
}





#endif