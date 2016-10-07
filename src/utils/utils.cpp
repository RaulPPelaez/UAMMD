#include"utils.h"
#include<stdlib.h>
#include<fstream>
#include"third_party/bravais/bravais.h"

Matrixf eye(uint n){
  Matrixf A(n,n);
  A.fill_with(0.0f);
  fori(0,n)
    A[i][i] = 1.0f;
  return A;
}

std::ostream& operator<<(std::ostream& out, const float3 &f){
  return out<<f.x<<" "<<f.y<<" "<<f.z;
}
std::ostream& operator<<(std::ostream& out, const float4 &f){
  return out<<f.x<<" "<<f.y<<" "<<f.z<<" "<<f.w;
}

//typedef enum {sc, bcc, fcc, dia, hcp, sq, tri} lattice;

Vector4 cubicLattice(float3 L, uint N){
  cerr<<"Starting in a cubic Lattice...";
  Vector4 pos(N);
  pos.fill_with(make_float4(0.0f));

  Bravais((float *) pos.data,
	  sc,/*Simple Cubic*/
	  N,
	  L.x, L.y, L.z,
	  0.0f, /*Color*/
	  NULL, NULL, /*Basis and vector files*/
	  true); /*Keep aspect ratio*/
  fori(0,N)
    pos[i] += make_float4(0.5f, 0.5f, 0.5f, 0.0f);
  
  return pos;
}


Vector4 readFile(const char * fileName){
  cerr<<"Reading initial positions from file...";
  uint N;
  ifstream in(fileName);
  in>>N;
  Vector4 p = Vector4(N);
  fori(0,N){
    in>>p[i].x>>p[i].y>>p[i].z>>p[i].w;
  }
  cerr<<"\tDONE!"<<endl;
  return p;
}



#define RANDESP (rand()/(float)RAND_MAX)
#define RANDL2 (RANDESP-0.5f)

bool randInitial(float4 *pos, float L, uint N){
  srand(time(NULL));
  pos[0] = make_float4(  RANDL2*L, RANDL2*L, RANDL2*L, 0.0f);
  float4 tempos, rij;
  bool accepted = true;
  uint trials = 0;
  float r2;
  cerr<<endl;
  fori(1,N){
    cerr<<"\rIntroducing "<<i<<"     ";
    tempos = make_float4(  RANDL2*L, RANDL2*L, RANDL2*L, 0.0f);
    forj(0,i+1){
      rij = tempos-pos[j];
      rij -= floorf(rij/L+0.5f)*L; 
      r2 = dot(rij, rij);
      if(r2<1.0f){
	accepted = false;
	break;
      }
    }
    if(!accepted){
      i--;
      trials++;
      accepted = true;
    }
    else{
      pos[i] = tempos;
      trials = 0;
    }
    if(trials > N*1000){
      return false;
    }

  }

  cerr<<endl;
  return true;
}




