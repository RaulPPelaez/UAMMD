#include"utils.h"
#include<stdlib.h>
#include<fstream>

Matrixf eye(uint n){
  Matrixf A(n,n);
  A.fill_with(0.0f);
  fori(0,n)
    A[i][i] = 1.0f;
  return A;
}


//Took from Fluam, adapted to use float4
void cubicLattice(float4 *pos, float L, uint N){
    float dx, dy, dz;
    int nx, ny, nz, n;
    int np = N;

    nx = ny = nz = 1;
    while((nx*ny*nz)<np){
      if((nx*ny*nz)<np) nx++;
      if((nx*ny*nz)<np) ny++;
      if((nx*ny*nz)<np) nz++;
    }
    dx = L/float(nx);
    dy = L/float(ny);
    dz = L/float(nz);

    n = 0;
    for (int i=0;i<nz;i++)
      for(int j=0;j<ny;j++)
	for(int k=0;k<nx;k++)
	  if(n<np){
	    n = n + 1;
	    pos[(n-1)].x = (k + 0.5f) * dx - L/2.0f;
	    pos[(n-1)].y = (j + 0.5f) * dy - L/2.0f;
	    pos[(n-1)].z = (i + 0.5f) * dz - L/2.0f;
	  }

}
//Took from Fluam, adapted to use float4
void cubicLattice2D(float4 *pos, float L, uint N){
    float dx, dy;
    int nx, ny, n;
    int np = N;

    nx = ny = 0;
    while((nx*ny)<np){
      if((nx*ny)<np) nx++;
      if((nx*ny)<np) ny++;
    }
    dx = L/float(nx);
    dy = L/float(ny);

    n = 0;
    for(int j=0;j<ny;j++)
      for(int k=0;k<nx;k++)
	if(n<np){
	  n = n + 1;
	  pos[(n-1)].x = (k + 0.5f) * dx - L/2.0f;
	  pos[(n-1)].y = (j + 0.5f) * dy - L/2.0f;
	  pos[(n-1)].z =  0.0f;
	}

}


Vector4 readFile(const char * fileName){
  uint N;
  ifstream in(fileName);
  in>>N;
  Vector4 p = Vector4(N);
  fori(0,N){
    in>>p[i].x>>p[i].y>>p[i].z>>p[i].w;
  }
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




