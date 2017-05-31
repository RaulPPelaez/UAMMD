#include"utils.h"
#include<stdlib.h>
#include<fstream>
#include<sstream>
#include<vector>
#include <iterator>
int checkFlag(int argc, char *argv[], const char *flag){
  fori(1, argc){
    if(strcmp(flag, argv[i])==0) return i;
  }
  return -1;
}

vector<string> stringSplit(string s){
  std::stringstream ss(s);
  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  return std::vector<std::string>(begin, end);
}

Matrixf eye(uint n){
  Matrixf A(n,n);
  A.fill_with(0.0f);
  fori(0,n)
    A[i][i] = 1.0f;
  return A;
}

std::ostream& operator<<(std::ostream& out, const real3 &f){
  return out<<f.x<<" "<<f.y<<" "<<f.z;
}
std::ostream& operator<<(std::ostream& out, const real4 &f){
  return out<<f.x<<" "<<f.y<<" "<<f.z<<" "<<f.w;
}

//typedef enum {sc, bcc, fcc, dia, hcp, sq, tri} lattice;

Vector4 initLattice(real3 L, uint N, BRAVAISLAT lat){
  
  cerr<<"Starting in a ";


  switch(lat){
  case sc:   cerr<<"cubic";     break;
  case fcc:  cerr<<"FCC";       break;
  case bcc:  cerr<<"BCC";       break;
  case hcp:  cerr<<"HCP";       break;
  case tri:  cerr<<"triangular";break;
  case sq:   cerr<<"square";    break;
  case dia:  cerr<<"zincblende";break;
  }
  
  cerr<<" Lattice...";
  Vector<float4> pos(N);
  pos.fill_with(make_float4(0.0f));

  Bravais((float *) pos.data,
	  lat,/*lattice type*/
 	  N,
 	  L.x, L.y, L.z,
	  0.0f, /*Color*/
 	  NULL, NULL, /*Basis and vector files*/
 	  false); /*Keep aspect ratio*/
  fori(0,N){
    pos[i] += make_float4(0.56f, 0.56f, 0.56f, 0.0f);
    if(L.z==real(0.0)) pos[i].z = 0.0f;
  }

  Vector4 pos_real(N);
  fori(0,N)
    pos_real[i] = make_real4(pos[i].x, pos[i].y, pos[i].z, real(0.0));

  cerr<<"\tDONE!"<<endl;
  return pos_real;
}

 // Vector4 cubicLattice(real3 l, uint N){
 //   real  L = l.x;
 //   Vector4 pos(N);
 //   real dx, dy, dz;
 //   int nx, ny, nz, n;
 //   int np = N;

 //   nx = ny = nz = 1;
 //   while((nx*ny*nz)<np){
 //     if((nx*ny*nz)<np) nx++;
 //     if((nx*ny*nz)<np) ny++;
 //     if((nx*ny*nz)<np) nz++;
 //   }
 //   dx = L/real(nx);
 //   dy = L/real(ny);
 //   dz = L/real(nz);

 //   n = 0;
 //   for (int i=0;i<nz;i++)
 //     for(int j=0;j<ny;j++)
 //       for(int k=0;k<nx;k++)
 //       if(n<np){
 //         n = n + 1;
 //         pos[(n-1)].x = (k + 0.5f) * dx - L/2.0f;
 //         pos[(n-1)].y = (j + 0.5f) * dy - L/2.0f;
 //         pos[(n-1)].z = (i + 0.5f) * dz - L/2.0f;
 //       }
 //   return pos;
 // }                                      



Vector4 readFile(const char * fileName){
  cerr<<"Reading initial positions from "<<fileName<<" file...";
  uint N;
  ifstream in(fileName);
  if(!in.good()){ cerr<<"\tERROR: File not found!!"<<endl; exit(1);}
  in>>N;
  Vector4 p = Vector4(N);
  fori(0,N){
     in>>p[i].x>>p[i].y>>p[i].z>>p[i].w;
  }
  cerr<<"\tDONE!"<<endl;
  return p;
}



#define RANDESP (rand()/(real)RAND_MAX)
#define RANDL2 (RANDESP-0.5)

bool randInitial(real4 *pos, real L, uint N){
  cerr<<"Starting in a random initial configuration..."<<endl;
  srand(time(NULL));
  pos[0] = make_real4(  RANDL2*L, RANDL2*L, RANDL2*L, 0.0);
  real4 tempos, rij;
  bool accepted = true;
  uint trials = 0;
  real r2;
  cerr<<endl;
  fori(1,N){
    cerr<<"\rIntroducing "<<i<<"     ";
    tempos = make_real4(  RANDL2*L, RANDL2*L, RANDL2*L, 0.0);
    forj(0,i+1){
      rij = tempos-pos[j];
      rij -= floorf(rij/L+real(0.5))*L; 
      r2 = dot(rij, rij);
      if(r2<real(1.0)){
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
  cerr<<"\tDONE!"<<endl;
  return true;
}



namespace printUtils{
  std::string prettySize(size_t size) {
    static const char *SIZES[] = { "B", "KB", "MB", "GB" };
    int div = 0;
    size_t rem = 0;

    while (size >= 1024 && div < (sizeof(SIZES)/ sizeof (*SIZES))) {
      rem = (size % 1024);
      div++;
      size /= 1024;
    }

    double size_d = (float)size + (float)rem / 1024.0;
    std::string result = std::to_string(size_d) + " " + std::string(SIZES[div]);
    return result;
  }
}
