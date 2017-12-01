/*Raul P. Pelaez 2016. Elastic Network Model bond constructor.

  Compile with:
  $ g++ -std=c++11 -Ofast Elastic_Network_Model.cpp -o enm

  Takes an UAMMD like initial position input and prints an UAMMD format bond list according to Elastic Network Model.
  
  Usage:
     enm fileName rcut Kspring Lx Ly Lz > particles.2bond
   
     rcut: maximum distance for the ENM
     Kspring: spring constant for the bonds
     LX: Box size, put -1 to have an infinite box.

  fileName has the following format:
  N
  x y z c
  .
  .
  .

  The output has the following format:
  Nbonds
  i j K r0
  .
  .
  .

The list of bonds is not ordered and each bond is printed only once. So only ij and not ji.

To convert the output to fluamlike bond input (duplicated bonds):
   $ tail -n+2 particles.2bond | awk '{ print $1,$2,$3,$4; print $2,$1,$3,$4}' | sort -g -k1 -k2 > fluam.2bonds
 */
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<cmath>
using namespace std;

//Vector type for convinience
struct float3{
  float x,y,z;
};
//substract two float3
float3 operator -(float3 a, float3 b){
  return {a.x-b.x, a.y-b.y, a.z-b.z};

}
//read a float3 from file using >>
istream & operator >>(istream &in, float3 &a){
  in>>a.x>>a.y>>a.z;
  return in;
}

//Bond struct containing bond information
struct Bond{
  int i,j; //The two particles involved
  float k, r0; //Spring constant and equilibrium distance
};
//Write a bond to disk using <<
ostream & operator << (ostream &out, Bond b){
  out<<b.i<<" "<<b.j<<" "<<b.k<<" "<<b.r0;
  return out;
}

//dot product between two float3
float dot(float3 a, float3 b){
  return a.x*b.x+a.y*b.y+a.z*b.z;
}

float Lx, Ly, Lz;

void apply_pbc(float3 &r12){
  if(Lx>0)
    r12.x -= floorf(r12.x/Lx+0.5)*Lx;
  if(Ly>0)
    r12.y -= floorf(r12.y/Ly+0.5)*Ly;
  if(Lz>0)
    r12.z -= floorf(r12.z/Lz+0.5)*Lz;
}

int main(int argc, char *argv[]){
  cerr<<"WARNING: Remember that the output needs further processing if intented to be used for FLUAM. Run enm for info"<<endl;
  if(argc<7){
    printf("  Usage:                                                                                                           \n"
	   "     enm fileName rcut Kspring Lx Ly Lz> particles.2bond						       \n"
	   "   														       \n"
	   "     rcut: maximum distance for the ENM									       \n"
	   "     Kspring: spring constant for the bonds									       \n"
	   "     LX: Box size, put -1 to ignore  									       \n"
	   "														       \n"
	   "  fileName has the following format:									       \n"
	   "  N														       \n"
	   "  x y z c													       \n"
	   "  .														       \n"
	   "  .														       \n"
	   "  .														       \n"
	   "														       \n"
	   "  The output has the following format:									       \n"
	   "  Nbonds													       \n"
	   "  i j K r0													       \n"
	   "  .														       \n"
	   "  .														       \n"
	   "  .														       \n"
	   "														       \n"
	   "  The list of bonds is not ordered and each bond is printed only once. So only ij and not ji.		       \n"
	   "														       \n"
	   "  To convert the output to fluamlike bond input (duplicated bonds):						       \n"
	   "    $ tail -n+2 particles.2bond | awk '{ print $1,$2,$3,$4; print $2,$1,$3,$4}' | sort -g -k1 -k2 > fluam.2bonds   \n");
    return 0;    
  }


  Lx = stod(argv[4], NULL);
  Ly = stod(argv[5], NULL);
  Lz = stod(argv[6], NULL);
  /*Open the first argument as the input file*/
  ifstream in(argv[1]);

  /*Read K, rcut and Nper_helix from argument list*/
  float rcut = stod(argv[2], NULL); //Rcut of the ENM
  float K = stod(argv[3], NULL);    //Spring constant
  
  int N; //number of particles in the input file
  
  /*Read N from file*/
  in>>N;

  /*Read all the particles from the file*/
  vector<float3> pos(N);
  float kk;
  for(int i=0; i<N; i++)
    in>>pos[i]>>kk;
  
  /*Store the bonds*/
  vector<Bond> bonds;
  
  float3 posi, posj, posij;





  for(int i=0; i<N; i++){
    /*Get the position of first particle i*/
    posi = pos[i];
    for(int j=i+1; j<N; j++){
      /*Get the position of particle j*/
      posj = pos[j];
      /*Compute the squared distance between them*/
      posij = posj-posi;

      apply_pbc(posij);
      
      float r2 = dot(posij, posij);
      
      /*If they are close enough add a bond to the system between i and j*/
      if(r2<= rcut*rcut){
	Bond b;
	b.i = i;
	b.j = j;
	b.r0 = sqrt(r2);
	b.k = K;
	/*Add to the vector of bonds*/
	bonds.push_back(b);
      }
    }
  }


  /*Print the bond list*/
  /*Nbonds*/
  cout<<bonds.size()<<endl;
  /*See the >> operator above to see the format*/
  for(int i=0; i<bonds.size(); i++){
    cout<<bonds[i]<<endl;
  }
  
  
  return 0;
}
