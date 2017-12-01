/*Raul P. Pelaez 2016
 *Computes the 3D Radial function distribution, 
 * averaged for all the particles and frames in the superpunto like input
 *
 * Usage: gdr -L [Lx Ly Lz || L] -rcut [rcut=5.0] -N [N] -nbins [nbins=100] -Nsteps [Nsteps=1]
 *  
 *
 *
 *
 */

#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<sstream>
#include<string.h>
#include<iostream>
#include <unistd.h>
#include<vector>
#include<cmath>
#include<omp.h>
using namespace std;
// PIPE HACK INSIDE
//tags: pipe input check

#define fori(x,y) for(int i=x; i<y; i++)
#define forj(x,y) for(int j=x; j<y; j++)

void parse_as_plain(istream &in, int Nsteps);
void read_frame(istream &in, float *r);
void rdf(const float *pos, bool write);

void apply_pbc(float *r);
float dist(const float *r1, const float *r2);

void print_help();

float L[3];
int N = 0;
float rcut = 5.0;
int RDF_SAMPLES = 100;

int main(int argc, char *argv[]){
  istream *in;
  int Nsteps = 1;
  float Lread = 0.0;
  float Lx=0, Ly=0, Lz=0;
  /* Parse cli input */
  //Check if L is in input
  fori(0,argc){
    /*With -L you can have one or three numbers*/
    if(strcmp(argv[i], "-L")==0){
      Lx = strtod(argv[i+1], NULL);
      if(argc>i+3){
	Ly = strtod(argv[i+2], NULL);
	Lz = strtod(argv[i+3], NULL);
      }
      if(!Ly || !Lz ) Lread = Lx;
    }
    if(strcmp(argv[i], "-Lx")==0)              Lx = strtod(argv[i+1], NULL);
    if(strcmp(argv[i], "-Ly")==0)              Ly = strtod(argv[i+1], NULL);
    if(strcmp(argv[i], "-Lz")==0)              Lz = strtod(argv[i+1], NULL);
    if(strcmp(argv[i], "-rcut")==0)            rcut = strtod(argv[i+1], NULL);
    if(strcmp(argv[i], "-N")==0)               N = atoi(argv[i+1]);
    if(strcmp(argv[i], "-nbins")==0)           RDF_SAMPLES = atoi(argv[i+1]);
    if(strcmp(argv[i], "-Nsteps")==0)          Nsteps = atoi(argv[i+1]);
    
    if(strcmp(argv[i], "-h")==0){ print_help(); exit(0); }

  }
  if(!Lread && !(Lx&&Ly&&Lz)){cout<<"ERROR!! NO VALID L WAS GIVEN!!"<<endl; print_help(); exit(1);}
  if(!N){cout<<"ERROR!! NO VALID N WAS GIVEN!!"<<endl; print_help(); exit(1);}

  //Look for the file, in stdin or in a given filename
  if(isatty(STDIN_FILENO)){ //If there is no pipe
    bool good_file = false;
    fori(1,argc){ //exclude the exe file
      in = new ifstream(argv[i]); //There must be a filename somewhere in the cli
      if(in->good()){good_file = true; break;}
    }
    if(!good_file){cout<< "ERROR!, NO INPUT DETECTED!!"<<endl; print_help(); exit(1);}
  }
  else in = &cin; //If there is something being piped

  if(!Lx||!Ly||!Lz)  L[0] = L[1] = L[2] = Lread;
  else{
    L[0] = Lx;
    L[1] = Ly;
    L[2] = Lz;
  }
  parse_as_plain(*in, Nsteps);

  return 0;
}

#include<limits>
void rdf(const float *r, bool write=false){
  static vector<unsigned long long int> gdr(RDF_SAMPLES, 0);
  static int avg_counter = 0;

  float rijmod;                                // distance between two particles
  float ratio = rcut/(float)(RDF_SAMPLES); //dr, distance interval of one bin
  int bin=0; 
  
  #pragma omp parallel for schedule(dynamic) private(rijmod, bin) 
  for(int i=0; i<N-1; i++){    
    for(int j=i+1; j<N; j++){
      rijmod = sqrt(dist(&r[3*j], &r[3*i]));
      if(rijmod<rcut){
	bin=floor((rijmod/ratio));
	#pragma omp atomic update
	gdr[bin]++;
      }	   
    }
  }
  avg_counter++;
  if(write){
    cout<<"\n";
    float V = L[0]*L[1]*L[2];
    float normalization = (float)avg_counter*2.0f*3.14159265359f*ratio*N*N/V;
    float R;
    for(int i=1; i<RDF_SAMPLES; i++){
      R = (i-0.5f)*ratio;
      cout<<R<<" "<<(double)gdr[i-1]/(normalization*R*R)<<"\n";
    }
    cout<<flush;
  }


}



void parse_as_plain(istream &in, int Nsteps){
  vector<float> pos(3*N, 0);// = new float[3*N];
  read_frame(in, &pos[0]);
  int write_frames = int(Nsteps/100.0)+1;
  fori(0, Nsteps){
    if( int(100.0*double(i)/Nsteps) != int(100.0*double(i+1)/Nsteps))
      cerr<<"                                      \rComputing..."<<int(100.0*double(i)/Nsteps)<<"%"; 
    rdf(&pos[0], i%write_frames==0);
    if( int(100.0*double(i)/Nsteps) != int(100.0*double(i+1)/Nsteps))
      cerr<<"   Reading step...";
    read_frame(in, &pos[0]);
  }
  cerr<<"DONE!"<<endl;
      rdf(&pos[0], true);
}

void read_frame(istream &in, float *r){
  string line;
  stringstream ss;
  getline(in, line);
  for(int i=0; i<N; i++){
    getline(in, line);
    ss.clear();
    ss.str(line);
    ss>>r[3*i]>>r[3*i+1]>>r[3*i+2];
  }
}

static inline int floor(float x){
  int i = (int)x;
  return i-(i>x);
}

inline void apply_pbc(float *r){
  fori(0,3) r[i] -= floor( r[i]/L[i] +0.5 )*L[i];
}

float dist(const float *r1, const float *r2){
  float r12[3];
  for(int i=0; i<3; i++) r12[i] = r2[i]-r1[i];
  apply_pbc(r12);
  return r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2];
}




void print_help(){
printf("Raul P. Pelaez 2016                                                                           \n");
printf("  Computes the 3D Radial function distribution, 						      \n");
printf("  averaged for all the particles and frames in the superpunto like input		      \n");
printf("											      \n");
printf("  Usage: gdr -L [Lx Ly Lz || L] -rcut [rcut=5.0] -N [N] -nbins [nbins=100] -Nsteps [Nsteps=1] \n");
  
}



