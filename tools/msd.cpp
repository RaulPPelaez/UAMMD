/*   Raul P. Pelaez 2016
 *   Computes the Mean Square Displacement for a superpunto file
 *   MSD(dt) = < ( x_i(t0+dt) - x_i(t0) )^2 >_t0,i
 *   
 *   MSD(dt) =  (1/(Np*Nt0)) sum_t0[ sum_i( (x_i(t0+dt)-x_i(t0))^2 )]
 *
 *
 *
 *    Usage: msd -N X -Nsteps X  file 
 *    You can use the ifile macro:  msd $(ifile file) file
 *
 */


#include"Timer.h"
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

#define fori(x,y) for(int i=x; i<y;i++)
#define forj(x,y) for(int j=x; j<y;j++)
#define fork(x,y) for(int k=x; k<y;k++)


void parse_as_plain(istream &in);
void read_frame(istream &in, float *r);

void print_help();

int N = 0;
int Nframes = 0;


int main(int argc, char *argv[]){
  istream *in;
  /* Parse cli input */
  fori(0,argc){
    if(strcmp(argv[i], "-h")==0){
      print_help();
      exit(0);
    }
    else if(strcmp(argv[i], "-N")==0) N = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-Nsteps")==0) Nframes = atoi(argv[i+1]); 
  }
  if(!N || !Nframes){cerr<<"ERROR!! SOME INPUT IS MISSING!!!"<<endl; print_help(); exit(1);}

  //Look for the file, in stdin or in a given filename
  if(isatty(STDIN_FILENO)){ //If there is no pipe
    bool good_file = false;
    fori(1,argc){ //exclude the exe file
      in = new ifstream(argv[i]); //There must be a filename somewhere in the cli
      if(in->good()){good_file = true; break;}
    }
    if(!good_file){cerr<< "ERROR!, NO INPUT DETECTED!!"<<endl; print_help(); exit(1);}
  }
  else in = &cin; //If there is something being piped
  parse_as_plain(*in);

  return 0;
}


/*Read from one # to the next (not included), rx, ry, rz*/
void read_frame(istream &in, float *r){
  string line;
  stringstream ss;
  getline(in, line);
  float kk;
  for(int i=0; i<N; i++){
    getline(in, line);
    ss.clear();
    ss.str(line);
    ss>>r[3*i]>>r[3*i+1]>>r[3*i+2];
  }
}

vector<vector<float>> read_all(istream &in){

 vector<vector<float>> pos(Nframes, vector<float>(3*N,0));
  
  
  /*   MSD(dt) = < ( x_i(t0+dt) - x_i(t0) )^2 >_t0,i
   *   
   *   MSD(dt) =  (1/(Np*Nt0)) sum_t0[ sum_i( (x_i(t0+dt)-x_i(t0))^2 )]
   */

  /*Read all the file into pos*/
  fori(0,Nframes) read_frame(in, &pos[i][0]);
  return pos;
}


void parse_as_plain(istream &in){
  int Nt = Nframes;
  int Tmax = Nframes;
  float *r;
  float *r0;
  cerr<<"Reading file..."<<endl;
  auto pos = read_all(in);

  vector<int> nmsd(Nt, 0); //Number of t0 samples per dt
  vector<double> msd(3*Nt, 0); //mean square displacement per dt
  

  
  Timer tim;
  tim.tic();
  int nmsd_tid = 0;
  double msd_tid[3] = {0,0,0};
  double rij; //Distance from particle i in t0 to t0+dt
  cerr<<"Computing..."<<endl;
#pragma omp parallel for default(shared) private(r, r0, rij, msd_tid, nmsd_tid) schedule(dynamic)
  for(int dt = 1; dt<Tmax; dt++){ //For all time intervals
    #pragma omp critical
      cerr<<"\rComputing dt "<<dt;
    /*Set to zero for each thread*/
    nmsd_tid = 0;
    fork(0,3) msd_tid[k] = 0;
    for(int t0 = 0; t0<Tmax-dt; t0++){ //For al possible t0
      /*Positions in time t0*/
      r0 = &pos[t0][0];
      /*Positions in time t*/
      r = &pos[t0+dt][0];
      nmsd_tid++;
      fori(0,N)//avg in particles
	fork(0,3){ //in each direction
	  rij = r[3*i+k]-r0[3*i+k];
	  msd_tid[k] += rij*rij; 
	}
    }
    /*Normalize*/
    fork(0,3) msd_tid[k] /= float(nmsd_tid); 
    /*Write results to global arrays*/
    nmsd[dt] = nmsd_tid;
    fork(0,3) msd[3*dt+k] = msd_tid[k];
  }
  cerr<<"\nElapsed time: "<<tim.toc()<<endl;
  fori(1,Nt){
    cout<<i<<" ";
    fork(0,3) cout<<msd[3*i+k]/(float)N<<" ";
    cout<<nmsd[i]<<"\n";
  }
}




void print_help(){
  printf(" Raul P. Pelaez 2016                                                   \n"); 
  printf(" Computes the Mean Square Displacement for a superpunto file	       \n");
  printf(" MSD(dt) = < ( x_i(t0+dt) - x_i(t0) )^2 >_t0,i			       \n");
  printf(" 								       \n");
  printf(" MSD(dt) =  (1/(Np*Nt0)) sum_t0[ sum_i( (x_i(t0+dt)-x_i(t0))^2 )]      \n");
  printf("								       \n");
  printf("								       \n");
  printf("								       \n");
  printf("  Usage: msd -N X -Nsteps X  file 				       \n");
  printf("  You can use the ifile macro:  msd $(ifile file) file                 \n");





}
