/*
  Raul P. Pelaez 2015

  Usage:
  ./pbc [opt] <file
  ./pbc [opt] filename
  
  [opt]:
    -L X                 To specify equal box length, X, in the three directions
    -L X Y Z             To specify independent lengths along each direction
    -Lx X -Ly Y -Lz Z    Same as above
  Periodic Boundary Conditions parser. Takes a file in the form:
  #
  x y z
  .
  .
  .
  #
  x2 y2 z2
  .
  .
  .
  (any superpunto file with only xyz or xyrc columns will work)
  and prints the same file with positions reduced to a box of size L
*/
#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<sstream>
#include<string.h>
#include<iostream>
#include<unistd.h>

using namespace std;
// PIPE HACK INSIDE
//tags: pipe input check

#define fori(x,y) for(int i=x; i<y;i++)

bool iscomment(std::string line){
  line.erase(0, line.find_first_not_of(" \t\r\n"));
  char iscomment = line[0];
  return iscomment=='#';
}

void parse_as_plain(istream &in, float *L);
void parse_as_binary(istream &in, float L);

void print_help();

int main(int argc, char *argv[]){
  istream *in;
  float Lread = 0;
  float Lx = 0, Ly = 0, Lz = 0;
  float L[3] = {0,0,0};
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
    if(strcmp(argv[i], "-Lx")==0){
      Lx = strtod(argv[i+1], NULL);
    }
    if(strcmp(argv[i], "-Ly")==0){
      Ly = strtod(argv[i+1], NULL);
    }
    if(strcmp(argv[i], "-Lz")==0){
      Lz = strtod(argv[i+1], NULL);
    }
    if(strcmp(argv[i], "-h")==0){
      print_help();
      exit(0);
    }
    
  }
  if(!Lread && !(Lx&&Ly&&Lz)){cerr<<"ERROR!! NO VALID L WAS GIVEN!!"<<endl; print_help(); exit(1);}

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
  
  if(!Lx||!Ly||!Lz)  Lx = Ly = Lz = Lread;
  L[0] = Lx;
  L[1] = Ly;
  L[2] = Lz;
  parse_as_plain(*in, L);

  return 0;
}

static inline int floor(float x){
  int i = (int)x;
  return i-(i>x);
}

void parse_as_plain(istream &in, float *L){
  string line, restofline;
  stringstream ss;
  float r[3];
  while(getline(in, line)){
    /*Read a line*/
    if(!iscomment(line)){
      ss.clear();
      ss.str(line);
      fori(0,3)ss>>r[i];
      //      ss>>x>>y>>z;
      getline(ss,restofline);
      
      fori(0,3) r[i] -= floor( r[i]/L[i] +0.5 )*L[i];
      fori(0,3) cout<<r[i]<<" ";
      cout<<restofline<<"\n";
      // /*Reduce to -0.5 0.5 box*/
      // x /= Lx;
      // y /= Ly;
      // z /= Lz;
      // /*Apply PBC*/
      // x-=int( ( (x<0)?-0.5:0.5 ) + x);
      // y-=int( ( (y<0)?-0.5:0.5 ) + y);
      // z-=int( ( (z<0)?-0.5:0.5 ) + z);
      // /*Write*/
      // cout<<x*Lx<<" "<<y*Ly<<" "<<z*Lz<<" "<<restofline<<"\n";
      
    }
    else{
      cout<<"#Lx="<<L[0]/2.0f<<";Ly="<<L[1]/2.0f<<";Lz="<<L[2]/2.0f<<";"<<endl;
    }
  }
}




void parse_as_binary(istream &in, float L){






}
void print_help(){

  printf("Raul P. Pelaez 2015                                                                   \n");
  printf("											\n");
  printf("Usage:										\n");
  printf("./pbc [opt] <file									\n");
  printf("./pbc [opt] filename									\n");
  printf("											\n");
  printf("[opt]:										\n");
  printf("  -L X                 To specify equal box length, X, in the three directions	\n");
  printf("  -L X Y Z             To specify independent lengths along each direction		\n");
  printf("  -Lx X -Ly Y -Lz Z    Same as above							\n");
  printf("Periodic Boundary Conditions parser. Takes a file in the form:			\n");
  printf("#											\n");
  printf("x y z											\n");
  printf(".											\n");
  printf(".											\n");
  printf(".											\n");
  printf("#											\n");
  printf("x2 y2 z2										\n");
  printf(".											\n");
  printf(".											\n");
  printf(".											\n");
  printf("(any superpunto file with only xyz or xyrc columns will work)				\n");
  printf("and prints the same file with positions reduced to a box of size L                    \n");

}
