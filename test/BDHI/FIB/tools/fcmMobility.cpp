//Raul P. Pelaez 1028. Taken from after eq. 20 in [1]
//  v = M·F
// Put two particles at a distance r, pull particle 0 with Fx = 1 and particle 1 with Fx = -1
//
//hydrodynamic force acting on particle 0 -> v = M0·F_0 + M(r)·F_1 = M0-M((r)
//
//This code returns:   M0(L)/v = M0(L)/(M0-M(r))
//References:
// [1] Hydrodynamic fluctuations in quasi-two dimensional diffusion. RP Pelaez et. al.
#include<iostream>
#include<stdlib.h>
#include<cmath>
#include<iomanip>

double rh, vis;
double f(double r){return (1.0/(8.0*M_PI*vis*r)) * (  (1+2*rh*rh/(M_PI*r*r))*erf(r*sqrt(M_PI)/(2*rh)) - 2*rh/(M_PI*r)*exp(-M_PI*r*r/(4*rh*rh)) );}
double g(double r){ return (1.0/(8.0*M_PI*vis*r)) * (  (1-6*rh*rh/(M_PI*r*r))*erf(r*sqrt(M_PI)/(2*rh)) + 6*rh/(M_PI*r)*exp(-M_PI*r*r/(4*rh*rh)) );}
int main(int argc, char *argv[]){

  rh = std::stod(argv[1]);
  vis = std::stod(argv[2]);
  double r = std::stod(argv[3]);

  double M0 = 1/(6*M_PIl*vis*rh);
  double corr = 1;//(1-2.8372979*rh/L);
  
  double fr = f(r);
  double gr = g(r);
    
  std::cout<<std::setprecision(15)<<r<<" "<<(fr+gr-M0)<<"\n";

  return 0;
}
