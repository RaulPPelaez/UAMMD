/*
Raul P. Pelaez 2016.
Force evaluator handler.
Takes a function to compute the force, a number of sampling points and a
 cutoff distance. Evaluates the force and saves it to upload as a texture 
 with interpolation.

TODO:
100- Use texture objects instead of creating and binding the texture in 
     initGPU.

 */
#include"Potential.h"
#include<fstream>

Potential::Potential(std::function<float(float)> Ffoo,
		     std::function<float(float)> Efoo,
		     int N, float rc):
  forceFun(Ffoo), energyFun(Efoo), N(N)
{

  F.resize(N);
  E.resize(N);

  float dr2 = rc*rc/(float)N;
  float r2 = 0.5f*dr2;

  fori(0,N){
    F[i] = forceFun(r2);
    E[i] = energyFun(r2);
    r2 += dr2;
  }
  //F[0] = 0xffffff;
  F[N-1] = 0.0f;
  E[N-1] = 0.0f;
  
}

void Potential::print(){
  ofstream out("potential.dat");
  fori(0,N) out<<F[i]<<" "<<E[i]<<"\n";
  out.close();
}
