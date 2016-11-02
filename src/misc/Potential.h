/*
Raul P. Pelaez 2016.
Force evaluator handler.
Takes a function to compute the force, a number of sampling points and a
 cutoff distance. Evaluates the force and saves it to upload as a texture 
 with interpolation.

TODO:
100- Use texture objects instead of creating and binding the texture in 
     initGPU.
100- This class is very bad written
 */
#ifndef POTENTIAL_H
#define POTENTIAL_H
#include<vector>
#include<fstream>
#include<functional>
#include<cuda.h>
#include"globals/defines.h"
#include"globals/globals.h"
#include"utils/utils.h"
using std::vector;
using std::ofstream;


class Potential{
public:
  Potential(){};
  Potential(std::function<real(real)> Ffoo,
	    std::function<real(real)> Efoo,
	    int N, real rc);
  size_t getSize(){ return N;}
  float *getForceData(){ return F.data();}
  float *getEnergyData(){ return E.data();}
  cudaTextureObject_t getForceTexture(){ return this->texForce;}
  cudaTextureObject_t getEnergyTexture(){ return this->texEnergy;}
  
  void print();

  uint N;
  vector<float> F;
  vector<float> E;

  cudaArray *FGPU, *EGPU;

  cudaTextureObject_t texForce, texEnergy;
  
  std::function<real(real)> forceFun;
  std::function<real(real)> energyFun;
};






#endif
