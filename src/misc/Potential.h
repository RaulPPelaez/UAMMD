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
#include"utils/utils.h"
using std::vector;
using std::ofstream;


class Potential{
public:
  Potential(){};
  Potential(std::function<float(float)> Ffoo,
	    std::function<float(float)> Efoo,
	    int N, float rc);
  size_t getSize(){ return N*sizeof(float);}
  float *getForceData(){ return F.data();}
  float *getEnergyData(){ return F.data();}
  void print();
private:

  uint N;
  vector<float> F;
  vector<float> E;

  std::function<float(float)> forceFun;
  std::function<float(float)> energyFun;
};






#endif
