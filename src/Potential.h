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
#ifndef POTENTIAL_H
#define POTENTIAL_H
#include<vector>
#include<fstream>
#include<functional>

#define fori(x,y) for(int i=x; i<y; i++)
#define forj(x,y) for(int j=x; j<y; j++)
using std::vector;
using std::ofstream;


class Potential{
public:
  Potential(){};
  Potential(std::function<float(float)> foo, int N, float rc);
  size_t getSize(){ return N*sizeof(float);}
  float *getData(){ return F.data();}
  void print();
private:

  vector<float> F;
  uint N;
  std::function<float(float)> forceFun;
};






#endif
