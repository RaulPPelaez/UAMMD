/*Raul P. Pelaez 2017. Constructions common to a Potential
  
  Usually a Potential can work in three modes: 
  energy, force and virial

 */
#ifndef POTENTIALBASE_CUH
#define POTENTIALBASE_CUH

namespace uammd{

  namespace Potential{   
    //The three different modes a Potential can work on
    enum class Mode{FORCE, ENERGY, VIRIAL};    
   
  }
}
#endif