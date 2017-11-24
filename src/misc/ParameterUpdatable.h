/*Raul P. Pelaez 2017.

  A parameter communication interface, anything that inherits from ParameterUpdatable can be called through update* to communicate a parameter change using a common interface. Parameters related with the particle data are communicated using ParticleData (like number of particles).

  Parameters like a simulation box or the current simulation time are updated through ParameterUpdatable.

  Interactors are ParameterUpdatable objects.

  If a module needs to be aware of a parameter change, it should override the particular virtual method. Wich will do nothing by default.
 */

#ifndef PARAMETERUPDATABLE_H
#define PARAMETERUPDATABLE_H

#include"global/defines.h"
#include"utils/Box.cuh"

namespace uammd{
  
  class ParameterUpdatable{
    
  public:
    
    virtual void updateTimeStep(real dt){};
    
    virtual void updateSimulationTime(real t){};


    
    virtual void updateBox(Box box){};



    virtual void updateTemperature(real T){};
    virtual void updateViscosity(real vis){};

    
  };

  


}

#endif
