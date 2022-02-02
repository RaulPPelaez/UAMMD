External Forces
==================

The ExternalForces :ref:`Interactor` module computes the effect of an external potential acting on each particle independently.



Usage
--------

External potential interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ExternalForces :ref:`Interactor` needs to be specialized with an user-defined functor specifying the particular interaction. This functor must abide to the following interface

.. code:: c++

  //Example external potential acting on each particle independently.
  struct HarmonicWall{
    real k, zwall;
    HarmonicWall(real k, real zwall):k(k), zwall(zwall){}

    //This function will be called for each particle
    //The arguments will be modified according to what was returned by getArrays below
    __device__ ForceEnergyVirial sum(Interactor::Computables comp, real4 pos,// real mass){
      //The decision to compute energy/virial and or force should come from the members of
      // comp (comp.force, comp.energy)
      real3 force = (comp.force or comp.virial)?make_real3(0.0f, 0.0f, -k*(pos.z-zwall)):real3();
      real energy = comp.energy?real(0.5)*k*pow(pos.z-zwall, 2):0;
      real virial = comp.virial?dot(f,make_real3(pos)):0;
      return {force,energy, virial};
    }
      
    auto getArrays(ParticleData* pd){
      auto pos = pd->getPos(access::gpu, access::read);    
      return pos.begin();
      //If more than one property is needed this would be the way to do it:
      //auto mass = pd->getMass(access::gpu, access::read);
      //return std::make_tuple(pos.begin(), mass.begin());
      //In this case the additional arrays must appear as additional arguments in "sum"
    }
  
Another example, in which we showcase the fact that the ExternalForces functor might be :cpp:class:`ParameterUpdatable`.

.. code:: c++

  struct ReallyComplexExternalForce: public ParameterUpdatable{
    real time = 0;
  
    __device__ ForceEnergyVirial sum(Interactor::Computables comp, real4 pos, real3 vel, int id, real mass){
      //The member variable "time" is available here
      //...
      //real3 force =...
      //real energy =...
      //real virial =...
      return {force, energy, virial};
    }
    
    auto getArrays(ParticleData* pd){
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto vel = pd->getVel(access::location::gpu, access::mode::read);
      auto id = pd->getId(access::location::gpu, access::mode::read);
      auto mass = pd->getMass(access::location::gpu, access::mode::read);
      return {pos.begin(), vel.begin(), id.begin(), mass.begin()};
    }

    //This function is part of the ParameterUpdatable interface
    virtual updateSimulationTime(real newTime) override{
      this->time = newTime;
    }
  };


.. cpp:class:: ForceEnergy

   A POD type holding members for the force, energy and virial

   .. cpp:member:: real3 force

   .. cpp:member:: real energy

   .. cpp:member:: real virial


Creating the :ref:`Interactor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the potential functor is crafted, the ExternalForces :ref:`Interactor` can be created as the rest, providing an instance to the functor to the constructor

.. code:: c++

  //You can use this function to create an interactor that can be directly added to an integrator
  std::shared_ptr<Interactor> createExternalPotentialInteractor(std::shared_ptr<ParticleData> pd){
    //You can pass an instance of the specialization as a shared_ptr, which allows you to modify it from outside the interactor module at any time.
    auto gr = std::make_shared<HarmonicWall>(1.0, 1.0);
    auto ext = std::make_shared<ExternalForces<HarmonicWall>>(pd, gr);
    return ext;  
  }
  
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...    
    auto ext = createExternalPotentialInteractor(pd);
    //We can now compute any of the possible particle quantities directly
    ext->sum({.force=true, .energy=false, .virial=false});
    //Or provide the Interactor to some Integrator
    //Assume some Integrator is available named "intergrator"
    integrator->addInteractor(ext);
    integrator->forwardTime();
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. note:: As usual, this :ref:`Interactor` can be added to an :ref:`Integrator`.

.. hint:: A :ref:`ParticleGroup` can be provided instead of a :ref:`ParticleData` for the module to act only on a subset of particles.
	  
