.. _pairforces:
.. _short_ranged_forces:

Pair Forces
==============

Computes the forces and energies acting on the particles due to its neighbours via a provided :ref:`Potential`.  
Two particles are neighbours if their distance is less than a given cutoff radius (which can be infinity).  
PairForces will delegate its :ref:`ParameterUpdatable` behavior to the provided :ref:`Potential`.


Usage
-------

PairForces is created as any other :ref:`Interactor` module, but it needs an additional argument in the constructor with a :cpp:any:`Potential` encoding the specific inter particle interaction. Some :ref:`Potentials <Potential>` are offered by UAMMD in the source :code:`Interactor/Potential/Potential.cuh`.

.. code:: c++
	  
  #include<uammd.cuh>
  #include"Interactor/PairForces.cuh"
  #include"Interactor/Potential/Potential.cuh" 
  using namespace uammd;
  int main(int argc, char *argv[]){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    using PairForces = PairForces<Potential::LJ>;
    auto pot = make_shared<Potential::LJ>();
    {
      //Each Potential describes the pair interactions with certain parameters.
      //The needed ones are in InputPairParameters inside each potential, in this case:
      Potential::LJ::InputPairParameters par;
      par.epsilon = 1.0;
      par.shift = false;
      par.sigma = 1.0;
      par.cutOff = 2.5*par.sigma;
      pot->setPotParameters(0, 0, par);
    }  
    PairForces::Parameters params;
    params.box = box;  //Box to work on, if no box is provided it defaults to an infinite, non-periodic box.
    //You can provide a neighbour list from the outside and PairForces will use it instead of creating one.
    //auto cl = make_shared<CellList>(pd);
    //params.nl = cl; 
    auto pairforces = make_shared<PairForces>(pd, params, pot);
    //If no box or instance of a neighbour list is needed and/or the potential requires no special initialization the last parameters can be omitted.
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: A :ref:`ParticleGroup` can be provided instead of a :ref:`ParticleData` for the module to act only on a subset of particles.
	  
.. note:: As usual, this :ref:`Interactor` can be added to an :ref:`Integrator`.

A second template argument can be passed to specify a neighbour list, :ref:`CellList` is used by default.
:code:`Potential` can be any :ref:`Potential`. :cpp:`Potential::LJ` for example.

PairForces will ask its :ref:`Potential` for a cut off distance and evaluate the interaction according to it. If the cut off distance is too large, it will fall back to an :ref:`NBody` interaction.  

