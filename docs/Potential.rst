Potential
==========

This interface is just a connection between the :ref:`Transverser` concept and the :ref:`PairForces`  :ref:`Interactor` module.

.. note:: Additionally, Potential aids with one limitation of the CUDA programming language and GPU programming in general. On the one hand, register memory in a GPU is quite limited, so it is not a good idea to use large objects in a kernel. On the other, there are some technical details that prevent certain objects from existing in a GPU kernel. For example, objects are provided by value to a kernel, which can incur undesired copies and/or destructors being called. Thus, it is some times worth it to make a conceptual and programmatic separation between CPU and GPU objects.
	  
:ref:`Transversers` are GPU objects, while :ref:`Interactors` or :ref:`Potentials` are meant to be used in the CPU.
     
Furthermore, while :ref:`Transverser` describes a very general computation, :ref:`Potential` only holds the logic on how to compute forces, energies and/or virials.
:ref:`Potential` s are used to provide force-, energy- and/or virial-calculating :ref:`Transversers` to an :ref:`Interactor`

.. hint:: The :ref:`PairForces` module needs a :ref:`Potential` encoding the specific particle interaction. 

The :ref:`Potential` interface is straightforward, requiring only two functions:

.. cpp:class:: Potential
	   
   .. cpp:function:: real getCutOff();
		     
      This function must return the highest cut off distance required by the interaction.

   .. cpp:function:: Transverser getTransverser(Interactor::Computables comp, Box box, std::shared_ptr<ParticleData> pd);
  
      This function must provide an instance of a :cpp:any:`Transverser` that, using the provided :cpp:class:`ParticleData` and :cpp:class:`Box` instances, computes anything requested by the :cpp:any:`Interactor::Computables` list (mainly forces, energies and/or virials).
      The return type of this function, called Transverser here, can be any valid :cpp:any:`Transverser` with only one restriction: The return type of the :cpp:`compute` function must be of type :cpp:`ForceEnergyVirial`.

   .. cpp:class:: ForceEnergyVirial

      A POD type that can hold a value for a force, energy and virial.
      
      .. cpp:member:: real3 force

      .. cpp:member:: real energy

      .. cpp:member:: real virial
		      

Example
--------

An example :cpp:any:`Potential` that computes Lennard-Jones forces, energies and/or virials. For simplicity, all relevant parameters are hardcoded here. In particular, :math:`\sigma_{lj} = 1`, :math:`\epsilon_{lj}=1` and the cut off is set at :math:`r_c = 2.5\sigma = 2.5`. The potential here defined (called :code:`SimpleLJ`) calculates forces, energies and virials. Note, however, that it does so only when provided to a :ref:`PairForces`  :ref:`Interactor` and, subsequently, to an :ref:`Integrator`. In other words, we use :ref:`Potentials` to define an :ref:`Interactor`, which will be used  by an :ref:`Integrator` to calculate forces, energies, etc.

.. code:: c++
	  
  //Some functions to compute forces/energies
  __device__ real lj_force(real r2){
    const real invr2 = real(1.0)/r2;
    const real invr6 = invr2*invr2*invr2;
    const real fmoddivr = (real(-48.0)*invr6 + real(24.0))*invr6*invr2;
    return fmoddivr;
  }

  __device__ real lj_energy(real r2){
    const real invr2 = real(1.0)/r2;
    const real invr6 = invr2*invr2*invr2;
    return real(4.0)*(invr6 - real(1.0))*invr6;
  }

  //A Transverser for computing, energy, virial and force (or just some of them).
  //It is the simplest form of Transverser, as it only provides the "compute" and "set" functions
  //When constructed, if the i_force, i_energy or i_virial pointers are null that computation will be avoided.
  struct LJTransverser{
    real4 *force;
    real *virial;
    real* energy;
    Box box;
    real rc;
    LJTransverser(Box i_box, real i_rc, real4* i_force, real* i_energy, real* i_virial):
    box(i_box), rc(i_rc), force(i_force), virial(i_virial), energy(i_energy){
      //All members will be available in the device functions
    }
    //For each pair computes and returns the LJ force and/or energy and/or virial based only on the positions
    __device__ ForceEnergyVirial compute(real4 pi, real4 pj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      if(r2>0 and r2< rc*rc){
        real3 f;
        real v, e;        
        f = (force or virial)?lj_force(r2)*rij:real3();	
        v = virial?dot(f, rij):0;
        e = energy?lj_energy(r2):0;
        return {f,e,v};
      }
      return {};
    }
    //Note that we are making use of the default behaviors by not defining an accumulate or zero functions.
    __device__ void set(int id, ForceEnergyVirial total){
      //Write the total result to memory if the pointer was provided
      if(force)  force[id] += make_real4(total.force, 0);
      if(virial) virial[id] += total.virial;
      if(energy) energy[id] += total.energy;
    }
  };

  //A simple LJ Potential, can compute force, energy, virial or all at the same time using the above Transverser.
  struct SimpleLJ{
    real rc = 2.5;
    //A function returning the maximum required cut off for the interaction
    real getCutOff(){
      return rc;
    }
    //This function is required to provide a Transverser that has the ability to compute the requested Computables.
    auto getTransverser(Interactor::Computables comp,
                        Box box,
                        std::shared_ptr<ParticleData> pd){
      auto force = comp.force?pd->getForce(access::gpu, access::readwrite).raw():nullptr;
      auto energy = comp.energy?pd->getEnergy(access::gpu, access::readwrite).raw():nullptr;
      auto virial = comp.virial?pd->getVirial(access::gpu, access::readwrite).raw():nullptr;
      return LJTransverser(box, rc, force, energy, virial);
    }
    
  };

.. note:: We defined two things in this code example; a :ref:`Transverser` called :cpp:`LJTransverser` and a :ref:`Potential` called :cpp:`SimpleLJ`

