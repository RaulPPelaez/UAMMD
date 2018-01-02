/*Raul P. Pelaez 2017. PairForces definition.

  PairForces Module is an interactor that computes short range forces.

  Computes the interaction between neighbour particles (pairs of particles closer tan rcut). 
  If the value of rcut reaches a certain threshold, the computation will be 
    
  For that, it uses a NeighbourList or an Nbody interaction and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

  See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces   for more info.

  See misc/Potential.cuh and https://github.com/RaulPPelaez/UAMMD/wiki/Potential for more info on potentials and how to implement them.
*/

#include<third_party/cub/cub.cuh>

#include"utils/GPUUtils.cuh"


namespace uammd{
  template<class Potential, class NL>
  PairForces<Potential, NL>::PairForces(shared_ptr<ParticleData> pd,
					shared_ptr<ParticleGroup> pg,
					shared_ptr<System> sys,		       
					Parameters par,
					shared_ptr<Potential> pot):
    Interactor(pd, pg, sys,
	       "PairForces/" +
	       stringUtils::removePattern(type_name<NL>(), "uammd::") +
	       "/" +
	       stringUtils::removePattern(type_name<Potential>(), "uammd::")),
    box(par.box),
    pot(pot),
    nl(nullptr),
    nb(nullptr)
  {
    this->setDelegate(pot.get());
  }

  
    
  template<class Potential, class NL>
  void PairForces<Potential, NL>::sumForce(cudaStream_t st){
    sys->log<System::DEBUG1>("[PairForces] Summing forces");

    this->rcut = pot->getCutOff();
    
    sys->log<System::DEBUG3>("[PairForces] Using cutOff: %f", this->rcut);
    

    bool useNeighbourList = true;
    

    //If the cutoff distance is too high, fall back to an NBody interaction
    int3 ncells = make_int3(box.boxSize/rcut+0.5);

    if(ncells.x <=3 || ncells.y <= 3 || ncells.z <=3){
      useNeighbourList = false;      
    }

    auto ft = pot->getForceTransverser(box, pd);
    
    if(useNeighbourList){
      if(!nl){
	//A neighbour list must know just my system information at construction
	nl = std::make_shared<NL>(pd, pg, sys);
      }

      //Update neighbour list. It is smart enough to do it only when it is necessary,
      // so do not fear calling it several times.
      nl->updateNeighbourList(box, rcut, st);   

      sys->log<System::DEBUG3>("[PairForces] Transversing neighbour list");
    
      nl->transverseList(ft, st);

    }
    else{
      if(!nb){
	nb = std::make_shared<NBody>(pd, pg, sys);
      }
      sys->log<System::DEBUG3>("[PairForces] Transversing NBody");
	
      nb->transverse(ft, st);

    }



  }
    



  template<class Potential, class NL>
  real PairForces<Potential, NL>::sumEnergy(){
    cudaStream_t st = 0;
    sys->log<System::DEBUG1>("[PairForces] Summing Energy");

    this->rcut = pot->getCutOff();
    
    sys->log<System::DEBUG3>("[PairForces] Using cutOff: %f", this->rcut);
    

    bool useNeighbourList = true;
    

    //If the cutoff distance is too high, fall back to an NBody interaction
    int3 ncells = make_int3(box.boxSize/rcut+0.5);

    if(ncells.x <=3 || ncells.y <= 3 || ncells.z <=3){
      useNeighbourList = false;      
    }

    auto ft = pot->getEnergyTransverser(box, pd);
    
    if(useNeighbourList){
      if(!nl){
	//A neighbour list must know just my system information at construction
	nl = std::make_shared<NL>(pd, pg, sys);
      }

      //Update neighbour list. It is smart enough to do it only when it is necessary,
      // so do not fear calling it several times.
      nl->updateNeighbourList(box, rcut, st);   

      sys->log<System::DEBUG3>("[PairForces] Transversing neighbour list");
    
      nl->transverseList(ft, st);

    }
    else{
      if(!nb){
	nb = std::make_shared<NBody>(pd, pg, sys);
      }
      sys->log<System::DEBUG3>("[PairForces] Transversing NBody");
	
      nb->transverse(ft, st);

    }


    return 0;
  }
    

}