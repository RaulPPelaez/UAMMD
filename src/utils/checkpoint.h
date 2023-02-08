/*Raul P. Pelaez 2021-2022. Utilities for storing/restoring the simulation state.

 */
#ifndef UAMMD_CHECKPOINT_H
#define UAMMD_CHECKPOINT_H
#include "ParticleData/ParticleData.cuh"
#include <uammd.cuh>
#include <fstream>
namespace uammd{

  namespace detail{
    template<class Container>
    void saveParticleContainer(Container &v, const int* id2index, std::string name, std::ofstream &out){
      if(name.compare("Id") == 0 ) return;
      out<<"# "<<name<<std::endl;
      fori(0, v.size()){
	out<<v[id2index[i]]<<"\n";
      }
    }

    template<class Container>
    void readParticleContainer(Container &v, std::ifstream &in){
      fori(0, v.size()){
        in>>v[i];
      }
    }

  }

  void saveParticleData(std::string fileName,
			std::shared_ptr<ParticleData> pd){
    std::ofstream out(fileName);
    out<<"# version "<<UAMMD_VERSION<<std::endl;
    int numberParticles = pd->getNumParticles();
    out<<"# "<<numberParticles<<std::endl;
    auto id2index = pd->getIdOrderedIndices(access::cpu);
#define WRITE_PROPERTIES_T(name)  WRITE_PROPERTIES_R(name)
#define WRITE_PROPERTIES_R(name) {auto prop = pd->get ## name ## IfAllocated(access::cpu, access::read); if(prop.raw()) detail::saveParticleContainer(prop, id2index, BOOST_PP_STRINGIZE(name), out);}
#define WRITE_PROPERTIES(r,data, tuple) WRITE_PROPERTIES_T(PROPNAME_CAPS(tuple))
    PROPERTY_LOOP(WRITE_PROPERTIES)
  }

  std::shared_ptr<ParticleData> restoreParticleData(std::string fileName, std::shared_ptr<System> sys){
    std::ifstream in(fileName);
    std::string str;
    in>>str>>str>>str;
    if(str != UAMMD_VERSION){
      sys->log<System::WARNING>("This restore file was saved with a different UAMMD version (%s)", str);

    }
    int numberParticles;
    in>>str>>numberParticles;
    auto pd = std::make_shared<ParticleData>(sys, numberParticles);
#define NAME_TO_GET_PROPERTY_T(Name) NAME_TO_GET_PROPERTY_R(Name)
#define NAME_TO_GET_PROPERTY_R(Name) if(propname.compare(BOOST_PP_STRINGIZE(Name)) == 0){ \
      sys->log<System::MESSAGE>("Found property %s in restore file", propname.c_str()); \
      auto prop = pd->get ## Name(access::cpu, access::write);		\
      detail::readParticleContainer(prop, in);				\
    }
#define NAME_TO_GET_PROPERTY(r, data, tuple)  NAME_TO_GET_PROPERTY_T(PROPNAME_CAPS(tuple))
    while(in>>str){
      std::string propname;
      in>>propname;
      PROPERTY_LOOP(NAME_TO_GET_PROPERTY)
	}
    return pd;
  }

}
#endif
