/*Raul P. Pelaez 2017-2021.

  A parameter communication interface, anything that inherits from ParameterUpdatable can be called through update* to communicate a parameter change using a common interface. Parameters related with the particle data are communicated using ParticleData (like number of particles).

  Parameters like a simulation box or the current simulation time are updated through ParameterUpdatable.

  Interactors are ParameterUpdatable objects.

  If a module needs to be aware of a parameter change, it should override the particular virtual method. Wich will do nothing by default.

  If a module needs to delegate the ParameterBehavior to a member (i.e ExternalForces) it must then inherit from ParameterUpdatableDelegate and
    call setDelegate(&member). From that moment, calls to update*() will be called on member.
    This will work even when member is not ParameterUpdatable (SFINAE).
 */

#ifndef PARAMETERUPDATABLE_H
#define PARAMETERUPDATABLE_H
#include"global/defines.h"
#include"utils/Box.cuh"
#include<third_party/boost/preprocessor.hpp>
#include<third_party/boost/preprocessor/stringize.hpp>
#include<third_party/boost/preprocessor/seq/for_each.hpp>
#include<third_party/boost/preprocessor/tuple/elem.hpp>
#ifndef EXTRA_UPDATABLE_PARAMETERS
#define EXTRA_UPDATABLE_PARAMETERS
#endif
//Add here any parameter you want along its type, after adding it here the function updateWHATEVER(type) will
//be available for all ParameterUpdatable modules
#define PARAMETER_LIST ((TimeStep, real)) \
                       ((SimulationTime, real)) \
                       ((Box, Box)) \
                       ((Temperature, real)) \
                       ((Viscosity, real)) EXTRA_UPDATABLE_PARAMETERS




//Please dont look at this :(


//Auto declare all parameter update functions

#define PARNAME(tuple) BOOST_PP_TUPLE_ELEM(2, 0 ,tuple)
  //Get the type from a tuple in the property list
#define PARTYPE(tuple) BOOST_PP_TUPLE_ELEM(2, 1 ,tuple)

#define PARAMETER_LOOP(macro)  BOOST_PP_SEQ_FOR_EACH(macro, _, PARAMETER_LIST)


#define DECLARE_UPDATE_FUNCTION_NAME_T(type, name) DECLARE_UPDATE_FUNCTION_NAME_R(type, name)
#define DECLARE_UPDATE_FUNCTION_NAME_R(type, name) virtual void update  ## name (type) {}
#define DECLARE_UPDATE_FUNCTION(r,data, tuple) DECLARE_UPDATE_FUNCTION_NAME_T(PARTYPE(tuple), PARNAME(tuple))



#define DECLARE_UPDATE_FUNCTION_NAME_DELEGATE_T(type, name) DECLARE_UPDATE_FUNCTION_NAME_DELEGATE_R(type, name)
#define DECLARE_UPDATE_FUNCTION_NAME_DELEGATE_R(type, name) \
  virtual void update  ## name (type  val) { if(delegate) delegate->update ## name (val);}

#define DECLARE_UPDATE_FUNCTION_DELEGATE(r,data, tuple) \
  DECLARE_UPDATE_FUNCTION_NAME_DELEGATE_T(PARTYPE(tuple), PARNAME(tuple))


//I warned you not to look :(


namespace uammd{

  class ParameterUpdatable{
  protected:

    virtual ~ParameterUpdatable(){}

  public:

    // virtual void updateTimeStep(real dt){};

    PARAMETER_LOOP(DECLARE_UPDATE_FUNCTION)

  };

  template<class T, bool isParameterUpdatable = std::is_base_of<ParameterUpdatable, T>::value>
    class ParameterUpdatableDelegate;

  template<class T>
  class ParameterUpdatableDelegate<T, true>: public virtual ParameterUpdatable{
  private:
    T *delegate = nullptr;
  public:
    void setDelegate(T* del){ this->delegate = del;}
    void setDelegate(std::shared_ptr<T> del){ this->delegate = del.get();}

    PARAMETER_LOOP(DECLARE_UPDATE_FUNCTION_DELEGATE)
  };

  template<class T>
  class ParameterUpdatableDelegate<T, false>: public virtual ParameterUpdatable{
  public:
    void setDelegate(T* del){}
    void setDelegate(std::shared_ptr<T> del){}
  };



}




#undef PARAMETER_LIST
#undef PARNAME
#undef PARTYPE
#undef PARAMETER_LOOP
#undef DECLARE_UPDATE_FUNCTION_NAME_T
#undef DECLARE_UPDATE_FUNCTION_NAME_R
#undef DECLARE_UPDATE_FUNCTION
#undef DECLARE_UPDATE_FUNCTION_NAME_DELEGATE_T
#undef DECLARE_UPDATE_FUNCTION_NAME_DELEGATE_R
#undef DECLARE_UPDATE_FUNCTION_DELEGATE



#endif
