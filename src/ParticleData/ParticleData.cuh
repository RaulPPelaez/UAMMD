/*Raul P. Pelaez 2019-2022. ParticleData.
  Handles and stores all properties a particle can have.
  However they are only initialized when they are asked for the first time.
  Offers a way to access this properties.

  Can change in size and periodically sorts the particles to increase spatial
  locality.

  All changes in the particle data are announced using boost signals.
  You can suscribe to this signals by asking for them with get*Signal()


  Entities using this class must take into account that the addresses of the
  properties and the order/number of the particles can change at any point. The
  former is solved by asking ParticleData for the address of a property each
  time it is going to be used, the latter are informed through signals so any
  needed computation can be performed.


  CREATION:

  auto pd = make_shared<ParticleData>(numberParticles, system);
  //System can be ommited if a handle to it is not needed.
  //It will be auto created by ParticleData.
  auto pd = make_shared<ParticleData>(numberParticles);

  USAGE:

  To get a certain property:

  You can get a property both in GPU or CPU memory
  and must specify the kind of access (read, write, readwrite)

  If the mode is set to write, the handle will gain exclusivity and no one else
  will be able to access it until it is realeased (the handle is deleted). You
  cannot write to an array that is currently being read. For this it is
  important to control the scope of the property handles.

  //Get a handle to it
  auto pos_handle = pd->getPos(access::location::cpu, access::mode::read);
  //Get a raw memory pointer if needed
  real4* pos_ptr = pos_handle.raw();

  To get the indices of particles in the original order (ordered by ID):
  int * originalOrder = pd->getIdOrderedIndices(access::location::cpu);
  particle zero would be: pos.raw()[originalOrder[0]];

  //To get a property only if it has been asked for before (i.e if the mass has
  been set) auto mass = pd->getMassIfAllocated(access::location::gpu,
  access::mode::read);
  //mass.raw() will be nullptr if mass has not been asked for before.
  //Note that this call will never allocate the property

  CONNECT TO A SIGNAL:

  When the particles are reordered, or the number of them changes a signal will
  be thrown. In order to hear this signals a user class must:

  class User{
    connection reorderConnection, numParticlesChangedConnection;
    public:
     User(std::shared_ptr<ParticleData> pd){
       reorderConnection = pd->getReorderSignal()->
         connect([this](){this->handle_reorder();});

       numParticlesChangedConnection = pd->getNumParticlesChangedSignal()->
         connect([this](int Nnew){this->handle_numChanged(Nnew);});
     }
     ~User(){
     //Remember to disconnect when the signal is not needed anymore!
       reorderConnection.disconnect();
       numParticlesChangedConnection.disconnect();
     }
     void handle_reorder(){
       std::cout<<"A reorder occured!!"<std::endl;
     }
     void handle_numChanged(int Nnew){
       std::cout<<"Particle number changed, now it is: "<<Nnew<<std::endl;
     }
  };

  LIST OF SIGNALS:

  numParticlesChangedSignal() -> int : Triggered when the total number of
  particles changes reorderSignal() -> void : Triggered when the global sorting
  of particles changes [PROPERTY]WriteRequestedSignal() -> void: Triggered when
  PROPERTY has been requested with the write or readwrite flag (notice that the
  signal is emitted at requesting of the property, so the requester has writing
  rights

*/
#ifndef PARTICLEDATA_CUH
#define PARTICLEDATA_CUH
#include "System/System.h"

#include "ParticleData/Property.cuh"
#include "utils/ParticleSorter.cuh"

#include <third_party/boost/preprocessor.hpp>
#include <third_party/boost/preprocessor/seq/for_each.hpp>
#include <third_party/boost/preprocessor/stringize.hpp>
#include <third_party/boost/preprocessor/tuple/elem.hpp>
#include <third_party/nod/nod.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system_error.h>

#include "utils/vector.cuh"

// List here all the properties with this syntax:
/*       ((PropertyName, propertyName, TYPE)) \      */
// The preprocessor ensures that they are included wherever is needed
#ifndef EXTRA_PARTICLE_PROPERTIES
#define EXTRA_PARTICLE_PROPERTIES
#endif
// clang-format off
#define IMPL_ALL_PROPERTIES_LIST        \
            ((Pos, pos, real4))         \
            ((Id, id, int))		\
	    ((Mass, mass, real))	\
	    ((Force, force, real4))	\
	    ((Virial, virial, real))	\
	    ((Energy, energy, real))	\
	    ((Vel, vel, real3))		\
	    ((Radius, radius, real))	\
	    ((Charge, charge, real))	\
	    ((Torque, torque, real4))	\
	    ((AngVel, angVel, real4))	\
	    ((Dir, dir, real4))		\
            EXTRA_PARTICLE_PROPERTIES   \
/*
                            ((Torque, torque, real4))  \
                            ((AngVel, angVel, real4))  \
                            ((Dir, dir, real4))        \

 */
// clang-format on

/**
 * @brief List of all properties available in @ref ParticleData .
 *
 * This macro is used to generate the list of all properties available in
 * ParticleData. It is used to generate getter functions and other related
 * functionality.
 *
 * You can add your own properties by adding them to the
@ref EXTRA_PARTICLE_PROPERTIES macro.
 *
 *
 * The format is:
 *
 * `((PropertyName, propertyName, TYPE))`
 *
 * where TYPE is the type of the property (e.g. @ref real4 , @ref int , @ref
real , etc.).
 * The property name will be used to generate the getter functions, so it must
be a valid C++ type.
 *
 * ## Example: Adding new particle properties
 * To add a new property called "MyProperty" of type @ref real3 you would define
the EXTRA_PARTICLE_PROPERTIES before including any UAMMD headers. This will
generate a getter function called `getMyProperty()` that returns a
 * @ref property_ptr handle to the property data with the requested type. For
example:
 *
 * @code{.cpp}
 * #define EXTRA_PARTICLE_PROPERTIES ((MyProperty, myProperty, real3))
 * #include "ParticleData/ParticleData.cuh"
 * // Now you can use the getMyProperty() function to access the property
 * auto pd = std::make_shared<ParticleData>(numberParticles);
 * auto myProperty_handle = pd->getMyProperty(access::cpu, access::write);
 * for(auto &p: myProperty_handle) {
 *   p = make_real3(1.0, 2.0, 3.0); // Set MyProperty to (1.0, 2.0, 3.0) for
each particle
 * }
 * @endcode
 */
#define ALL_PROPERTIES_LIST IMPL_ALL_PROPERTIES_LIST

// Get the Name (first letter capital) from a tuple in the property list
#define PROPNAME_CAPS(tuple) BOOST_PP_TUPLE_ELEM(3, 0, tuple)
// Get the name (no capital) from a tuple in the property list
#define PROPNAME(tuple) BOOST_PP_TUPLE_ELEM(3, 1, tuple)
// Get the type from a tuple in the property list
#define PROPTYPE(tuple) BOOST_PP_TUPLE_ELEM(3, 2, tuple)
// This macro iterates through all properties applying some macro
#define PROPERTY_LOOP(macro)                                                   \
  BOOST_PP_SEQ_FOR_EACH(macro, _, ALL_PROPERTIES_LIST);

namespace uammd {

/**
 * @brief Signal type for ParticleData.
 *
 * This is a convenience type alias for nod::unsafe_signal<void(void)>.
 * It is used to signal events in ParticleData, such as reordering of
 * particles or changes in the number of particles.
 * @tparam T The type returned by the signal.
 */
template <class T> using signal = typename nod::unsafe_signal<T>;

/**
 * @brief Connection type for ParticleData signals.
 *
 * This is a convenience type alias for nod::connection.
 * It is used to manage connections to signals in ParticleData.
 */
using connection = nod::connection;
/**
 * @class ParticleData
 * @brief Central container for all per-particle properties in the simulation.
 *
 * The ParticleData class manages particle properties such as position,
 * velocity, mass, charge, force, energy, and more. Properties are allocated
 * lazily — they are only initialized upon first access. It supports both CPU
 * and GPU memory access modes and controls exclusive write access to ensure
 * data integrity across computational modules.
 *
 * It also manages spatial sorting of particles to enhance memory locality and
 * performance. This sorting can change the memory layout of particles, which
 * is announced through signals.
 *
 * Most UAMMD modules will need to be provided with a reference to a
 * ParticleData instance, or its sister class, ParticleGroup.
 *
 * ## Accessing Properties
 *
 * Getter functions are provided to access each available property.
 * @code{.cpp}
 * // Get a CPU read handle to positions
 * auto pd = std::make_shared<ParticleData>(numberParticles);
 * auto pos_handle = pd->getPos(access::cpu, access::read);
 * for(auto &p: pos_handle) {
 *   p = make_real4(0.0, 0.0, 0.0, 0.0); // Set position to origin, type 0
 * }
 * real4* pos_ptr = pos_handle.raw();  // Get raw pointer if needed
 * @endcode
 *
 * @warning Raw pointers to properties are invalidated when the handle goes
 * out of scope.
 *
 */

class ParticleData {
public:
#ifndef DOXYGEN
  struct Hints {
    bool orderByHash = false;
    Box hash_box = Box(make_real3(128));
    real3 hash_cutOff = make_real3(10.0);
    bool orderByType = false;
  };
#endif // DOXYGEN

private:
  shared_ptr<System> sys;
#define DECLARE_PROPERTIES_T(type, name) Property<type> name;
#define DECLARE_PROPERTIES(r, data, tuple)                                     \
  DECLARE_PROPERTIES_T(PROPTYPE(tuple), PROPNAME(tuple))

  // Declare all property containers
  PROPERTY_LOOP(DECLARE_PROPERTIES)

  int numberParticles;
  shared_ptr<signal<void(void)>> reorderSignal =
      std::make_shared<signal<void(void)>>();
  shared_ptr<signal<void(int)>> numParticlesChangedSignal =
      std::make_shared<signal<void(int)>>();

// Declare write access signals for all properties
#define DECLARE_SIGNAL_PROPERTIES_T(type, name)                                \
  shared_ptr<signal<void(void)>> BOOST_PP_CAT(                                 \
      name, WriteRequestedSignal = std::make_shared<signal<void(void)>>();)
#define DECLARE_SIGNAL_PROPERTIES(r, data, tuple)                              \
  DECLARE_SIGNAL_PROPERTIES_T(PROPTYPE(tuple), PROPNAME(tuple))
  // Declare all property write signals
  PROPERTY_LOOP(DECLARE_SIGNAL_PROPERTIES)

  std::shared_ptr<ParticleSorter> particle_sorter;
  thrust::host_vector<int> originalOrderIndexCPU;
  bool originalOrderIndexCPUNeedsUpdate;
  Hints hints;

public:
  ParticleData() = delete;

  /**
   * @brief Initializes ParticleData with a specified number of
   * particles.
   *
   * If no System instance is provided, a default one will be created.
   *
   * @param numberParticles Number of particles to initialize.
   * @param sys Optional shared pointer to a System instance.
   */
  ParticleData(int numberParticles, shared_ptr<System> sys = nullptr);

  ParticleData(shared_ptr<System> sys, int numberParticles)
      : ParticleData(numberParticles, sys) {}

  ~ParticleData() { sys->log<System::DEBUG>("[ParticleData] Destroyed"); }

  // Return the System instance used by this instance of ParticleData
  auto getSystem() { return this->sys; }

  // Generate getters for all properties except ID
#define GET_PROPERTY_T(Name, name) GET_PROPERTY_R(Name, name);
#define GET_PROPERTY_R(Name, name)                                             \
  inline auto get##Name(access::location dev, access::mode mode)               \
      ->decltype(name.data(dev, mode)) {                                       \
    if (!name.isAllocated())                                                   \
      name.resize(numberParticles);                                            \
    if (!name.isAllocated() or mode == access::mode::write or                  \
        mode == access::mode::readwrite) {                                     \
      (*name##WriteRequestedSignal)();                                         \
    }                                                                          \
    return name.data(dev, mode);                                               \
  }

#define GET_PROPERTY(r, data, tuple)                                           \
  GET_PROPERTY_T(PROPNAME_CAPS(tuple), PROPNAME(tuple));

  // Define getProperty() functions for all properties in list
  PROPERTY_LOOP(GET_PROPERTY);

  // Generate getters for all properties except ID
#define GET_PROPERTY_IF_ALLOC_T(Name, name) GET_PROPERTY_IF_ALLOC_R(Name, name)
#define GET_PROPERTY_IF_ALLOC_R(Name, name)                                    \
  inline auto get##Name##IfAllocated(access::location dev, access::mode mode)  \
      ->decltype(name.data(dev, mode)) {                                       \
    if (!name.isAllocated()) {                                                 \
      decltype(name.data(dev, mode)) tmp;                                      \
      return tmp;                                                              \
    }                                                                          \
    return this->get##Name(dev, mode);                                         \
  }

#define GET_PROPERTY_IF_ALLOC(r, data, tuple)                                  \
  GET_PROPERTY_IF_ALLOC_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))

  // Define getProperty() functions for all properties in list
  PROPERTY_LOOP(GET_PROPERTY_IF_ALLOC)

  // Generate isPropAllocated for all properties
#define IS_ALLOCATED_T(Name, name) IS_ALLOCATED_R(Name, name)
#define IS_ALLOCATED_R(Name, name)                                             \
  inline bool is##Name##Allocated() { return name.isAllocated(); }

#define IS_ALLOCATED(r, data, tuple)                                           \
  IS_ALLOCATED_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))

  PROPERTY_LOOP(IS_ALLOCATED)

  // Trigger a particle sort, which assigns an spatial hash to each particle
  // and then reorders them in memory, you can access the original order via
  // getIdOrderedIndices
  /**
   * @brief Sort particles to improve data locality.
   *
   * This function sorts the particles based on their spatial hash or type.
   * It updates the particle order and notifies any listeners about the reorder
   * event.
   *
   * @param st Optional CUDA stream for asynchronous execution.
   */
  void sortParticles(cudaStream_t st = 0);
  /**
   * @brief Get the indices of particles ordered by their IDs.
   *
   * This function returns an array of indices that represent the current
   * location of each particle in the order of their IDs. The particle with
   * ID=i can be found at index getIdOrderedIndices()[i].
   *
   * @param dev The location where the indices should be returned (CPU or GPU).
   * @return A pointer to an array of indices ordered by particle IDs.
   */
  const int *getIdOrderedIndices(access::location dev) {
    sys->log<System::DEBUG5>(
        "[ParticleData] Id order requested for %d (0=cpu, 1=gpu)", dev);
    auto id = getId(access::location::gpu, access::mode::read);
    int *sortedIndex =
        particle_sorter->getIndexArrayById(id.raw(), numberParticles);
    sys->log<System::DEBUG6>("[ParticleData] Id reorder completed.");
    if (dev == access::location::gpu) {
      return sortedIndex;
    } else {
      if (originalOrderIndexCPUNeedsUpdate) {
        sys->log<System::DEBUG1>(
            "[ParticleData] Updating CPU original order array");
        originalOrderIndexCPU.resize(numberParticles);
        int *sortedIndexCPU =
            thrust::raw_pointer_cast(originalOrderIndexCPU.data());
        CudaSafeCall(cudaMemcpy(sortedIndexCPU, sortedIndex,
                                numberParticles * sizeof(int),
                                cudaMemcpyDeviceToHost));
        originalOrderIndexCPUNeedsUpdate = false;
        return sortedIndexCPU;
      } else {
        return thrust::raw_pointer_cast(originalOrderIndexCPU.data());
      }
    }
  }

  /**
   * @brief Apply the current particle order to a range of elements.
   *
   * This function applies the current particle order to a range of elements
   * from an input iterator to an output iterator. It is useful for reordering
   * data based on the current particle order.
   *
   * @tparam InputIterator Type of the input iterator.
   * @tparam OutputIterator Type of the output iterator.
   * @param in Input iterator pointing to the data to be reordered.
   * @param out Output iterator where the reordered data will be written.
   * @param numElements Number of elements to reorder.
   */
  template <class InputIterator, class OutputIterator>
  void applyCurrentOrder(InputIterator in, OutputIterator out,
                         int numElements) {
    particle_sorter->applyCurrentOrder(in, out, numElements);
  }

  const int *getCurrentOrderIndexArray() {
    return particle_sorter->getSortedIndexArray(numberParticles);
  }

  void changeNumParticles(int Nnew);

  /**
   * @brief Get the current number of particles.
   */
  int getNumParticles() { return this->numberParticles; }

  /**
   * @brief Get the signal that is emitted when particles are reordered.
   *
   * This function returns a signal that is emitted when particles are
   * reordered. This can be used to synchronize access to the particle data
   * after a reorder operation.
   * ## Example: Subscribing to the reorder signal
   * @code{.cpp}
   * auto pd = std::make_shared<ParticleData>(numberParticles);
   * auto reorderConnection = pd->getReorderSignal()->connect(
   *     []() {std::cout << "Particles reordered!" << std::endl;});
   * pd->sortParticles(); // This will trigger the signal and print the message
   * // Remember to disconnect when the signal is not needed anymore
   * reorderConnection.disconnect();
   * @endcode
   * @return A shared pointer to the reorder signal.
   */
  shared_ptr<signal<void(void)>> getReorderSignal() {
    sys->log<System::DEBUG>("[ParticleData] Reorder signal requested");
    return this->reorderSignal;
  }

#define GET_PROPERTY_SIGNAL_T(Name, name) GET_PROPERTY_SIGNAL_R(Name, name)
#define GET_PROPERTY_SIGNAL_R(Name, name)                                      \
  inline shared_ptr<signal<void(void)>> get##Name##WriteRequestedSignal() {    \
    return this->name##WriteRequestedSignal;                                   \
  }
#define GET_PROPERTY_SIGNAL(r, data, tuple)                                    \
  GET_PROPERTY_SIGNAL_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))
  PROPERTY_LOOP(GET_PROPERTY_SIGNAL)

  shared_ptr<signal<void(int)>> getNumParticlesChangedSignal() {
    return this->numParticlesChangedSignal;
  }

  void hintSortByHash(Box hash_box, real3 hash_cutOff) {
    hints.orderByHash = true;
    hints.hash_box = hash_box;
    hints.hash_cutOff = hash_cutOff;
  }

#ifdef DOXYGEN
  /**
   * @brief Get a handle to a property.
   *
   * This function returns a handle to the property, which can be used to
   * access the data in the specified location and mode.
   *
   * @note A family of `get*()` functions is generated for each property.
   * Replace "Property" by any of the property names in the @ref
   * ALL_PROPERTIES_LIST
   * @param dev The location of the data (CPU or GPU).
   * @param mode The access mode (read, write, readwrite).
   * @return A property_ptr handle to the property data. The type of the handle
   * will depend on the property.
   */
  property_ptr<type> getProperty(access::location dev, access::mode mode);

  /**
   * @brief Get a handle to a property if it has been allocated.
   *
   * This function returns a handle to the property if it has been allocated,
   * otherwise it returns a null pointer.
   *
   * @note A family of `get*IfAllocated()` functions is generated for each
   * property. Replace "Property" by any of the property names in the @ref
   * ALL_PROPERTIES_LIST
   * @param dev The location of the data (CPU or GPU).
   * @param mode The access mode (read, write, readwrite).
   * @return A property_ptr handle to the property data, or a null pointer if
   * the property has not been allocated.
   */
  property_ptr<type> getPropertyIfAllocated(access::location dev,
                                            access::mode mode);
  /**
   * @brief Check if a property is allocated.
   *
   * This function checks if a property is allocated.
   *
   * @note A family of `is*Allocated()` functions is generated for each
   * property. Replace "Property" by any of the property names in the @ref
   * ALL_PROPERTIES_LIST
   * @return true if the property is allocated, false otherwise.
   */
  bool isPropertyAllocated();
  /**
   * @brief Get the signal that is emitted when a property is requested for
   * writing.
   *
   * This function returns a signal that is emitted when a property is
   * requested for writing. This can be used to synchronize access to the
   * property.
   *
   * @note A family of `get*WriteRequestedSignal()` functions is generated for
   * each property. Replace "Property" by any of the property names in the @ref
   * ALL_PROPERTIES_LIST
   * @return A shared pointer to the signal that is emitted when the property is
   * requested for writing.
   */
  shared_ptr<signal<void(void)>> getPropertyWriteRequestedSignal();

#endif // DOXYGEN

private:
  void emitNumParticlesChanged(int Nnew) { (*numParticlesChangedSignal)(Nnew); }

  void emitReorder() {
    sys->log<System::DEBUG>("[ParticleData] Emitting reorder signal...");
    (*this->reorderSignal)();
  }
};

#define INIT_PROPERTIES_T(NAME, name) , name(BOOST_PP_STRINGIZE(NAME), sys)
#define INIT_PROPERTIES(r, data, tuple)                                        \
  INIT_PROPERTIES_T(PROPNAME_CAPS(tuple), PROPNAME(tuple))

ParticleData::ParticleData(int numberParticles, shared_ptr<System> sys)
    : numberParticles(numberParticles), originalOrderIndexCPUNeedsUpdate(true),
      sys(sys) PROPERTY_LOOP(INIT_PROPERTIES) {
  if (!sys) {
    this->sys = std::make_shared<System>();
    sys->log<System::DEBUG>(
        "[ParticleData] No System provided, creating a default one.");
  }
  sys->log<System::MESSAGE>("[ParticleData] Created with %d particles.",
                            numberParticles);
  if (numberParticles == 0) {
    sys->log<System::WARNING>(
        "[ParticleData] Initialized with zero particles.");
  }
  id.resize(numberParticles);
  CudaCheckError();
  auto id_prop = id.data(access::location::gpu, access::mode::write);
  thrust::sequence(thrust::cuda::par, id_prop.begin(), id_prop.end(), 0);
  particle_sorter = std::make_shared<ParticleSorter>();
}

void ParticleData::sortParticles(cudaStream_t st) {
  sys->log<System::DEBUG>("[ParticleData] Sorting particles...");

  {
    auto posPtr = pos.data(access::gpu, access::read);
    if (hints.orderByHash || !hints.orderByType) {
      int3 cellDim = make_int3(hints.hash_box.boxSize / hints.hash_cutOff);
      particle_sorter->updateOrderByCellHash(posPtr.raw(), numberParticles,
                                             hints.hash_box, cellDim, st);
    }
  }
  // This macro reorders to the newest order a property given its name
#define APPLY_CURRENT_ORDER(r, data, tuple)                                    \
  APPLY_CURRENT_ORDER_R(PROPNAME(tuple))
#define APPLY_CURRENT_ORDER_R(name)                                            \
  {                                                                            \
    if (name.isAllocated()) {                                                  \
      auto devicePtr = name.data(access::gpu, access::write);                  \
      auto device_altPtr = name.getAltGPUBuffer();                             \
      particle_sorter->applyCurrentOrder(devicePtr.raw(), device_altPtr,       \
                                         numberParticles);                     \
      name.swapInternalBuffers();                                              \
    }                                                                          \
  }
  // Apply current order to all allocated properties. See APPLY_CURRENT_ORDER
  // macro
  PROPERTY_LOOP(APPLY_CURRENT_ORDER)

  originalOrderIndexCPUNeedsUpdate = true;
  this->emitReorder();
}

void ParticleData::changeNumParticles(int Nnew) {
  sys->log<System::CRITICAL>(
      "[ParticleData] CHANGE PARTICLES FUNCTIONALITY NOT IMPLEMENTED YET!!!");
  sys->log<System::DEBUG>("[ParticleData] Adding/Removing particles...");
  this->numberParticles = Nnew;
  pos.resize(Nnew);
#define RESIZE_PROPERTY_R(name)                                                \
  {                                                                            \
    if (this->name.isAllocated()) {                                            \
      this->name.resize(this->numberParticles);                                \
    }                                                                          \
  }
#define RESIZE_PROPERTY(r, data, tuple) RESIZE_PROPERTY_R(PROPNAME(tuple))

  PROPERTY_LOOP(RESIZE_PROPERTY)

  originalOrderIndexCPUNeedsUpdate = true;
  this->emitNumParticlesChanged(Nnew);
}
} // namespace uammd

// #undef ALL_PROPERTIES_LIST
// #undef PROPNAME_CAPS
// #undef PROPNAME
// #undef PROPTYPE
#undef PROPERTY_LOOP
#undef DECLARE_PROPERTIES_T
#undef DECLARE_PROPERTIES
#undef DECLARE_SIGNAL_PROPERTIES_T
#undef DECLARE_SIGNAL_PROPERTIES
#undef GET_PROPERTY_T
#undef GET_PROPERTY_R
#undef GET_PROPERTY
#undef GET_PROPERTY_SIGNAL_T
#undef GET_PROPERTY_SIGNAL_R
#undef GET_PROPERTY_SIGNAL
#undef IS_ALLOCATED_T
#undef IS_ALLOCATED_R
#undef IS_ALLOCATED
#undef GET_PROPERTY_IF_ALLOC
#undef GET_PROPERTY_IF_ALLOC_T
#undef GET_PROPERTY_IF_ALLOC_R
#undef APPLY_CURRENT_ORDER
#undef APPLY_CURRENT_ORDER_R
#undef RESIZE_PROPERTY_R
#undef RESIZE_PROPERTY

#endif
