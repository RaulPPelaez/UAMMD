/* Raul P. Pelaez 2021. Python wrapper example
   
   This example shows how to expose any UAMMD functionality as a python library.
   We will create a PairForces module, specialized for a LJ potential and expose to python a function that computes forces between particles.
   
   Given a list of positions the resulting python library will expose a function that computes LJ forces between them.

   UAMMD stores positions and forces as real4, but lets make the python library work for tightly packed positions/forces by allowing real3 vectors as input.

 */
//The pybind11 library takes care of interfacing python and c++, it has much more functionality than what is shown here.
//Look up its documentation if you need.
#include<pybind11/pybind11.h> //Basic interfacing utilities
#include<pybind11/numpy.h>    //Utilities to work with numpy arrays
#include <uammd.cuh>
#include <Interactor/PairForces.cuh>
namespace py = pybind11;
using namespace uammd;

//This struct holds the basic parameters required
struct Parameters{
  real sigma, epsilon, cutOff;
  Box box;
};

//The LJ force for a distance rij
__device__ real3 lj_force(real3 rij, real epsilon, real sigma, real rc){
  const real r2 = dot(rij, rij);
  if(r2 > 0 and r2 < rc*rc){
    const real invr2 = sigma/r2;
    const real invr6 = invr2*invr2*invr2;
    const real fmoddivr = epsilon*(real(-48.0)*invr6 + real(24.0))*invr6*invr2/sigma;
    return fmoddivr*rij;
  }
  return real3();  
}

//The operator () of this object returns its input as a real4
struct ToReal4{
  template<class vectype>
  __host__ __device__ real4 operator()(vectype i){
    auto pr4 = make_real4(i);
    return pr4;
  }
};

//A simple LJ Potential, see advanced/customPotential.cu or the wiki page for PairForces if you need more info about UAMMD Potentials.
struct LJPotential{
  real epsilon, sigma, rc;
  
  LJPotential(Parameters par): epsilon(par.epsilon), sigma(par.sigma), rc(par.cutOff){}
  
  real getCutOff(){
    return rc;
  }
  
  struct LJTransverser{
    real4 *force;
    real epsilon, sigma;
    Box box;
    real rc;
    LJTransverser(Box i_box, real i_rc, real4* i_force, real epsilon, real sigma):
      box(i_box), rc(i_rc), force(i_force), epsilon(epsilon), sigma(sigma){
    }

    __device__ ForceEnergyVirial compute(real4 pi, real4 pj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      return {lj_force(rij, epsilon, sigma, rc), 0, 0};
    }
    
    __device__ void set(int id, ForceEnergyVirial total){
      force[id] += make_real4(total.force);
    }
  };

  auto getForceTransverser(Box box, std::shared_ptr<ParticleData> pd){
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();    
    return LJTransverser(box, rc, force, epsilon, sigma);
  }
};

//This is the struct that will get exposed to Python, it holds the UAMMD state (a System and a ParticleData)
// and a PairForces module specialized with the above LJ Potential
//We will also expose the sumForce function
//From the python side a user would create an instance of this class and then call sumForce as many times as needed.
struct UAMMD {
  using LJForces = PairForces<LJPotential>;
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<LJForces> pf;
  thrust::device_vector<real3> tmp;
  cudaStream_t st;
  UAMMD(Parameters par, int numberParticles){
    //Initialize UAMMD
    this->sys = std::make_shared<System>();
    this->pd = std::make_shared<ParticleData>(numberParticles);
    //Create the PairForces module
    LJForces::Parameters pfpar;
    pfpar.box = par.box;
    auto pot = std::make_shared<LJPotential>(par);
    this->pf = std::make_shared<LJForces>(pd, pfpar, pot);   
    tmp.resize(numberParticles);
    CudaSafeCall(cudaStreamCreate(&st));
  }
  //This function expects two CPU numpy arrays:
  // h_pos: Stores positions in the format [x0, y0, z0, x1, y1,...]
  // h_F: The forces in a similar format, LJ forces will be added to this array
  void sumForce(py::array_t<real> h_pos, py::array_t<real> h_F){
    const int numberParticles = pd->getNumParticles();
    { //Transform and upload the input vectors to UAMMD
      auto pos = pd->getPos(access::location::gpu, access::mode::write);
      thrust::copy((real3*)h_pos.data(), (real3*)h_pos.data() + numberParticles, tmp.begin());
      thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), pos.begin(), ToReal4());
      auto forces = pd->getForce(access::location::gpu, access::mode::write);
      thrust::copy((real3*)h_F.data(), (real3*)h_F.data() + numberParticles, tmp.begin());
      thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), forces.begin(), ToReal4());
    }
    //Compute LJ forces
    pf->sum({.force=true, .energy=false, .virial=false}, st);
    //Copy particle forces to the input array, transformed again into a real3 layout
    auto forces = pd->getForce(access::cpu, access::read);
    std::transform(forces.begin(), forces.end(), (real3*) h_F.mutable_data(),
		   [](real4 f){ return make_real3(f);});
  }

  ~UAMMD(){
    cudaDeviceSynchronize();
    cudaStreamDestroy(st);
    sys->finish();
  }
};


//This is the interfacing code. If you are familiar with pybind11 you will see it is a quite basic module.
using namespace pybind11::literals;
//Lets call the library "uammd" and assign it the identifier "m"
PYBIND11_MODULE(uammd, m) {
  //This text will appear if you type help(uammd) in python
  m.doc() = "UAMMD Python interface example";
  //Lets expose the UAMMD class defined above under the name "LJ"
  py::class_<UAMMD>(m, "LJ"). //Two functions will be available in python: The constructor and the sumForce method (the destructor is called automatically when needed)
    def(py::init<Parameters, int>(),"Parameters"_a, "numberParticles"_a). //This is the constructor
    def("sumForce", &UAMMD::sumForce, "Computes LJ forces between particles given a list of positions", //This is the sumForce function, the third argument is the doc
	"positions"_a,"forces"_a); //These allow to have python-style named arguments
  //Lets expose also a part of the Box class just for the sake of it.
  py::class_<Box>(m, "Box", "Domain bounds, the box is periodic by default. Note that you can make the domain aperiodic in any direction by using an infinite length.").
    def(py::init<real>()).
    def(py::init([](real x, real y, real z) {
      return std::unique_ptr<Box>(new Box(make_real3(x,y,z)));
    }));
  //In a similar way lets expose the Parameters struct
  py::class_<Parameters>(m, "Parameters").
    def(py::init([](real sigma, real epsilon, real cutOff, Box box) { //This is the constructor
      auto tmp = std::unique_ptr<Parameters>(new Parameters);
      tmp->box = box;
      tmp->sigma = sigma;
      tmp->epsilon = epsilon;
      tmp->cutOff = cutOff;
      return tmp;	
    }),"sigma"_a = 1.0, "epsilon"_a  = 1.0, "cutOff"_a = 1.0, "box"_a = Box()). //You can also set default values for the arguments
    def_readwrite("sigma", &Parameters::sigma, "LJ sigma"). //You can also allow each member to be modifiable and add a doc string to it
    def_readwrite("epsilon", &Parameters::epsilon, "LJ epsilon").
    def_readwrite("cutOff", &Parameters::cutOff, "LJ cutOff").
    def_readwrite("box", &Parameters::box, "Domain").
    def("__str__", [](const Parameters &p){ //This allows to print the object in python
      return "sigma = "+std::to_string(p.sigma)+"\n"+
	"epsilon = " + std::to_string(p.epsilon) +"\n"+
	"cutOff = " + std::to_string(p.cutOff) +"\n"+
	"box (L = " + std::to_string(p.box.boxSize.x) +
	"," + std::to_string(p.box.boxSize.y) + "," + std::to_string(p.box.boxSize.z) + ")\n";
    });
    
    }
