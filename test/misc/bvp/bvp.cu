/*Raul P. Pelaez 2022. Tests for the BVP solver

 Run with:
./bvp 2>&1 | awk '!/\[/{for(i=1; i<=NF; i++){printf("%25s ", $i);} printf("\n");next}1'

Awk formats the output in columns.

 */
#include <iomanip>
#include <iterator>
#include <uammd.cuh>
#include "misc/BoundaryValueProblem/BVPSolver.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/InputFile.h"

#include<random>
using namespace uammd;
// Solves the equation:
// y''(z)-k y(z)^2 = f(z)
// With the boundary conditions:
// tfi y(1)' + tsi y(1) = \alpha ;      bfi y(-1)' + bsi y(-1) = \beta ;
// Where tfi, tsi, bfi and bsi are some arbitrary factors.
using complex = cufftComplex_t<real>;

struct Parameters{
  real k = 2;
  real H = 1;
  real gamma = 1;
  int nz = 16;
  complex alpha = {1,0};
  complex beta = {1,0};
};

class TopBoundaryConditions{
  real k, H;
public:
  TopBoundaryConditions(real k, real H):k(k),H(H){
  }

  real getFirstIntegralFactor() const{
    return (k!=0)*H;
  }

  real getSecondIntegralFactor() const{
    return k!=0?(k*H*H):(1.0);
  }
};

class BottomBoundaryConditions{
  real k, H;
public:
  BottomBoundaryConditions(real k, real H):k(k),H(H){
  }

  real getFirstIntegralFactor() const{
    return (k!=0)*H;
  }

  real getSecondIntegralFactor() const{
    return k!=0?(-k*H*H):(1.0);
  }
};

template<class BoundaryConditions, class Klist>
class BoundaryConditionsDispatch{
  Klist klist;
  real H;
public:
  BoundaryConditionsDispatch(Klist klist, real H):klist(klist), H(H){}

  BoundaryConditions operator()(int i) const{
    return BoundaryConditions(klist[i], H);
  }
};

template<class BoundaryConditions, class Klist>
auto make_boundary_dispatcher(Klist &klist, real H){
  return thrust::make_transform_iterator(
					 thrust::make_counting_iterator<int>(0),
					 BoundaryConditionsDispatch<BoundaryConditions, Klist>(klist, H));
}

#define ncopy 1000
template<class Solver>
__global__ void solve(Solver solver, complex* fn, complex* an, complex *cn,
		      Parameters par){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id>= ncopy) return;
  int offset = id*par.nz;
  an += offset;
  cn += offset;
  solver.solve(id, fn+offset, par.alpha, par.beta, an, cn);
}

template <class Container>
auto cheb2real(Container &cn_gpu) {
  std::vector<complex> cn(cn_gpu.size());
  thrust::copy(cn_gpu.begin(), cn_gpu.end(), cn.begin());
  int nz = cn.size();
  std::vector<complex> res(nz,  complex());
  fori(0, cn.size()){
    real z = i*M_PI/(nz-1);
    forj(0, cn.size()){
      res[i] += cn[j]*complex{cos(j*z), cos(j*z)};
    }
  }
  return res;
}

template <class Container>
auto real2cheb(Container &cn_gpu) {
  std::vector<complex> cn(cn_gpu.size());
  thrust::copy(cn_gpu.begin(), cn_gpu.end(), cn.begin());
  int nz = cn.size();
  std::vector<complex> res(nz, complex());
  fori(0, cn.size()){
    real pm = i==0?1:2;
    res[i] += pm/(nz-1)*(0.5*(cn[0]*pow(-1, i) + cn[nz-1]));
    forj(1, cn.size()-1){
      real z = j/(nz-1.0);
      res[i] += (pm/(nz-1))*cn[j]*pow(-1,i)*cospi(i*z);
    }
  }
  return res;
}

auto abs(complex z) { return sqrt(z.x * z.x + z.y * z.y); }

template <class Container>
auto getErrorNormalization(Container &v){
  complex max = complex();
  for(auto a: v){
    if(abs(a.x)>abs(max.x)) max.x = abs(a.x);
    if(abs(a.y)>abs(max.y)) max.y = abs(a.y);
  }
  return max;
}

auto computeRightHandSideExpGamma(Parameters par){
  int nz = par.nz;
  real H = par.H;
  thrust::device_vector<complex> fn(nz);
  std::vector<complex> f(nz, complex());
  for(int i = 0; i<nz; i++){
    real z = H*cospi((real(i))/(nz-1));
    f[i] = complex{1,1}*(exp(-par.gamma*z*z)*H*H);
  }
  auto fcheb = real2cheb(f);
  thrust::copy(fcheb.begin(), fcheb.end(), fn.begin());
  return fn;
}

auto computeRightHandSideRandom(Parameters par){
  int nz = par.nz;
  thrust::device_vector<complex> fn(nz);
  std::vector<complex> f(nz, complex());
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_real_distribution<real> uniform(-1.0, 1.0);
  for(int i = 0; i<nz; i++){
    f[i] = complex{uniform(e1), uniform(e1)};
  }
  auto fcheb = real2cheb(f);
  thrust::copy(fcheb.begin(), fcheb.end(), fn.begin());
  return fn;
}

real evaluateSolution(real x, real H, real a, real k, real b, real c){
  long double pi = M_PIl;
  //Solution for f(z) = exp(-gamma*z*z)
  return -(expl(-powl((-2*a-k),2)/(4*a)-powl((k-2*a),2)/(4*a)-k*x-2*k)*(-2*sqrtl(a)*b*k*expl(powl((-2*a-k),2)/(4*a)+powl((k-2*a),2)/(4*a)+2*k*x+k)+2*sqrtl(a)*c*k*expl(powl((-2*a-k),2)/(4*a)+powl((k-2*a),2)/(4*a)+k)+sqrtl(pi)*k*erf((2*a+k)/(2*sqrtl(a)))*expl(powl(k,2)/(4*a)+powl((-2*a-k),2)/(4*a)+powl((k-2*a),2)/(4*a)+2*k*x+2*k)+sqrtl(pi)*k*expl(powl(k,2)/(4*a)+powl((-2*a-k),2)/(4*a)+powl((k-2*a),2)/(4*a)+2*k)*erf((2*a*x-k)/(2*sqrtl(a)))-sqrtl(pi)*k*erf((2*a*x+k)/(2*sqrtl(a)))*expl(powl(k,2)/(4*a)+powl((-2*a-k),2)/(4*a)+powl((k-2*a),2)/(4*a)+2*k*x+2*k)-sqrtl(pi)*k*erf((-2*a-k)/(2*sqrtl(a)))*expl(powl(k,2)/(4*a)+powl((-2*a-k),2)/(4*a)+powl((k-2*a),2)/(4*a)+2*k)-sqrtl(a)*expl(powl(k,2)/(4*a)+powl((-2*a-k),2)/(4*a)+2*k*x)+sqrtl(a)*expl(powl(k,2)/(4*a)+powl((k-2*a),2)/(4*a)+2*k*x+2*k)-sqrtl(a)*expl(powl(k,2)/(4*a)+powl((-2*a-k),2)/(4*a))+sqrtl(a)*expl(powl(k,2)/(4*a)+powl((k-2*a),2)/(4*a)+2*k)))/(4*sqrtl(a)*powl(k,2));
  //Solution for f(z)=0
  //return exp(-k*(x+1))*(c-b*exp(2*k*x))/(2*k);
}

auto computeSolutionCheb(Parameters par){
  std::vector<complex> sol(par.nz);
  for(int i = 0; i<par.nz; i++){
    real x = -cospi((real(i))/(par.nz-1));
    real sr = evaluateSolution(x, par.H, par.gamma, par.k, par.alpha.x, par.beta.x);
    real si = evaluateSolution(x, par.H, par.gamma, par.k, par.alpha.y, par.beta.y);
    sol[i] = complex{sr,si};
  }
  return real2cheb(sol);
}

auto getParameters(){
  Parameters par;
  InputFile in("data.main");
  in.getOption("k", InputFile::Required)>>par.k;
  in.getOption("gamma", InputFile::Required)>>par.gamma;
  in.getOption("nz", InputFile::Required)>>par.nz;
  in.getOption("H", InputFile::Required)>>par.H;
  in.getOption("alpha", InputFile::Required)>>par.alpha.x;
  in.getOption("beta", InputFile::Required)>>par.beta.x;
  System::log<System::MESSAGE>("k: %g", par.k);
  System::log<System::MESSAGE>("nz: %d", par.nz);
  System::log<System::MESSAGE>("H: %g", par.H);
  System::log<System::MESSAGE>("alpha: %g %g", par.alpha.x, par.alpha.y);
  System::log<System::MESSAGE>("beta: %g %g", par.beta.x, par.beta.y);
  return par;
}

auto createBVP(std::vector<real> klist, Parameters par){
  auto bvp = std::make_shared<BVP::BatchedBVPHandler> (klist,
			     make_boundary_dispatcher<TopBoundaryConditions>(klist, par.H),
			     make_boundary_dispatcher<BottomBoundaryConditions>(klist, par.H),
			     klist.size(), par.H, par.nz);
  return bvp;
}

template <class Container, class Solver>
void callBVPSolver(Container &fn, Container &cn,
		   Solver &solver, Parameters par){
  int nz = par.nz;
  fn.resize(nz*ncopy);
  cn.resize(nz*ncopy);
  fori(1, ncopy)
    thrust::copy(fn.begin(), fn.begin()+nz, fn.begin() + nz*i);
  complex* d_fn = thrust::raw_pointer_cast(fn.data());
  auto an = fn;
  complex* d_an = thrust::raw_pointer_cast(an.data());
  complex* d_cn = thrust::raw_pointer_cast(cn.data());
  auto gpu_solver = solver->getGPUSolver();
  int nblocks = ncopy/128 +1;
  solve<<<nblocks, 128>>>(gpu_solver, d_fn, d_an, d_cn, par);
}

auto computeChebyshevExtrema(real H, int nz){
  std::vector<real> z(nz, 0);
  fori(0, nz) z[i] = H*cospi((real(i))/(nz-1));
  return z;
}

template<class Container>
void printComparisonResults(Container &cn, Parameters par){
  auto theory_vec = computeSolutionCheb(par);
  auto z = computeChebyshevExtrema(par.H, par.nz);
  std::cerr<<"-----------------------------Chebyshev coefficients---------------------------------------"<<std::endl;
  std::cerr<<"z real(y_n) imag(y_n) real(theory) imag(theory) real(error) imag(error)"<<std::endl;
  {
    auto chebres = theory_vec;
    auto norm = getErrorNormalization(chebres);
    fori(0, par.nz){
      auto theory = chebres[i];
      auto err = (cn[i]-theory)/norm;
      std::cerr<<z[i]<<" "<<cn[i]<<" "<<theory<<" "<<err<<std::endl;
    }
  }
  // std::cerr<<"-----------------------------REAL SPACE---------------------------------------"<<std::endl;
  // auto res = cheb2real(cn);
  // auto realtheory = cheb2real(theory_vec);
  // auto norm = getErrorNormalization(realtheory);
  // fori(0, nz){
  //   auto theory = realtheory[i];
  //   auto err = (res[i]-theory)/norm;
  //   std::cerr<<z[i]<<" "<<res[i]<<" "<<theory<<" "<<err<<std::endl;
  // }
}

template<class Container>
void printResults(Container &fn, Container &cn, Parameters par){
  auto z = computeChebyshevExtrema(par.H, par.nz);
  std::cerr<<std::setprecision(16);
  std::cerr<<"PRINTING SOLUTION"<<std::endl;
  std::cerr<<"z real(f_n) imag(f_n) real(y_n) imag(y_n)"<<std::endl;
  {
    fori(0, par.nz){
      std::cerr<<z[i]<<" "<<fn[i]<<" "<<cn[i]<<std::endl;
    }
  }
}

bool compare(complex a, complex b) {
  if(a.x != b.x) return false;
  if(a.y != b.y) return false;
  return true;
}

template <class Container>
void checkCopies(Container &cn, Parameters par){
  std::vector<complex> h_cn(cn.size());
  thrust::copy(cn.begin(), cn.end(), h_cn.begin());
  bool error = false;
  forj(0, ncopy){
    fori(0, par.nz){
      complex a = h_cn[par.nz*j + i];
      complex b = h_cn[i];
      if(not compare(a,b)){
	error = true;
	System::log<System::ERROR>("Copy %d is not identical on element %d: a: %g %g b: %g %g",
				   j, i, a.x, a.y, b.x, b.y);
      }
    }
  }
  if(not error)
    System::log<System::MESSAGE>("SUCCESS: all copies are identical");
}

template <class Container>
void checkError(Container &cn, std::vector<real> klist, Parameters par) {
  constexpr real errorThreshold = 1e-10;
  real maxErr = 0;
  real kMaxErr = 0;
  real zMaxErr = 0;
  auto z = computeChebyshevExtrema(par.H, par.nz);
  forj(0, klist.size()){
    par.k = klist[j];
    auto theory_vec = computeSolutionCheb(par);
    auto chebres = theory_vec;
    auto norm = getErrorNormalization(chebres);
    fori(0, par.nz){
      auto theory = chebres[i];
      auto err = (cn[i+par.nz*j]-theory)/norm;
      if(abs(err.x)>maxErr or abs(err.y)>maxErr){
	kMaxErr = par.k;
	zMaxErr = z[i];
      }
      maxErr = std::max({maxErr, abs(err.x), abs(err.y)});
    }
  }
  System::log<System::MESSAGE>("Maximum error found: %g in z=%g for k=%g",
			       maxErr, zMaxErr, kMaxErr);
  if(maxErr > errorThreshold){
    System::log<System::ERROR>("Error was higher than tolerance");
  }
  else{
    System::log<System::MESSAGE>("SUCCESS: Results are wihtin tolerance of theory");
  }
}

int main(int argc, char* argv[]){
  auto par = getParameters();
  if(abs(par.H - real(1.0))>1e-10){
    System::log<System::WARNING>("The analitical solution only works for H=1");
  }
  { //Identical copies test
    System::log<System::MESSAGE>("Starting known solution test");
    std::vector<real> klist(ncopy, par.k);
    auto solver = createBVP(klist, par);
    auto fn = computeRightHandSideExpGamma(par);
    auto cn = fn;
    callBVPSolver(fn, cn, solver, par);
    checkError(cn, klist, par);
    checkCopies(cn,par);
    //printComparisonResults(cn, par);
  }
  //Random k test
  {
    System::log<System::MESSAGE>("Starting random K test");
    std::vector<real> klist(ncopy, 0);
    std::random_device r;
    std::default_random_engine e1(r());
    //The analitical solution is not very good for large k
    std::uniform_real_distribution<real> uniform(0.1, 2.0);
    for(int i = 0; i<klist.size(); i++){
      klist[i] = uniform(e1);
    }
    auto solver = createBVP(klist, par);
    auto fn = computeRightHandSideExpGamma(par);
    auto cn = fn;
    callBVPSolver(fn, cn, solver, par);
    checkError(cn, klist, par);
  }
  //Random right hand side for matlab comparison
  {
    System::log<System::MESSAGE>("Starting random RHS test for matlab comparison");
    std::vector<real> klist(ncopy, par.k);
    auto solver = createBVP(klist, par);
    auto fn = computeRightHandSideRandom(par);
    auto cn = fn;
    callBVPSolver(fn, cn, solver, par);
    checkCopies(cn, par);
    printResults(fn, cn, par);
  }
  return 0;
}
