#pragma once

#include <uammd.cuh>
#include "misc/IBM.cuh"
#include "misc/ChevyshevUtils.cuh"
#include<thrust/complex.h>
using namespace uammd;
using complex = thrust::complex<real>;
using real = uammd::real;

struct Gaussian{
  int3 support;
  Gaussian(real width, real h, real H, real nz, int supportxy):
    nz(nz){
    this-> prefactor = 1.0/(width*sqrt(2*M_PI));
    this-> tau = -1.0/(2.0*width*width);
    this->rmax = supportxy*h*0.5;
    support.x = supportxy;
    support.y = support.x;
    this->Htot = H;
    real smax = std::max((H*0.5-rmax), -H*0.5);
    int czmax = ceil((nz-1)*(acos(smax/(0.5*Htot))/real(M_PI)));
    support.z = 2*czmax+1;
  }

  inline __host__  __device__ int3 getMaxSupport() const{
    return make_int3(support.x, support.y, support.z);
  }

  inline __host__  __device__ int3 getSupport(real3 pos, int3 cell) const{
    real bound = Htot*real(0.5);
    real ztop = thrust::min(pos.z+rmax, bound);
    real zbot = thrust::max(pos.z-rmax, -bound);
    int czb = int((nz-1)*(acos(ztop/bound)/real(M_PI)));
    int czt = int((nz-1)*(acos(zbot/bound)/real(M_PI)));
    int sz = 2*thrust::max(cell.z - czb, czt - cell.z)+1;
    return make_int3(support.x, support.y, sz);
  }

  __host__  __device__ real phi(real r, real3 pos) const{
    real val = 0;
    if(abs(r)<=rmax*real(1.0001)){
      val = prefactor*exp(tau*r*r);
    }
    return val;
  }

private:
  real prefactor;
  real tau;
  real rmax;
  int nz;
  real Htot;
};

auto spreadParticles(std::vector<real3> pos, std::vector<complex> values,
		     int3 n, real sigma, int supp, real3 L){
  std::vector<complex> fr(n.x*n.y*n.z, complex());
  auto h = L/make_real3(n);
  auto kernel = std::make_shared<Gaussian>(sigma, h.x, L.z, n.z, supp);
  using Grid = chebyshev::doublyperiodic::Grid;
  Grid grid(Box(L), n);
  IBM<Gaussian, Grid> ibm(kernel, grid);
  thrust::device_vector<real3> d_pos = pos;
  auto pos_ptr = thrust::raw_pointer_cast(d_pos.data());
  thrust::device_vector<complex> d_values = values;
  auto values_ptr = thrust::raw_pointer_cast(d_values.data());
  thrust::device_vector<complex> d_fr = fr;
  auto fr_ptr = thrust::raw_pointer_cast(d_fr.data());
  ibm.spread(pos_ptr, (real2*)values_ptr, (real2*)fr_ptr, 1);
  thrust::copy(d_fr.begin(), d_fr.end(), fr.begin());
  return fr;
}

auto interpolateField(std::vector<real3> pos, std::vector<complex> field,
		      int3 n, real sigma, int supp, real3 L){
  std::vector<complex> values(pos.size(), complex());
  auto h = L/make_real3(n);
  auto kernel = std::make_shared<Gaussian>(sigma, h.x, L.z, n.z, supp);
  using Grid = chebyshev::doublyperiodic::Grid;
  Grid grid(Box(L), n);
  IBM<Gaussian, Grid> ibm(kernel, grid);
  thrust::device_vector<real3> d_pos = pos;
  auto pos_ptr = thrust::raw_pointer_cast(d_pos.data());
  thrust::device_vector<complex> d_values = values;
  auto values_ptr = thrust::raw_pointer_cast(d_values.data());
  thrust::device_vector<complex> d_fr = field;
  auto fr_ptr = thrust::raw_pointer_cast(d_fr.data());
  chebyshev::doublyperiodic::QuadratureWeights qw(L.z, h.x, h.y, n.z);
  IBM_ns::DefaultWeightCompute wc;
  ibm.gather(pos_ptr, (real2*)values_ptr, (real2*)fr_ptr, qw, wc, 1);
  thrust::copy(d_values.begin(), d_values.end(), values.begin());
  return values;
}
