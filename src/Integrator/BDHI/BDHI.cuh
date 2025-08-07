/*Raul P. Pelaez 2017. Some Brownian Hydrodynamics utilities and definitions
 */
#ifndef BDHI_CUH
#define BDHI_CUH

#include "global/defines.h"
#include "utils/Box.cuh"
#include <vector>
namespace uammd {
namespace BDHI {

// Parameters that BDHI modules may need
struct Parameters {
  // The 3x3 shear matrix is encoded as an array of 3 real3
  std::vector<real3> K;
  real temperature;
  real viscosity;
  real hydrodynamicRadius =
      -1; // If not provided it will be taken from pd->getRadius if possible
  real tolerance = 1e-3;
  real dt;
  bool is2D = false;
  Box box;
};

// The Rotne-Prager-Yamakawa tensor
struct RotnePragerYamakawa {
  real M0;
  RotnePragerYamakawa(real viscosity) { M0 = 1 / (6 * M_PI * viscosity); }
  // RPY tensor as a function of distance, r

  // Version for particles of equal size
  /*M(r) = 0.75*M0*( f(r)*I + g(r)*r(diadic)r )
    c12.x = f(r) * 0.75*M0
    c12.y = g(r) * 0.75*M0/r^2  -> r^2 to normalize the diadic
  Where M0 is the self mobility: 1/(6*pi*eta*rh)*/
  inline __host__ __device__ real2 RPY(real r, real rh) const {
    /*Distance in units of rh*/
    const real invrh = real(1.0) / rh;
    r *= invrh;
    real2 c12;
    if (r >= real(2.0)) {
      const real invr = real(1.0) / r;
      const real invr2 = invr * invr;
      c12.x = (real(0.75) + real(0.5) * invr2) * invr;
      c12.y = (real(0.75) - real(1.5) * invr2) * invr * invr2;
    } else {
      c12.x = real(1.0) - real(0.28125) * r;
      if (r > real(0.0))
        c12.y = real(0.09375) / r;
    }

    return c12 * (M0 * invrh);
  }

  // Taken from "Rotne-Prager-Yamakawa approximation for different-sized
  // particles in application to macromolecular bead models", P.J. Zik et.al.
  // 2014
  inline __host__ __device__ real2 RPY_differentSizes(real r, real ai,
                                                      real aj) const {

    const real asum = ai + aj;
    const real asub = fabs(ai - aj);
    real2 c12;
    if (r > asum) {
      const real invr = real(1.0) / r;
      const real pref = M0 * real(3.0) * real(0.25) * invr;
      const real denom = (ai * ai + aj * aj) / (real(3.0) * r * r);
      c12.x = pref * (real(1.0) + denom);
      c12.y = pref * (real(1.0) - real(3.0) * denom) * invr * invr;
    } else if (r > asub) {
      const real pref = M0 / (ai * aj * real(32.0) * r * r * r);
      real num = asub * asub + real(3.0) * r * r;
      c12.x = pref * (real(16.0) * r * r * r * asum - num * num);
      num = asub * asub - r * r;
      c12.y = pref * (real(3.0) * num * num) / (r * r);

    } else {
      c12.x = M0 / (ai > aj ? ai : aj);
      c12.y = real(0.0);
    }

    return c12;
  }

  // Computes f(r) and g(r)/r^2 given the distance r and the particle radius of
  // both particles
  inline __host__ __device__ real2 operator()(real r, real a_i,
                                              real a_j) const {
    if (a_i == a_j)
      return RPY_differentSizes(r, a_i, a_i);
    else
      return RPY_differentSizes(r, a_i, a_j);
  }
};

} // namespace BDHI
} // namespace uammd

#endif