
#include "utils.cuh"
#include "gmock/gmock.h"
#include <gtest/gtest.h>
#include <random>
#include <vector>

real phi(real r, real rmax, real sigma) {
  r = abs(r);
  real res = 0;
  if (r <= rmax)
    res =
        (1.0 / (sigma * sqrt(2 * M_PI))) * exp(-0.5 * r * r / (sigma * sigma));
  return res;
}

real phiz(real r, real rmax, real sigma, real H, real pz) {
  if (abs(r) <= rmax) {
    real bot_rimg = fabs(-H - pz + r);
    return phi(r, rmax, sigma) -
           ((bot_rimg >= rmax) ? real(0.0) : phi(bot_rimg, rmax, sigma));
  } else
    return 0;
}

real3 distanceToCell(int3 cell, real3 pos, int3 n, real3 L) {
  real3 h = L / make_real3(n);
  Box box(L);
  box.setPeriodicity(1, 1, 0);
  real3 r;
  r.x = -L.x * 0.5 + (cell.x) * h.x - pos.x;
  r.y = -L.y * 0.5 + (cell.y) * h.y - pos.y;
  r.z = L.z * cospi(cell.z / (n.z - 1.0)) - pos.z;
  r = box.apply_pbc(r);
  return r;
}

bool runSpreadingTest(real3 qi, real3 L, int3 n, int supp) {
  real sigma = L.x / n.x;
  real rmax = supp * L.x * 0.5 / n.x;
  complex Fi = complex{1, 0};
  bool error = false;
  int numberParticles = 1;
  thrust::device_vector<real3> pos(numberParticles);
  thrust::device_vector<complex> forces(numberParticles);
  pos[0] = qi;
  forces[0] = Fi;
  auto fr = spreadParticles(pos, forces, n, sigma, supp, L);
#ifdef DOUBLE_PRECISION
  constexpr real errorThreshold = 1e-12;
#else
  constexpr real errorThreshold = 1e-5;
#endif
  for (int i = 0; i < n.x; i++) {
    for (int j = 0; j < n.y; j++) {
      for (int k = 0; k < n.z; k++) {
        real3 r = distanceToCell({i, j, k}, qi, n, L);
        complex fijk = {0, 0};
        fijk = Fi * phi(r.x, rmax, sigma) * phi(r.y, rmax, sigma) *
               phiz(r.z, rmax, sigma, L.z, qi.z);
        int id = i + (j + k * n.y) * n.x;
        real errnorm = std::max(fijk.real(), fijk.imag());
        if (errnorm == 0) {
          errnorm = std::max(fr[id].real(), fr[id].imag());
          if (errnorm == 0)
            errnorm = 1;
        }
        complex err = (fr[id] - fijk) / errnorm;
        err.real(abs(err.real()));
        err.imag(abs(err.imag()));
        if (norm(err) > errorThreshold or
            (norm(fr[id]) == 0 and norm(fijk) != 0) or
            (norm(fijk) == 0 and norm(fr[id]) != 0)) {
          error = true;
          int3 ci = chebyshev::doublyperiodic::Grid(Box(L), n).getCell(qi);
          System::log<System::MESSAGE>(
              "Spreading a particle at %g %g %g (cell %d %d %d)", qi.x, qi.y,
              qi.z, ci.x, ci.y, ci.z);
          System::log<System::ERROR>("Difference in cell %d %d %d: Found %g "
                                     "%g, expected %g %g (error %g)",
                                     i, j, k, fr[id].real(), fr[id].imag(),
                                     fijk.real(), fijk.imag(),
                                     thrust::norm(err));
          return not error;
        }
      }
    }
  }
  return not error;
}

TEST(SpreadingTest, RandomPlacements) {
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_real_distribution<real> uniform(-0.5, 0.5);
  real3 L = {1, 1, 1};
  int3 n = {16, 16, 16};
  // I check an even and an odd support
  int supp = 5;
  int ntest = 5;
  for (int test = 0; test < ntest; test++) {
    real3 qi = make_real3(uniform(e1), uniform(e1), uniform(e1)) * L;
    ASSERT_TRUE(runSpreadingTest(qi, L, n, supp));
  }
  supp = 6;
  for (int test = 0; test < ntest; test++) {
    real3 qi = make_real3(uniform(e1), uniform(e1), uniform(e1)) * L;
    ASSERT_TRUE(runSpreadingTest(qi, L, n, supp));
  }
}

TEST(SpreadingTest, DangerousPlacementsOddSupport) {
  real3 L = {1, 1, 1};
  int3 n = {16, 16, 16};
  int supp = 5;
  ASSERT_TRUE(runSpreadingTest(make_real3(0, 0, -L.z * 0.5), L, n, supp));
  ASSERT_TRUE(runSpreadingTest(make_real3(0, 0, L.z * 0.5), L, n, supp));
  ASSERT_TRUE(
      runSpreadingTest(make_real3(0, 0, -L.z * 0.5 + 0.001), L, n, supp));
}

TEST(SpreadingTest, DangerousPlacementsEvenSupport) {
  real3 L = {1, 1, 1};
  int3 n = {16, 16, 16};
  int supp = 6;
  ASSERT_TRUE(runSpreadingTest(make_real3(0, 0, -L.z * 0.5), L, n, supp));
  ASSERT_TRUE(runSpreadingTest(make_real3(0, 0, L.z * 0.5), L, n, supp));
  ASSERT_TRUE(
      runSpreadingTest(make_real3(0, 0, -L.z * 0.5 + 0.001), L, n, supp));
}

TEST(SpreadingTest, WholeDomain) {
  real3 L = {1, 1, 1};
  ASSERT_TRUE(
      runSpreadingTest(make_real3(0, 0, 0.001), L, {128, 128, 128}, 128));
}

void runInterpolationTest(real3 qi, real3 L, int3 n, int supp) {
  complex Fi = complex{1, 0};
  real sigma = 0.1;
  thrust::device_vector<real3> pos(1);
  thrust::device_vector<complex> forces(1);
  pos[0] = qi;
  forces[0] = Fi;
  thrust::device_vector<complex> fr =
      spreadParticles(pos, forces, n, sigma, supp, L);
  auto res = interpolateField(pos, fr, n, sigma, supp, L);
  real l = L.x;
  // The double integral of a Gaussian of width sigma in a cubic domain of size
  // l
  real solution = (9278850.0 * pow(M_PI, 1.5) * pow(erf(0.5 * l / sigma), 3)) /
                  (2301620723.0 * pow(sigma, 3));
  real JS = complex(res[0]).real();
  real error = abs(JS - solution) / solution;
  ASSERT_LE(error, 1e-11);
}
// Checks that JS1=\int \delta_a(\vec{r})^2 d\vec{r}, i.e interpolating
// after spreading returns the inverse of the volume.
// Here \delta_a is the spreading kernel
TEST(SpreadingTest, InterpolateAfterSpreadYieldsInverseVolume) {
  int3 n = {128, 128, 128};
  real l = 1;
  real3 L = {l, l, l};
  int supp = n.x;
  real3 qi = make_real3(0, 0, 0) * L;
  runInterpolationTest(qi, L, n, supp);
}
