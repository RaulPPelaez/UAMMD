#include "Interactor/SpectralEwaldPoisson.cuh"
#include "uammd.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace uammd;
using std::endl;
using std::make_shared;

// Helper function to calculate theoretical electric field
real calculateTheoreticalField(real r, real gw) {
  const auto pi = M_PIl;
  return -exp(-r * r / (4.0 * gw * gw)) / (4 * pi * sqrt(pi) * gw * r) -
         erf(r / (2.0 * gw)) / (4 * pi * r * r);
}

// Helper function to calculate theoretical potential
real calculateTheoreticalPotential(real r, real gw) {
  const real pi = M_PI;
  return 1 / (4 * gw * pow(pi, 1.5)) - erf(r / (2.0 * gw)) / (4 * pi * r);
}

// Simple polynomial fit to extrapolate field to infinite box size
class PolynomialFit {
public:
  // Fit data to a model: y = a + b/x + c/x^2 + d/x^3 + e/x^4 + f/x^5
  static real fitAndGetConstantTerm(const std::vector<real> &x,
                                    const std::vector<real> &y) {
    // Implementation of a simple least squares fit
    // We'll use a simplified approach with normal equations
    int n = x.size();
    int terms = 6; // a + b/x + c/x^2 + d/x^3 + e/x^4 + f/x^5
    // Build the design matrix
    std::vector<std::vector<real>> A(n, std::vector<real>(terms, 0.0));
    for (int i = 0; i < n; i++) {
      A[i][0] = 1.0;
      for (int j = 1; j < terms; j++) {
        A[i][j] = A[i][j - 1] / x[i]; // 1/x, 1/x^2, 1/x^3, etc.
      }
    }
    // Compute A^T * A
    std::vector<std::vector<real>> ATA(terms, std::vector<real>(terms, 0.0));
    for (int i = 0; i < terms; i++) {
      for (int j = 0; j < terms; j++) {
        for (int k = 0; k < n; k++) {
          ATA[i][j] += A[k][i] * A[k][j];
        }
      }
    }

    // Compute A^T * y
    std::vector<real> ATy(terms, 0.0);
    for (int i = 0; i < terms; i++) {
      for (int k = 0; k < n; k++) {
        ATy[i] += A[k][i] * y[k];
      }
    }

    // Solve the system ATA * x = ATy using Gaussian elimination
    std::vector<real> coefficients = solveLinearSystem(ATA, ATy);

    // Return the constant term (a), which represents the field as x approaches
    // infinity
    return coefficients[0];
  }

private:
  // Simple Gaussian elimination to solve the linear system
  static std::vector<real> solveLinearSystem(std::vector<std::vector<real>> A,
                                             std::vector<real> b) {
    int n = b.size();

    // Forward elimination
    for (int i = 0; i < n; i++) {
      // Find pivot
      int maxRow = i;
      real maxVal = std::abs(A[i][i]);
      for (int j = i + 1; j < n; j++) {
        if (std::abs(A[j][i]) > maxVal) {
          maxVal = std::abs(A[j][i]);
          maxRow = j;
        }
      }

      // Swap rows if needed
      if (maxRow != i) {
        std::swap(A[i], A[maxRow]);
        std::swap(b[i], b[maxRow]);
      }

      // Eliminate
      for (int j = i + 1; j < n; j++) {
        real factor = A[j][i] / A[i][i];
        b[j] -= factor * b[i];
        for (int k = i; k < n; k++) {
          A[j][k] -= factor * A[i][k];
        }
      }
    }

    // Back substitution
    std::vector<real> x(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
      x[i] = b[i];
      for (int j = i + 1; j < n; j++) {
        x[i] -= A[i][j] * x[j];
      }
      x[i] /= A[i][i];
    }

    return x;
  }
};

class PoissonTest : public ::testing::Test {
protected:
  std::shared_ptr<System> sys;

  void SetUp() override { sys = make_shared<System>(); }

  void TearDown() override { sys->finish(); }

  real4 runPoissonSimulation(real L, real r, real tolerance, real gw,
                             real split) {
    int N = 2;
    auto pd = make_shared<ParticleData>(N, sys);
    Box box(L);
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::write);
      auto charge = pd->getCharge(access::location::cpu, access::mode::write);
      auto ori = make_real4(make_real3(sys->rng().uniform3(-0.5, 0.5)) * L, 0);
      pos[0] = make_real4(-r * 0.5, 0, 0, 0) + ori;
      pos[1] = make_real4(r * 0.5, 0, 0, 0) + ori;
      charge[0] = 1;
      charge[1] = -1;
    }
    Poisson::Parameters par;
    par.box = box;
    par.epsilon = 1;
    par.gw = gw;
    par.tolerance = tolerance;
    par.split = split;
    auto poisson = make_shared<Poisson>(pd, par);
    {
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
    }
    poisson->sum({.force = true, .energy = false, .virial = false});
    real4 result;
    auto force = pd->getForce(access::location::cpu, access::mode::read);
    result = force[0];
    return result;
  }
};

TEST_F(PoissonTest, SingleSimulationTest) {
  real L = 100.0;
  real r = 2.0;
  real tolerance = 1e-7;
  real gw = 0.001;
  real split = 0.2;
  real4 force;
  force = runPoissonSimulation(L, r, tolerance, gw, split);
  ASSERT_LT(force.y, 1e-10);
  ASSERT_LT(force.z, 1e-10);
  ASSERT_GT(force.x, 0);
  real theoreticalField = calculateTheoreticalField(r, gw);
  real relativeDifference =
      std::abs(1.0 - std::abs(force.x / theoreticalField));
  ASSERT_LT(relativeDifference, 1e-3);
}

TEST_F(PoissonTest, InfiniteBoxSizeTest) {
  real tolerance = 1e-7;
  real gw = 0.001;
  real maxDeviation = 0.0;
  real maxDeviationDistance = 0.0;
  for (real r = 2.0; r <= 24.0; r += 4.0) {
    std::vector<real> boxSizes;
    std::vector<real> fieldValues;
    for (real L = std::max(real(16.0), 4 * r); L <= 450.0; L += 4.0) {
      real split = std::max(1.0 - (L - 16.0) / (128.0 - 16.0) * 0.9, 0.1);
      real3 force =
          make_real3(runPoissonSimulation(L, r, tolerance, gw, split));
      real fieldMagnitude = sqrt(dot(force, force));
      boxSizes.push_back(L);
      fieldValues.push_back(fieldMagnitude);
    }
    real extrapolatedField =
        PolynomialFit::fitAndGetConstantTerm(boxSizes, fieldValues);
    real theoreticalField = calculateTheoreticalField(r, gw);
    real deviation =
        std::abs(1.0 - std::abs(extrapolatedField / theoreticalField));
    if (deviation > maxDeviation) {
      maxDeviation = deviation;
      maxDeviationDistance = r;
    }
    std::cout << "Distance " << r << ": deviation = " << deviation
              << ", extrapolated = " << extrapolatedField
              << ", theoretical = " << theoreticalField << std::endl;
  }
  ASSERT_LT(maxDeviation, 1e-4)
      << "Maximum deviation too large at distance " << maxDeviationDistance;
}
