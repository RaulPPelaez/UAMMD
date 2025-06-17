/*Raul P. Pelaez 2022. Tests for the VQCM DPStokes algorithm.
  Most tests should take about a second to run. A few of them will take several
  minutes, though.
 */
#include "misc/Chebyshev/FastChebyshevTransform.cuh"
#include "utils/container.h"
#include "gmock/gmock.h"
#include <fstream>
#include <gtest/gtest.h>
#include <random>

using namespace uammd;
using namespace chebyshev;
using namespace chebyshev::detail;

using complex = thrust::complex<real>;
template <class T> using cached_vector = uninitialized_cached_vector<T>;

__host__ __device__ int2 waveNumberFromId(int id, int nx, int ny) {
  int i = id % nx;
  int j = id / nx;
  i -= nx * (i >= (nx / 2 + 1));
  j -= ny * (j >= (ny / 2 + 1));
  return {i, j};
}

__host__ __device__ real2 waveVectorFromId(int id, real Lx, real Ly, int nx,
                                           int ny) {
  int2 kn = waveNumberFromId(id, nx, ny);
  real kx = 2 * M_PI * kn.x / Lx;
  real ky = 2 * M_PI * kn.y / Ly;
  return {kx, ky};
}

// Given a container with values of a function f(z=cospi((real(i))/(nz-1)))
// returns the Chebyshev coefficients of f.
template <class Iterator> auto cheb2realNaive(Iterator cn_gpu, int size) {
  std::vector<complex> cn(size);
  thrust::copy(cn_gpu, cn_gpu + size, cn.begin());
  int nz = cn.size();
  std::vector<complex> res(nz, complex());
  fori(0, cn.size()) {
    real z = real(i) / (nz - 1);
    forj(0, cn.size()) {
      // res[i] += cn[j]*pow(-1,j)*cospi(j*z);
      res[i] += cn[j] * cospi(j * z);
    }
  }
  return res;
}

// Given a container with values of a function f(z=cospi((real(i))/(nz-1)))
// returns the Chebyshev coefficients of f.
template <class Container> auto cheb2realNaive(Container &&cn_gpu) {
  return cheb2realNaive(cn_gpu.begin(), cn_gpu.size());
}

// Given the Chebyshev coefficients of a given function, f,
//  this function returns the values of f(z=cospi((real(i))/(nz-1))).
template <class Iterator> auto real2chebNaive(Iterator cn_gpu, int size) {
  std::vector<complex> cn(size);
  thrust::copy(cn_gpu, cn_gpu + size, cn.begin());
  int nz = cn.size();
  std::vector<complex> res(nz, complex());
  fori(0, cn.size()) {
    real pm = (i == 0 or i == (nz - 1)) ? 1 : 2;
    res[i] += pm / (nz - 1) * (0.5 * (cn[0] + cn[nz - 1] * pow(-1, i)));
    forj(1, cn.size() - 1) {
      real z = j / (nz - 1.0);
      res[i] += (pm / (nz - 1)) * cn[j] * cospi(i * z);
    }
  }
  return res;
}

// Given the Chebyshev coefficients of a given function, f,
//  this function returns the values of f(z=cospi((real(i))/(nz-1))).
template <class Container> auto real2chebNaive(Container &&cn_gpu) {
  return real2chebNaive(cn_gpu.begin(), cn_gpu.size());
}

TEST(NaiveChebyshevTransform, ChebyshevNaiveOfConstantIsCorrect) {
  int n = 16;
  std::vector<complex> f(n, 1);
  f = real2chebNaive(f);
  std::vector<complex> truth(n, 0);
  truth[0] = 1;
  for (int i = 0; i < n; i++) {
    ASSERT_THAT(f[i].real(), ::testing::DoubleNear(truth[i].real(), 1e-14));
    ASSERT_THAT(f[i].imag(), ::testing::DoubleNear(truth[i].imag(), 1e-14));
  }
}

TEST(NaiveChebyshevTransform, ChebyshevNaiveOfXIsCorrect) {
  int n = 16;
  std::vector<complex> f(n);
  for (int i = 0; i < n; i++) {
    real x = cos(M_PI * i / (n - 1));
    f[i] = x;
  }
  f = real2chebNaive(f);
  std::vector<complex> truth(n, 0);
  truth[1] = 1;
  for (int i = 0; i < n; i++) {
    ASSERT_THAT(f[i].real(), ::testing::DoubleNear(truth[i].real(), 1e-14));
    ASSERT_THAT(f[i].imag(), ::testing::DoubleNear(truth[i].imag(), 1e-14));
  }
}

TEST(NaiveChebyshevTransform, ChebyshevNaiveOfAChebyshevPolynomialIsCorrect) {
  int n = 64;
  std::vector<complex> f(n);
  for (int mode = 0; mode < n; mode++) {
    for (int i = 0; i < n; i++) {
      real x = cos(M_PI * i * mode / (n - 1));
      f[i] = x;
    }
    f = real2chebNaive(f);
    std::vector<complex> truth(n, 0);
    truth[mode] = 1;
    for (int i = 0; i < n; i++) {
      ASSERT_THAT(f[i].real(), ::testing::DoubleNear(truth[i].real(), 1e-14));
      ASSERT_THAT(f[i].imag(), ::testing::DoubleNear(truth[i].imag(), 1e-14));
    }
  }
}

TEST(NaiveChebyshevTransform,
     InverseChebyshevNaiveOfAChebyshevTransformIsOriginalSignal) {
  for (int n = 2; n < 128; n++) {
    std::vector<complex> f(n);
    std::random_device r;
    std::default_random_engine e1(1234);
    std::uniform_real_distribution<real> uniform(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
      f[i] = complex{uniform(e1), uniform(e1)};
    }
    auto ftest = cheb2realNaive(real2chebNaive(f));
    for (int i = 0; i < n; i++) {
      ASSERT_THAT(ftest[i].real(), ::testing::DoubleNear(f[i].real(), 1e-13));
      ASSERT_THAT(ftest[i].imag(), ::testing::DoubleNear(f[i].imag(), 1e-13));
    }
  }
}

template <class Container> auto periodicExtendNaive(Container v, int3 n) {
  v.resize(n.x * n.y * (2 * n.z - 2));
  for (int k = n.z; k < 2 * n.z - 2; k++) {
    for (int j = 0; j < n.y; j++) {
      for (int i = 0; i < n.x; i++) {
        int src = i + (j + (2 * n.z - 2 - k) * n.y) * n.x;
        int dest = i + (j + k * n.y) * n.x;
        v[dest] = v[src];
      }
    }
  }
  return v;
}

TEST(FastPeriodicExtension, FastPeriodicExtensionHasTheCorrectSize) {
  for (int nx = 1; nx <= 16; nx++) {
    for (int ny = 1; ny <= 16; ny++) {
      for (int n = 2; n < 128; n++) {
        cached_vector<int> signal(n);
        thrust::sequence(signal.begin(), signal.end(), 0);
        auto extendedSignal = periodicExtend(signal, nx * ny, n);
        ASSERT_EQ(extendedSignal.size(), nx * ny * (2 * n - 2));
      }
    }
  }
}

TEST(FastPeriodicExtension, periodicExtensionNaiveIsCorrectForSize3) {
  int n = 3;
  cached_vector<int> signal(n);
  thrust::sequence(signal.begin(), signal.end(), 0);
  thrust::host_vector<int> extendedSignal =
      periodicExtendNaive(signal, {1, 1, n});
  std::vector<int> truth = {0, 1, 2, 1};
  for (int i = 0; i < n; i++) {
    ASSERT_EQ(extendedSignal[i], truth[i]);
  }
}

TEST(FastPeriodicExtension, FastPeriodicExtensionYieldsCorrectResultForSize3) {
  constexpr int n = 3;
  cached_vector<int> signal(n);
  thrust::sequence(signal.begin(), signal.end(), 0);
  thrust::host_vector<int> extendedSignal = periodicExtend(signal, 1, n);
  std::vector<int> truth = {0, 1, 2, 1};
  for (int i = 0; i < n; i++) {
    ASSERT_EQ(extendedSignal[i], truth[i]);
  }
}

void runPeriodicExtensionTest(int n) {
  cached_vector<int> signal(n);
  thrust::sequence(signal.begin(), signal.end(), 0);
  thrust::host_vector<int> extendedSignal = periodicExtend(signal, 1, n);
  thrust::host_vector<int> extendedSignalTrue =
      periodicExtendNaive(signal, {1, 1, n});
  for (int i = 0; i < n; i++) {
    ASSERT_EQ(extendedSignal[i], extendedSignalTrue[i]);
  }
}

TEST(FastPeriodicExtension, FastPeriodicExtensionMatchesNaiveForOddSizes) {
  for (int n = 3; n < 129; n += 2)
    runPeriodicExtensionTest(n);
}

TEST(FastPeriodicExtension, FastPeriodicExtensionMatchesNaiveForEvenSizes) {
  for (int n = 2; n < 128; n += 2)
    runPeriodicExtensionTest(n);
}

TEST(FastPeriodicExtension, FastPeriodicExtensionMatchesNaiveIn3D) {

  for (int nx = 1; nx <= 8; nx++) {
    for (int ny = 1; ny <= 5; ny++) {
      for (int n = 2; n < 32; n++) {
        cached_vector<int> signal(nx * ny * n);
        thrust::sequence(signal.begin(), signal.end(), 0);
        thrust::host_vector<int> extendedSignal =
            periodicExtend(signal, nx * ny, n);
        thrust::host_vector<int> extendedSignalTrue =
            periodicExtendNaive(signal, {nx, ny, n});
        for (int i = 0; i < extendedSignalTrue.size(); i++) {
          ASSERT_EQ(extendedSignal[i], extendedSignalTrue[i])
              << "Failed for size n (" << nx << "," << ny << "," << n
              << ") at element i " << i << std::endl;
        }
      }
    }
  }
}

TEST(FourierChebyshevTransform, Transform1DMatchesNaive) {
  for (int n = 2; n < 128; n++) {
    std::vector<complex> f(n);
    std::random_device r;
    std::default_random_engine e1(1234);
    std::uniform_real_distribution<real> uniform(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
      f[i] = complex{uniform(e1), uniform(e1)};
    }
    auto naive = real2chebNaive(f);
    cached_vector<complex> d_f(n);
    thrust::copy(f.begin(), f.end(), d_f.begin());
    auto fast = chebyshevTransform1DCufft(d_f);
    for (int i = 0; i < n; i++) {
      complex fast_i = fast[i];
      ASSERT_THAT(fast_i.real(), ::testing::DoubleNear(naive[i].real(), 1e-13));
      ASSERT_THAT(fast_i.imag(), ::testing::DoubleNear(naive[i].imag(), 1e-13));
    }
  }
}

TEST(FourierChebyshevTransform,
     InverseTransformOfDirectTransformMatchesOriginalSignal1D) {
  for (int n = 2; n < 128; n++) {
    std::vector<complex> f(n);
    std::random_device r;
    std::default_random_engine e1(1234);
    std::uniform_real_distribution<real> uniform(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
      f[i] = complex{uniform(e1), uniform(e1)};
    }
    cached_vector<complex> d_f(n);
    thrust::copy(f.begin(), f.end(), d_f.begin());
    auto fast =
        inverseChebyshevTransform1DCufft(chebyshevTransform1DCufft(d_f), n);
    for (int i = 0; i < n; i++) {
      complex fast_i = fast[i];
      ASSERT_THAT(fast_i.real(), ::testing::DoubleNear(f[i].real(), 1e-13));
      ASSERT_THAT(fast_i.imag(), ::testing::DoubleNear(f[i].imag(), 1e-13));
    }
  }
}

TEST(FourierChebyshevTransform, Transform3DMathches1DWithSize1) {
  for (int n = 2; n < 128; n++) {
    std::vector<complex> f(n);
    std::default_random_engine e1(1234);
    std::uniform_real_distribution<real> uniform(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
      f[i] = complex{uniform(e1), uniform(e1)};
    }
    cached_vector<complex> d_f(n);
    thrust::copy(f.begin(), f.end(), d_f.begin());
    auto c1d = chebyshevTransform1DCufft(d_f);
    auto c3d = fourierChebyshevTransform3DCufft(d_f, {1, 1, n});
    for (int i = 0; i < n; i++) {
      complex c3d_i = c3d[i];
      complex c1d_i = c1d[i];
      ASSERT_THAT(c3d_i.real(), ::testing::DoubleNear(c1d_i.real(), 1e-14))
          << "Failed at n " << n << std::endl;
      ASSERT_THAT(c3d_i.imag(), ::testing::DoubleNear(c1d_i.imag(), 1e-14));
    }
  }
}

TEST(FourierChebyshevTransform, InverseTransform3DMathches1DWithSize1) {
  for (int n = 2; n < 128; n++) {
    std::vector<complex> f(n);
    std::default_random_engine e1(1234);
    std::uniform_real_distribution<real> uniform(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
      f[i] = complex{uniform(e1), uniform(e1)};
    }
    cached_vector<complex> d_f(n);
    thrust::copy(f.begin(), f.end(), d_f.begin());
    auto c1d = inverseChebyshevTransform1DCufft(d_f, n);
    auto c3d = inverseFourierChebyshevTransform3DCufft(d_f, {1, 1, n});
    for (int i = 0; i < n; i++) {
      complex c3d_i = c3d[i];
      complex c1d_i = c1d[i];
      ASSERT_THAT(c3d_i.real(), ::testing::DoubleNear(c1d_i.real(), 1e-14));
      ASSERT_THAT(c3d_i.imag(), ::testing::DoubleNear(c1d_i.imag(), 1e-14));
    }
  }
}

TEST(FourierChebyshevTransform,
     InverseTransformOfDirectTransformMatchesOriginalSignal3D) {
  for (int nx = 1; nx <= 16; nx += 3) {
    for (int ny = 1; ny <= 32; ny += 3) {
      for (int n = 2; n < 32; n += 3) {
        std::vector<complex> f(nx * ny * n);
        std::default_random_engine e1(1234);
        std::uniform_real_distribution<real> uniform(-1.0, 1.0);
        for (int i = 0; i < f.size(); i++) {
          f[i] = complex{uniform(e1), uniform(e1)};
        }
        cached_vector<complex> d_f(f.size());
        thrust::copy(f.begin(), f.end(), d_f.begin());
        auto fast = inverseFourierChebyshevTransform3DCufft(
            fourierChebyshevTransform3DCufft(d_f, {nx, ny, n}), {nx, ny, n});
        for (int i = 0; i < f.size(); i++) {
          complex fast_i = fast[i];
          ASSERT_THAT(fast_i.real(), ::testing::DoubleNear(f[i].real(), 1e-13))
              << "Failed at size nx " << nx << " ny " << ny << " nz " << n
              << std::endl;
          ASSERT_THAT(fast_i.imag(), ::testing::DoubleNear(f[i].imag(), 1e-13));
        }
      }
    }
  }
}

TEST(FourierChebyshevTransform, TransformOfGaussian1DIsCorrect) {
  real H = 1;
  real3 L = {H, H, H};
  int3 n{6, 9, 32};
  cached_vector<complex> fxyz(n.x * n.y * n.z);
  for (int ik = 0; ik < n.x * n.y; ik++) {
    auto fx_k = make_interleaved_iterator(fxyz.begin(), ik, n.x * n.y);
    for (int i = 0; i < n.z; i++) {
      real z = H * cospi(i / (n.z - 1.0));
      real r2 = z * z;
      fx_k[i] = exp(-r2 * 0.25);
    }
  }
  fxyz = fourierChebyshevTransform3DCufft(fxyz, n);
  int nk = n.x * n.y;
  for (int i = 0; i < nk; i++) {
    auto k = waveVectorFromId(i, L.x, L.y, n.x, n.y);
    real knorm = sqrt(k.x * k.x + k.y * k.y);
    auto vx_k = make_interleaved_iterator(fxyz.begin(), i, nk);
    std::vector<complex> theoryFou(n.z);
    std::vector<complex> resultFou(n.z);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cos(M_PI * iz / (n.z - 1.0));
      complex theory = exp(-z * z * 0.25);
      if (k.x != 0 or k.y != 0)
        theory *= 0;
      theoryFou[iz] = theory;
      resultFou[iz] = vx_k[iz];
    }
    resultFou = cheb2realNaive(resultFou);
    // theoryFou = real2chebNaive(theoryFou);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cos(M_PI * iz / (n.z - 1.0));
      complex vy_k_z = resultFou[iz];
      auto theory = theoryFou[iz];
      ASSERT_THAT(vy_k_z.real(), ::testing::DoubleNear(theory.real(), 1e-15))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
      ASSERT_THAT(vy_k_z.imag(), ::testing::DoubleNear(theory.imag(), 1e-15))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
    }
  }
}

TEST(FourierChebyshevTransform, TransformOfSin3DIsCorrect) {
  real H = 1;
  real3 L = {H, H, H};
  int3 n{16, 8, 5};
  cached_vector<complex> fxyz(n.x * n.y * n.z);
  for (int ik = 0; ik < n.x * n.y; ik++) {
    auto fx_k = make_interleaved_iterator(fxyz.begin(), ik, n.x * n.y);
    real x = ((ik % n.x) / real(n.x)) * L.x;
    for (int i = 0; i < n.z; i++) {
      real z = H * cospi(i / (n.z - 1.0));
      real r2 = z * z;
      fx_k[i] = exp(-r2 * 0.25) * sin(2 * M_PI * 2 * x);
    }
  }
  fxyz = fourierChebyshevTransform3DCufft(fxyz, n);
  int nk = n.x * n.y;
  for (int i = 0; i < nk; i++) {
    auto k = waveVectorFromId(i, L.x, L.y, n.x, n.y);
    auto ik = waveNumberFromId(i, n.x, n.y);
    real knorm = sqrt(k.x * k.x + k.y * k.y);
    auto vx_k = make_interleaved_iterator(fxyz.begin(), i, nk);
    std::vector<complex> theoryFou(n.z);
    std::vector<complex> resultFou(n.z);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cospi(iz / (n.z - 1.0));
      complex j{0, 1};
      complex theory = -0.5 * exp(-z * z * 0.25) * j *
                       ((ik.x == 2) - (ik.x == -2)) * (ik.y == 0);
      theoryFou[iz] = theory;
      complex res = vx_k[iz];
      resultFou[iz] = res;
    }
    resultFou = cheb2realNaive(resultFou);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cos(M_PI * iz / (n.z - 1.0));
      complex vy_k_z = resultFou[iz];
      auto theory = theoryFou[iz];
      ASSERT_THAT(vy_k_z.real(), ::testing::DoubleNear(theory.real(), 1e-11))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
      ASSERT_THAT(vy_k_z.imag(), ::testing::DoubleNear(theory.imag(), 1e-11))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
    }
  }
}

auto naiveDFTGauss(real2 k, real2 L, int2 n) {
  complex res{};
  for (int i = 0; i < n.x * n.y; i++) {
    real x = ((i % n.x) / real(n.x)) * L.x;
    real y = (((i / n.x) % n.y) / real(n.y)) * L.y;
    res += exp(-(x * x + y * y) * 0.25) *
           exp(-complex{0, 1} * (k.x * x + k.y * y)) / (n.x * n.y);
  }
  return res;
}

TEST(FourierChebyshevTransform, TransformOfGaussianOfZ3DIsCorrect) {
  real H = 1;
  real3 L = {H, H, H};
  int3 n{18, 16, 3}; // Arbitrary size
  cached_vector<complex> fxyz(n.x * n.y * n.z);
  for (int ik = 0; ik < n.x * n.y; ik++) {
    auto fx_k = make_interleaved_iterator(fxyz.begin(), ik, n.x * n.y);
    real x = ((ik % n.x) / real(n.x)) * L.x;
    real y = (((ik / n.x) % n.y) / real(n.y)) * L.y;
    for (int i = 0; i < n.z; i++) {
      real z = H * cospi(i / (n.z - 1.0));
      real r2 = x * x + y * y + z * z;
      fx_k[i] = exp(-r2 * 0.25);
    }
  }
  fxyz = fourierChebyshevTransform3DCufft(fxyz, n);
  int nk = n.x * n.y;
  for (int i = 0; i < nk; i++) {
    auto k = waveVectorFromId(i, L.x, L.y, n.x, n.y);
    auto vx_k = make_interleaved_iterator(fxyz.begin(), i, nk);
    std::vector<complex> theoryFou(n.z);
    std::vector<complex> resultFou(n.z);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cospi(iz / (n.z - 1.0));
      // Cannot compare with the theoretical FT of a Gaussian, since the DFT
      //  is actually interpreting an infinite repetition of Gaussians (it is
      //  not a periodic function)
      complex theory =
          exp(-z * z * 0.25) * naiveDFTGauss(k, {L.x, L.y}, {n.x, n.y});
      theoryFou[iz] = theory;
      complex res = vx_k[iz];
      resultFou[iz] = res;
    }
    resultFou = cheb2realNaive(resultFou);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cos(M_PI * iz / (n.z - 1.0));
      complex vy_k_z = resultFou[iz];
      auto theory = theoryFou[iz];
      ASSERT_THAT(vy_k_z.real(), ::testing::DoubleNear(theory.real(), 1e-12))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
      ASSERT_THAT(vy_k_z.imag(), ::testing::DoubleNear(theory.imag(), 1e-12))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
    }
  }
}

TEST(FastChebyshevTransform, TransformOfGaussianOfZ3DIsCorrect) {
  real H = 1;
  real3 L = {H, H, H};
  int3 n{18, 16, 3}; // Arbitrary size
  cached_vector<complex> fxyz(n.x * n.y * n.z);
  for (int ik = 0; ik < n.x * n.y; ik++) {
    auto fx_k = make_interleaved_iterator(fxyz.begin(), ik, n.x * n.y);
    auto k = waveVectorFromId(ik, L.x, L.y, n.x, n.y);
    for (int i = 0; i < n.z; i++) {
      real z = H * cospi(i / (n.z - 1.0));
      real r2 = z * z;
      fx_k[i] = exp(-r2 * 0.25) * exp(complex{0, 1} * (k.x + k.y));
    }
  }
  fxyz = chebyshevTransform3DCufft(fxyz, n);
  int nk = n.x * n.y;
  for (int i = 0; i < nk; i++) {
    auto k = waveVectorFromId(i, L.x, L.y, n.x, n.y);
    auto vx_k = make_interleaved_iterator(fxyz.begin(), i, nk);
    std::vector<complex> theoryFou(n.z);
    std::vector<complex> resultFou(n.z);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cospi(iz / (n.z - 1.0));
      complex theory = exp(-z * z * 0.25) * exp(complex{0, 1} * (k.x + k.y));
      theoryFou[iz] = theory;
      complex res = vx_k[iz];
      resultFou[iz] = res;
    }
    resultFou = cheb2realNaive(resultFou);
    for (int iz = 0; iz < n.z; iz++) {
      real z = H * cos(M_PI * iz / (n.z - 1.0));
      complex vy_k_z = resultFou[iz];
      auto theory = theoryFou[iz];
      ASSERT_THAT(vy_k_z.real(), ::testing::DoubleNear(theory.real(), 1e-12))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
      ASSERT_THAT(vy_k_z.imag(), ::testing::DoubleNear(theory.imag(), 1e-12))
          << "Failed at k[" << i << "] = (" << k.x << "," << k.y << ") z[" << iz
          << "] = " << z << std::endl;
    }
  }
}
