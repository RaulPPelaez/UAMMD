/*Raul P. Pelaez 2022. Tests for the Lanczos algorithm.
  Tests the result of sqrt(M)*v for increasingly complex matrices and vectors of several sizes.
 */
#include <fstream>
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include"misc/LanczosAlgorithm.cuh"
#include <memory>
#include<random>
#include<thrust/host_vector.h>

using namespace uammd;

TEST(Lanczos, CanBeCreated){
  auto lanczos = std::make_shared<lanczos::Solver>();
}

//Encodes the matrix dot product of the diagonal matrix M_{ij} = (i+1)\delta_{ij}
struct IdentityMatrixDot: public lanczos::MatrixDot{
  real mval;

  IdentityMatrixDot(real val = 1):mval(val){}

  void operator()(real* v, real*mv) override{
    auto cit = thrust::make_constant_iterator<real>(mval);
    thrust::transform(thrust::cuda::par,
		      cit, cit + m_size,
		      v,
		      mv,
		      thrust::multiplies<real>());
  }

};

TEST(Lanczos, WorksForIdentityMatrixAndVectorOfOnes){
  real tol = 1e-7;
  auto lanczos = std::make_shared<lanczos::Solver>();
  IdentityMatrixDot dot;
  for(int size = 1; size<128; size++){
    thrust::device_vector<real> Mv(size);
    thrust::fill(Mv.begin(), Mv.end(), real());
    thrust::device_vector<real> v(size);
    thrust::fill(v.begin(), v.end(), 1);
    lanczos->run(dot, Mv.data().get(), v.data().get(), tol, size);
    thrust::host_vector<real> h_mv = Mv;
    thrust::host_vector<real> h_v = v;
    for(int i = 0; i<size; i++){
      real h_mv_i = h_mv[i];
      real theory = h_v[i];
      ASSERT_THAT(h_mv_i, ::testing::DoubleNear(theory, tol))<<
	"Failed at index "<<i<<" for size "<<size<<" after "<<lanczos->getLastRunRequiredSteps()<<" steps"<<std::endl;
    }
  }
}


TEST(Lanczos, WorksForIdentityMatrixOfTwosAndVectorOfOnes){
  real tol = 1e-7;
  auto lanczos = std::make_shared<lanczos::Solver>();
  IdentityMatrixDot dot(2);
  for(int size = 1; size<128; size++){
    thrust::device_vector<real> Mv(size);
    thrust::fill(Mv.begin(), Mv.end(), real());
    thrust::device_vector<real> v(size);
    thrust::fill(v.begin(), v.end(), 1);
    lanczos->run(dot, Mv.data().get(), v.data().get(), tol, size);
    thrust::host_vector<real> h_mv = Mv;
    thrust::host_vector<real> h_v = v;
    for(int i = 0; i<size; i++){
      real h_mv_i = h_mv[i];
      real theory = sqrt(2)*h_v[i];
      ASSERT_THAT(h_mv_i, ::testing::DoubleNear(theory, tol))<<
	"Failed at index "<<i<<" for size "<<size<<" after "<<lanczos->getLastRunRequiredSteps()<<" steps"<<std::endl;
    }
  }
}

TEST(Lanczos, IdentityMatrixOfTwosInUnder5Steps){
  real tol = 1e-7;
  auto lanczos = std::make_shared<lanczos::Solver>();
  IdentityMatrixDot dot(2);
  for(int size = 1; size<128; size++){
    thrust::device_vector<real> Mv(size);
    thrust::fill(Mv.begin(), Mv.end(), real());
    thrust::device_vector<real> v(size);
    thrust::fill(v.begin(), v.end(), 1);
    lanczos->run(dot, Mv.data().get(), v.data().get(), tol, size);
    thrust::host_vector<real> h_mv = Mv;
    thrust::host_vector<real> h_v = v;
    int steps = lanczos->getLastRunRequiredSteps();
    ASSERT_LE(steps, 5)<<"Failed for size "<<size;
  }
}

//Encodes the matrix dot product of the diagonal matrix M_{ij} = m(i)\delta_{ij}
struct DiagonalMatrixDot: public lanczos::MatrixDot{

  real*m;
  DiagonalMatrixDot(real* m):m(m){}
  void operator()(real* v, real*mv) override{
    thrust::transform(thrust::cuda::par,
		      m, m + m_size,
		      v,
		      mv,
		      thrust::multiplies<real>());
  }

};

TEST(Lanczos, WorksForDiagonalMatrixAndVectorOfOnes){
  real tol = 1e-7;
  auto lanczos = std::make_shared<lanczos::Solver>();
  int maxSize = 128;
  thrust::device_vector<real> m(maxSize);
  std::mt19937 mersenne_engine {29374238};  // Generates random integers
  std::uniform_real_distribution<real> dist {1,2};
  auto gen = [&](){return dist(mersenne_engine);};
  std::vector<real> vecm(maxSize);
  std::generate(begin(vecm), end(vecm), gen);
  m = vecm;
  DiagonalMatrixDot dot(m.data().get());
  for(int size = 1; size<maxSize; size++){
    thrust::device_vector<real> Mv(size);
    thrust::fill(Mv.begin(), Mv.end(), real());
    thrust::device_vector<real> v(size);
    thrust::fill(v.begin(), v.end(), 1);
    lanczos->run(dot, Mv.data().get(), v.data().get(), tol, size);
    thrust::host_vector<real> h_mv = Mv;
    thrust::host_vector<real> h_v = v;
    for(int i = 0; i<size; i++){
      real h_mv_i = h_mv[i];
      real theory = sqrt(vecm[i])*h_v[i];
      real err = abs((h_mv_i-theory)/theory);
      if(theory == 0) err = abs(h_mv_i);
      ASSERT_LE(err, tol)<<
	"Failed at index "<<i<<" for size "<<size<<" after "<<lanczos->getLastRunRequiredSteps()<<" steps, found "<<
	h_mv_i<<" while expecting "<< theory<<std::endl;
    }
  }
}

TEST(Lanczos, WorksForDiagonalMatrixAndRandomVector){
  real tol = 1e-7;
  auto lanczos = std::make_shared<lanczos::Solver>();
  int maxSize = 128;
  thrust::device_vector<real> m(maxSize);
  std::mt19937 mersenne_engine {29374238};  // Generates random integers
  std::uniform_real_distribution<real> dist {1,2};
  auto gen = [&](){return dist(mersenne_engine);};
  std::vector<real> vecm(maxSize);
  std::generate(begin(vecm), end(vecm), gen);
  m = vecm;
  DiagonalMatrixDot dot(m.data().get());
  for(int size = 1; size<maxSize; size++){
    thrust::device_vector<real> Mv(size);
    thrust::fill(Mv.begin(), Mv.end(), real());
    thrust::device_vector<real> v(size);
    std::mt19937 mersenne_engine {1234567};  // Generates random integers
    std::uniform_real_distribution<real> dist {-10,10};
    auto gen = [&](){return dist(mersenne_engine);};
    std::vector<real> vec(size);
    std::generate(begin(vec), end(vec), gen);
    v = vec;
    lanczos->run(dot, Mv.data().get(), v.data().get(), tol, size);
    thrust::host_vector<real> h_mv = Mv;
    thrust::host_vector<real> h_v = v;
    for(int i = 0; i<size; i++){
      real h_mv_i = h_mv[i];
      real theory = sqrt(vecm[i])*h_v[i];
      real err = abs((h_mv_i-theory)/theory);
      if(theory == 0) err = abs(h_mv_i);
      ASSERT_LE(err, tol)<<
	"Failed at index "<<i<<" for size "<<size<<" after "<<lanczos->getLastRunRequiredSteps()<<" steps, found "<<
	h_mv_i<<" while expecting "<< theory<<std::endl;
    }
  }
}

struct DenseMatrixDot: public lanczos::MatrixDot{

  real*m;
  cublasHandle_t handle;
  DenseMatrixDot(real* m):m(m){
    cublasCreate_v2(&handle);
  }

  ~DenseMatrixDot(){
    cublasDestroy_v2(handle);
  }

  void operator()(real* v, real*mv) override{
    real alpha = 1;
    real beta = 0;
    cublasgemv(handle, CUBLAS_OP_N, m_size, m_size, &alpha, m, m_size, v, 1, &beta, mv, 1);
  }

};

auto generateRandomSymmetricPositiveMatrix(int n){
  std::mt19937 mersenne_engine {29374238};  // Generates random integers
  std::uniform_real_distribution<real> dist {0,1};
  auto gen = [&](){return dist(mersenne_engine);};
  std::vector<real> vecm(n*n);
  std::generate(begin(vecm), end(vecm), gen);
  auto vecmp = vecm;
  for(int i= 0;i<n;i++)
    for(int j= 0;j<n;j++){
      vecmp[i+n*j] = 0.5*(vecm[i+n*j] + vecm[j+n*i]) + 5*n*(i==j);
    }
  return vecmp;
}

auto computeSquaredMatrix(thrust::device_vector<real> &M, int n){
  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  auto M2 = M;
  real alpha = 1;
  real beta = 0;
  cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
		 &alpha, M.data().get(), n, M.data().get(), n, &beta, M2.data().get(), n);
  cublasDestroy_v2(handle);
  return M2;
}

auto computeMatrixVectorProduct(thrust::device_vector<real> &M, thrust::device_vector<real> &v, int n){
  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  auto mv = v;
  real alpha = 1;
  real beta = 0;
  cublasgemv(handle, CUBLAS_OP_N, n,n, &alpha, M.data().get(), n, v.data().get(), 1, &beta, mv.data().get(), 1);
  cublasDestroy_v2(handle);
  return mv;
}

TEST(Lanczos, WorksForDensePositiveDefiniteMatrixAndRandomVector){
  real tol = 1e-7;
  auto lanczos = std::make_shared<lanczos::Solver>();
  int maxSize = 512;
  for(int size = 1; size<maxSize; size++){
    auto vecm = generateRandomSymmetricPositiveMatrix(size);
    thrust::device_vector<real> m(vecm);
    auto m2 = computeSquaredMatrix(m, size);
    DenseMatrixDot dot(m2.data().get());
    thrust::device_vector<real> Mv(size);
    thrust::fill(Mv.begin(), Mv.end(), real());
    thrust::device_vector<real> v(size);
    std::mt19937 mersenne_engine {1234567};  // Generates random integers
    std::uniform_real_distribution<real> dist {-10,10};
    auto gen = [&](){return dist(mersenne_engine);};
    std::vector<real> vec(size);
    std::generate(begin(vec), end(vec), gen);
    v = vec;
    lanczos->run(dot, Mv.data().get(), v.data().get(), tol, size);
    thrust::host_vector<real> h_mv = Mv;
    thrust::host_vector<real> theo = computeMatrixVectorProduct(m, v, size);
    for(int i = 0; i<size; i++){
      real h_mv_i = h_mv[i];
      real theory = theo[i];
      real err = abs((h_mv_i-theory)/theory);
      if(theory == 0) err = abs(h_mv_i);
      ASSERT_LE(err, tol)<<
	"Failed at index "<<i<<" for size "<<size<<" after "<<lanczos->getLastRunRequiredSteps()<<" steps, found "<<
	h_mv_i<<" while expecting "<< theory<<std::endl;
    }
  }
}
