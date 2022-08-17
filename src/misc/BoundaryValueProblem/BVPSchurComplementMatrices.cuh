#ifndef BVPSCHURCOMPLEMENTMATRICES_CUH
#define BVPSCHURCOMPLEMENTMATRICES_CUH
#include"global/defines.h"
#include<vector>
#include"MatrixUtils.h"
namespace uammd{

  namespace BVP{

    class SecondIntegralMatrix{
      int nz;
    public:

      __host__ __device__ SecondIntegralMatrix(int nz): nz(nz){}

      real getElement(int i, int j){
	if(i == nz and j== 0) return 1; //c0
	if(i == nz+1 and j==1) return 1; //d0
	if(i >= nz or j >= nz) return 0;
	if(std::max(i,j) >= 3){
	  if(i==j) return -(0.5/i*(1/(2.0*i-2) + (i<nz-1)/(2.0*i+2)));
	  if(i==j+2) return 1.0/(2.0*j)*(j<nz-2)/(2.0*j+2);
	  if(i==j-2) return 1.0/(2.0*j)*1.0/(2.0*j-2);
	}
	else if(i==j) return -((i>0)/8.0+(i>1)/24.0);
	else if(j==2 and i==0) return 1.0/4.0;
	else if(j==2 and i==4) return 1.0/24.0;
	else return 0;
	return 0;
      }

      template<class T, class AnIterator>
      __host__ __device__ T computeSecondIntegralCoefficient(int i,  const AnIterator &a, T c0, T d0){
	if(i==0) return c0;
	else if(i==1) return d0 - real(1/8.0)*(a[1]-a[3]);
	else if(i==2) return real(0.25)*(real(0.5)*(real(2.0)*a[i-2]-a[i]) - real(1/6.0)*(a[i] - a[i+2]));
	const T di_m1 = real(0.5)/((i-real(1.0)))*(a[i-2] - a[i]);
	T di_p1 = T();
	if(i<nz-1) di_p1 = real(0.5)/((i+real(1.0)))*(a[i] - ((i<nz-2)?a[i+2]:T()) );
	const T ci = real(0.5)/i*( di_m1 - di_p1);
	return ci;
      }

    };

    class FirstIntegralMatrix{
      int nz;

    public:
      FirstIntegralMatrix(int nz):nz(nz){}

      real getElement(int i, int j){
	if(i==nz+1 and j==0) return 1; //d0
	if(i>=nz or j>=nz) return 0;
	if(j==1 and i==0) return 1;
	if(j==1 and i==2) return -0.5;
	if(j>1){
	  if(i==j-1) return 1.0/(2.0*j);
	  if(i==j+1 and i<=nz-1) return -1.0/(2.0*j);
	}
	return 0;
      }

    };

    class BCRows{
      int nz;
    public:
      BCRows(int nz): nz(nz){}

      std::vector<real> topFirstIntegral(){
	return std::move(firstIntegralCoefficients());
      }

      std::vector<real> topSecondIntegral(){
	return std::move(secondIntegralCoefficients());
      }

      std::vector<real> bottomFirstIntegral(){
	auto bfd = firstIntegralCoefficients();
	for(int i = 0; i<nz; i++){
	  bfd[i] *= pow(-1, i+1);
	}
	return std::move(bfd);
      }

      std::vector<real> bottomSecondIntegral(){
	auto si = secondIntegralCoefficients();
	for(int i = 0; i<nz; i++){
	  si[i] *= -pow(-1, i+1);
	}
	si[nz+1] = -1;
	return std::move(si);
      }

    private:
      std::vector<real> firstIntegralCoefficients(){
	std::vector<real> coefficients(nz+2, 0);
	FirstIntegralMatrix fdm(nz);
	for(int i = 0; i<nz+2; i++){
	  for(int j = 0; j<nz+2; j++){
	    coefficients[i] += fdm.getElement(i,j);
	  }
	}
	return std::move(coefficients);
      }

      std::vector<real> secondIntegralCoefficients(){
	std::vector<real> coefficients(nz+2, 0);
	SecondIntegralMatrix sim(nz);
	for(int i = 0; i<nz+2; i++){
	  for(int j = 0; j<nz+2; j++){
	    coefficients[i] += sim.getElement(i,j);
	  }
	}
	return std::move(coefficients);
      }
    };

    class SchurBoundaryCondition{
      int nz;
      real H;
      BCRows bcs;
    public:
      SchurBoundaryCondition(int nz, real H):nz(nz), H(H), bcs(nz){}

      template<class TopBC, class BottomBC>
      auto computeBoundaryConditionMatrix(const TopBC &top, const BottomBC &bottom){
	std::vector<complex> CandD(2*nz+4, 0);
	auto topRow = computeTopRow(top, bottom);
	auto bottomRow = computeBottomRow(top, bottom);
	std::copy(topRow.begin(), topRow.end()-2, CandD.begin());
	std::copy(bottomRow.begin(), bottomRow.end()-2, CandD.begin() + nz);
	fori(0, 2){
	  CandD[2*nz + i] = topRow[i+nz];
	  CandD[2*nz + 2 + i] = bottomRow[i+nz];
	}
	return CandD;
      }

    private:

      template<class TopBC, class BottomBC>
      auto computeTopRow(const TopBC &top, const BottomBC &bottom){
	std::vector<complex> topRow(nz+2, 0);
	auto tfi = bcs.topFirstIntegral();
	auto tsi = bcs.topSecondIntegral();
	auto tfiFactor = top.getFirstIntegralFactor();
	auto tsiFactor = top.getSecondIntegralFactor();
	fori(0,nz+2){
	  auto topSecondIntegral = tsiFactor*tsi[i];
	  auto topFirstIntegral = tfiFactor*tfi[i];
	  topRow[i] = topFirstIntegral + topSecondIntegral;
	}
	return std::move(topRow);
      }

      template<class TopBC, class BottomBC>
      auto computeBottomRow(const TopBC &top, const BottomBC &bottom){
	std::vector<complex> bottomRow(nz+2, 0);
	auto bfi = bcs.bottomFirstIntegral();
	auto bsi = bcs.bottomSecondIntegral();
	auto bfiFactor = bottom.getFirstIntegralFactor();
	auto bsiFactor = bottom.getSecondIntegralFactor();
	fori(0,nz+2){
	  auto bottomSecondIntegral = bsiFactor*bsi[i];
	  auto bottomFirstIntegral = bfiFactor*bfi[i];
	  bottomRow[i] = bottomFirstIntegral + bottomSecondIntegral;
	}
	return std::move(bottomRow);
      }

    };

    auto computeSecondIntegralMatrix(complex k, real H, int nz){
      std::vector<complex> A(nz*nz, complex());
      SecondIntegralMatrix sim(nz);
      auto kH2 = k*k*H*H;
      fori(0, nz){
	forj(0,nz){
	  A[i+nz*j] = (i==j)?real(1.0):real(0.0) - kH2*sim.getElement(i, j);
	}
      }
      return std::move(A);
    }

    auto computeInverseSecondIntegralMatrix(complex k, real H, int nz){
      if(k.real()==0 and k.imag() == 0){
	std::vector<complex> invA(nz*nz, 0);
	fori(0, nz){
	  invA[i+nz*i] = 1;
	}
	return std::move(invA);
      }
      else{
	auto A = computeSecondIntegralMatrix(k, H, nz);
	auto invA = BVP::invertSquareMatrix(A, nz);
	return std::move(invA);
      }
    }


  }
}
#endif
