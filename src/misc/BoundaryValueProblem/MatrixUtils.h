/*Raul P. Pelaez 2019. Boundary Value Problem Matrix algebra utilities
 */
#ifndef BVP_MATRIX_UTILS_H
#define BVP_MATRIX_UTILS_H
#include<global/defines.h>
#include<utils/exception.h>
#include<vector>
#ifdef USE_MKL
#include<mkl.h>
#else
#include<lapacke.h>
#endif

namespace uammd{
  namespace BVP{

    template<class T> struct LapackeUAMMD;
    template<>
    struct LapackeUAMMD<float>{
      template<class ...T> static int getrf(T... args){return LAPACKE_sgetrf(args...);}
      template<class ...T> static int getri(T... args){return LAPACKE_sgetri(args...);}
    };
    template<>
    struct LapackeUAMMD<double>{
      template<class ...T> static int getrf(T... args){return LAPACKE_dgetrf(args...);}
      template<class ...T> static int getri(T... args){return LAPACKE_dgetri(args...);}
    };

    std::vector<real> invertSquareMatrix(const std::vector<real> &A, lapack_int N){
      lapack_int pivotArray[N];
      int errorHandler;
      auto invA = A;
      errorHandler = LapackeUAMMD<real>::getrf(LAPACK_ROW_MAJOR, N, N, invA.data(), N, pivotArray);
      if(errorHandler){
	throw std::runtime_error("Lapacke getrf failed with error code: " + std::to_string(errorHandler));
      }
      errorHandler = LapackeUAMMD<real>::getri(LAPACK_ROW_MAJOR, N, invA.data(), N, pivotArray);
      if(errorHandler){
	throw std::runtime_error("Lapacke getri failed with error code: " + std::to_string(errorHandler));
      }
      return invA;
    }

    template<class T, class T2>
    std::vector<typename T::value_type> matmul(const T &A, int ncol_a, int nrow_a,
					       const T2 &B, int ncol_b, int nrow_b){
      std::vector<typename T::value_type> C;
      C.resize(ncol_b*nrow_a);
      for(int i = 0; i<nrow_a; i++){
	for(int j = 0; j<ncol_b; j++){
	  real tmp = 0;
	  for(int k = 0; k<ncol_a; k++){
	    tmp += A[k+ncol_a*i] * B [j+ncol_b*k];
	  }
	  C[j+ncol_b*i] = tmp;
	}
      }
      return C;
    }

  }
}
#endif
