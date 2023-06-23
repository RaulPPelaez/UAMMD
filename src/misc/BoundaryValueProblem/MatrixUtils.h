/*Raul P. Pelaez 2019-2021. Boundary Value Problem Matrix algebra utilities
*/
#ifndef BVP_MATRIX_UTILS_H
#define BVP_MATRIX_UTILS_H
#include "utils/cufftPrecisionAgnostic.h"
#include<global/defines.h>
#include<utils/exception.h>
#include<thrust/pair.h>
#include<vector>
#ifdef USE_MKL
#include<mkl.h>
#else
#include<lapacke.h>
#endif
#include<thrust/complex.h>
namespace uammd{
  namespace BVP{
    using complex = thrust::complex<real>;
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

    template<>
      struct LapackeUAMMD<lapack_complex_float>{
        template<class ...T> static int getrf(T... args){return LAPACKE_cgetrf(args...);}
        template<class ...T> static int getri(T... args){return LAPACKE_cgetri(args...);}
      };
    template<>
      struct LapackeUAMMD<lapack_complex_double>{
        template<class ...T> static int getrf(T... args){return LAPACKE_zgetrf(args...);}
        template<class ...T> static int getri(T... args){return LAPACKE_zgetri(args...);}
      };
    template<>
      struct LapackeUAMMD<cufftComplex_t<float>>{
        static int getrf(int matrix_layout, lapack_int n, lapack_int m,
            cufftComplex_t<float>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_cgetrf(matrix_layout, n, m, (lapack_complex_float*)(a), lda, ipiv);
        }
        static int getri(int matrix_layout, lapack_int n,
            cufftComplex_t<float>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_cgetri(matrix_layout, n, (lapack_complex_float*)(a), lda, ipiv);
        }
      };
    template<>
      struct LapackeUAMMD<cufftComplex_t<double>>{
        static int getrf(int matrix_layout, lapack_int n, lapack_int m,
            cufftComplex_t<double>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_zgetrf(matrix_layout, n, m, (lapack_complex_double*)(a), lda, ipiv);
        }
        static int getri(int matrix_layout, lapack_int n,
            cufftComplex_t<double>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_zgetri(matrix_layout, n, (lapack_complex_double*)(a), lda, ipiv);
        }
      };
    template<>
      struct LapackeUAMMD<thrust::complex<float>>{
        static int getrf(int matrix_layout, lapack_int n, lapack_int m,
            thrust::complex<float>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_cgetrf(matrix_layout, n, m, (lapack_complex_float*)(a), lda, ipiv);
        }
        static int getri(int matrix_layout, lapack_int n,
            thrust::complex<float>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_cgetri(matrix_layout, n, (lapack_complex_float*)(a), lda, ipiv);
        }
      };
    template<>
      struct LapackeUAMMD<thrust::complex<double>>{
        static int getrf(int matrix_layout, lapack_int n, lapack_int m,
            thrust::complex<double>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_zgetrf(matrix_layout, n, m, (lapack_complex_double*)(a), lda, ipiv);
        }
        static int getri(int matrix_layout, lapack_int n,
            thrust::complex<double>* a, lapack_int lda,
            lapack_int* ipiv){
          return LAPACKE_zgetri(matrix_layout, n, (lapack_complex_double*)(a), lda, ipiv);
        }
      };


    template<class T>
      std::vector<T> invertSquareMatrix(const std::vector<T> &A, lapack_int N){
        lapack_int pivotArray[N];
        int errorHandler;
        auto invA = A;
        errorHandler = LapackeUAMMD<T>::getrf(LAPACK_ROW_MAJOR, N, N, invA.data(), N, pivotArray);
        if(errorHandler){
          throw std::runtime_error("Lapacke getrf failed with error code: " + std::to_string(errorHandler));
        }
        errorHandler = LapackeUAMMD<T>::getri(LAPACK_ROW_MAJOR, N, invA.data(), N, pivotArray);
        if(errorHandler){
          throw std::runtime_error("Lapacke getri failed with error code: " + std::to_string(errorHandler));
        }
        return invA;
      }

    template<class T, class T2>
      auto matmul(const T &A, int ncol_a, int nrow_a,
          const T2 &B, int ncol_b, int nrow_b){
        using type = typename T::value_type;
        std::vector<type> C;
        C.resize(ncol_b*nrow_a);
        for(int i = 0; i<nrow_a; i++){
          for(int j = 0; j<ncol_b; j++){
            type tmp{};
            for(int k = 0; k<ncol_a; k++){
              tmp += A[k+ncol_a*i] * B [j+ncol_b*k];
            }
            C[j+ncol_b*i] = tmp;
          }
        }
        return C;
      }
    //Solves the linear system A*x = b
    //Given A (2x2 matrix) and b as two numbers of an arbitrary type (could be complex, real...)
    template<class T, class T2>
    __device__ thrust::pair<T,T> solve2x2System(T2 A[4], thrust::pair<T,T> b){
      const auto det  = A[0]*A[3] - A[1]*A[2];
      const T c0 = (b.first*A[3] - A[1]*b.second)/det;
      const T d0 = (b.second*A[0] - b.first*A[2])/det;
      return thrust::make_pair(c0, d0);
    }
    
    template<class T>
    __device__ thrust::pair<T,T> solve2x2System(real4 A, thrust::pair<T,T> b){
      const real det  = A.x*A.w - A.y*A.z;
      const T c0 = (b.first*A.w - A.y*b.second)/det;
      const T d0 = (b.second*A.x - b.first*A.z)/det;
      return thrust::make_pair(c0, d0);
    }

  }
}
#endif
