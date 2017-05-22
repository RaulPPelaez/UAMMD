
#ifndef LAPACK_AND_BLAS_DEFINES_H
#define LAPACK_AND_BLAS_DEFINES_H

#ifdef SINGLE_PRECISION

#define LAPACKE_steqr LAPACKE_ssteqr
#define cblas_gemv  cblas_sgemv

#else

#define LAPACKE_steqr LAPACKE_dsteqr
#define cblas_gemv  cblas_dgemv

#endif

#endif
