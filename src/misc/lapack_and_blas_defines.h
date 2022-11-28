
#ifndef LAPACK_AND_BLAS_DEFINES_H
#define LAPACK_AND_BLAS_DEFINES_H
#ifdef USE_MKL
#include<mkl.h>
#else
#include<lapacke.h>
#include<cblas.h>
#endif
#ifdef SINGLE_PRECISION
#define LAPACKE_steqr LAPACKE_ssteqr
#define cblas_gemv  cblas_sgemv
#define cblas_axpy  cblas_saxpy
#define cblas_scal  cblas_sscal
#define cblas_nrm2  cblas_snrm2
#define cblas_dot  cblas_sdot
#else
#define LAPACKE_steqr LAPACKE_dsteqr
#define cblas_gemv  cblas_dgemv
#define cblas_axpy  cblas_daxpy
#define cblas_scal  cblas_dscal
#define cblas_nrm2  cblas_dnrm2
#define cblas_dot  cblas_ddot
#endif
#endif
